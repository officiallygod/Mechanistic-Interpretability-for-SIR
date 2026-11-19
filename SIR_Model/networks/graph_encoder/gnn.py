import hashlib
import hydra
import logging
import numpy as np
from omegaconf import OmegaConf
from torch import nn, Tensor
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.nn.norm import LayerNorm
from torch_geometric.typing import Adj, OptTensor, PairTensor, SparseTensor
from torch_geometric.utils import add_self_loops, remove_self_loops, is_torch_sparse_tensor, softmax, to_dense_batch
from torch_geometric.utils.sparse import set_sparse_value
try:
    import torch_sparse
except ImportError:
    torch_sparse = None

from networks.graph_encoder.utils import debatch_graphs_masks, soft_histogram_loss, soft_histrogram_except_middle_loss
from networks.transformer.transformer_decoders import TransformerFiLMDecoder
from utils.generate_graph_dataset_robocasa import get_num_relevant_nodes_per_task, is_node_relevant_for_task
from utils.node_scores import load_node_whitelist

log = logging.getLogger(__name__)


def adapt_alpha_with_edge_weight(alpha, edge_index, edge_attr, edge_weight):
    if edge_weight is None and edge_attr is not None and edge_attr.dim() == 1:
        edge_weight = edge_attr
    elif edge_weight is None:
        print("Warning: No edge weights provided, using default edge weights of 1.0.")

    if edge_weight is not None:
        if edge_weight.shape[0] != edge_index.shape[1]:
            # because of adding of self loops
            edge_weight_sl = torch.ones(edge_index.shape[1], device=edge_index.device, dtype=edge_weight.dtype)
            edge_weight_sl[edge_index[0] != edge_index[1]] = edge_weight
        else:
            edge_weight_sl = edge_weight
        alpha = alpha * edge_weight_sl.unsqueeze(1)

    return alpha


def renormalize_alpha(alpha: Tensor, edge_index: Tensor, num_nodes: int) -> Tensor:
    """Rescale gated attention so each target node's incoming coefficients sum to 1.

    `softmax` in `edge_update` runs before the edge weight is applied, so the gated
    coefficients no longer form a distribution: a pruned neighbour's share of the mass
    is simply deleted rather than redistributed, and the surviving neighbours' messages
    are scaled down by however much the pruned node had claimed. That residual
    dependence on the pruned node is the leak. Dividing by the surviving mass is exactly
    a softmax taken over the surviving edges only.

    See the NOT-TRAINABLE warning in WeightedGATv2Conv.__init__ before enabling.
    """
    target = edge_index[1]
    denom = torch.zeros(num_nodes, alpha.size(1), dtype=alpha.dtype, device=alpha.device)
    denom.index_add_(0, target, alpha)
    # A target whose incoming edges were all gated to zero keeps zero, not NaN.
    return alpha / denom.clamp(min=1e-16)[target]


class WeightedGATv2Conv(GATv2Conv):
    """GATv2Conv extended with a multiplicative edge_weight on attention scores.

    After the standard attention is computed, each edge's attention coefficient
    is multiplied by the corresponding edge_weight. This lets the sparsification
    module gate messages in a fully differentiable way via probs[src]*probs[dst].
    """

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        edge_dim: Optional[int] = None,
        fill_value: Union[float, Tensor, str] = 1.0,
        bias: bool = True,
        share_weights: bool = False,
        residual: bool = False,
        attn_renorm: bool = False,
        **kwargs,
    ):
        super().__init__(in_channels=in_channels, out_channels=out_channels, heads=heads, concat=concat,
                         negative_slope=negative_slope, dropout=dropout, add_self_loops=add_self_loops,
                         edge_dim=edge_dim, fill_value=fill_value, bias=bias, share_weights=share_weights,
                         residual=residual, **kwargs)
        # Leak path 2: `edge_update` softmaxes the attention logits BEFORE the edge
        # weight gates them, so a pruned node's logit stays in its neighbours'
        # softmax denominator even when its edge weight is zero. Renormalising after
        # gating is algebraically a softmax restricted to the surviving edges.
        #
        # !! NOT TRAINABLE AS IT STANDS -- ANALYSIS AND EVALUATION ONLY !!
        # A pruned target node's denominator is only its own self-loop attention, so
        # the backward blows up (measured at ~7e11x alone, ~7e4x even with
        # norm_mode='node'). For a leak-free TRAINING path use `true_removal`, which
        # drops the pruned nodes' edges instead of gating them.
        self.attn_renorm = attn_renorm

    def forward(
        self,
        x: Union[Tensor, PairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        edge_weight: OptTensor = None,
        return_attention_weights: Optional[bool] = None,
    ) -> Union[Tensor, Tuple[Tensor, Tuple[Tensor, Tensor]], Tuple[Tensor, SparseTensor]]:
        H, C = self.heads, self.out_channels

        res: Optional[Tensor] = None

        x_l: OptTensor = None
        x_r: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2

            if self.res is not None:
                res = self.res(x)

            x_l = self.lin_l(x).view(-1, H, C)
            if self.share_weights:
                x_r = x_l
            else:
                x_r = self.lin_r(x).view(-1, H, C)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2

            if x_r is not None and self.res is not None:
                res = self.res(x_r)

            x_l = self.lin_l(x_l).view(-1, H, C)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)

        assert x_l is not None
        assert x_r is not None

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(edge_index, edge_attr, fill_value=self.fill_value,
                                                       num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = torch_sparse.set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' simultaneously is currently "
                        "not yet supported for 'edge_index' in a 'SparseTensor' form")

        alpha = self.edge_updater(edge_index, x=(x_l, x_r), edge_attr=edge_attr)
        alpha = adapt_alpha_with_edge_weight(alpha, edge_index, edge_attr, edge_weight)
        if self.attn_renorm:
            alpha = renormalize_alpha(alpha, edge_index, num_nodes=x_l.size(0))

        out = self.propagate(edge_index, x=(x_l, x_r), alpha=alpha)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if res is not None:
            out = out + res

        if self.bias is not None:
            out = out + self.bias

        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                if is_torch_sparse_tensor(edge_index):
                    adj = set_sparse_value(edge_index, alpha)
                    return out, (adj, alpha)
                else:
                    return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def edge_update(self, x_j: Tensor, x_i: Tensor, edge_attr: OptTensor,
                    index: Tensor, ptr: OptTensor, dim_size: Optional[int]) -> Tensor:
        x = x_i + x_j

        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            assert self.lin_edge is not None
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            x = x + edge_attr

        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1)
        alpha = softmax(alpha, index, ptr, dim_size)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha

    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        return x_j * alpha.unsqueeze(-1)


class Multi_GNN(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 edge_dim,
                 num_layer,
                 layer_name,
                 pool_name,
                 heads,
                 dropout,
                 modalities,
                 norm_mode="graph",
                 attn_renorm=False,
                ) -> None:
        super(Multi_GNN, self).__init__()

        self.models = nn.ModuleDict()
        for mod in modalities:
            self.models[mod] = GNN(input_dim[mod], hidden_dim, output_dim, edge_dim, num_layer, layer_name, pool_name, heads, dropout,
                                   norm_mode=norm_mode, attn_renorm=attn_renorm)

    def forward(self, input, lang_emb=None):
        # lang_emb is accepted but unused, so the caller can invoke Multi_GNN and
        # Multi_XAI_GNN through the same signature.
        out = {}
        for key in input:
            out[key] = self.models[key](input[key])
        return out

class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, edge_dim, num_layer, layer_name, pool_name, heads, dropout,
                 norm_mode: str = "graph", attn_renorm: bool = False) -> None:
        """norm_mode: 'graph' (shipped) | 'node'.

        PyG's LayerNorm defaults to mode='graph', which reduces mean/std ACROSS nodes,
        so a pruned node's activations enter every surviving node's normalisation --
        leak path 1. Worse, these norms are called without `batch`, so 'graph' mode
        pools over the whole batch tensor and a graph's embedding depends on its
        batch-mates (train batch vs a rollout batch mixing several envs is then a
        train/eval asymmetry). mode='node' normalises each node over its own feature
        dim and is equivalent to torch.nn.LayerNorm.

        Default stays 'graph' so existing checkpoints re-evaluate as they trained --
        an eval-only run reloads the checkpoint's method config, which for anything
        trained before this change carries no norm_mode at all.
        """
        super(GNN, self).__init__()

        if norm_mode not in ("graph", "node"):
            raise ValueError(f"norm_mode must be 'graph' or 'node', got {norm_mode!r}")
        self.norm_mode = norm_mode
        self.attn_renorm = attn_renorm
        if attn_renorm and layer_name != "WeightedGATv2":
            raise ValueError(f"attn_renorm needs layer_name='WeightedGATv2', got {layer_name!r}")

        if layer_name == "GATv2":
            self.layer_class = GATv2Conv
        elif layer_name == "WeightedGATv2":
            self.layer_class = WeightedGATv2Conv
        else:
            raise ValueError(f"Unsupported GNN layer: {layer_name}")

        if pool_name == "mean":
            self.pooling = global_mean_pool
        elif pool_name == "add":
            self.pooling = global_add_pool
        elif pool_name == "max":
            self.pooling = global_max_pool
        elif pool_name == "none":
            # Per-node readout: skip pooling and emit one embedding per node, so the
            # action generator can cross-attend to each object separately instead of
            # to a single vector summarizing the whole scene.
            self.pooling = None
        else:
            raise ValueError(f"Unsupported pooling method: {pool_name}")

        extra = {"attn_renorm": attn_renorm} if layer_name == "WeightedGATv2" else {}
        if layer_name in ("GATv2", "WeightedGATv2"):
            self.input_layer = self.layer_class(input_dim, hidden_dim, edge_dim=edge_dim, heads=heads, dropout=dropout, concat=False, **extra)

        self.input_norm = LayerNorm(hidden_dim, mode=norm_mode)
        self.input_projection_layer = nn.Linear(input_dim, hidden_dim)

        self.layers = nn.ModuleList()
        for _ in range(num_layer):
            if layer_name in ("GATv2", "WeightedGATv2"):
                self.layers.append(self.layer_class(hidden_dim, hidden_dim, edge_dim=edge_dim, heads=heads, dropout=dropout, concat=False, **extra))

        self.norms = nn.ModuleList()
        for _ in range(num_layer):
            self.norms.append(LayerNorm(hidden_dim, mode=norm_mode))

        self.output = nn.Linear(hidden_dim, output_dim)

        self.leaky_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Initialize weights of Linear and Embedding layers
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, GATv2Conv):
            # Initialize GATv2Conv layers (attention and projections)
            if hasattr(module, 'att'):
                nn.init.xavier_uniform_(module.att, gain=1.0)  # Attention coefficients

            # GATv2Conv typically has its projection weights internally as Linear layers.
            # Use `apply` recursively to initialize them.
            for attr in ('lin_l', 'lin_r'):
                lin = getattr(module, attr, None)
                if lin is None:
                    continue
                # GATv2 builds these lazily when in_channels is inferred; touching an
                # UninitializedParameter raises, so skip it and let PyG initialize it.
                if isinstance(lin.weight, nn.parameter.UninitializedParameter):
                    continue
                nn.init.xavier_uniform_(lin.weight, gain=1.0)
                if lin.bias is not None:
                    nn.init.zeros_(lin.bias)

        elif isinstance(module, nn.LayerNorm):
            # Initialize LayerNorm layers
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(self, batch, node_weights=None):
        x, edge_index = batch.x, batch.edge_index
        edge_weight = batch.edge_attr
        edge_index = edge_index.to(torch.int64)

        # Input projection — WeightedGATv2Conv takes edge_weight as a separate multiplicative kwarg
        if isinstance(self.input_layer, WeightedGATv2Conv):
            input = self.input_layer(x, edge_index, edge_attr=None, edge_weight=edge_weight)
        else:
            input = self.input_layer(x, edge_index, edge_weight)
        input = self.leaky_relu(self.input_norm(self.input_projection_layer(x) + input))
        if self.training:
            input = self.dropout(input)

        x = input
        # Iterating through the layers
        for layer, norm in zip(self.layers, self.norms):
            residual = x
            if isinstance(layer, WeightedGATv2Conv):
                x = layer(x, edge_index, edge_attr=None, edge_weight=edge_weight)
            else:
                x = layer(x, edge_index, edge_weight)
            x = self.leaky_relu(norm(x + residual))
            if self.training:
                x = self.dropout(x)

        output = x
        if node_weights is not None:
            output = output * node_weights.view(-1, 1)

        if self.pooling is None:
            # Per-node readout. to_dense_batch pads every graph out to the largest node
            # count in the batch and returns the mask marking the real nodes.
            dense, mask = to_dense_batch(output, batch.batch)   # [B, N_max, hidden], [B, N_max]
            if node_weights is not None:
                # Nodes the sparsifier zeroed out carry no information, so drop them from
                # the mask too — otherwise they stay attendable as all-zero tokens.
                kept, _ = to_dense_batch((node_weights.view(-1) > 0), batch.batch)
                mask = mask & kept
            embedding = self.output(dense)                      # [B, N_max, output_dim]
            return embedding, mask

        if node_weights is not None and self.pooling is global_max_pool:
            # A removed node is ZEROED, and zero WINS a max over any dimension where every
            # survivor is negative -- so the removed node still reaches the output. Zeroing
            # is not removal for a max, exactly as it is not removal for a softmax
            # denominator. Push them to -inf so they cannot win.
            masked = output.masked_fill((node_weights.view(-1, 1) <= 0), float("-inf"))
            pooling = global_max_pool(masked, batch.batch)
            # A graph with no survivors at all would pool to -inf; fall back to zeros.
            pooling = torch.where(torch.isinf(pooling), torch.zeros_like(pooling), pooling)
        elif node_weights is not None and self.pooling is global_mean_pool:
            # Weighted mean: sum(x*w) / count(w > 0), avoids dropped nodes diluting the pool
            sum_pool = global_add_pool(output, batch.batch)
            count_pool = global_add_pool((node_weights.view(-1, 1) > 0).float(), batch.batch)
            pooling = sum_pool / count_pool.clamp(min=1e-6)
        else:
            pooling = self.pooling(output, batch.batch)

        embedding = self.output(pooling)                        # [B, output_dim]

        return embedding, None


class Sparsification_Module(nn.Module):
    """Language-conditioned node sparsification: TransformerFiLMDecoder scoring + topk_PerTask selection."""

    def __init__(self,
                 in_dim,
                 hidden_dim,
                 num_heads,
                 num_layers,
                 dropout_prob=0.2,
                 sampling_strategy="topk_PerTask",
                 sample_abs=5,
                 threshold=0.5,
                 threshold_sum_weight=0.0,
                 variance_weight=0.0,
                 sum_mask_weight=0.0,
                 target_sum_node_probs=0.0,
                 entropy_reg_weight=0.0,
                 uniform_dist_weight=0.0,
                 uniform_except_middle_weight=0.0,
                 uniform_except_middle_offset=0.0,
                 vgib_noise_kl_weight=0.0,
                 vgib_noise_kl_mode="full",
                 eval_prune_mode="soft",
                 node_whitelist=None,
                 whitelist_bce_weight=0.0,
                 whitelist_pos_weight=None,
                 relevance_whitelist=None,
                 task_thresholds=None,
                 # TRUE REMOVAL: zero the pruned features AND drop every edge touching
                 # them, so the subgraph IS the graph. Applies to fixed, topk,
                 # topk_PerTask and threshold -- every strategy that selects a hard subset.
                 # Default False keeps shipped behaviour, under which a pruned node keeps
                 # its features and its share of the kept nodes' attention softmax.
                 # Only exact together with norm_mode='node' on the graph encoder.
                 true_removal=False):
        super().__init__()

        assert sampling_strategy in ("topk_PerTask", "topk", "threshold", "vgib", "fixed"), \
            f"Unsupported sampling_strategy: {sampling_strategy}"
        assert vgib_noise_kl_mode in ("full", "truncated", "eq10"), \
            f"Unsupported vgib_noise_kl_mode: {vgib_noise_kl_mode}"
        assert eval_prune_mode in ("soft", "noise", "hard"), \
            f"Unsupported eval_prune_mode: {eval_prune_mode}"

        self.coarsening_loss = dict()
        self.coarsening_layer = TransformerCoarsening(in_dim, hidden_dim, num_heads, num_layers, dropout_prob)

        self.sampling_strategy = sampling_strategy
        if true_removal and sampling_strategy == "vgib":
            raise ValueError("true_removal has no effect under sampling_strategy='vgib': "
                             "VGIB dissolves dropped nodes into noise instead of selecting "
                             "a hard subset. Use eval_prune_mode='hard' for a pruned graph.")
        self.true_removal = bool(true_removal)
        # Edge mask of the subgraph for the caller to slice `batch.edge_index` with, set
        # only on the true-removal paths and cleared on every call.
        self._edge_keep = None
        self.sample_abs = sample_abs
        self.threshold = threshold
        self.threshold_sum_weight = threshold_sum_weight

        self.variance_weight = variance_weight
        self.sum_mask_weight = sum_mask_weight
        self.target_sum_node_probs = target_sum_node_probs
        self.entropy_reg_weight = entropy_reg_weight
        self.uniform_dist_weight = uniform_dist_weight
        self.uniform_except_middle_weight = uniform_except_middle_weight
        self.uniform_except_middle_offset = uniform_except_middle_offset

        self.vgib_noise_kl_weight = vgib_noise_kl_weight
        self.vgib_noise_kl_mode = vgib_noise_kl_mode
        self.eval_prune_mode = eval_prune_mode

        # Stage-2 subgraph. Derived from stage 1's own unsupervised node scores via
        # `utils.select_subgraph` — never from RELEVENT_NODES, so distilling it back into
        # the scorer does not smuggle the relevance table in. Under 'fixed' it *is* the
        # mask; with whitelist_bce_weight > 0 it is the BCE target while the scorer still
        # runs and decides selection itself.
        self.node_whitelist = load_node_whitelist(node_whitelist)

        # Swaps the SUPERVISION TARGET of the SIR recipe (top-k-per-task + CBM loss) from
        # the hand-written RELEVENT_NODES to a discovered list. It replaces both places
        # the table is consulted -- which nodes the CBM pushes up, and what `k` is per
        # task -- so the recipe is unchanged and only its target moves. That isolates
        # "is the discovered list any good?" from "is VGIB's training mechanism any
        # good?", which no run comparing a VGIB-trained model to SIR can separate.
        #
        # Note what is lost: RELEVENT_NODES is consulted through
        # `is_node_relevant_for_task(name, task, lang_goal)`, which on the four door
        # tasks resolves the door named in THIS episode's instruction (and takes k+1).
        # A per-task list cannot express that, so on those tasks this arm is handicapped
        # by the loss of language grounding, independently of list quality.
        self.relevance_whitelist = load_node_whitelist(relevance_whitelist)
        self.whitelist_bce_weight = whitelist_bce_weight
        self.whitelist_pos_weight = whitelist_pos_weight
        self._last_whitelist_match = None
        self._warned_empty_tasks = set()

        # Eval-only diagnostics, set through Multi_XAI_GNN rather than the config so a
        # checkpoint can be re-evaluated under them without retraining.
        self.graph_swap = 'none'
        # Bump to draw a different random subgraph while keeping it fixed per scene.
        self.graph_swap_seed = 0
        self._warned_unseeded_swap = False
        # Per-task eval cutoff; tasks not listed fall back to the scalar `threshold`.
        # The gap in the score distribution sits in a different place per task (measured
        # 0.878 on CloseDrawer against 0.797 on CoffeePressButton), so one global bar
        # necessarily over-prunes one and under-prunes the other.
        self.task_thresholds = dict(task_thresholds) if task_thresholds else None
        if sampling_strategy == "fixed" and not self.node_whitelist:
            raise ValueError("sampling_strategy 'fixed' requires node_whitelist")
        if whitelist_bce_weight > 0:
            if not self.node_whitelist:
                raise ValueError("whitelist_bce_weight > 0 requires node_whitelist")
            if sampling_strategy == "fixed":
                raise ValueError("whitelist_bce_weight has no effect under 'fixed': the "
                                 "whitelist is already the mask, so there is nothing to "
                                 "supervise. Use sampling_strategy 'threshold'.")

    def forward(self, x, edge_index=None, edge_attr=None, edge_weights=None, lang_emb=None, batch=None, task_names=None):

        self.coarsening_loss = dict()
        # Only the true-removal paths assign this. Left over from a previous call, the
        # consumer would apply it as `batch.edge_index[:, _edge_keep]` against an edge
        # count with no reason to match.
        self._edge_keep = None

        node_mask, edge_mask, num_graphs = debatch_graphs_masks(x, None, batch)

        if self.sampling_strategy == "fixed":
            # Stage 2, frozen variant: the subgraph was decided offline, so the scorer
            # never runs and training, validation and rollout all see the same nodes.
            probs = self._whitelist_probs(x, batch, node_mask, num_graphs, task_names)
            if self.true_removal:
                # TRUE REMOVAL. The shipped path below zeroes nothing and only gates the
                # edge WEIGHTS, which leaves the pruned nodes' real features in the kept
                # nodes' attention softmax -- during training as well as at eval.
                # Here the node is actually gone: its features are zeroed AND every edge
                # with a pruned endpoint is dropped, so it is absent from the softmax
                # denominator too. _whitelist_probs writes literal 0/1, so survivors pass
                # through unattenuated.
                keep = probs > 0
                self._edge_keep = keep[edge_index[0]] & keep[edge_index[1]]
                x_out = x * probs.unsqueeze(-1)
                sampled_nodes = keep.nonzero(as_tuple=False).squeeze(-1)
                ew = None if edge_weights is None else edge_weights[self._edge_keep]
                return ew, sampled_nodes, probs, x_out, probs.clone()
            if edge_weights is not None:
                edge_weights = edge_weights * probs[edge_index[0]] * probs[edge_index[1]]
            sampled_nodes = (probs != 0).nonzero(as_tuple=False).squeeze(-1)
            return edge_weights, sampled_nodes, probs, None, probs.clone()

        scores = self.coarsening_layer(x, edge_index, edge_attr, node_mask, lang_emb, num_graphs)
        probs = torch.sigmoid(scores)

        assert (probs != probs).sum() == 0, "NaN values in probabilities after coarsening layer."

        # The scores before any selection masks them. `probs` is zeroed for dropped nodes
        # below, which would make every dropped node look equally uninteresting — but
        # ranking the *rejected* nodes is exactly what setting a threshold needs.
        raw_probs = probs.clone()

        # Stage-2 supervision: push whitelisted nodes to 1 and the rest to 0. On the
        # logits rather than on `probs` so the gradient survives sigmoid saturation.
        # `pos_weight` matters because graphs carry 18-27 node types while a whitelist
        # keeps 1-5 — at ~19% positives, plain BCE is mostly "push everything down" and a
        # constant output becomes attractive.
        wl_target = None
        if self.whitelist_bce_weight > 0:
            wl_target = self._whitelist_probs(x, batch, node_mask, num_graphs, task_names)
            pos_weight = None
            if self.whitelist_pos_weight:
                pos_weight = torch.tensor(self.whitelist_pos_weight,
                                          device=scores.device, dtype=scores.dtype)
            self.coarsening_loss['whitelist_bce_loss'] = {
                'value': F.binary_cross_entropy_with_logits(scores, wl_target,
                                                            pos_weight=pos_weight),
                'weight': self.whitelist_bce_weight}

        # Compute coarsening losses always — during training they contribute to gradients,
        # during validation they're logged for monitoring, during rollout they're ignored.
        #
        # Under `vgib` the sparsity pressure is the compression KL below, so these are all
        # off by default -- but only ONE of them is actually forbidden there.
        # `soft_histrogram_except_middle_loss` needs `scores_in_upper`, which comes from
        # `get_num_relevant_nodes_per_task`, i.e. from the hand-written relevance table;
        # running it would put that supervision back into a path whose whole claim is that
        # it does not use it. It is also the expensive one -- it sizes itself through
        # `_calc_k_for_topk`, whose `.item()` is a host-device sync once per graph, which
        # cost 115 s/epoch on the 24-task config. So it is forced to zero under vgib
        # regardless of config, and the config comment saying so is load-bearing.
        #
        # The other four read nothing but the scores themselves and are legitimate for an
        # unsupervised sparsifier. `coars_uniform_loss` in particular is the soft histogram
        # that pushes scores toward the ends of [0,1]; it is the only lever we have when the
        # learned distribution sits too bunched for any threshold to separate cleanly.
        # Only the terms with a non-zero weight are computed. `method/diffusion.py` folds
        # each in as `value * weight`, so a zero-weighted term contributes exactly 0.0 to
        # the loss AND logs as exactly 0.0 whether or not it was evaluated -- computing it
        # bought nothing. Skipped terms are still registered at zero so the wandb key set
        # is unchanged.
        legacy_terms = (
            ('coars_variance_loss', self.variance_weight),
            ('coars_sum_mask_loss', self.sum_mask_weight),
            ('coars_entropy_loss', self.entropy_reg_weight),
            ('coars_uniform_loss', self.uniform_dist_weight),
            ('coars_uniform_except_middle_loss',
             0.0 if self.sampling_strategy == "vgib" else self.uniform_except_middle_weight),
        )
        zero = torch.zeros((), device=x.device, dtype=probs.dtype)
        for term_name, weight in legacy_terms:
            if weight == 0:
                self.coarsening_loss[term_name] = {'value': zero, 'weight': weight}

        active = {name for name, weight in legacy_terms if weight != 0}
        for i in range(num_graphs) if active else ():
            p_i = probs[node_mask == i]

            if 'coars_variance_loss' in active:
                l = -torch.var(p_i, unbiased=False) / num_graphs
                self._accumulate_coarsening_loss('coars_variance_loss', l, self.variance_weight)

            if 'coars_sum_mask_loss' in active:
                l = (p_i.sum() - self.target_sum_node_probs).abs() / num_graphs
                self._accumulate_coarsening_loss('coars_sum_mask_loss', l, self.sum_mask_weight)

            if 'coars_entropy_loss' in active:
                l = torch.sum(p_i * torch.log(p_i + 1e-20)) / num_graphs
                self._accumulate_coarsening_loss('coars_entropy_loss', l, self.entropy_reg_weight)

            if 'coars_uniform_loss' in active:
                l = soft_histogram_loss(p_i) / num_graphs
                self._accumulate_coarsening_loss('coars_uniform_loss', l, self.uniform_dist_weight)

            if 'coars_uniform_except_middle_loss' in active:
                l = soft_histrogram_except_middle_loss(
                    p_i,
                    offset=self.uniform_except_middle_offset,
                    scores_in_upper=self._calc_k_for_topk(
                        p_i.numel(),
                        task_name=task_names[i] if task_names else None
                    )
                ) / num_graphs
                self._accumulate_coarsening_loss(
                    'coars_uniform_except_middle_loss', l, self.uniform_except_middle_weight)

        # Selection: topk_PerTask / topk use per-graph top-k; threshold uses probs > threshold.
        # Both paths apply the straight-through estimator so gradients flow back into the scorer.
        if self.sampling_strategy in ("topk_PerTask", "topk"):
            # Per-graph top-k without a Python loop. The loop this replaces called
            # `.item()` once per graph to size k, and every one of those is a
            # host-device sync -- 128 per step at batch 128, measured at 224 s/epoch on
            # the 24-task config against 7 s here. The selection is the same: `k` is
            # still min(per-task budget, nodes in graph), and ranking by descending
            # score picks the same set.
            counts = torch.bincount(node_mask, minlength=num_graphs)
            if self.sampling_strategy == "topk_PerTask":
                # Through `_k_per_task`, NOT `get_num_relevant_nodes_per_task` directly.
                # This path used to read the hand-written table while `_calc_k_for_topk`
                # honoured `relevance_whitelist`, so a run with a discovered list trained
                # the CBM against that list while the selection budget stayed on the
                # hand-written one -- confounding "is the list any good?" with "is the
                # budget any good?". One function, both callers.
                k_budget = torch.tensor(
                    [self._k_per_task(task_names[i] if task_names else None)
                     for i in range(num_graphs)],
                    device=probs.device, dtype=counts.dtype)
            else:
                k_budget = torch.full_like(counts, self.sample_abs)
            k = torch.minimum(k_budget, counts)

            # Padding sorts below every sigmoid output (which is > 0), so it can never
            # enter the top k; `valid` masks it a second time regardless.
            dense, valid = to_dense_batch(probs, node_mask, batch_size=num_graphs,
                                          fill_value=-1.0)
            rank = dense.argsort(dim=1, descending=True).argsort(dim=1)
            kept = ((rank < k.unsqueeze(1)) & valid)[valid]

            indexes = (kept.to(probs.dtype) - probs).detach() + probs
            probs = probs * indexes
        elif self.sampling_strategy == "threshold":
            # Per-task cutoff at eval; training always uses the scalar, since the bar the
            # BCE trains against has to be the one it was calibrated on.
            thr = self._eval_threshold(probs, node_mask, num_graphs, task_names)
            indexes = ((probs > thr).to(probs.dtype) - probs).detach() + probs
            probs = probs * indexes
            if self.graph_swap in ('full', 'random') and not self.training:
                probs = self._apply_graph_swap(probs, node_mask, num_graphs, task_names,
                                               getattr(batch, 'node_names', None))

            l = probs.sum() / num_graphs
            self.coarsening_loss['coars_threshold_sum_loss'] = {'value': l, 'weight': self.threshold_sum_weight}
        swap_applied = False
        if self.sampling_strategy == "vgib":
            probs, x_out, keep = self._vgib(x, probs, node_mask, num_graphs, task_names)
            if self.graph_swap in ('full', 'random') and not self.training:
                # Diagnostic: replace the learned selection with all-nodes or a random
                # subgraph of the same size, so "the policy needs THIS subgraph" can be
                # separated from "the policy needs A subgraph of this size".
                probs = self._apply_graph_swap(probs, node_mask, num_graphs, task_names,
                                               getattr(batch, 'node_names', None))
                keep = probs > 0
                x_out = None
                swap_applied = True

        if self.sampling_strategy == "vgib":
            # The soft path keeps the full topology on purpose: nodes are suppressed by
            # being replaced with noise, not by having their edges cut. Gating edges by
            # `probs` on top of that would prune the graph structurally and undo the one
            # property that makes the discard differentiable. Only `hard` eval, which is
            # explicitly asking for a pruned subgraph, gates them.
            #
            # A graph swap is the other case that has to gate them. The swap discards
            # `x_out`, so the noise attenuation is gone; if the edges then stayed intact,
            # a "random subgraph of size k" would still let every unselected node's real
            # features reach the selected ones through message passing, and the control
            # would read almost as strong as `full` no matter which k nodes it picked —
            # which is precisely the comparison it exists to make.
            if (not self.training and edge_weights is not None
                    and (self.eval_prune_mode == "hard" or swap_applied)):
                edge_weights = edge_weights * probs[edge_index[0]] * probs[edge_index[1]]
        else:
            # topk / topk_PerTask / threshold.
            #
            # Note: `probs` is NOT a 0/1 mask here. The selection block above ends in
            # `probs = probs * indexes`, and `indexes` has VALUE 1 for a kept node -- so a
            # dropped node leaves at 0 and a kept node leaves at p, its own sigmoid score.
            keep = probs != 0
            if self.true_removal:
                # TRUE REMOVAL. The shipped path below leaves x_out=None -- the pruned
                # nodes keep their real features -- and only gates the edge WEIGHTS after
                # the attention softmax, so those features still reach every kept node
                # through the softmax denominator, during training. As shipped, SIR
                # re-weights; it does not remove.
                #
                # Survivors go through at the HARD mask, not at p: `x * probs` would
                # attenuate every survivor in proportion to the scorer's confidence, which
                # is not what a graph built without the pruned nodes looks like. The
                # straight-through form has VALUE `keep` and gradient 1 w.r.t. probs, so
                # the selection stays exactly as learnable -- d(x_out)/d(probs) is `x`
                # either way. Applied to `probs` ITSELF because `probs` is also the node
                # weight the readout receives under multiplicative_pooling.
                hard = keep.to(probs.dtype)
                probs = (hard - probs).detach() + probs
                x_out = x * probs.unsqueeze(-1)
                self._edge_keep = keep[edge_index[0]] & keep[edge_index[1]]
                if edge_weights is not None:
                    edge_weights = edge_weights[self._edge_keep]
            else:
                x_out = None
                if edge_weights is not None:
                    edge_weights = edge_weights * probs[edge_index[0]] * probs[edge_index[1]]

        sampled_nodes = keep.nonzero(as_tuple=False).squeeze(-1)

        # Fraction of nodes whose hard selection agrees with the whitelist. Should reach
        # ~1.0 within the first stage-2 epochs; if it does not, the BCE weight is wrong.
        self._last_whitelist_match = None
        if wl_target is not None:
            self._last_whitelist_match = float(
                (keep.to(wl_target.dtype) == wl_target).to(torch.float32).mean())

        return edge_weights, sampled_nodes, probs, x_out, raw_probs

    def _accumulate_coarsening_loss(self, name, value, weight):
        """Sum a per-graph term into its running entry, registering it on first sight."""
        entry = self.coarsening_loss.get(name)
        if entry is None:
            self.coarsening_loss[name] = {'value': value, 'weight': weight}
        else:
            entry['value'] = entry['value'] + value

    def _vgib(self, x, probs, node_mask, num_graphs, task_names=None):
        """VGIB information filtration (Yu et al., CVPR 2022).

        During training nothing is pruned. Node v's features are blended into its own
        graph's Gaussian N(mu_G, sigma_G^2) with weight (1 - p_v), so a node is discarded
        by being dissolved into noise rather than removed — which is what makes the
        discard differentiable at all. The compression term is then the closed-form KL
        between the noisy node distribution N(p*x + (1-p)*mu, (1-p)^2 * sigma^2) and the
        noise prior N(mu, sigma^2): keeping a node costs in proportion to how far its
        features deviate from the graph mean, so a node has to earn its place against the
        prediction loss. Nothing here consults the task-relevance table — that is the
        entire point of the strategy.

        What happens at eval is `eval_prune_mode`:

          soft  — no threshold at all. Every node is attenuated toward the graph mean in
                  proportion to (1 - p), exactly as in training, using the expectation
                  `p*x + (1-p)*mu` rather than a fresh noise draw. Same relationship to
                  training that eval-time dropout has to training-time dropout, and it is
                  the only mode in which a number reflects the learned scores alone
                  rather than the scores *and* a threshold chosen after the fact.
          noise — threshold, but sub-threshold nodes are replaced by a draw from the
                  graph's own Gaussian instead of being removed. The graph keeps its size
                  and topology; the dropped nodes carry no information.
          hard  — threshold, and sub-threshold nodes are zeroed out and their edges cut.
                  The deterministic sparse subgraph you would ship as the explanation.

        `hard` is the honest end product, but it is also the furthest from what training
        ever showed the policy: on the first sweep it cost CloseDrawer 0.600 -> 0.185 on
        the same checkpoint, purely from the cut.

        Returns `(probs, x_out, keep)` — the (at eval, thresholded) node scores, the
        features to run the GNN on (None at eval, where the originals are used
        unchanged), and the boolean node mask to report as selected.
        """
        # Per-graph mean and std over nodes, per feature dim. Dense so this costs two
        # kernels rather than one indexing round-trip per graph.
        x_dense, dense_mask = to_dense_batch(x, node_mask, batch_size=num_graphs)
        valid = dense_mask.unsqueeze(-1)
        counts = dense_mask.sum(dim=1).clamp(min=1).unsqueeze(-1)          # [B, 1]
        mu_g = (x_dense * valid).sum(dim=1) / counts                       # [B, D]
        var_g = ((x_dense - mu_g.unsqueeze(1)).pow(2) * valid).sum(dim=1) / counts
        std_g = var_g.clamp(min=0).sqrt() + 1e-6

        mu = mu_g[node_mask]
        std = std_g[node_mask]

        p = probs.unsqueeze(-1)
        z_mean = p * x + (1 - p) * mu
        z_std = (1 - p) * std

        if self.vgib_noise_kl_mode == "eq10":
            # Graph-level bound (Prop. 4.1 / Eq. 10): one KL between the pooled noisy
            # representation and the pooled prior rather than a per-node sum. -log(A) is
            # a barrier against keeping everything whose pressure decays as ~1/A once
            # eviction starts, so no single node can run away; deviations of kept nodes
            # can cancel inside B. B^2 is averaged over feature dims so the scale does
            # not depend on in_dim.
            p_dense, _ = to_dense_batch(probs, node_mask, batch_size=num_graphs)
            dev_dense, _ = to_dense_batch((x - mu) / std, node_mask, batch_size=num_graphs)
            n_g = dense_mask.sum(dim=1).clamp(min=1)                        # [B]
            A = ((1.0 - p_dense).pow(2) * dense_mask).sum(dim=1).clamp(min=1e-6)
            B = (p_dense.unsqueeze(-1) * dev_dense * valid).sum(dim=1)      # [B, D]
            vgib_noise_kl = (-0.5 * torch.log(A) + A / (2 * n_g)
                             + B.pow(2).mean(dim=-1) / (2 * n_g)).mean()
        else:
            quad = 0.5 * ((z_std / std).pow(2).mean(dim=-1)
                          + ((z_mean - mu) / std).pow(2).mean(dim=-1))
            if self.vgib_noise_kl_mode == "full":
                # Full Gaussian KL. Restores the -log(1 - p) - 1/2 that the truncated
                # form drops, which is what removes the p* = 1/(1 + d^2) fixed point:
                # this version is zero at p=0 and strictly increasing in p, so it is pure
                # down-pressure that only the prediction loss can hold a node against.
                # The truncated form instead has an interior attractor, and nodes park at
                # a middling score rather than committing either way.
                vgib_noise_kl = (-torch.log1p(-probs.clamp(max=1 - 1e-6)) + quad - 0.5).mean()
            else:
                vgib_noise_kl = quad.mean()

        self.coarsening_loss['vgib_noise_kl_loss'] = {
            'value': vgib_noise_kl, 'weight': self.vgib_noise_kl_weight}

        # Optional linear rent on scores above the eval cutoff, off by default (plain
        # VGIB). It exists because once the KL has been paid, sitting just above the bar
        # is free, and a scorer that learns to do that makes `threshold` meaningless.
        # Uses the same cutoff train and eval so the two agree.
        above = (probs * (probs > self.threshold).to(probs.dtype)).sum() / num_graphs
        self.coarsening_loss['coars_threshold_sum_loss'] = {
            'value': above, 'weight': self.threshold_sum_weight}

        eval_thr = self._eval_threshold(probs, node_mask, num_graphs, task_names)

        # Reported selection is the threshold cut in both phases — during training
        # `probs` is soft and never exactly zero, so `probs != 0` would report every node
        # as kept and the logged sparsity would be a flat 1.0 all run.
        keep = probs > eval_thr

        if self.training:
            return probs, z_mean + torch.randn_like(z_std) * z_std, keep

        # ---- inference. All three branches are out of place: `probs` is still the
        # sigmoid output that autograd needs for the validation pass.
        if self.eval_prune_mode == "soft":
            # No threshold. `keep` is still reported against it so the logged
            # kept-node ratio stays comparable across modes, but nothing is cut and
            # every node is attenuated exactly as it was in training.
            return probs, z_mean, keep

        if self.eval_prune_mode == "noise":
            # Sub-threshold nodes become pure graph-mean noise; survivors are untouched.
            # Topology is preserved, so the GNN sees a graph of the same shape.
            noised = mu + torch.randn_like(std) * std
            x_out = torch.where(keep.unsqueeze(-1), x, noised)
            return probs, x_out, keep

        # hard: zero the score, which also gates the edges upstream.
        return probs * keep.to(probs.dtype), None, keep

    def _eval_threshold(self, probs, node_mask, num_graphs, task_names):
        """Scalar `threshold`, or a per-node tensor when task_thresholds is set so a
        mixed-task batch still gets the right cutoff per graph."""
        if self.training or not self.task_thresholds or not task_names:
            return self.threshold
        per_graph = torch.tensor(
            [self.task_thresholds.get(task_names[min(i, len(task_names) - 1)], self.threshold)
             for i in range(num_graphs)],
            device=probs.device, dtype=probs.dtype)
        return per_graph[node_mask]

    def _selected_count(self, probs_i, n_i, task_name):
        """How many nodes the active strategy would keep — the size `random` must match."""
        if self.sampling_strategy in ("topk_PerTask", "topk"):
            return self._calc_k_for_topk(n_i, task_name)
        if self.sampling_strategy == "fixed":
            return int((probs_i > 0).sum().item())
        thr = self.threshold
        if self.task_thresholds and task_name is not None:
            thr = self.task_thresholds.get(task_name, self.threshold)
        return int((probs_i > thr).sum().item())

    def _apply_graph_swap(self, probs, node_mask, num_graphs, task_names, node_names=None):
        """Eval-only. 'full' keeps every node; 'random' keeps a random subgraph of the
        same size the active strategy selected. Hard 0/1 either way.

        The random draw is seeded from the graph's own identity (task + node-name set),
        NOT redrawn per forward pass. That matters: the learned subgraph is stable for as
        long as the scene is, so a random one resampled every timestep would differ from
        it in two ways at once — wrong nodes *and* flickering — and a drop could not be
        attributed to either. Seeding this way keeps one fixed subgraph per scene while
        still varying across scenes, and makes the control reproducible between runs.
        """
        if self.graph_swap == 'full':
            return torch.ones_like(probs)

        if node_names is not None and not isinstance(node_names[0], (list, tuple)):
            node_names = [node_names]          # rollout feeds a flat list (one graph)

        new_probs = torch.zeros_like(probs)
        for i in range(num_graphs):
            idx_i = (node_mask == i).nonzero(as_tuple=False).squeeze(1)
            n_i = idx_i.numel()
            task = task_names[i] if task_names else None
            k = self._selected_count(probs[idx_i], n_i, task)
            if k <= 0:
                continue

            gen = None
            if node_names is not None and i < len(node_names):
                key = f"{task}|{'|'.join(map(str, node_names[i]))}|{self.graph_swap_seed}"
                seed = int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(),
                                      'big') & 0x7FFFFFFFFFFFFFFF
                gen = torch.Generator().manual_seed(seed)
            elif not self._warned_unseeded_swap:
                self._warned_unseeded_swap = True
                log.warning("graph_swap='random' has no node_names to seed from, so the "
                            "subgraph is redrawn every step. The comparison against the "
                            "learned subgraph is confounded by that flicker.")

            perm = (torch.randperm(n_i, generator=gen) if gen is not None
                    else torch.randperm(n_i)).to(probs.device)[:k]
            new_probs[idx_i[perm]] = 1.0
        return new_probs

    def _whitelist_probs(self, x, batch, node_mask, num_graphs, task_names):
        """Hard 0/1 mask from the stage-2 whitelist. Constant per (task, node name), so
        it carries no gradient. The mask under 'fixed', the BCE target otherwise."""
        names = getattr(batch, 'node_names', None)
        if not names:
            raise ValueError("the node whitelist needs node_names on the graph batch")
        # PyG collation gives one name-list per graph; the sequential tester feeds a flat
        # list (single graph).
        if not isinstance(names[0], (list, tuple)):
            names = [names]
        if len(names) != num_graphs:
            raise ValueError(f"node_names has {len(names)} entries for {num_graphs} graphs")

        probs = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        for i in range(num_graphs):
            idx_i = (node_mask == i).nonzero(as_tuple=False).squeeze(1)
            if len(names[i]) != idx_i.numel():
                raise ValueError(f"graph {i}: {len(names[i])} node names vs {idx_i.numel()} nodes")
            task = task_names[min(i, len(task_names) - 1)] if task_names else None
            keep = self.node_whitelist.get(task)
            if keep is None:
                raise KeyError(f"task '{task}' is not in the node whitelist "
                               f"(has: {sorted(self.node_whitelist)})")
            sel = [j for j, name in enumerate(names[i]) if name in keep]
            if sel:
                probs[idx_i[torch.tensor(sel, device=x.device)]] = 1.0
            elif task not in self._warned_empty_tasks:
                # An empty subgraph pools to a zero embedding — the policy goes scene-blind.
                self._warned_empty_tasks.add(task)
                log.warning(f"stage-2 subgraph for task '{task}' matched no node in this "
                            f"graph (whitelist: {sorted(keep)}, graph: {sorted(set(names[i]))})")
        return probs

    def _k_per_task(self, task_name):
        """The per-task k budget: the single place the per-task count is decided.

        `relevance_whitelist` swaps the CBM's target list; it must swap the budget with
        it, or the run measures a discovered list held to a hand-written cardinality.
        Both the selection path and `_calc_k_for_topk` call this. Not capped at the graph
        size: each caller applies that cap itself.
        """
        if self.relevance_whitelist is not None:
            if task_name not in self.relevance_whitelist:
                raise KeyError(
                    f"relevance_whitelist has no entry for task '{task_name}' "
                    f"(has: {sorted(self.relevance_whitelist)})")
            return len(self.relevance_whitelist[task_name])
        return get_num_relevant_nodes_per_task(task_name)

    def _calc_k_for_topk(self, num_nodes_in_graph, task_name=None):
        if self.sampling_strategy == "topk_PerTask":
            k = self._k_per_task(task_name)
        else:
            # topk (task-agnostic) and threshold (k only used as scores_in_upper fallback)
            k = self.sample_abs
        return min(k, num_nodes_in_graph)


class TransformerCoarsening(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, dropout_prob=0.2):
        super().__init__()

        self.embedding = nn.Linear(input_dim, embed_dim)
        self.encoder = TransformerFiLMDecoder(
            embed_dim=embed_dim,
            n_heads=num_heads,
            attn_pdrop=dropout_prob,
            resid_pdrop=dropout_prob,
            n_layers=num_layers,
            film_cond_dim=embed_dim,
            causal=False,
            # The scorer is self-attention only — forward() never passes a context,
            # so a cross_att here would allocate parameters that never receive a
            # gradient. Language enters through the FiLM conditioning instead.
            use_cross_attention=False)
        self.scorer = nn.Sequential(
            nn.Linear(embed_dim, 1)
        )

    def forward(self, x, edge_index=None, edge_attr=None, node_mask=None, lang_emb=None, num_graphs=None):

        x = self.embedding(x)  # [N, D]
        assert (x != x).sum() == 0, "NaN-Values in x."

        # One padded [B, N_max, D] call rather than one transformer call per graph. The
        # loop this replaces cost ~1.2 ms per graph forward+backward, i.e. ~280 s of pure
        # scorer time per ALL-task epoch, and it scaled with the batch size instead of
        # amortising over it.
        dense, valid = to_dense_batch(x, node_mask, batch_size=num_graphs)  # [B, N_max, D], [B, N_max]

        # Mask KEYS only. Masking padded queries as well would leave those rows with no
        # attendable key at all, and softmax over an all-masked row is NaN — padded rows
        # are simply discarded below instead.
        attn_mask = valid[:, None, None, :]                                 # [B, 1, 1, N_max]

        # FiLM conditioning is per graph. `modulate` broadcasts the [D] shift/scale
        # against the token axis, so `c` has to carry an explicit length-1 token
        # dimension — a bare [B, D] would align against N_max and either fail or, when
        # B == N_max, silently modulate each node with a different graph's language.
        dense = self.encoder(dense, lang_emb[:num_graphs].unsqueeze(1), custom_attn_mask=attn_mask)

        x = dense[valid]  # back to [N, D] in the original node order

        scores = self.scorer(x).squeeze(-1)  # [N]

        assert (scores != scores).sum() == 0, "NaN-Values in scores."

        return scores


class Multi_XAI_GNN(nn.Module):
    """Wraps one Sparsification_Module + one GNN per graph modality."""

    def __init__(self, sparsification_layer, input_dim, hidden_dim, output_dim, edge_dim, num_layer, layer_name, pool_name, heads, dropout, modalities,
                 use_cbm_loss=None, lcbm_weight=0.0, multiplicative_pooling=False,
                 norm_mode="graph", attn_renorm=False):
        super().__init__()
        self.sparsification_modules = nn.ModuleDict()
        self.gnn_models = nn.ModuleDict()
        self.aux_losses = {}
        self.last_sparsification = {}

        self.use_cbm_loss = use_cbm_loss
        self.lcbm_weight = lcbm_weight
        self.multiplicative_pooling = multiplicative_pooling

        for mod in modalities:
            spars_dict = OmegaConf.to_container(sparsification_layer, resolve=True)
            spars_dict['in_dim'] = input_dim[mod]
            self.sparsification_modules[mod] = hydra.utils.instantiate(OmegaConf.create(spars_dict))
            self.gnn_models[mod] = GNN(input_dim[mod], hidden_dim, output_dim, edge_dim, num_layer, layer_name, pool_name, heads, dropout,
                                       norm_mode=norm_mode, attn_renorm=attn_renorm)

    def set_graph_swap(self, mode):
        """Eval-only ablation of the selection itself:
          none   — the learned subgraph
          full   — every node kept, i.e. no sparsification (in-distribution upper bound)
          random — a random subgraph of the SAME SIZE the model selected

        `random` is the control that separates "the policy needs this subgraph" from
        "the policy needs a subgraph of this size" — without it, a high score on a
        3-node explanation says nothing about which 3 nodes.
        """
        assert mode in ('none', 'full', 'random'), f"unknown graph_swap: {mode}"
        for spars in self.sparsification_modules.values():
            spars.graph_swap = mode
            # The swap always produces a hard 0/1 subgraph with its edges cut. Under
            # `soft`/`noise` the unswapped run it is meant to be compared against does
            # not, so the pair differs by the pruning as well as by the node choice and
            # the comparison answers nothing. Run the swaps against `hard`.
            if (mode != 'none' and spars.sampling_strategy == 'vgib'
                    and spars.eval_prune_mode != 'hard'):
                log.warning(
                    f"graph_eval_mode='{mode}' with eval_prune_mode="
                    f"'{spars.eval_prune_mode}': the swap yields a pruned subgraph while "
                    f"the graph_eval_mode='none' reference does not, so the two are not "
                    f"comparable. Set eval_prune_mode=hard for selection ablations.")

    def set_task_thresholds(self, thresholds):
        """Per-task eval cutoff, {task: threshold}. Tasks not listed keep the scalar."""
        thresholds = dict(thresholds) if thresholds else None
        for spars in self.sparsification_modules.values():
            spars.task_thresholds = thresholds

    def forward(self, input, lang_emb, task_names, lang_goal_texts=None):
        self.aux_losses = {}
        self.last_sparsification = {}
        out = {}
        for mod in input:
            batch = input[mod]
            spars = self.sparsification_modules[mod]

            edge_weights_out, sampled_nodes, probs, x_out, raw_probs = spars(
                batch.x, batch.edge_index, edge_attr=None, edge_weights=batch.edge_attr,
                lang_emb=lang_emb, batch=batch, task_names=task_names
            )

            # VGIB replaces discarded nodes' features with graph-level noise instead of
            # pruning them, so the GNN has to run on the noised features — otherwise the
            # compression KL is computed against a corruption the model never sees and
            # trains nothing at all.
            if x_out is not None:
                batch.x = x_out

            # True removal: the pruned nodes' edges are physically gone, so they are
            # absent from the attention softmax rather than merely down-weighted in it.
            # Applied here because the module cannot rewrite the caller's batch.
            # The total edge count is taken before the slice, or the kept ratio reads ~1.
            n_total_edges = batch.edge_index.shape[1]
            _ek = spars._edge_keep
            if _ek is not None:
                batch.edge_index = batch.edge_index[:, _ek]

            # Track kept nodes/edges sparsification ratio
            n_total = batch.x.shape[0]
            n_kept = sampled_nodes.shape[0]
            n_kept_edges = int((edge_weights_out > 0).sum().item()) if edge_weights_out is not None else n_total_edges

            self.last_sparsification[mod] = {
                'sampled_nodes': sampled_nodes.detach().cpu(),
                'probs': probs.detach().cpu(),
                # Pre-selection scores + the task each graph belongs to: together these
                # are what utils.node_scores needs to rank nodes per task, which is the
                # only way to choose a VGIB threshold (there is no k budget to fall
                # back on).
                'raw_probs': raw_probs.detach().cpu(),
                'task_names': list(task_names) if task_names is not None else None,
                'node_names': getattr(batch, 'node_names', None),
                'kept_nodes_ratio': n_kept / n_total if n_total > 0 else 1.0,
                'kept_edges_ratio': n_kept_edges / n_total_edges if n_total_edges > 0 else 1.0,
                # Stage-2 canary: fraction of nodes whose hard selection matches the
                # whitelist. None outside stage 2.
                'whitelist_match': spars._last_whitelist_match,
            }

            if edge_weights_out is not None:
                batch.edge_attr = edge_weights_out

            # Stash aux losses keyed by modality so they don't collide across modalities
            for loss_name, loss_entry in spars.coarsening_loss.items():
                self.aux_losses[f"{mod}/{loss_name}"] = loss_entry

            # CBM loss: push scores of task-relevant nodes UP, irrelevant nodes DOWN
            if self.training and self.use_cbm_loss == "nodes_att":
                cbm_loss = self._compute_cbm_loss(probs, batch, task_names, lang_goal_texts)
                if cbm_loss is not None:
                    self.aux_losses[f"{mod}/cbm_loss"] = {'value': cbm_loss, 'weight': self.lcbm_weight}

            out[mod] = self.gnn_models[mod](batch, node_weights=probs if self.multiplicative_pooling else None)

        return out

    @property
    def relevance_whitelist(self):
        """The discovered list that replaces RELEVENT_NODES, or None.

        It is configured on the sparsification module (which needs it for `k`), but the
        CBM loss lives up here, so read it back off the modules rather than duplicating
        the config key in two places and letting them drift apart. All modalities share
        one sparsification config, so the first is representative.
        """
        for spars in self.sparsification_modules.values():
            return getattr(spars, 'relevance_whitelist', None)
        return None

    def _compute_cbm_loss(self, probs, batch, task_names, lang_goal_texts):
        """CBM loss: (-sum(probs[relevant]) + sum(probs[~relevant])) / N.
        Negative = model correctly assigns high scores to relevant nodes.
        """
        node_names_attr = getattr(batch, 'node_names', None)
        if not node_names_attr or task_names is None:
            return None

        node_mask = batch.batch
        num_graphs = batch.num_graphs if hasattr(batch, 'num_graphs') else int(node_mask.max().item()) + 1

        # After PyG batching: node_names_attr is list-of-lists (one per graph);
        # for batch_size=1 (rollout) it may be a flat list of strings.
        batched = isinstance(node_names_attr[0], (list, tuple))

        # Relevance is a pure function of (node name, task, language goal), so it is
        # resolved on the host and moved across as one tensor. The nested loop this
        # replaces wrote ~2560 individual CUDA scalars per step and called `.nonzero()`
        # once per graph, measured at 122 s/epoch on the 24-task config against 1 s here.
        # The memo matters as much as the batching: a batch of 128 graphs over 24 tasks
        # holds only a few hundred distinct (name, task, goal) triples.
        counts = torch.bincount(node_mask, minlength=num_graphs).tolist()
        cache, flags = {}, []
        for i in range(num_graphs):
            task_name = task_names[i] if isinstance(task_names, (list, tuple)) else task_names
            lang_text = lang_goal_texts[i] if lang_goal_texts is not None else ""
            names_i = node_names_attr[i] if batched else node_names_attr
            for j_local in range(counts[i]):
                if j_local >= len(names_i):
                    # Graph carries more nodes than names; the old loop left these False.
                    flags.append(False)
                    continue
                key = (names_i[j_local], task_name, lang_text)
                if key not in cache:
                    if self.relevance_whitelist is not None:
                        cache[key] = names_i[j_local] in self.relevance_whitelist.get(task_name, ())
                    else:
                        cache[key] = bool(is_node_relevant_for_task(names_i[j_local], task_name, lang_text))
                flags.append(cache[key])

        relevant_mask = torch.tensor(flags, dtype=torch.bool, device=probs.device)

        N = probs.shape[0]
        if relevant_mask.sum() == 0 or (~relevant_mask).sum() == 0:
            return None

        cbm_loss = (-probs[relevant_mask].sum() + probs[~relevant_mask].sum()) / N
        return cbm_loss
