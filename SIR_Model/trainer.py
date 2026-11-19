import gc
import logging
import hydra
from omegaconf import OmegaConf
import torch
import numpy as np
import random
from tqdm import tqdm
import wandb
import os
import yaml

from method.utils import EMA
from utils.node_scores import NodeScoreTracker, wandb_log_dict

log = logging.getLogger(__name__)

class Trainer():
    def __init__(self, manager, method, config):
        self.seed = config.seed
        self.device = config.device
        self.epochs = config.epochs
        self.train_bool = config.train_bool
        self.test_during_training_bool = config.test_during_training_bool
        self.test_bool = config.test_bool
        self.log_dir = config.log_dir
        # Absolute, because Hydra chdir's into the run directory before the Trainer is
        # built: a relative path given on the command line would otherwise be looked up
        # inside this run's own (empty) output dir.
        self.model_path = hydra.utils.to_absolute_path(config.model_path)
        self.eval_n_times = config.eval_n_times
        self.store_videos = config.store_videos
        self.use_ema = config.use_ema
        # Warm-start a TRAINING run from another run's weights (stage 2 points this at
        # the stage-1 run dir). Weights only: this run keeps its own config, optimizer
        # and LR schedule, so the LR restarts high instead of inheriting a decayed tail.
        # Ignored when train_bool=false — evaluation uses model_path. MUST stay null for
        # stage 1, or every run silently continues someone else's checkpoint.
        self.init_from = OmegaConf.select(config, "init_from", default=None)
        self.init_from_suffix = OmegaConf.select(config, "init_from_suffix", default="_final")
        # Init from the run's EMA shadow weights — the ones its val node scores, and so
        # the threshold you picked, were measured under.
        self.init_from_ema = OmegaConf.select(config, "init_from_ema", default=True)
        # Eval-only diagnostics. Read from THIS config, not the checkpoint's, so a
        # trained model can be re-examined under them without retraining.
        self.graph_eval_mode = OmegaConf.select(config, "graph_eval_mode", default="none")
        self.task_thresholds = OmegaConf.select(config, "task_thresholds", default=None)
        # Which checkpoint an eval-only run loads: '_final' (last epoch) or '_best' (the
        # lowest val_total_loss seen during training). Both are written by every training
        # run, so a model can be scored either way without retraining. Kept as a plain
        # suffix rather than a bool so '_epoch_<n>' stays reachable too.
        self.eval_checkpoint = OmegaConf.select(config, "eval_checkpoint", default="_final")
        self.ema_decay = OmegaConf.select(config, "ema_decay", default=0.999)
        # Lowest val_total_loss seen so far, for the '_best' checkpoint.
        self.best_val_loss = float('inf')
        # 0 disables gradient clipping, which is the previous behaviour.
        self.grad_clip = OmegaConf.select(config, "grad_clip", default=0.0)


        # set seed for random stuff
        # Note: GNNs are non-deterministic by nature when executing them on the GPU -> https://github.com/pyg-team/pytorch_geometric/issues/92
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # Note: this is a function call, not an attribute assignment — assigning to it
        # silently replaces the function and leaves determinism off.
        torch.use_deterministic_algorithms(True, warn_only=True)

        log.info(f'Used seed: {self.seed}')
        
        manager, method = self.check_if_model_is_loaded(manager, method)
        
        self.manager = hydra.utils.instantiate(manager)
        method = self.set_runtime_parameter(method)
        self.method = hydra.utils.instantiate(method)
        
        self.working_dir = hydra.core.hydra_config.HydraConfig.get()['runtime']['output_dir']
    
    def check_if_model_is_loaded(self, manager, method):
        new_task_names = manager.task_names
        new_datapath = manager.data_path
        self.load_model_bool = not self.train_bool
        if self.load_model_bool:
            if os.path.exists(self.model_path + "/.hydra/config.yaml"):
                cfg_reloaded = OmegaConf.load(self.model_path + "/.hydra/config.yaml")
            else:
                raise FileNotFoundError("Could not find config file to reload model parameters.")
            manager = cfg_reloaded.manager
            manager.load_dataset = False
            manager.task_names = new_task_names
            manager.data_path = new_datapath
            method = cfg_reloaded.method
            method.device = self.device

            # Architecture and modality settings must come from the checkpoint or the
            # weights will not load — but everything else on the manager was just
            # replaced wholesale by the checkpoint's copy, which silently discarded any
            # `manager.*` / `method.*` override given on the command line. An eval-time
            # knob set explicitly for this invocation (num_envs, parallel_eval,
            # worker_seed_base, ...) would inherit the training run's value and measure
            # something other than what was asked for. Re-apply the explicit overrides
            # on top, so the checkpoint stays the default and the command line wins.
            overrides = hydra.core.hydra_config.HydraConfig.get().get("overrides", {}).get("task", [])
            manager_dotlist = []
            method_dotlist = []
            for override in overrides:
                normalized = override.lstrip("+~")
                if normalized.startswith("manager."):
                    manager_dotlist.append(normalized)
                elif normalized.startswith("method."):
                    method_dotlist.append(normalized)

            if manager_dotlist:
                log.info(f"Re-applying manager overrides onto reloaded config: {manager_dotlist}")
                manager = OmegaConf.merge(manager, OmegaConf.from_dotlist(manager_dotlist).manager)
            if method_dotlist:
                log.info(f"Re-applying method overrides onto reloaded config: {method_dotlist}")
                method = OmegaConf.merge(method, OmegaConf.from_dotlist(method_dotlist).method)

            self.use_ema = cfg_reloaded.trainer.use_ema
        
        return manager, method
    
    def start(self):
        self.instantiate_models()

        if self.train_bool and self.init_from:
            self.warm_start()

        if self.train_bool:
            log.info("Starting training...")
            self.train()
            log.info("Training done.")
        
        if self.test_bool:
            if self.load_model_bool:
                self.load_model(epoch=None)

            if self.graph_eval_mode != 'none':
                self.method.set_graph_eval_mode(self.graph_eval_mode)
                log.info(f"Graph eval mode ENABLED: '{self.graph_eval_mode}' — this is an "
                         f"ablation, not a normal evaluation.")
            if self.task_thresholds:
                thresholds = self._resolve_task_thresholds(self.task_thresholds)
                self.method.set_task_thresholds(thresholds)
                log.info(f"Per-task eval thresholds: {thresholds}")
            log.info("Starting testing...")
            self.test()
            log.info("Testing done.")
    
    def _resolve_task_thresholds(self, spec):
        """{task: cutoff} inline, or a path to a utils.select_subgraph YAML.

        Passing 24 cutoffs on a Hydra command line is unreadable and easy to get
        silently wrong, so the K-means selector writes them to a file and this reads
        the `task_thresholds` mapping back out of it.
        """
        if isinstance(spec, str):
            path = hydra.utils.to_absolute_path(spec)
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"task_thresholds points at '{path}', which does not exist. "
                    f"Pass an inline mapping or a utils.select_subgraph output.")
            with open(path) as f:
                doc = yaml.safe_load(f)
            thresholds = doc.get("task_thresholds") if isinstance(doc, dict) else None
            if not thresholds:
                raise ValueError(
                    f"'{path}' carries no non-empty `task_thresholds` mapping — it was "
                    f"probably written without --kmeans/--task-threshold.")
            log.info(f"Per-task eval thresholds loaded from {path}")
            return dict(thresholds)
        return (OmegaConf.to_container(spec, resolve=True)
                if OmegaConf.is_config(spec) else dict(spec))

    def train(self):
        for epoch in tqdm(range(self.epochs)):
            log.info(f"Epoch {epoch+1}/{self.epochs}")
            
            if self.use_ema:
                self.ema.apply_shadow()
            
            val_loss_accum = {}
            val_loss_counts = {}
            # Per-node score snapshot over the whole validation set. The sparsification
            # log below only ever sees the last batch, which is not enough to rank nodes
            # — and ranking them per task is the only way to choose a threshold for a
            # sparsifier that has no k budget (VGIB). Cheap: the scores are already on
            # the CPU in `last_sparsification`.
            val_scores = NodeScoreTracker()
            for val_batch in tqdm(self.manager.valid_dataloader):
                val_loss_dic = self.val_step(val_batch)
                val_scores.update_from_method(self.method)
                for k, v in val_loss_dic.items():
                    if k not in val_loss_accum:
                        val_loss_accum[k] = 0.0
                        val_loss_counts[k] = 0
                    val_loss_accum[k] += v.item()
                    val_loss_counts[k] += 1

            if self.use_ema:
                self.ema.restore()

            # Average each key over the batches it actually appeared in — a key that is
            # only emitted for some batches would otherwise be scaled down by the ratio.
            val_log_dict = {
                f"val_{k}": (v / max(val_loss_counts[k], 1))
                for k, v in val_loss_accum.items()
            }

            if val_scores:
                val_scores.save_json(
                    os.path.join(self.working_dir, 'node_scores', f'val_epoch{epoch:03d}.json'),
                    metadata={'split': 'val', 'epoch': epoch},
                )
                val_log_dict.update(wandb_log_dict(val_scores, 'val', epoch))

            # Log sparsification kept_nodes/kept_edges ratios (from last val batch)
            if hasattr(self.method, 'graph_encoder') and hasattr(self.method.graph_encoder, 'last_sparsification'):
                for mod, info in self.method.graph_encoder.last_sparsification.items():
                    if 'kept_nodes_ratio' in info:
                        val_log_dict[f"sparsification/val_kept_nodes_{mod}"] = info['kept_nodes_ratio']
                        val_log_dict[f"sparsification/val_kept_edges_{mod}"] = info['kept_edges_ratio']
                    if info.get('whitelist_match') is not None:
                        val_log_dict[f"sparsification/val_whitelist_match_{mod}"] = info['whitelist_match']

            # Save the best-val checkpoint HERE rather than at the end of the epoch.
            # Validation runs before training, so `val_total_loss` describes the weights
            # as they are at this moment; by the end of the epoch they have already moved
            # and would no longer be the model the number was measured on. The EMA shadow
            # is what validation ran under, and what `test()` applies, so the pair saved
            # here is exactly the model that scored this loss.
            val_loss = val_log_dict.get("val_total_loss")
            if val_loss is not None and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_model(name="_best")
                log.info(f"new best val_total_loss {val_loss:.6f} at epoch {epoch} -> saved '_best'")

            wandb.log(val_log_dict, step=epoch)
            for k, v in val_log_dict.items():
                # Node scores go to wandb only — ~80 of them an epoch would bury every
                # other line in main.log. The full breakdown is in node_scores/*.json.
                if not k.startswith("node_scores/"):
                    log.info(f"{k}: {v:.6f}")
            
            train_loss_accum = {}
            train_loss_counts = {}
            train_scores = NodeScoreTracker()
            for batch in tqdm(self.manager.train_dataloader):
                loss_dic = self.train_step(batch)
                train_scores.update_from_method(self.method)
                for k, v in loss_dic.items():
                    if k not in train_loss_accum:
                        train_loss_accum[k] = 0.0
                        train_loss_counts[k] = 0
                    train_loss_accum[k] += v.item()
                    train_loss_counts[k] += 1

            train_log_dict = {
                f"train_{k}": (v / max(train_loss_counts[k], 1))
                for k, v in train_loss_accum.items()
            }

            if train_scores:
                train_scores.save_json(
                    os.path.join(self.working_dir, 'node_scores', f'train_epoch{epoch:03d}.json'),
                    metadata={'split': 'train', 'epoch': epoch},
                )
                train_log_dict.update(wandb_log_dict(train_scores, 'train', epoch))

            # Log sparsification kept_nodes/kept_edges ratios (from last train batch)
            if hasattr(self.method, 'graph_encoder') and hasattr(self.method.graph_encoder, 'last_sparsification'):
                for mod, info in self.method.graph_encoder.last_sparsification.items():
                    if 'kept_nodes_ratio' in info:
                        train_log_dict[f"sparsification/train_kept_nodes_{mod}"] = info['kept_nodes_ratio']
                        train_log_dict[f"sparsification/train_kept_edges_{mod}"] = info['kept_edges_ratio']
                    if info.get('whitelist_match') is not None:
                        train_log_dict[f"sparsification/train_whitelist_match_{mod}"] = info['whitelist_match']

            wandb.log(train_log_dict, step=epoch)
            for k, v in train_log_dict.items():
                # Node scores go to wandb only — ~80 of them an epoch would bury every
                # other line in main.log. The full breakdown is in node_scores/*.json.
                if not k.startswith("node_scores/"):
                    log.info(f"{k}: {v:.6f}")

            self.save_model()
            
            if self.test_during_training_bool and epoch % 5 == 0 and epoch != 0:
                self.test(20, epoch)
                self.save_model(epoch)
            
            wandb.log({"learn_rate": self.method.lr_scheduler.get_last_lr()[0]}, step=epoch)
            
            self.method.lr_scheduler.step()
    
    def val_step(self, batch):
        self.method.eval()
        state, action, goal = self.method.preprocess_batch(batch)
        
        with torch.no_grad():
            loss_dic = self.method.compute_validation_loss(state, action['action'], goal)
        
        return loss_dic
    
    def train_step(self, batch):
        self.method.train()
        state, action, goal = self.method.preprocess_batch(batch)
        
        loss_dic = self.method.compute_training_loss(state, action['action'], goal)
        loss = loss_dic['total_loss']
        
        self.method.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip > 0:
            for group in self.method.optimizer.param_groups:
                torch.nn.utils.clip_grad_norm_(group['params'], self.grad_clip)
        self.method.optimizer.step()
        
        if self.use_ema:
            self.ema.update()
        
        return loss_dic
    
    def _cleanup_before_eval(self, drop_training_state):
        """Free training-time memory before spinning up the eval environments.

        Rollout builds a full MuJoCo scene per env on top of the model, so peak
        memory here is much higher than during training. `drop_training_state`
        additionally releases the optimizer state and the dataloaders, which is only
        safe once no further training will happen.
        """
        optimizer = getattr(self.method, 'optimizer', None)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            if drop_training_state:
                optimizer.state.clear()

        if drop_training_state:
            for attr in ('train_dataloader', 'valid_dataloader'):
                if hasattr(self.manager, attr):
                    setattr(self.manager, attr, None)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def test(self, eval_n_times=None, epoch=None):
        self.method.eval()

        during_training = False

        if eval_n_times is not None:
            eval_n_times = eval_n_times
            during_training = True
        else:
            eval_n_times = self.eval_n_times

        self._cleanup_before_eval(drop_training_state=not during_training)

        if self.use_ema:
            self.ema.apply_shadow()


        test_results = self.manager.test_method(self.method, self.store_videos, eval_n_times, self.working_dir, during_training, epoch)
        
        if self.use_ema:
            self.ema.restore()
        
        # Sub-graph faithfulness travels on the manager rather than in
        # `test_results`, which the manager sums into Avg_Test_Reward.
        faith_log = getattr(self.manager, 'last_faithfulness_log', None) or {}
        faith_table = getattr(self.manager, 'last_faithfulness_table', None)
        if faith_table:
            faith_log = dict(faith_log)
            faith_log[faith_table['key']] = wandb.Table(
                columns=faith_table['columns'], data=faith_table['rows'],
            )

        if epoch is not None:
            wandb.log({f"test_during_training_{k}": v
                       for k, v in {**test_results, **faith_log}.items()}, step=epoch)
        else:
            wandb.log({**test_results, **faith_log})
    
    def instantiate_models(self):
        if self.manager.use_proprioceptive:
            self.method.instantiate_proprioceptive()
        if self.manager.use_image:
            self.method.instantiate_vision_encoder()
        if self.manager.use_graph:
            self.method.instantiate_graph_encoder()
            
        self.method.instantiate_optimizer()
        
        if self.use_ema:
            ema_modules = []
            
            # Check standard attributes (adjust names to match your Method class)
            if hasattr(self.method, 'action_generator'): # The MDT model
                ema_modules.append(self.method.action_generator)
            if hasattr(self.method, 'vision_encoder') and self.method.vision_encoder:
                ema_modules.append(self.method.vision_encoder)
            if hasattr(self.method, 'graph_encoder') and self.method.graph_encoder:
                ema_modules.append(self.method.graph_encoder)
            
            # Initialize EMA if we found modules
            if len(ema_modules) > 0:
                log.info(f"Initializing EMA for {len(ema_modules)} modules (decay={self.ema_decay}).")
                self.ema = EMA(ema_modules, decay=self.ema_decay)
        
    def set_runtime_parameter(self, method):
        method.action_generator.proprio_dim = self.manager.prop_dim
        method.action_generator.graph_mod = self.manager.adapted_graph_modalities
        method.graph_encoder.input_dim = self.manager.graph_dim
        method.graph_encoder.edge_dim = self.manager.graph_edge_dim
        method.graph_encoder.modalities = self.manager.adapted_graph_modalities

        # A pooled graph readout emits one token per modality, so the token budget (and
        # with it the position embedding) must stay at obs_window — otherwise every
        # existing checkpoint would fail to load on a pos_emb shape mismatch. Only the
        # per-node readout actually needs room for max_graph_nodes tokens.
        if OmegaConf.select(method, "graph_encoder.pool_name", default="mean") != "none":
            method.action_generator.max_graph_nodes = self.manager.obs_window

        return method
    
    def warm_start(self):
        """Load another run's weights into this run, without its config or optimizer.

        Distinct from `check_if_model_is_loaded`, which is the eval-only path and *does*
        reload the source config: stage 2 must keep its own encoder config (threshold +
        whitelist BCE) rather than reverting to stage 1's `vgib`.

        With `init_from_ema` the source run's EMA shadow is copied into the live
        parameters before this run's own EMA is re-registered from them, so training
        continues from exactly the weights whose node scores were thresholded — not from
        the raw weights, which are a different model.
        """
        src, name = self.init_from, self.init_from_suffix
        log.info(f"Warm start from {src} ({name}, from_ema={self.init_from_ema})")
        self.method.load_models(src, name)

        if self.init_from_ema:
            ema_path = os.path.join(src, "ema_weights" + name + ".pth")
            if not os.path.exists(ema_path):
                raise FileNotFoundError(
                    f"init_from_ema is set but {ema_path} does not exist — the source run "
                    f"was trained with use_ema: false. Pass trainer.init_from_ema=false to "
                    f"warm start from its raw weights instead.")
            shadow = torch.load(ema_path, map_location=self.device)
            if not self.use_ema:
                raise ValueError(
                    "init_from_ema needs use_ema: true in this run too — there is no EMA "
                    "object to load the shadow into otherwise.")
            self.ema.load_state_dict(shadow)
            self.ema.apply_shadow()   # shadow -> live parameters
            self.ema.backup = {}      # never restore the pre-warm-start weights
            self.ema.register()       # this run's EMA starts from the loaded weights

    def save_model(self, epoch=None, name=None):
        if name is None:
            name = '_final' if epoch is None else '_epoch_' + str(epoch)
        self.method.save_models(self.working_dir, name)
        if self.use_ema:
            torch.save(self.ema.state_dict(), os.path.join(self.working_dir, "ema_weights" + name + ".pth"))
    
    def load_model(self, epoch=None):
        if epoch is None:
            name = self.eval_checkpoint
        else:
            name = '_epoch_' + str(epoch)
        log.info(f"Evaluating checkpoint '{name}' from {self.model_path}")
        self.method.load_models(self.model_path, name)
        if self.use_ema:
            self.ema.load_state_dict(torch.load(os.path.join(self.model_path, "ema_weights" + name + ".pth")))