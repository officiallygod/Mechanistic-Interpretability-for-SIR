import logging
import math
import clip
from einops import einops
from omegaconf import DictConfig
import torch
import torch.nn as nn
from functools import partial

from method import utils
from method.base_method import Base_Method

log = logging.getLogger(__name__)

class Diffusion(Base_Method):
    def __init__(self,
                action_generator: DictConfig,
                vision_encoder: DictConfig,
                graph_encoder: DictConfig,
                optimization: DictConfig,
                lr_scheduler: DictConfig,
                device: str,
                act_window_size: int,
                obs_window_size: int,
                action_dim: int,
                sampler_type: str,
                num_sampling_steps: int,
                noise_scheduler: str,
                sigma_data: float,
                sigma_min: float,
                sigma_max: float,
                sigma_sample_density_type: str,
                ):
        super().__init__(action_generator,
                         vision_encoder,
                         graph_encoder,
                         optimization,
                         lr_scheduler,
                         device)
        
        self.act_window_size = act_window_size
        self.obs_window_size = obs_window_size
        self.action_dim = action_dim
        # diffusion stuff
        self.sampler_type = sampler_type
        self.num_sampling_steps = num_sampling_steps
        self.noise_scheduler = noise_scheduler
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_sample_density_type = sigma_sample_density_type
        
    def set_graph_eval_mode(self, mode):
        """none | zero | full | random. `zero` blanks the graph embedding after the
        encoder (token-level, and therefore out of distribution); `full`/`random` act on
        the node selection inside the sparsifier and stay in distribution."""
        self.graph_eval_mode = mode
        if mode in ('full', 'random'):
            # `graph_encoder` only exists once instantiate_graph_encoder has run, i.e.
            # never on an image-only or proprio-only run. getattr, not attribute access,
            # or setting this knob on the image baseline raises AttributeError instead of
            # saying it does not apply.
            encoder = getattr(self, 'graph_encoder', None)
            if hasattr(encoder, 'set_graph_swap'):
                encoder.set_graph_swap(mode)
            else:
                log.warning(f"graph_eval_mode='{mode}' needs a sparsifying graph encoder "
                            f"(Multi_XAI_GNN); this run has no node selection — ignored.")

    def set_task_thresholds(self, thresholds):
        encoder = getattr(self, 'graph_encoder', None)
        if hasattr(encoder, 'set_task_thresholds'):
            encoder.set_task_thresholds(thresholds)
        else:
            log.warning("task_thresholds needs a sparsifying graph encoder "
                        "(Multi_XAI_GNN); this run has no node selection — ignored.")

    def preprocess_batch(self, batch):
        
        goal = {}
        state = {}
        action = {}
        
        lang_goal = batch['goal']['lang']
        if isinstance(lang_goal, str):
            lang_goal_tokenized = clip.tokenize(lang_goal).to(self.device)
            lang_goal = self.language_encoder.encode_text(lang_goal_tokenized).to(torch.float32)
        goal['lang'] = lang_goal.to(self.device)
        
        for key in batch['observation']:
            if key == "obs_prop":
                state[key] = batch["observation"][key].to(self.device).float()
            elif key == "obs_img":
                # The encoders run per frame, so the observation window is folded into
                # the batch axis and unfolded again afterwards.
                img_input = {
                    k: einops.rearrange(v.to(self.device).float(), "b w ... -> (b w) ...")
                    for k, v in batch["observation"][key].items()
                }
                img_out = self.vision_encoder(img_input)
                state[key] = {
                    k: einops.rearrange(v, "(b w) ... -> b w ...", w=self.obs_window_size)
                    for k, v in img_out.items()
                }
            elif key == "obs_graph":
                graph_input = {k: v.to(self.device) for k, v in batch["observation"][key].items()}
                if hasattr(self.graph_encoder, 'aux_losses'):
                    # Multi_XAI_GNN: pass language embedding, task names, and raw text for CBM loss
                    lang_goal_texts = batch.get('goal', {}).get('lang_text', None)
                    graph_out = self.graph_encoder(
                        graph_input,
                        lang_emb=goal['lang'],
                        task_names=batch.get('task_name', None),
                        lang_goal_texts=lang_goal_texts,
                    )
                else:
                    graph_out = self.graph_encoder(graph_input, lang_emb=goal['lang'])

                # Normalize both readouts to (tokens, mask): pooled graph encoders emit a
                # single [B, output_dim] vector, per-node ones emit [B, N_max, output_dim]
                # plus the mask marking the real nodes.
                state[key] = {}
                for m, (tensor, mask) in graph_out.items():
                    if tensor.dim() == 2:
                        tensor = tensor.unsqueeze(1)
                        mask = torch.ones(tensor.shape[0], 1, dtype=torch.bool, device=tensor.device)
                    state[key][m] = (tensor, mask)
            else:
                raise NotImplementedError(f"Modality {key} not implemented in diffusion method.")


        # Zero ablation: blank the graph embedding so the MDT's graph token carries no
        # scene information at all.
        if getattr(self, 'graph_eval_mode', 'none') == 'zero' and "obs_graph" in state:
            state["obs_graph"] = {
                m: (torch.zeros_like(v[0]), v[1]) if isinstance(v, tuple) else torch.zeros_like(v)
                for m, v in state["obs_graph"].items()
            }

        if "action" not in list(batch.keys()):
            action['action'] = None
        else:  
            action['action'] = batch["action"]['eef'].to(self.device).float()
            
        return state, action, goal
    
    def compute_training_loss(self, state, action, goal):
        loss_dict = {}

        loss_dict['diffusion_loss'] = self.diffusion_loss(perceptual_emb=state, latent_goal=goal, actions=action)

        loss_dict['total_loss'] = loss_dict['diffusion_loss']

        # Aggregate sparsification auxiliary losses when using Multi_XAI_GNN
        if hasattr(self, 'graph_encoder') and hasattr(self.graph_encoder, 'aux_losses') and self.graph_encoder.aux_losses:
            sparsification_loss = torch.tensor(0.0, device=self.device)
            for key, entry in self.graph_encoder.aux_losses.items():
                weighted = entry['value'] * entry['weight']
                loss_dict[f'spars/{key}'] = weighted
                sparsification_loss = sparsification_loss + weighted
            loss_dict['sparsification_loss'] = sparsification_loss
            loss_dict['total_loss'] = loss_dict['total_loss'] + sparsification_loss

        return loss_dict
    
    def compute_validation_loss(self, state, action, goal):
        loss_dict = {}

        action_pred = self.denoise_actions(perceptual_emb=state, latent_goal=goal)

        mse_loss = nn.functional.mse_loss(action_pred, action)

        loss_dict['mse_loss'] = mse_loss
        loss_dict['total_loss'] = loss_dict['mse_loss']

        # Log sparsification aux losses during validation (not added to total_loss, monitoring only)
        if hasattr(self, 'graph_encoder') and hasattr(self.graph_encoder, 'aux_losses') and self.graph_encoder.aux_losses:
            for key, entry in self.graph_encoder.aux_losses.items():
                loss_dict[f'spars/{key}'] = entry['value'] * entry['weight']

        return loss_dict
    
    def predict(self, state, goal, noise=None):
        action_pred = self.denoise_actions(perceptual_emb=state, latent_goal=goal,
                                           noise=noise)

        return action_pred
    
    def diffusion_loss(
        self,
        perceptual_emb: torch.Tensor,
        latent_goal: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the score matching loss given the perceptual embedding, latent goal, and desired actions.
        """
        sigmas = self.make_sample_density()(shape=(len(actions),), device=self.device).to(self.device)
        noise = torch.randn_like(actions).to(self.device)
        loss, _ = self.action_generator.loss(perceptual_emb, actions, latent_goal, noise, sigmas)
        return loss
    
    def denoise_actions(
        self,
        perceptual_emb: torch.Tensor,
        latent_goal: torch.Tensor,
        noise: torch.Tensor = None,
    ):
        """
        Denoise the next sequence of actions

        `noise`, when given, replaces the starting-noise draw below. It must already be
        scaled by sigma_max and shaped [B, act_window_size, action_dim]. It lets the eval
        harness make the starting noise a deterministic function of the rollout instead
        of whatever the global RNG happens to hold (see envs.robocasa.utils.eval_noise).
        The distribution is unchanged; only its reproducibility is.
        """
        input_state = perceptual_emb
        sigmas = self.get_noise_schedule(self.num_sampling_steps, self.noise_scheduler)
        if noise is not None:
            x = noise.to(self.device)
        else:
            x = torch.randn((latent_goal['lang'].shape[0], self.act_window_size, self.action_dim), device=self.device) * self.sigma_max

        actions = self.sample_loop(sigmas, x, input_state, latent_goal, self.sampler_type)

        return actions
    
    def sample_loop(
        self, 
        sigmas, 
        x_t: torch.Tensor,
        state: torch.Tensor, 
        goal: torch.Tensor, 
        sampler_type: str,
        ):
        """
        Main method to generate samples depending on the chosen sampler type. DDIM is the default as it works well in all settings.
        """

        if sampler_type == 'ddim':
            x_0 = utils.sample_ddim(self.action_generator, state, x_t, goal, sigmas, disable=True)
        else:
            raise ValueError('desired sampler type not found!')
        return x_0
    
    def get_noise_schedule(self, n_sampling_steps, noise_schedule_type):
        """
        Get the noise schedule for the sampling steps. Describes the distribution over the noise levels from sigma_min to sigma_max.
        """
        if noise_schedule_type == 'exponential':
            return utils.get_sigmas_exponential(n_sampling_steps, self.sigma_min, self.sigma_max, self.device)
        else:
            raise ValueError('Unknown noise schedule type')
    
    def make_sample_density(self):
        if self.sigma_sample_density_type == 'loglogistic':
            loc = math.log(self.sigma_data)
            scale = 0.5
            min_value = self.sigma_min
            max_value = self.sigma_max
            return partial(utils.rand_log_logistic, loc=loc, scale=scale, min_value=min_value, max_value=max_value)
        else:
            raise NotImplementedError(f"Sigma sample density type {self.sigma_sample_density_type} not implemented.")