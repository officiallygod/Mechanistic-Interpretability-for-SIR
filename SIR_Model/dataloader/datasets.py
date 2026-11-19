import json
import clip
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from einops import einops
import logging

from dataloader.utils import combine_graph_modalities, fuse_graphs, pad_sequence

log = logging.getLogger(__name__)

# Fixed width the variable-length `object` observation is padded to; must match
# RoboCasa_Manager.OBJECT_OBS_DIM.
OBJECT_OBS_DIM = 56

# The normalisation that was applied when the image tensors were written. It has to be
# reproduced exactly to invert it — including the 0.255 blue-channel std, which differs
# from the 0.225 used to denormalise graph crops in
# `utils/generate_graph_dataset_robocasa.py` and `envs/robocasa/kitchen.py:424`. That
# inconsistency is pre-existing and baked into every img_tensor on disk; matching it here
# is what makes the round trip below exact, so do not "fix" it to 0.225. The rollout
# transform at `envs/robocasa/kitchen.py:49` uses 0.255 too, which is what makes training
# and rollout see the same pixels.
_IMG_NORM_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMG_NORM_STD = torch.tensor([0.229, 0.224, 0.255]).view(1, 3, 1, 1)


def _to_uint8_pixels(normalised: torch.Tensor) -> torch.Tensor:
    """Invert the export normalisation and store 8-bit pixels.

    Float32 costs 0.59 MB per frame per camera. Across the 24 tasks that is 211 GB
    resident on a 123 GB machine — the image baseline simply cannot be built. uint8
    brings it to 53 GB.

    Lossless here, not merely cheap: these tensors are the original natively-128
    recording, so inverting the normalisation lands them exactly on the 8-bit grid
    (measured max off-grid error 2e-5, round-trip error 0.0 on all 24 tasks). fp16 would
    have been the lossy option at twice the size.
    """
    pixels = normalised * _IMG_NORM_STD + _IMG_NORM_MEAN
    return pixels.mul_(255.0).round_().clamp_(0.0, 255.0).to(torch.uint8)


def _from_uint8_pixels(pixels: torch.Tensor) -> torch.Tensor:
    """Re-apply the export normalisation to what `_to_uint8_pixels` stored."""
    return (pixels.to(torch.float32).div_(255.0) - _IMG_NORM_MEAN) / _IMG_NORM_STD


class RoboCasaDataset(Dataset):
    def __init__(self,
                 data: any,
                 keys: list,
                 offset: int,
                 data_directory: str,
                 obs_window: int,
                 action_window: int,
                 prop_mod: list,
                 img_mod: list,
                 graph_mod: list,
                 use_graph_fusion: bool,
                 use_splitted_modalities: bool,
                 cropped_img_dim: int,
                 model_name: str = None,
                 task_name: str = None
    ):
        self.data_directory = data_directory
        
        self.use_prop = True if len(prop_mod) > 0 else False
        self.use_img = True if len(img_mod) > 0 else False
        self.use_graph = True if len(graph_mod) > 0 else False
        
        self.use_graph_fusion = use_graph_fusion
        self.task_name = task_name

        # `__getitem__` takes `raw_graph[0]` regardless of the window, and
        # `TransformerCoarsening` indexes `lang_emb` by *graph* index — which equals the
        # batch index only while one sample contributes one graph. Both are correct at
        # obs_window == 1 and silently wrong above it (the policy would see the window's
        # first frame repeated, with FiLM conditioning drawn from the wrong sample), so
        # fail here rather than train something that looks fine.
        if self.use_graph and obs_window > 1:
            raise NotImplementedError(
                f"obs_window={obs_window} is not supported with graph modalities: the "
                f"dataset serves only the window's first graph and the scorer's language "
                f"conditioning assumes one graph per sample. Use obs_window=1, or "
                f"implement per-frame graph batching first.")

        self.obs_window = obs_window
        self.action_window = action_window
        self.prop_mod = prop_mod
        self.img_mod = img_mod
        self.graph_mod = graph_mod
        
        self.prop_obs = []
        self.img_obs = []
        if use_graph_fusion:
            self.graph_obs = []
        else:
            self.graph_obs_left = []
            self.graph_obs_right = []
        
        self.actions = []
        self.lang_goals = []
        self.lang_goals_text = []
        
        self.dataset_length = 0
        self.demo_lengths = []
        
        language_encoder, _ = clip.load("ViT-B/32", device='cpu')
        
        self.is_cropped_fusion = False
        self.merged_mod_name = None

        if self.use_graph:
            self.graph_data_left = {}
            self.graph_data_right = {}
            for mod in self.graph_mod:
                if mod == "cropped_image_feature":
                    crop_str = "_" + str(cropped_img_dim)
                    if "fusion" in model_name:
                        crop_str = "_fusion" + crop_str
                        self.is_cropped_fusion = True
                else:
                    crop_str = ""
                left_graph_path = self.data_directory + "/" + mod + crop_str + "_left_image.pth"
                right_graph_path = self.data_directory + "/" + mod + crop_str + "_right_image.pth"
                log.info(f"Loading graph modality '{mod}' from:\n Left: {left_graph_path}\n Right: {right_graph_path}")
                self.graph_data_left[mod] = (torch.load(left_graph_path, weights_only=False))
                self.graph_data_right[mod] = (torch.load(right_graph_path, weights_only=False))
            if not use_splitted_modalities and len(self.graph_mod) > 1:
                self.merged_mod_name = "_".join(self.graph_mod)
        
        for key in tqdm(keys):
            
            dl = len(data['data'][key]['actions'][()])
            self.dataset_length += dl
            self.demo_lengths.append(dl)
            
            idx = int(key.split('_')[-1]) - offset
            
            if self.use_prop:
                single_prop_obs = self.get_proprioceptive_obs(data['data'][key]['obs'], prop_mod)
                self.prop_obs.append(single_prop_obs)
            
            if self.use_img:
                single_img_obs = self.get_image_obs(img_mod, idx)
                self.img_obs.append(single_img_obs)
                
            if self.use_graph:
                if use_graph_fusion:
                    self.graph_obs.append(self.get_graph_obs(idx, use_splitted_modalities))
                else:
                    left, right = self.get_graph_obs(idx, use_splitted_modalities)
                    self.graph_obs_left.append(left)
                    self.graph_obs_right.append(right)
                    
            action = torch.tensor(data['data'][key]['actions'][()])
            self.actions.append(action)
                        
            lang_text = json.loads(data['data'][key].attrs.get("ep_meta", None))['lang']
            self.lang_goals.append(lang_text)
            self.lang_goals_text.append(lang_text)

        with torch.no_grad():
            lang_goal_tokenized = clip.tokenize(self.lang_goals)
            self.lang_goals = language_encoder.encode_text(lang_goal_tokenized).to(torch.float32)
            
        if self.use_graph and not use_splitted_modalities and len(self.graph_mod) > 1:
            # Update the class attribute so __getitem__ knows the new key
            self.graph_mod = [self.merged_mod_name]

    def get_proprioceptive_obs(self, demo_data, mod):
        modalities = []
        for m in mod:
            values = torch.as_tensor(np.asarray(demo_data[m][()]))
            if m == 'object':
                # Variable width per task (14 per sampled object); pad to the fixed
                # budget the manager assumes. Only the padded tensor is appended —
                # appending the raw one as well silently doubled the feature width
                # and broke the dim the manager had computed.
                padded = torch.zeros((values.shape[0], OBJECT_OBS_DIM))
                padded[:, :values.shape[1]] = values
                values = padded
            modalities.append(values.to(torch.float32))

        observation = torch.concatenate(modalities, dim=-1).type(torch.float32)
        return observation
    
    def get_image_obs(self, mod, idx):
        img_tensor = torch.load(self.data_directory + "/img_tensor_demo_" + str(idx) + ".pth")
        img_tensor = einops.rearrange(img_tensor, "step (num c h w) -> step num c h w", num=3, c=3, h=128, w=128)
        
        img_obs = {}

        # `img_tensor[:, k]` is a *view*, so keeping one pins the storage of all three
        # cameras — asking for a single camera used to cost exactly as much RAM as asking
        # for three. `_to_uint8_pixels` materialises a new tensor, which drops the parent.
        camera_index = {'inhand': 0, 'left': 1, 'right': 2}
        for m in mod:
            if m in camera_index:
                img_obs[m] = _to_uint8_pixels(img_tensor[:, camera_index[m]])

        return img_obs
    
    def get_graph_obs(self, idx, use_splitted_modalities):
        local_graph_mod = self.graph_mod
        # Which modalities each key is built from, in concatenation order.
        mod_order_map = {mod: [mod] for mod in self.graph_mod}

        # Handle merging modalities if required
        if not use_splitted_modalities and len(self.graph_mod) > 1:
            new_data_left = []
            new_data_right = []
                
            num_steps = len(self.graph_data_left[self.graph_mod[0]][idx])
            
            for j in range(num_steps):
                new_data_left.append(combine_graph_modalities(self.graph_data_left, self.graph_mod, idx, j))
                new_data_right.append(combine_graph_modalities(self.graph_data_right, self.graph_mod, idx, j))
            
            graph_data_left = {self.merged_mod_name: new_data_left}
            graph_data_right = {self.merged_mod_name: new_data_right}
            
            local_graph_mod = [self.merged_mod_name]
            mod_order_map = {self.merged_mod_name: list(self.graph_mod)}
        else:
            graph_data_left = {}
            graph_data_right = {}
            for mod in self.graph_mod:
                graph_data_left[mod] = self.graph_data_left[mod][idx]
                graph_data_right[mod] = self.graph_data_right[mod][idx]
                        
        # Handle merging graphs if required
        if self.use_graph_fusion:
            fused_graph_data = {}
            for mod in local_graph_mod:
                fused_graph_data[mod] = []
                
                for step_idx in range(len(graph_data_left[mod])):
                    fused_graph_data[mod].append(fuse_graphs(graph_data_left, graph_data_right, mod, mod_order_map[mod], step_idx, self.is_cropped_fusion))
            return fused_graph_data
        else:
            return graph_data_left, graph_data_right
    
    def get_indices(self, idx, window):
        """
        Resolves a global dataset index into a specific demonstration index
        and the start/end indices within that demonstration.
        Ensures indices do not exceed the length of the specific demonstration.
        """
        demo_index = 0
        inside_start_index = 0
        
        # 1. Find which demo this index belongs to
        for i in range(len(self.demo_lengths)):
            temp = idx - self.demo_lengths[i]
            if temp < 0:
                demo_index = i
                inside_start_index = idx
                break
            else:
                idx = temp
        
        # 2. Calculate the end index based on the window size
        # We clamp the end index to the max length of the current demo
        demo_len = self.demo_lengths[demo_index]
        inside_end_index = min(inside_start_index + window, demo_len)
        
        return demo_index, inside_start_index, inside_end_index
    
    def __getitem__(self, idx):
        item = {
            'observation': {},
            'action': {},
            'goal': {}
        }
        
        # --- OBSERVATIONS ---
        # Get indices for Observation Window
        demo_idx, obs_start, obs_end = self.get_indices(idx, self.obs_window)
        
        if self.use_prop:
            # Slicing
            raw_prop = self.prop_obs[demo_idx][obs_start:obs_end]
            # Padding if out of bounds (handled by helper)
            item['observation']['obs_prop'] = pad_sequence(raw_prop, self.obs_window)
            
        if self.use_img:
            item['observation']['obs_img'] = {}
            for m in self.img_mod:
                raw_img = _from_uint8_pixels(self.img_obs[demo_idx][m][obs_start:obs_end])
                item['observation']['obs_img'][m] = pad_sequence(raw_img, self.obs_window)
                
        if self.use_graph:
            item['observation']['obs_graph'] = {}
            for m in self.graph_mod:
                if not self.use_graph_fusion:
                    raw_graph_left = self.graph_obs_left[demo_idx][m][obs_start:obs_end]
                    raw_graph_right = self.graph_obs_right[demo_idx][m][obs_start:obs_end]
                    # TODO padding
                    item['observation']['obs_graph'][m + '_left'] = raw_graph_left[0]
                    item['observation']['obs_graph'][m + '_right'] = raw_graph_right[0]
                else:
                    raw_graph = self.graph_obs[demo_idx][m][obs_start:obs_end]
                    # Graph padding usually implies repeating the last graph object TODO
                    if len(raw_graph) < self.obs_window:
                        last = raw_graph[-1]
                        raw_graph.extend([last] * (self.obs_window - len(raw_graph)))
                    item['observation']['obs_graph'][m] = raw_graph[0]

        # --- ACTIONS ---
        # Get indices for Action Window
        _, act_start, act_end = self.get_indices(idx, self.action_window)
        
        raw_action = self.actions[demo_idx][act_start:act_end]
        item['action']['eef'] = pad_sequence(raw_action, self.action_window)

        # --- GOALS ---
        # Language goal is static per demonstration
        item['goal']['lang'] = self.lang_goals[demo_idx]
        item['goal']['lang_text'] = self.lang_goals_text[demo_idx]

        item['task_name'] = self.task_name

        return item
            
    def __len__(self):
        return self.dataset_length