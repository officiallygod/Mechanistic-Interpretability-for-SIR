import os
import cv2
import imageio
import numpy as np
import torch
from torchvision import transforms

from torchvision.ops import masks_to_boxes
from dataloader.datasets import OBJECT_OBS_DIM
from dataloader.utils import combine_graph_modalities, fuse_graphs
from utils.node_scores import NodeScoreTracker
from envs.robocasa.utils import (
    accumulate_faithfulness, create_kitchen_env, eval_noise, mean_faithfulness,
    per_rollout_seed, rollout_artifact_dir, seed_rollout,
    visualize_sparsification_rollout,
)

from tqdm import tqdm
import networkx as nx

import logging

from networks.vision_encoder.cnn import SimpleImageEncoder
from networks.vision_encoder.utils import crop_and_resize_to_64, crop_and_resize_to_64_for_fusion
from utils.create_graphs import create_graph_datapoint
from utils.generate_graph_dataset_robocasa import OBJECT_NAMES_IMAGES, get_bb_pos
from utils.point_geom import (
    POINT_GEOM_VOXEL_SIZE, build_robot_mask, point_geom_features_from_views,
    robot_class_ids, stack_point_geom,
)

log = logging.getLogger(__name__)


class EmptyObservationError(RuntimeError):
    """No object was detectable in a camera view.

    Reached when a rollout drives the robot into a pose where the agentview cameras
    frame nothing in the known class list, or when a dead EGL context returns blank
    segmentation buffers. The graph pipeline assumes at least one node at every stage
    (networkx -> PyG conversion, fusion, dense batching, and the MDT cross-attention
    mask, whose all-masked softmax row is NaN), so there is no valid observation to hand
    the policy.

    Previously `helper_function` returned a 2-tuple here, which raised an unrelated
    "not enough values to unpack" ValueError in `get_data_from_img`. Raising a named
    error lets the callers tell this case apart from a genuine crash: the parallel
    worker scores an isolated blank as a policy failure and exits on a run of them
    (dead renderer), and the sequential tester scores it as a failure.
    """


class RoboCasaKitchenTester():
    def __init__(self,
                 dataset_path: str,
                 task_list: list[str],
                 use_depth: bool,
                 cropped_image_feature_encoder: torch.nn.Module = None,
                 text_encoder: tuple[torch.nn.Module, torch.nn.Module] = None,
                 seed: int = 42,
                 ):

        self.env_list, self.task_list = create_kitchen_env(dataset_path, task_list, use_depth, seed)
        
        self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((128, 128)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.255])
            ])
        
        self.transform_cropped_feature = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((128, 128)),
            ])
        
        self.cropped_image_feature_encoder = cropped_image_feature_encoder
        self.text_encoder = text_encoder

        # Set per task in test(); point_geom needs the live simulator.
        self._current_env = self.env_list[0] if self.env_list else None

    def test(self, method, manager, store_videos: bool, eval_n_times: int, working_dir: str, test_count: int, epoch: int):
        result_dict = {}
        # {task: {rollout_idx: success}} / {task: {rollout_idx: {mod: {metric: v}}}},
        # mirroring what the parallel tester exposes so the manager can aggregate
        # both eval paths through the same code.
        self.per_scene_success = {}
        self.per_scene_faithfulness = {}
        # Rollout-time node scores, mirroring the parallel path's server-side tracker.
        self.node_scores = NodeScoreTracker()

        for env, task_name in zip(self.env_list, self.task_list):
            total_reward = 0
            self.per_scene_success[task_name] = {}
            self.per_scene_faithfulness[task_name] = {}
            log.info("Evaluation of: " + task_name)
            # point_geom reads camera/base/gripper poses straight off the simulator, so
            # the feature builders need a handle on the env currently being evaluated.
            self._current_env = env
            
            for i in range(eval_n_times):
                
                frames = []

                # The sequential path did not seed per rollout at all, so its
                # scenes were not reproducible between runs and could not be
                # paired against the parallel path. Use the same scheme as the
                # parallel workers so both agree.
                seed_rollout(env, per_rollout_seed(
                    task_name, i, getattr(manager, 'worker_seed_base', 42)))
                obs_all = env.reset()
                
                self.cls_list = {cls: i for i, cls in enumerate(list(env.model.classes_to_ids.keys()))}
                self.cls_numbers = []
                for class_name in list(self.cls_list.keys()):
                    self.cls_numbers.append(self.cls_list[class_name] + 1)
                self.cls_numbers = torch.tensor(self.cls_numbers)
                
                obs = {}
                
                left_graph = {}
                right_graph = {}
                for mod in manager.graph_modalities:
                    left_graph[mod] = nx.DiGraph()
                    right_graph[mod] = nx.DiGraph()

                if manager.use_proprioceptive:
                    prop_obs = self.create_prop_obs(obs_all, manager)
                    obs['obs_prop'] = prop_obs
                if manager.use_image:
                    img_obs = self.create_image_obs(obs_all, manager)
                    obs['obs_img'] = img_obs
                # A view with no known object ends the rollout as a failure rather than
                # crashing the whole evaluation. There is no worker to respawn here, so
                # a dead renderer shows up as every rollout blanking -- logged each time.
                blind = False
                if manager.use_graph:
                    try:
                        graph_obs = self.create_graph_obs(obs_all, left_graph, right_graph, manager)
                        obs['obs_graph'] = graph_obs
                    except EmptyObservationError as e:
                        log.warning(f"{task_name} rollout {i}: blank view at reset ({e}); "
                                    f"scoring as a failure")
                        blind = True

                lang_goal = env.get_ep_meta()['lang']
                print("Goal: ", lang_goal)
                
                success = 0
                step_counter = 0
                pbar = tqdm(total=env.horizon)

                presence_counts = {}
                selection_timeline = {}
                total_rollout_steps = 0
                faith_sums = {}
                faith_steps = {}

                while not blind and step_counter < env.horizon:
                    batch = {'observation': obs,
                             'goal': {'lang': lang_goal},
                             'task_name': [task_name]
                             }

                    state, _, goal = method.preprocess_batch(batch)

                    self.node_scores.update_from_method(method)

                    if hasattr(method.graph_encoder, 'last_sparsification'):
                        for mod_key, info in method.graph_encoder.last_sparsification.items():
                            node_names = info['node_names']
                            if node_names is None:
                                continue
                            bucket = presence_counts.setdefault(mod_key, {})
                            sampled_idx = info['sampled_nodes'].tolist()
                            step_names = [node_names[idx] for idx in sampled_idx]
                            for name in step_names:
                                bucket[name] = bucket.get(name, 0) + 1
                            selection_timeline.setdefault(mod_key, []).append(step_names)
                            accumulate_faithfulness(
                                faith_sums, faith_steps, mod_key, step_names,
                                node_names, task_name, lang_goal,
                            )
                        total_rollout_steps += 1

                    if getattr(manager, 'deterministic_eval_noise', False):
                        # Same key as the parallel server, so both paths draw the same
                        # noise for the same scene and step.
                        noise = eval_noise(method, manager.eval_noise_seed,
                                           [(task_name, i, step_counter)])
                        pred_action = method.predict(state, goal, noise=noise)
                    else:
                        pred_action = method.predict(state, goal)
                    for a in range(pred_action.shape[1]):
                        if store_videos:
                            img = obs_all['robot0_agentview_left_image']
                            frame = cv2.resize(img, dsize=(640, 480), interpolation=cv2.INTER_CUBIC)
                            frames.append(frame)
                        action = method.clip_action(pred_action[0][a], manager.y_bounds).reshape(-1).detach().cpu().numpy()
                        obs_all, _, _, _ = env.step(action)
                        step_counter += 1
                        pbar.update(1)

                    obs = {}
                    if manager.use_proprioceptive:
                        prop_obs = self.create_prop_obs(obs_all, manager)
                        obs['obs_prop'] = prop_obs
                    if manager.use_image:
                        img_obs = self.create_image_obs(obs_all, manager)
                        obs['obs_img'] = img_obs
                    if manager.use_graph:
                        try:
                            graph_obs = self.create_graph_obs(obs_all, left_graph, right_graph, manager)
                            obs['obs_graph'] = graph_obs
                        except EmptyObservationError as e:
                            log.warning(f"{task_name} rollout {i}: blank view at step "
                                        f"{step_counter} ({e}); scoring as a failure")
                            break

                    if env._check_success():
                        log.info(f"Success in {i}")
                        success = 1
                        total_reward += 1
                        break

                add_str = ''
                if epoch is not None:
                    add_str = f"epoch_{epoch}_"

                self.per_scene_success[task_name][i] = success
                rollout_faith = mean_faithfulness(faith_sums, faith_steps)
                if rollout_faith:
                    self.per_scene_faithfulness[task_name][i] = rollout_faith

                if store_videos:
                    video_filename = add_str + f"env_{task_name}_{test_count}_rollout_{i}_success_{success}.mp4"
                    video_dir = rollout_artifact_dir(working_dir, 'videos', task_name, test_count)
                    video_filepath = os.path.join(video_dir, video_filename)
                    # Save the frames as a video using imageio
                    imageio.mimsave(video_filepath, frames, fps=30)

                if (presence_counts and total_rollout_steps > 0
                        and getattr(manager, 'store_sparsification_plots', False)):
                    spars_dir = rollout_artifact_dir(working_dir, 'sparsification', task_name, test_count)
                    for mod_key, counts in presence_counts.items():
                        png_name = add_str + f"env_{task_name}_{test_count}_rollout_{i}_{mod_key}_success_{success}.png"
                        visualize_sparsification_rollout(
                            counts,
                            total_rollout_steps,
                            os.path.join(spars_dir, png_name),
                            timeline=selection_timeline.get(mod_key),
                            title=f"{task_name} rollout {i} — {mod_key}",
                        )

            log.info(f"Total reward in {task_name}: {total_reward}")
            result_dict["Avg_Test_Reward_" + task_name] = total_reward / eval_n_times

            env.close()
        return result_dict
    
    def create_prop_obs(self, obs_all, manager):
        """Build the proprioceptive observation from a live env step.

        Mirrors `RoboCasaDataset.get_proprioceptive_obs` so a policy sees the same
        layout at rollout as it did in training: modalities concatenated in config
        order, with the variable-width `object` observation zero-padded to
        OBJECT_OBS_DIM. Returned as (batch=1, time=1, prop_dim), matching the
        convention `create_image_obs` uses.

        This was a `pass` stub, so any run with `use_proprioceptive` trained happily
        and then died at the first rollout on `obs[obs_prop]` being None.
        """
        modalities = []
        for m in manager.prop_modalities:
            if m not in obs_all:
                # `object` in particular exists in the demo HDF5s but is not exposed
                # by the live env, so such a run would only fail here at rollout.
                raise KeyError(
                    f"proprioceptive modality '{m}' is not provided by the live "
                    f"environment (available: {sorted(obs_all.keys())})"
                )
            values = torch.as_tensor(np.asarray(obs_all[m]), dtype=torch.float32).reshape(-1)
            if m == 'object':
                padded = torch.zeros(OBJECT_OBS_DIM, dtype=torch.float32)
                padded[:values.shape[0]] = values
                values = padded
            modalities.append(values)

        observation = torch.concatenate(modalities, dim=-1)
        return observation.unsqueeze(0).unsqueeze(0)
    
    def create_image_obs(self, obs_all, manager):
        obs = {}
        for mod in manager.image_modalities:
            if mod == 'left':
                img = obs_all['robot0_agentview_left_image']
            elif mod == 'right':
                img = obs_all['robot0_agentview_right_image']
            elif mod == 'inhand':
                img = obs_all['robot0_eye_in_hand_image']
            else:
                raise ValueError(f'Unknown image modality: {mod}')
            
            img = self.transform(img).unsqueeze(0).unsqueeze(0)  # add batch and time dimensions
            
            obs[mod] = img
        return obs
    
    def create_graph_obs(self, obs_all, left_graph, right_graph, manager):
        left_data_point = {}
        right_data_point = {}
        
        for mod in manager.graph_modalities:
            left_object_names, right_object_names, left_objects, right_objects = self.get_data_from_img(obs_all, mod, self.cls_numbers, self.cls_list)
            left_data_point[mod] = create_graph_datapoint(left_graph[mod], left_object_names, left_objects)
            right_data_point[mod] = create_graph_datapoint(right_graph[mod], right_object_names, right_objects)

        local_graph_mods = manager.graph_modalities
        # Which modalities each key is built from, in concatenation order.
        mod_order_map = {mod: [mod] for mod in manager.graph_modalities}

        if not manager.use_splitted_modalities and len(manager.graph_modalities) > 1:
            merged_graph_modalities = "_".join(manager.graph_modalities)
            graph_data_left = {merged_graph_modalities: combine_graph_modalities(left_data_point, manager.graph_modalities)}
            graph_data_right = {merged_graph_modalities: combine_graph_modalities(right_data_point, manager.graph_modalities)}
            local_graph_mods = [merged_graph_modalities]
            mod_order_map = {merged_graph_modalities: list(manager.graph_modalities)}
        else:
            graph_data_left = left_data_point
            graph_data_right = right_data_point

        if manager.use_graph_fusion:
            graph_obs = {}
            for mod in local_graph_mods:
                fused_graph = fuse_graphs(graph_data_left, graph_data_right, mod, mod_order_map[mod], is_cropped_fusion=manager.is_cropped_fusion)
                graph_obs[mod] = fused_graph
        else:
            graph_obs = {}
            for mod in local_graph_mods:
                graph_obs[mod + '_left'] = graph_data_left[mod]
                graph_obs[mod + '_right'] = graph_data_right[mod]
        return graph_obs
                
    def get_data_from_img(self, obs, mod, cls_numbers: list, cls_list: list):

        left_object_names, left_boxes, left_morph_masks, left_img = self.helper_function(obs, 'agentview_left', cls_numbers, cls_list)
        right_object_names, right_boxes, right_morph_masks, right_img = self.helper_function(obs, 'agentview_right', cls_numbers, cls_list)

        if mod == "point_geom":
            return self.get_point_geom_from_obs(
                obs, left_object_names, left_morph_masks, right_object_names, right_morph_masks)

        if mod == "one_hot_labels":
            objects = []
            for name in left_object_names:
                one_hot = torch.zeros(len(OBJECT_NAMES_IMAGES))
                index = OBJECT_NAMES_IMAGES.index(name)
                one_hot[index] = 1.0
                objects.append(one_hot)
            left_objects = torch.stack(objects)
        elif mod == "siglip_text_emb":
            prompts = [f"The current observed object in the image is {name}" for name in left_object_names]
            inputs = self.text_encoder[1](
                text=prompts, 
                padding="max_length", 
                return_tensors="pt"
            )
            with torch.no_grad():
                text_features = self.text_encoder[0].text_model(**inputs)
                left_objects = text_features.pooler_output
        elif mod == "bb_coordinates":
            objects = []
            for i in range(left_boxes.shape[0]):
                objects.append(get_bb_pos(left_boxes[i]) / 127)
            left_objects = torch.stack(objects)
        elif mod == "cropped_image_feature":
            object_img = []
            for i in range(left_morph_masks.shape[0]):
                if isinstance(self.cropped_image_feature_encoder, SimpleImageEncoder):
                    if left_object_names[i] in right_object_names:
                        r_obj_idx = right_object_names.index(left_object_names[i])
                        obj_r = right_morph_masks[r_obj_idx]
                    else:
                        obj_r = None
                    object_img.append(crop_and_resize_to_64_for_fusion(left_img, right_img, left_morph_masks[i], obj_r))
                else:
                    object_img.append(crop_and_resize_to_64(left_img, left_morph_masks[i]))
            # no_grad is not just an optimization: with grad enabled these node features
            # are non-leaf tensors carrying a grad_fn, and the parallel tester cannot ship
            # them over a multiprocessing queue ("Cowardly refusing to serialize non-leaf
            # tensor which requires_grad") — the put fails inside the queue's feeder
            # thread, so the worker hangs waiting for a reply that never comes.
            with torch.no_grad():
                left_objects = self.cropped_image_feature_encoder(torch.stack(object_img)).squeeze(0)
            
        if mod == "one_hot_labels":
            objects = []
            for name in right_object_names:
                one_hot = torch.zeros(len(OBJECT_NAMES_IMAGES))
                index = OBJECT_NAMES_IMAGES.index(name)
                one_hot[index] = 1.0
                objects.append(one_hot)
            right_objects = torch.stack(objects)
        elif mod == "siglip_text_emb":
            prompts = [f"The current observed object in the image is {name}" for name in right_object_names]
            inputs = self.text_encoder[1](
                text=prompts, 
                padding="max_length", 
                return_tensors="pt"
            )
            with torch.no_grad():
                text_features = self.text_encoder[0].text_model(**inputs)
                right_objects = text_features.pooler_output
        elif mod == "bb_coordinates":
            objects = []
            for i in range(right_boxes.shape[0]):
                objects.append(get_bb_pos(right_boxes[i]) / 127)
            right_objects = torch.stack(objects)
        elif mod == "cropped_image_feature":
            object_img = []
            for i in range(right_morph_masks.shape[0]):
                if isinstance(self.cropped_image_feature_encoder, SimpleImageEncoder):
                    if right_object_names[i] in left_object_names:
                        l_obj_idx = left_object_names.index(right_object_names[i])
                        obj_l = left_morph_masks[l_obj_idx]
                    else:
                        obj_l = None
                    object_img.append(crop_and_resize_to_64_for_fusion(left_img, right_img, obj_l, right_morph_masks[i]))
                else:
                    object_img.append(crop_and_resize_to_64(right_img, right_morph_masks[i]))
            with torch.no_grad():
                right_objects = self.cropped_image_feature_encoder(torch.stack(object_img)).squeeze(0)
        
        return left_object_names, right_object_names, left_objects, right_objects

    def get_point_geom_from_obs(self, obs, left_names, left_masks, right_names, right_masks):
        """Online twin of utils/generate_point_geom_dataset.

        Must stay numerically identical to the offline pass — same voxel size, same
        frame, same masks — otherwise the policy is evaluated on a different feature
        distribution than it was trained on.
        """
        sim = self._current_env.sim
        robot_ids = robot_class_ids(self.cls_list)

        view_inputs = []
        for view, names, masks in (('agentview_left', left_names, left_masks),
                                   ('agentview_right', right_names, right_masks)):
            camera = 'robot0_' + view
            robot_mask = build_robot_mask(obs.get(camera + '_segmentation_class'), robot_ids)
            masks_by_name = {name: np.asarray(masks[i]) for i, name in enumerate(names)}
            view_inputs.append((camera, obs.get(camera + '_depth'), masks_by_name, robot_mask))

        features = point_geom_features_from_views(sim, view_inputs, POINT_GEOM_VOXEL_SIZE)

        return (left_names, right_names,
                stack_point_geom(features, left_names),
                stack_point_geom(features, right_names))

    def helper_function(self, obs, view, cls_numbers, cls_list):
        img = self.transform(obs['robot0_' + view + '_image'])
        mask = torch.tensor(obs['robot0_' + view + '_segmentation_class'][:, :, 0])
        
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

        img = img*std + mean

        obj_ids_unique = torch.unique(mask)
        obj_ids = []
        for id in cls_numbers:
            if id in obj_ids_unique:
                obj_ids.append(id)
        obj_ids = torch.tensor(obj_ids)

        masks = mask == obj_ids[:, None, None]

        new_obj_ids = []
        morph_masks = []

        for r, mask in enumerate(masks):
            new_mask = cv2.morphologyEx(mask.numpy().astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2)))
            if np.sum(new_mask) == 0:
                pass
            else:
                morph_masks.append(new_mask)
                new_obj_ids.append(obj_ids[r].item())

        if new_obj_ids == []:
            raise EmptyObservationError(
                f"no known object visible in {view}; rollout cannot continue")
        
        morph_masks = torch.tensor(np.array(morph_masks))

        boxes = masks_to_boxes(morph_masks)

        object_names = []

        for i, id in enumerate(new_obj_ids):
            obj_name = list(cls_list.keys())[id - 1]
            object_names.append(obj_name)
            
        return object_names, boxes, morph_masks, img