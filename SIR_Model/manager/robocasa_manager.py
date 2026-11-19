import json
import os

import torch
from dataloader.dataloader import dataset_split_robocasa
from envs.robocasa.kitchen import RoboCasaKitchenTester
from envs.robocasa.parallel.parallel_tester import RoboCasaKitchenParallelTester
from envs.robocasa.utils import TASK_LIST
from manager.base_manager import Base_Manager
from networks.vision_encoder.utils import load_pretrained_image_encoder
from utils.generate_graph_dataset_robocasa import (
    FAITHFULNESS_METRICS, OBJECT_NAMES_IMAGES, create_graphs_and_save,
)
from utils.point_geom import POINT_GEOM_DIM
#from transformers import SiglipProcessor, SiglipModel, SiglipTokenizer

import logging

log = logging.getLogger(__name__)

FAITHFULNESS_TABLE_KEY = "faithfulness_per_scene"


def _mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def _mean_metrics(dicts):
    """Element-wise mean over a list of {metric: value} dicts (None if empty)."""
    dicts = [d for d in dicts if d]
    if not dicts:
        return None
    metrics = set()
    for d in dicts:
        metrics.update(d.keys())
    return {m: _mean([d[m] for d in dicts if m in d]) for m in sorted(metrics)}


def _summarize_faithfulness(faith_all):
    """Aggregate the sub-graph faithfulness grid into wandb payloads + JSON.

    `faith_all` is {task: {round_idx: {rollout_idx: {mod: {metric: value}}}}},
    every leaf already averaged over one rollout's inference steps.

    Two groupings are produced, matching the two questions asked of an eval:
      - per repeat-round: mean over that round's scenes (`times_repeat` values)
      - per scene:        mean over that scene's rounds (`eval_n_times` values)

    Returns (log_dict, table_rows, json_payloads). `log_dict` holds only scalars;
    the per-scene grouping goes to `table_rows` because 24 tasks x 50 scenes x 4
    metrics would be thousands of unusable wandb series.
    """
    # Flatten to {mod: {task: {round: {scene: {metric: v}}}}}.
    by_mod = {}
    for task, rounds in (faith_all or {}).items():
        for round_idx, scenes in (rounds or {}).items():
            for scene_idx, per_mod in (scenes or {}).items():
                for mod, metrics in (per_mod or {}).items():
                    (by_mod.setdefault(mod, {}).setdefault(task, {})
                          .setdefault(int(round_idx), {})[int(scene_idx)]) = metrics

    log_dict = {}
    table_rows = []
    json_payloads = {}

    for mod in sorted(by_mod):
        per_task_means = []
        for task in sorted(by_mod[mod]):
            rounds = by_mod[mod][task]

            # --- per repeat-round: mean over the scenes of that round ---
            per_round = {}
            for round_idx in sorted(rounds):
                round_mean = _mean_metrics(list(rounds[round_idx].values()))
                if round_mean is None:
                    continue
                per_round[round_idx] = round_mean
                for metric, value in round_mean.items():
                    log_dict[f"faithfulness/{mod}/{task}/round_{round_idx}_{metric}"] = value

            # --- per scene: mean over the rounds that ran that scene ---
            per_scene = {}
            scene_idxs = sorted({s for r in rounds.values() for s in r})
            for scene_idx in scene_idxs:
                runs = [r[scene_idx] for r in rounds.values() if scene_idx in r]
                scene_mean = _mean_metrics(runs)
                if scene_mean is None:
                    continue
                per_scene[scene_idx] = {"n_rounds": len(runs), **scene_mean}
                table_rows.append([task, mod, scene_idx, len(runs)] +
                                  [scene_mean.get(m) for m in FAITHFULNESS_METRICS])

            # --- per task: mean over every rollout of the task ---
            task_mean = _mean_metrics([m for r in rounds.values() for m in r.values()])
            if task_mean is None:
                continue
            per_task_means.append(task_mean)
            for metric, value in task_mean.items():
                log_dict[f"faithfulness/{mod}/{task}/{metric}"] = value

            payload = json_payloads.setdefault(task, {"task": task, "modalities": {}})
            payload["modalities"][mod] = {
                "mean": task_mean,
                "per_round": {str(k): v for k, v in per_round.items()},
                "per_scene": {str(k): v for k, v in per_scene.items()},
                "raw": {str(r): {str(s): m for s, m in scenes.items()}
                        for r, scenes in rounds.items()},
            }

        # --- grand mean over tasks (unweighted, so a task is a task) ---
        grand = _mean_metrics(per_task_means)
        if grand is not None:
            for metric, value in grand.items():
                log_dict[f"faithfulness/{mod}/{metric}"] = value

    return log_dict, table_rows, json_payloads


class RoboCasa_Manager(Base_Manager):
    def __init__(
        self,
        data_path: str,
        load_dataset: bool,
        batch_size: int,
        train_per: float,
        num_workers: int,
        obs_window: int,
        act_window: int,
        action_dim: int,
        times_repeat: int,
        num_train_demos: int = None,
        task_names: list = None,
        prop_modalities: list = None,
        image_modalities: list = None,
        graph_modalities: list = None,
        pretrained_img_encoder_name: str = None,
        use_graph_fusion: bool = False,
        use_splitted_modalities: bool = False,
        # When `parallel_eval` is True and `num_envs > 1`, `test_method` spawns
        # `num_envs` MuJoCo env worker processes and routes their observations
        # through a single GPU-resident policy server thread. With num_envs==1
        # or parallel_eval=False, falls back to the original sequential tester.
        parallel_eval: bool = False,
        num_envs: int = 1,
        batch_timeout_ms: float = 20.0,
        max_eval_batch: int = 16,
        worker_seed_base: int = 42,
        # Start every eval prediction's diffusion from noise keyed on
        # (task, rollout_idx, step) instead of the global RNG, so an evaluation is
        # reproducible rollout-for-rollout (see envs.robocasa.utils.eval_noise). False is
        # the legacy behaviour: an eval-only run reloads the checkpoint's manager config,
        # so checkpoints from before this flag keep evaluating as they did.
        deterministic_eval_noise: bool = False,
        # Seed mixed into that key. None = worker_seed_base. Set separately to measure
        # noise sensitivity on identical scenes.
        eval_noise_seed: int = None,
        # Per-rollout node-selection diagnostic plots, one PNG per rollout per graph
        # modality. OFF by default: nothing reads them back -- every reported number
        # comes from eval_per_scene_<task>.json and node_scores/ -- and a 24-task x
        # 4-seed sweep wrote 359,376 of them (72.9 GB). Turn on per run to eyeball a
        # specific rollout's selection.
        store_sparsification_plots: bool = False,
    ):
        super().__init__(
            data_path,
            load_dataset,
            batch_size,
            train_per,
            num_workers,
            obs_window,
            act_window,
            action_dim,
            task_names,
            prop_modalities,
            image_modalities,
            graph_modalities,
            pretrained_img_encoder_name,
        )
        
        self.times_repeat = times_repeat
        self.num_train_demos = num_train_demos

        self.use_graph_fusion = use_graph_fusion
        self.use_splitted_modalities = use_splitted_modalities
        self.adapted_graph_modalities = graph_modalities

        self.parallel_eval = parallel_eval
        self.num_envs = num_envs
        self.batch_timeout_ms = batch_timeout_ms
        self.max_eval_batch = max_eval_batch
        self.worker_seed_base = worker_seed_base
        self.deterministic_eval_noise = bool(deterministic_eval_noise)
        self.eval_noise_seed = worker_seed_base if eval_noise_seed is None else int(eval_noise_seed)
        self.store_sparsification_plots = bool(store_sparsification_plots)
        
        self.is_cropped_fusion = False
        
        self.cropped_img_dim = 0
        self.cropped_image_feature_encoder = None
        self.text_encoder = (None, None)
        
        # point_geom is the only modality that needs depth rendering at rollout time.
        self.use_point_geom = "point_geom" in self.graph_modalities

        self.prop_dim = self.calculate_proprioceptive_dim(prop_modalities)
        self.graph_dim, self.graph_edge_dim = self.calculate_graph_dim() if self.use_graph else (0, 0)

        self.init_data()
        
    def init_data(self):
        if self.use_graph:
            self.check_and_create_graph_dataset()
        
        if self.load_dataset:
            self.train_dataloader, self.valid_dataloader = dataset_split_robocasa(
                data_directory=self.data_path,
                task_names=self.task_names,
                split=self.train_per,
                obs_window=self.obs_window,
                act_window=self.action_window,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                prop_mod=self.prop_modalities,
                img_mod=self.image_modalities,
                graph_mod=self.graph_modalities,
                use_graph_fusion=self.use_graph_fusion,
                use_splitted_modalities=self.use_splitted_modalities,
                cropped_img_dim=self.cropped_img_dim,
                cropped_model_name=self.pretrained_img_encoder_name,
                num_train_demos=self.num_train_demos,
            )

        self.y_bounds = torch.tensor(
                    [[-1.0000, -1.0000, -1.0000, -1.0000, -0.9629, -0.9857, -1.0000,  0.0000,
                    -0.0000, -0.0286, -0.2071, -1.0000],
                    [ 1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  0.8971,  1.0000,  0.1214,
                    0.6929,  0.0000,  0.0000, -1.0000]])
    
    def check_and_create_graph_dataset(self):
        # Check if the requested dataset is there and if not create it
        datasets_to_create = []

        for mod in self.graph_modalities:
            if "cropped_image_feature" == mod:
                crop_str = "_" + str(self.cropped_img_dim)
                if self.is_cropped_fusion:
                    crop_str = "_fusion" + crop_str
            else:
                crop_str = ""
            
            if self.task_names[0] == "ALL":
                task_names = TASK_LIST
            else:
                task_names = self.task_names
            for task_name in task_names:
                right_graph = not os.path.isfile(os.path.join(self.data_path, task_name, mod + crop_str + "_right_image.pth"))
                left_graph = not os.path.isfile(os.path.join(self.data_path, task_name, mod + crop_str + "_left_image.pth"))
                if right_graph or left_graph:
                    datasets_to_create.append((task_name, mod))
        
        if "siglip_text_emb" in self.graph_modalities:
            pass
            # model_id = "google/siglip-base-patch16-224"
            # model = SiglipModel.from_pretrained(model_id)
            # processor = SiglipProcessor.from_pretrained(model_id)
            # self.text_encoder = (model, processor)
        
        if "cropped_image_feature" in self.graph_modalities:
            self.cropped_image_feature_encoder = load_pretrained_image_encoder(self.pretrained_img_encoder_name)
        
        # point_geom needs a full simulator replay to recover depth, so it cannot be
        # produced by create_graphs_and_save (which reads the pre-extracted object hdf5).
        missing_point_geom = [t for t, m in datasets_to_create if m == "point_geom"]
        if missing_point_geom:
            raise FileNotFoundError(
                "point_geom datasets are missing for: " + ", ".join(sorted(set(missing_point_geom)))
                + "\nThey are built by replaying the demos with depth enabled. Run:\n"
                + "\n".join(f"  python -m utils.generate_point_geom_dataset --task {t}"
                            for t in sorted(set(missing_point_geom)))
            )
        datasets_to_create = [(t, m) for t, m in datasets_to_create if m != "point_geom"]

        if datasets_to_create == []:
            log.info("Requested datasets are available")
        else:
            for task, mod in datasets_to_create:
                log.info(f"Creating dataset for task: {task} and modality: {mod}")
                create_graphs_and_save(
                    dataset_path=self.data_path,
                    task_name=task,
                    graph_modality=mod,
                    encoder_model=self.cropped_image_feature_encoder,
                    text_encoder=self.text_encoder
                )
    
    def _sampler_fields(self):
        """Which action sampler produced a result, written INTO the result file.

        "seeded" = the diffusion starting noise was keyed on (task, rollout_idx, step)
        with `noise_seed`; "legacy" = it came from the global RNG. The two are not
        comparable rollout-for-rollout, so this is recorded at run time rather than
        inferred later from a date. Files written before this field existed carry none,
        and readers must treat a missing field as "legacy".
        """
        if self.deterministic_eval_noise:
            return {"sampler": "seeded", "noise_seed": self.eval_noise_seed}
        return {"sampler": "legacy"}

    def test_method(self, method, store_videos, eval_n_times, working_dir, during_training=False, epoch=None):
        times_repeat = self.times_repeat #if during_training else self.times_repeat
        results_all = {}
        
        task_names = ["CloseDrawer"] if during_training else self.task_names
        if task_names[0] == "ALL":
            task_names = TASK_LIST
        
        # Per-scene accumulation across rounds for paired comparison:
        #   per_scene_all[task][rollout_idx]    = total successes over rounds
        #   per_scene_rounds[task][rollout_idx] = number of rounds that scene ran
        per_scene_all = {}
        per_scene_rounds = {}
        # Sub-graph faithfulness grid: {task: {round: {rollout_idx: {mod: {metric: v}}}}}.
        faith_all = {}

        use_parallel = self.parallel_eval and self.num_envs > 1
        for i in range(times_repeat):
            if use_parallel:
                # `worker_seed_base` is constant across rounds so that for every
                # task, rollout j sees the same scene in every round — matching
                # the round-to-round reproducibility of the sequential tester.
                tester = RoboCasaKitchenParallelTester(
                    num_envs=self.num_envs,
                    batch_timeout_ms=self.batch_timeout_ms,
                    max_eval_batch=self.max_eval_batch,
                    worker_seed_base=self.worker_seed_base,
                    deterministic_noise=self.deterministic_eval_noise,
                    noise_seed=self.eval_noise_seed,
                )
                results = tester.test(
                    method=method,
                    manager=self,
                    store_videos=store_videos,
                    eval_n_times=eval_n_times,
                    working_dir=working_dir,
                    test_count=i,
                    epoch=epoch,
                    task_names=task_names,
                )
            else:
                tester = RoboCasaKitchenTester(
                    dataset_path=self.data_path,
                    task_list=task_names,
                    # point_geom reconstructs its features from depth, so the rollout
                    # cameras must render depth or the online twin has nothing to read.
                    use_depth=self.use_point_geom,
                    cropped_image_feature_encoder=self.cropped_image_feature_encoder,
                    text_encoder=self.text_encoder
                )

                results = tester.test(
                    method=method,
                    manager = self,
                    store_videos=store_videos,
                    eval_n_times=eval_n_times,
                    working_dir=working_dir,
                    test_count=i,
                    epoch=epoch
                )

            # add results over all repeated times
            for key in results.keys():
                if key not in list(results_all.keys()):
                    results_all[key] = results[key]
                else:
                    results_all[key] += results[key]

            # Accumulate per-scene success across rounds. Both testers expose
            # `per_scene_success` / `per_scene_faithfulness`.
            round_scene = getattr(tester, 'per_scene_success', {}) or {}
            for task, scenes in round_scene.items():
                d = per_scene_all.setdefault(task, {})
                r = per_scene_rounds.setdefault(task, {})
                for ri, ns in scenes.items():
                    d[ri] = d.get(ri, 0) + int(ns)
                    r[ri] = r.get(ri, 0) + 1

            # Keep the faithfulness grid keyed by round so both the per-round
            # (mean over scenes) and per-scene (mean over rounds) groupings stay
            # derivable downstream.
            # Rollout-time node scores for this round, alongside the train/val ones.
            # `utils.compare_node_scores` reads these against the training snapshots.
            round_scores = getattr(tester, 'node_scores', None)
            if round_scores:
                tag = f"epoch{epoch}" if epoch is not None else "final"
                round_scores.save_json(
                    os.path.join(working_dir, 'node_scores', f'eval_{tag}_round{i}.json'),
                    metadata={'split': 'eval', 'round': i, 'epoch': epoch,
                              'eval_n_times': eval_n_times},
                )

            round_faith = getattr(tester, 'per_scene_faithfulness', {}) or {}
            for task, scenes in round_faith.items():
                if scenes:
                    faith_all.setdefault(task, {})[i] = scenes

        avg_test_reward = 0 # average over all tasks

        # divide by the number of repeated times
        for key in results_all.keys():
            results_all[key] = results_all[key]/times_repeat
            avg_test_reward += results_all[key]
        
        num_tasks = len(task_names) if task_names else 1
        results_all['Avg_Test_Reward'] = avg_test_reward / num_tasks

        # ---- Write per-scene success JSON for paired comparison ----
        # One file per task: {rollout_idx: {n_success, n_rounds}}. Two configs
        # evaluated with the same eval_n_times + worker_seed_base see identical
        # scenes, so they can be paired per rollout_idx (far less variance than
        # comparing the two aggregate numbers).
        suffix = f"_epoch{epoch}" if epoch is not None else ""
        for task in task_names:
            scenes = per_scene_all.get(task, {})
            if not scenes:
                continue  # sequential path / no per-scene data
            rounds = per_scene_rounds.get(task, {})
            payload = {
                "task": task,
                "eval_n_times": eval_n_times,
                "times_repeat": times_repeat,
                "worker_seed_base": self.worker_seed_base,
                **self._sampler_fields(),
                "per_scene": {
                    str(ri): {"n_success": int(scenes[ri]), "n_rounds": int(rounds.get(ri, 0))}
                    for ri in sorted(scenes)
                },
            }
            out_path = os.path.join(working_dir, f"eval_per_scene_{task}{suffix}.json")
            try:
                with open(out_path, "w") as f:
                    json.dump(payload, f, indent=2)
                log.info(f"[test_method] wrote per-scene success to {out_path}")
            except Exception:
                log.exception(f"[test_method] failed to write per-scene JSON for {task}")

        # ---- Sub-graph faithfulness ----
        # Deliberately kept OUT of `results_all`: every key in there is divided by
        # `times_repeat` and summed into `Avg_Test_Reward` above, so an extra key
        # would silently corrupt the headline success number. The trainer picks
        # this up from the attribute instead.
        faith_log, faith_rows, faith_json = _summarize_faithfulness(faith_all)
        self.last_faithfulness_log = faith_log
        self.last_faithfulness_table = (
            {
                "key": FAITHFULNESS_TABLE_KEY,
                "columns": ["task", "modality", "scene", "n_rounds"] + list(FAITHFULNESS_METRICS),
                "rows": faith_rows,
            }
            if faith_rows else None
        )

        for task, payload in faith_json.items():
            payload.update({
                "eval_n_times": eval_n_times,
                "times_repeat": times_repeat,
                "worker_seed_base": self.worker_seed_base,
                **self._sampler_fields(),
            })
            out_path = os.path.join(working_dir, f"eval_faithfulness_{task}{suffix}.json")
            try:
                with open(out_path, "w") as f:
                    json.dump(payload, f, indent=2)
                log.info(f"[test_method] wrote sub-graph faithfulness to {out_path}")
            except Exception:
                log.exception(f"[test_method] failed to write faithfulness JSON for {task}")

        return results_all
    
    # Widths of the proprioceptive observations as they actually appear in the
    # demo HDF5s. The previous table mis-stated gripper_qpos as 4 (it is 2) and
    # joint_pos as 3 (it is 7), and omitted the base_to_eef modalities entirely.
    # `object` is variable per task (14 per sampled object) and is zero-padded to
    # OBJECT_OBS_DIM by the dataset.
    OBJECT_OBS_DIM = 56
    PROP_MODALITY_DIMS = {
        'robot0_eef_pos': 3,
        'robot0_eef_quat': 4,
        'robot0_base_pos': 3,
        'robot0_base_quat': 4,
        'robot0_base_to_eef_pos': 3,
        'robot0_base_to_eef_quat': 4,
        'robot0_gripper_qpos': 2,
        'robot0_gripper_qvel': 2,
        'robot0_joint_pos': 7,
        'robot0_joint_pos_cos': 7,
        'robot0_joint_pos_sin': 7,
        'robot0_joint_vel': 7,
        'object': OBJECT_OBS_DIM,
    }

    def calculate_proprioceptive_dim(self, modalities):
        dim = 0
        for mod in modalities:
            if mod not in self.PROP_MODALITY_DIMS:
                raise ValueError(f'Unknown proprioceptive modality: {mod}')
            dim += self.PROP_MODALITY_DIMS[mod]
        return dim

    def calculate_graph_dim(self):
        factor = 2
        if not self.use_splitted_modalities:
            self.adapted_graph_modalities = ["_".join(self.graph_modalities)]
        if not self.use_graph_fusion:
            factor = 1
            temp_graph_mods = self.adapted_graph_modalities
            self.adapted_graph_modalities = []
            for key in temp_graph_mods:
                self.adapted_graph_modalities.append(key + "_left")
                self.adapted_graph_modalities.append(key + "_right")
                
        dim = {}
        for mod in self.adapted_graph_modalities:
            dim[mod] = 0
        
        # Only meaningful when cropped image features are actually used — otherwise the
        # encoder name is irrelevant and must not be able to fail the run.
        uses_cropped = any("cropped_image_feature" in mod for mod in self.adapted_graph_modalities)
        if uses_cropped:
            if "256" in self.pretrained_img_encoder_name:
                self.cropped_img_dim = 256
            elif "37" in self.pretrained_img_encoder_name:
                self.cropped_img_dim = 37
            else:
                raise ValueError(f"Unknown embed_dim for pretrained image encoder: {self.pretrained_img_encoder_name}")


        graph_edge_dim = 1
        for mod in self.adapted_graph_modalities:
            if "one_hot_labels" in mod:
                dim[mod] += len(OBJECT_NAMES_IMAGES)
            if "bb_coordinates" in mod:
                dim[mod] += 10 * factor
                # graph_edge_dim = 4
            if "cropped_image_feature" in mod:
                if "fusion" in self.pretrained_img_encoder_name:
                    dim[mod] += self.cropped_img_dim
                    self.is_cropped_fusion = True
                else:
                    dim[mod] += self.cropped_img_dim * factor
            if "siglip_text_emb" in mod:
                dim[mod] += 768
            if "point_geom" in mod:
                # Reconstructed in the robot-base frame from both views at once, so it is
                # view invariant and is not multiplied by `factor`.
                dim[mod] += POINT_GEOM_DIM
        return dim, graph_edge_dim