import hashlib
import os
import random
import robocasa
import robosuite
import h5py
import json

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx

from utils.generate_graph_dataset_robocasa import (
    FAITHFULNESS_METRICS, score_subgraph_faithfulness,
)

TASK_LIST = [
    'CloseDrawer', 'OpenDrawer',
    'CloseDoubleDoor', 'CloseSingleDoor', 'OpenDoubleDoor', 'OpenSingleDoor',
    'TurnOffMicrowave', 'TurnOnMicrowave',
    'TurnOnSinkFaucet', 'TurnOffSinkFaucet', 'TurnSinkSpout',
    'TurnOnStove', 'TurnOffStove',
    'CoffeePressButton', 'CoffeeServeMug', 'CoffeeSetupMug',
    'PnPCabToCounter', 'PnPCounterToCab', 'PnPCounterToMicrowave', 'PnPCounterToSink',
    'PnPCounterToStove', 'PnPMicrowaveToCounter', 'PnPSinkToCounter', 'PnPStoveToCounter'
]

HORIZON_LIST = [
                500, 500,
                700, 500, 1000, 500,
                500, 500,
                500, 500, 500,
                500, 500,
                300, 600, 600,
                500, 500, 600, 700,
                500, 500, 500, 500
            ]

def create_kitchen_env(dataset_path: str, task_list: list[str], use_depth: bool, seed: int,
                       env_overrides: dict | None = None):
    """Build one robosuite env per task from the DEMO FILE's own `env_args`.

    Every scene-generating parameter -- `layout_ids`, `style_ids`, `obj_instance_split`,
    `generative_textures`, `randomize_cameras` -- is inherited verbatim from the dataset,
    which is why rollout scenes are fresh draws from exactly the generator that produced
    the demonstrations. `env_overrides` is the one way to break that tie on purpose.

    The use for it is RoboCasa's own held-out asset split: training data carries
    `obj_instance_split='A'` (all but the last 3 instances of each category, or the first
    half, whichever is larger), so `{"obj_instance_split": "B"}` evaluates on object
    meshes the policy has never seen. Held-out layout/style is NOT available as an
    alternative -- pooled over all 24 tasks' 1293 demos, every one of the 100
    (layout, style) pairs appears in training.
    """
    if task_list[0] == "ALL":
        dataset_path = os.path.dirname(dataset_path)
        data_folder_name_list = TASK_LIST
        task_list = TASK_LIST
        print("Task list: ", task_list)
        env_meta_list = []
        for name in data_folder_name_list:
            env_meta_list.append(get_env_metadata_from_dataset(dataset_path=os.path.join(dataset_path, name, "demo_gentex_im128_randcams.hdf5")))
    else:
        env_meta_list = []
        for task in task_list:
            env_meta_list.append(get_env_metadata_from_dataset(dataset_path=os.path.join(dataset_path, task, "demo_gentex_im128_randcams.hdf5")))

    env_list = []
    for env_meta, env_n in zip(env_meta_list, task_list):
        config = env_meta['env_kwargs']

        config['camera_heights'] = 128
        config['camera_widths'] = 128
        config['horizon'] = HORIZON_LIST[TASK_LIST.index(env_n)]
        config['seed'] = seed
        config['camera_depths'] = use_depth
        if env_overrides:
            # layout_and_style_ids is mutually exclusive with layout_ids/style_ids in
            # robocasa's Kitchen.__init__ (it asserts both others are None), so setting it
            # has to clear them rather than sit alongside them.
            if 'layout_and_style_ids' in env_overrides:
                config.pop('layout_ids', None)
                config.pop('style_ids', None)
            config.update(env_overrides)
            print(f"Env overrides for {env_n}: {env_overrides}")

        print("Initializing environment...")

        env = robosuite.make(
            **config,
            env_name=env_n
        )

        env_list.append(env)

    return env_list, task_list

def get_env_metadata_from_dataset(dataset_path):
    """
    Retrieves env metadata from dataset.

    Args:
        dataset_path (str): path to dataset

    Returns:
        env_meta (dict): environment metadata. Contains 3 keys:

            :`'env_name'`: name of environment
            :`'type'`: type of environment, should be a value in EB.EnvType
            :`'env_kwargs'`: dictionary of keyword arguments to pass to environment constructor
    """
    dataset_path = os.path.expanduser(dataset_path)
    f = h5py.File(dataset_path, "r")
    env_meta = json.loads(f["data"].attrs["env_args"])
    f.close()
    return env_meta

def accumulate_faithfulness(faith_sums, faith_steps, mod_key, step_names,
                            node_names, task_name, lang_goal):
    """Score one inference step's kept sub-graph and fold it into the running sums.

    `faith_sums` is {mod_key: {metric: sum}}, `faith_steps` is {mod_key: n_scored}.
    Steps the scorer cannot score (unknown task, no relevant node in the graph)
    are skipped entirely rather than counted as zero.
    """
    scores = score_subgraph_faithfulness(step_names, node_names, task_name, lang_goal)
    if scores is None:
        return
    bucket = faith_sums.setdefault(mod_key, {m: 0.0 for m in FAITHFULNESS_METRICS})
    for metric, value in scores.items():
        bucket[metric] += value
    faith_steps[mod_key] = faith_steps.get(mod_key, 0) + 1


def mean_faithfulness(faith_sums, faith_steps):
    """Collapse the running sums into {mod_key: {metric: mean over scored steps}}.

    Averaging over steps first means every rollout carries equal weight
    regardless of its length, matching how success is counted.
    """
    return {
        mod: {metric: total / faith_steps[mod] for metric, total in sums.items()}
        for mod, sums in faith_sums.items()
        if faith_steps.get(mod, 0) > 0
    }


def per_rollout_seed(task_name: str, rollout_idx: int, base_seed: int) -> int:
    """Deterministic seed for a single (task, rollout_idx) pair.

    Stable across rounds, workers and processes, and independent of which worker
    happened to pick the rollout up, so rollout j is the same scene in every run
    that shares `base_seed`.
    """
    h = hashlib.blake2b(f"{task_name}|{rollout_idx}|{base_seed}".encode(), digest_size=4).digest()
    return int.from_bytes(h, 'big') & 0x7FFFFFFF


def eval_noise(method, noise_seed, keys):
    """Starting diffusion noise for a batch of predictions, as a function of WHERE each
    prediction sits in the evaluation -- `keys` is one (task_name, rollout_idx, step) per
    batch row -- rather than of the global torch RNG.

    Why. Every prediction of the diffusion policy starts from fresh Gaussian noise. Drawn
    from the global RNG, the noise a rollout receives depends on everything that drew
    before it: in the parallel path, the order in which the workers' requests reached the
    shared policy server. The same unit re-run unchanged then disagrees on a large share
    of scenes even though `seed_rollout` pins every scene. Keying the noise on
    (task, rollout_idx, step) -- the same identity that pins the scene -- makes it
    independent of batch composition, arrival order, which worker ran the rollout,
    `num_envs`, retry rounds, and parallel vs sequential evaluation.

    The distribution is unchanged (standard normal times sigma_max); only the realisation
    is fixed. Generated on CPU, because a CUDA generator's stream is not reproducible
    across differing batch shapes. blake2b rather than hash(): Python salts str hashing
    per process.

    Returns [B, act_window_size, action_dim] on CPU.
    """
    shape = (method.act_window_size, method.action_dim)
    g = torch.Generator(device="cpu")
    rows = []
    for task_name, rollout_idx, step in keys:
        h = hashlib.blake2b(f"{noise_seed}|{task_name}|{rollout_idx}|{step}".encode(),
                            digest_size=8).digest()
        g.manual_seed(int.from_bytes(h, "big") & 0x7FFFFFFFFFFFFFFF)
        rows.append(torch.randn(shape, generator=g) * method.sigma_max)
    return torch.stack(rows)


def seed_rollout(env, seed):
    """Make the next `env.reset()` produce a reproducible scene.

    Setting `env.rng` alone is not enough: it fixes the kitchen layout and style,
    but the object sampling and task variant (e.g. which drawer the instruction
    names) are drawn from the *global* numpy/random state, which advances with
    every reset. Without seeding those too, rollout j of one run is a different
    scene from rollout j of the next, which silently breaks any paired
    comparison across runs — the same seed produced "close the right drawer" in
    one run and "close the left drawer" in the next.
    """
    np.random.seed(seed)
    random.seed(seed)
    env.rng = np.random.default_rng(seed)
    try:
        env.seed = seed
    except Exception:
        pass


def rollout_artifact_dir(working_dir, kind, task_name, test_count):
    """`<working_dir>/<kind>/<task_name>/repeat_<test_count>/`, created on demand.

    kind is 'sparsification' or 'videos'. Nesting keeps a 24-task x 50-scene x
    4-round eval browsable instead of dumping thousands of files side by side.
    """
    d = os.path.join(working_dir, kind, task_name, f"repeat_{test_count}")
    os.makedirs(d, exist_ok=True)
    return d

def visualize_sparsification_rollout(presence_counts, total_steps, save_path, timeline=None, title=None):
    """Render aggregate presence graph + optional per-step topk selection timeline.

    presence_counts: dict[str, int] mapping node label -> step count kept
    total_steps:     total rollout steps (denominator for the percentage)
    save_path:       absolute PNG path
    timeline:        optional list[list[str]] — per-step list of selected node names
    title:           optional figure title
    """
    if total_steps <= 0 or not presence_counts:
        return

    nodes = list(presence_counts.keys())
    percents = [100.0 * presence_counts[n] / total_steps for n in nodes]

    G = nx.Graph()
    for n in nodes:
        G.add_node(n)
    for i, a in enumerate(nodes):
        for b in nodes[i + 1:]:
            G.add_edge(a, b)

    pos = nx.spring_layout(G, seed=42)
    labels = {n: f"{n}\n{p:.0f}%" for n, p in zip(nodes, percents)}
    node_sizes = [300 + 20 * p for p in percents]

    if timeline:
        fig, (ax_graph, ax_timeline) = plt.subplots(
            1, 2, figsize=(18, 8), gridspec_kw={'width_ratios': [1, 1.3]}
        )
    else:
        fig, ax_graph = plt.subplots(figsize=(10, 8))
        ax_timeline = None

    nx.draw_networkx_edges(G, pos, alpha=0.2, ax=ax_graph)
    nodes_artist = nx.draw_networkx_nodes(
        G, pos,
        node_color=percents,
        cmap=plt.cm.viridis,
        vmin=0, vmax=100,
        node_size=node_sizes,
        ax=ax_graph,
    )
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, ax=ax_graph)

    cbar = plt.colorbar(nodes_artist, ax=ax_graph)
    cbar.set_label('Presence (%)')

    ax_graph.set_title('aggregate presence')
    ax_graph.set_axis_off()

    if ax_timeline is not None:
        nodes_sorted = sorted(presence_counts, key=lambda n: -presence_counts[n])
        name_to_row = {n: r for r, n in enumerate(nodes_sorted)}
        T = len(timeline)
        M = np.zeros((len(nodes_sorted), T), dtype=np.float32)
        for t, step_names in enumerate(timeline):
            for name in step_names:
                row = name_to_row.get(name)
                if row is not None:
                    M[row, t] = 1.0

        ax_timeline.imshow(
            M, aspect='auto', cmap='Greys', interpolation='nearest', vmin=0, vmax=1
        )
        ax_timeline.set_yticks(range(len(nodes_sorted)))
        ax_timeline.set_yticklabels(nodes_sorted, fontsize=8)
        ax_timeline.set_xlabel('rollout step')
        ax_timeline.set_title('per-step topk selection')

    full_title = f"{title}\n{total_steps} rollout steps" if title else f"{total_steps} rollout steps"
    fig.suptitle(full_title)

    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)