"""In-process policy-server based parallel evaluation for RoboCasa.

Activated when `parallel_eval: true` and `num_envs > 1` in the manager config.
The main process owns the GPU diffusion policy (run as a thread); worker
subprocesses each own one MuJoCo env and ship observations to the server via
multiprocessing queues. See `parallel_tester.RoboCasaKitchenParallelTester`.
"""
