import os
import threading

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
import logging

from trainer import Trainer
from utils.run_naming import derive_run_name

log = logging.getLogger(__name__)


def naming_config(cfg, resolved):
    """The config the run name should describe.

    For a training run that is simply this config. For an eval-only run it is not:
    `Trainer.check_if_model_is_loaded` replaces `manager` and `method` wholesale with the
    checkpoint's copy, so the variant that actually runs is the checkpoint's, while this
    config still holds whatever the defaults list composed. Naming from the latter gets a
    re-evaluated VGIB checkpoint logged as "SIR_s42".

    The seed and everything else still come from this run.
    """
    if cfg.trainer.train_bool:
        return resolved
    # Hydra chdir's into the run dir before main() is called, so a relative model_path
    # would resolve against the *new* run's directory. That fails the exists() check
    # below and silently falls back to naming the run after whatever the defaults list
    # composed — the exact mislabel this function exists to prevent.
    ckpt = os.path.join(hydra.utils.to_absolute_path(cfg.trainer.model_path),
                        ".hydra", "config.yaml")
    if not os.path.exists(ckpt):
        return resolved
    # resolve=False on purpose: the checkpoint config carries interpolations pointing at
    # keys that are not in scope here. Nothing the naming reads is an interpolation.
    reloaded = OmegaConf.to_container(OmegaConf.load(ckpt), resolve=False)
    return {**resolved,
            "manager": reloaded.get("manager", resolved.get("manager")),
            "method": reloaded.get("method", resolved.get("method"))}

@hydra.main(config_path="configs", config_name="main.yaml")
def main(cfg: DictConfig) -> None:
    # init wandb logger and config from hydra path
    wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    # `cfg.run_name` never existed — the key is `cfg.wandb.run_name` — so this silently
    # passed name=None and every run got a random wandb codename, however carefully
    # `wandb.run_name=` was set on the command line. With the key fixed, an explicit
    # name still wins; otherwise one is derived from the config that actually ran, e.g.
    # "TopK_s43", so a variant cannot be mislabelled by a stale override.
    run_name = cfg.wandb.get("run_name") or derive_run_name(naming_config(cfg, wandb.config))
    log.info(f"wandb run name: {run_name}")

    mode = cfg.wandb.get("mode", "online")
    init_kwargs = dict(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=wandb.config,
        name=run_name,
    )
    try:
        run = wandb.init(mode=mode, **init_kwargs)
    except Exception as e:
        # Logging is not the job. On 2026-09-06 the network dropped, wandb.init timed out
        # after 90 s, and three jobs -- an 8.5 h training and two 8 h evaluations -- died
        # before touching the GPU, purely because the metrics sink was unreachable. Fall
        # back to offline: the run proceeds, the data lands in the run dir, and
        # `wandb sync <dir>` uploads it afterwards.
        log.error(f"wandb.init failed ({type(e).__name__}: {e}); falling back to offline. "
                  f"Sync later with: wandb sync {os.path.join(os.getcwd(), 'wandb')}")
        run = wandb.init(mode="offline", **init_kwargs)

    trainer = Trainer(cfg.manager, cfg.method, cfg.trainer)

    trainer.start()

    log.info("done")
    # Bounded, for the same reason. The same outage left a finished training run blocked
    # in wandb's upload for 44 hours, holding the GPU against the next job in the queue.
    # The work is done and saved by this point, so a stuck flush must not hold the process.
    finisher = threading.Thread(target=run.finish, daemon=True)
    finisher.start()
    finisher.join(timeout=300)
    if finisher.is_alive():
        log.error("wandb run.finish() still blocked after 300 s; exiting anyway "
                  "(results are already on disk, metrics may be incomplete)")
        os._exit(0)
    
if __name__ == "__main__":
    main()
