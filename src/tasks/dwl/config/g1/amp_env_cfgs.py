"""G1 AMP environment adapted for recurrent DWL training."""

from mjlab.envs import ManagerBasedRlEnvCfg

from src.tasks.amp_loco.config.g1.env_cfgs import g1_amp_flat_env_cfg


def g1_amp_dwl_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create the flat AMP task with temporal history handled by the DWL LSTM."""
    cfg = g1_amp_flat_env_cfg(play=play)
    cfg.observations["actor"].history_length = 1
    cfg.observations["critic"].history_length = 1
    return cfg
