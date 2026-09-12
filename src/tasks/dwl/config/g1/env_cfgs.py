"""Unitree G1 flat-terrain environment configuration for DWL."""

from mjlab.envs import ManagerBasedRlEnvCfg

from src.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg


def g1_dwl_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the G1 DWL task using the established flat velocity environment."""
  return unitree_g1_flat_env_cfg(play=play)
