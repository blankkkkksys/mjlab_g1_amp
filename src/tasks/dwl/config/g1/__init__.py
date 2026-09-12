"""Register Unitree G1 DWL locomotion tasks."""

from mjlab.tasks.registry import register_mjlab_task

from src.tasks.amp_loco.rl import AMPOnPolicyRunner
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .amp_env_cfgs import g1_amp_dwl_flat_env_cfg
from .amp_rl_cfg import g1_amp_dwl_ppo_runner_cfg
from .env_cfgs import g1_dwl_flat_env_cfg
from .rl_cfg import g1_dwl_ppo_runner_cfg


register_mjlab_task(
  task_id="Unitree-G1-DWL-Flat",
  env_cfg=g1_dwl_flat_env_cfg(),
  play_env_cfg=g1_dwl_flat_env_cfg(play=True),
  rl_cfg=g1_dwl_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-AMP-DWL-Flat",
  env_cfg=g1_amp_dwl_flat_env_cfg(),
  play_env_cfg=g1_amp_dwl_flat_env_cfg(play=True),
  rl_cfg=g1_amp_dwl_ppo_runner_cfg(),
  runner_cls=AMPOnPolicyRunner,
)
