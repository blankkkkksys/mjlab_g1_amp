"""Register Unitree G1 DWL locomotion tasks."""

from mjlab.tasks.registry import register_mjlab_task

from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import g1_dwl_flat_env_cfg
from .rl_cfg import g1_dwl_ppo_runner_cfg


register_mjlab_task(
  task_id="Unitree-G1-DWL-Flat",
  env_cfg=g1_dwl_flat_env_cfg(),
  play_env_cfg=g1_dwl_flat_env_cfg(play=True),
  rl_cfg=g1_dwl_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
