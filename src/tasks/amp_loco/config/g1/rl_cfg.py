"""RL configuration for Unitree G1 AMP locomotion task."""

import os
from dataclasses import dataclass, field
from typing import List

from src.rsl_rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)
from src.tasks.amp_loco.config.g1.env_cfgs import G1_AMP_BODY_NAMES

# Prepared AMP motion clips (run scripts/prepare_amp_motions.py first).
_MOTION_DATA_DIR = os.path.join(
  os.path.dirname(os.path.abspath(__file__)),
  os.pardir,
  os.pardir,
  os.pardir,
  os.pardir,
  "assets",
  "motions",
  "g1",
  "amp",
)


@dataclass
class RslRlAmpRunnerCfg(RslRlOnPolicyRunnerCfg):
  """Extended runner config with AMP-specific parameters."""
  amp_reward_coef: float = 0.05
  amp_motion_files: str = ""
  amp_num_preload_transitions: int = 200000
  amp_task_reward_lerp: float = 0.75
  amp_discr_hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
  amp_discriminator_learning_rate: float = 1.0e-4
  amp_discriminator_updates_per_iteration: int = 2
  amp_grad_penalty_coef: float = 10.0
  min_normalized_std: List[float] = field(default_factory=lambda: [0.05] * 29)
  amp_body_names: tuple = ()
  amp_anchor_name: str = ""


def g1_amp_ppo_runner_cfg() -> RslRlAmpRunnerCfg:
  """Create RL runner configuration for Unitree G1 AMP locomotion task."""
  return RslRlAmpRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
      symmetry_cfg={
        "use_data_augmentation": True,
        "use_mirror_loss": True,
        "mirror_loss_coeff": 0.1,
        "data_augmentation_func": (
          "src.tasks.amp_loco.config.g1.symmetry:mirror_g1_amp"
        ),
      },
      class_name="src.rsl_rl.algorithms.amp_ppo:AMPPPO",
    ),
    experiment_name="g1_amp_locomotion",
    logger="tensorboard",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=100001,
    # AMP parameters
    amp_reward_coef=0.10,
    amp_motion_files=os.path.normpath(_MOTION_DATA_DIR),
    amp_num_preload_transitions=200000,
    # Final reward = lerp * task reward + (1 - lerp) * AMP style reward.
    amp_task_reward_lerp=0.60,
    amp_discr_hidden_dims=[512, 256, 128],
    amp_discriminator_learning_rate=5.0e-5,
    amp_discriminator_updates_per_iteration=2,
    min_normalized_std=[0.05] * 29,
    amp_body_names=G1_AMP_BODY_NAMES,
    amp_anchor_name="torso_link",
  )
