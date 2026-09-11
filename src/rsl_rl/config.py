"""RSL-RL configuration shared by the repository tasks."""

from dataclasses import dataclass, field
from typing import Any, Literal, Tuple


@dataclass
class RslRlModelCfg:
  """Configuration for one actor or critic model."""

  hidden_dims: Tuple[int, ...] = (128, 128, 128)
  activation: str = "elu"
  obs_normalization: bool = False
  cnn_cfg: dict[str, Any] | None = None
  distribution_cfg: dict[str, Any] | None = None
  class_name: str = "MLPModel"


@dataclass
class RslRlPpoAlgorithmCfg:
  """Configuration for the standard PPO algorithm."""

  num_learning_epochs: int = 5
  num_mini_batches: int = 4
  learning_rate: float = 1e-3
  schedule: Literal["adaptive", "fixed"] = "adaptive"
  gamma: float = 0.99
  lam: float = 0.95
  entropy_coef: float = 0.005
  desired_kl: float = 0.01
  max_grad_norm: float = 1.0
  value_loss_coef: float = 1.0
  use_clipped_value_loss: bool = True
  clip_param: float = 0.2
  normalize_advantage_per_mini_batch: bool = False
  optimizer: Literal["adam", "adamw", "sgd", "rmsprop"] = "adam"
  share_cnn_encoders: bool = False
  symmetry_cfg: dict[str, Any] | None = None
  class_name: str = "PPO"


@dataclass
class RslRlBaseRunnerCfg:
  """Configuration common to repository-local runners."""

  seed: int = 42
  num_steps_per_env: int = 24
  max_iterations: int = 300
  obs_groups: dict[str, tuple[str, ...]] = field(
    default_factory=lambda: {"actor": ("actor",), "critic": ("critic",)}
  )
  save_interval: int = 50
  experiment_name: str = "exp1"
  run_name: str = ""
  logger: Literal["wandb", "tensorboard"] = "wandb"
  wandb_project: str = "mjlab"
  wandb_tags: Tuple[str, ...] = ()
  resume: bool = False
  load_run: str = ".*"
  load_checkpoint: str = "model_.*.pt"
  clip_actions: float | None = None
  upload_model: bool = True


@dataclass
class RslRlOnPolicyRunnerCfg(RslRlBaseRunnerCfg):
  """Configuration for the standard on-policy runner."""

  class_name: str = "OnPolicyRunner"
  actor: RslRlModelCfg = field(
    default_factory=lambda: RslRlModelCfg(
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      }
    )
  )
  critic: RslRlModelCfg = field(default_factory=RslRlModelCfg)
  algorithm: RslRlPpoAlgorithmCfg = field(default_factory=RslRlPpoAlgorithmCfg)
