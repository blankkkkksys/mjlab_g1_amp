"""RSL-RL 5.x PPO extension for adversarial motion priors."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from .ppo import PPO
from src.rsl_rl.env import VecEnv
from src.rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from src.rsl_rl.models import MLPModel
from src.rsl_rl.modules.amp_discriminator import Discriminator
from src.rsl_rl.modules.amp_normalizer import AmpNormalizer
from src.rsl_rl.storage import RolloutStorage
from src.rsl_rl.storage.amp_replay_buffer import AmpReplayBuffer
from src.rsl_rl.utils import resolve_callable, resolve_obs_groups
from src.rsl_rl.utils.amp_motion_loader import AMPLoader


class AMPPPO(PPO):
  """PPO with a separately optimized AMP discriminator."""

  def __init__(
    self,
    actor: MLPModel,
    critic: MLPModel,
    storage: RolloutStorage,
    *,
    amp_loader: AMPLoader,
    amp_observation_dim: int,
    amp_reward_coef: float,
    amp_task_reward_lerp: float,
    amp_discr_hidden_dims: list[int],
    amp_replay_buffer_size: int,
    amp_discriminator_learning_rate: float = 1.0e-4,
    amp_discriminator_updates_per_iteration: int = 4,
    amp_grad_penalty_coef: float = 10.0,
    min_normalized_std: list[float],
    max_normalized_std: list[float] | None = None,
    **kwargs,
  ) -> None:
    super().__init__(actor, critic, storage, **kwargs)
    self.amp_loader = amp_loader
    self.amp_grad_penalty_coef = amp_grad_penalty_coef
    if amp_discriminator_updates_per_iteration < 1:
      raise ValueError("amp_discriminator_updates_per_iteration must be positive")
    self.amp_discriminator_updates_per_iteration = (
      amp_discriminator_updates_per_iteration
    )
    self.amp_discriminator_learning_rate = amp_discriminator_learning_rate
    self.amp_normalizer = AmpNormalizer(amp_observation_dim).to(self.device)
    self.discriminator = Discriminator(
      observation_dim=amp_observation_dim,
      reward_coef=amp_reward_coef,
      hidden_dims=amp_discr_hidden_dims,
      task_reward_lerp=amp_task_reward_lerp,
    ).to(self.device)
    self.discriminator_optimizer = torch.optim.Adam(
      self.discriminator.parameters(), lr=self.amp_discriminator_learning_rate
    )
    self.amp_replay_buffer = AmpReplayBuffer(
      amp_observation_dim, amp_replay_buffer_size, self.device
    )
    self.min_normalized_std = torch.as_tensor(
      min_normalized_std, dtype=torch.float32, device=self.device
    )
    self.max_normalized_std = (
      torch.as_tensor(
        max_normalized_std, dtype=torch.float32, device=self.device
      )
      if max_normalized_std is not None
      else None
    )
    if (
      self.max_normalized_std is not None
      and self.max_normalized_std.shape != self.min_normalized_std.shape
    ):
      raise ValueError("min_normalized_std and max_normalized_std must match")
    if (
      self.max_normalized_std is not None
      and torch.any(self.max_normalized_std < self.min_normalized_std)
    ):
      raise ValueError("max_normalized_std must not be below its minimum")
    self._current_amp_obs: torch.Tensor | None = None
    self._rollout_task_reward_sum = 0.0
    self._rollout_style_reward_sum = 0.0
    self._rollout_mixed_reward_sum = 0.0
    self._rollout_logit_sum = 0.0
    self._rollout_step_count = 0

  def act(self, obs: TensorDict) -> torch.Tensor:
    self._current_amp_obs = obs["amp"].detach().clone()
    return super().act(obs)

  def process_env_step(
    self,
    obs: TensorDict,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    extras: dict[str, torch.Tensor],
  ) -> None:
    if self._current_amp_obs is None:
      raise RuntimeError("AMPPPO.process_env_step called before act")
    next_amp_obs = obs["amp"].detach().clone()
    terminal_mask = dones.bool().reshape(-1)
    next_amp_obs[terminal_mask] = self._current_amp_obs[terminal_mask]
    self.amp_replay_buffer.insert(self._current_amp_obs, next_amp_obs)

    task_rewards = rewards.detach().clone()
    style_rewards, logits = self.discriminator.predict_style_reward(
      self._current_amp_obs, next_amp_obs, self.amp_normalizer
    )
    mixed_rewards = (
      (1.0 - self.discriminator.task_reward_lerp) * style_rewards
      + self.discriminator.task_reward_lerp * task_rewards
    )
    rewards.copy_(mixed_rewards)

    step_count = task_rewards.numel()
    self._rollout_task_reward_sum += task_rewards.sum().item()
    self._rollout_style_reward_sum += style_rewards.sum().item()
    self._rollout_mixed_reward_sum += mixed_rewards.sum().item()
    self._rollout_logit_sum += logits.sum().item()
    self._rollout_step_count += step_count

    super().process_env_step(obs, rewards, dones, extras)
    self._current_amp_obs = None

  def update(self) -> dict[str, float]:
    loss_dict = super().update()
    loss_dict.update(self._build_rollout_metrics())
    if self.amp_replay_buffer.size == 0:
      self._reset_rollout_metrics()
      return loss_dict

    batch_size = max(
      1,
      self.storage.num_envs
      * self.storage.num_transitions_per_env
      // self.num_mini_batches,
    )
    policy_state, policy_next_state = self.amp_replay_buffer.sample(batch_size)
    expert_state, expert_next_state = self.amp_loader.sample(batch_size)
    self.amp_normalizer.update(
      torch.cat(
        (policy_state, policy_next_state, expert_state, expert_next_state), dim=0
      )
    )

    mean_disc_loss = 0.0
    mean_policy_loss = 0.0
    mean_expert_loss = 0.0
    mean_grad_penalty = 0.0
    mean_policy_logit = 0.0
    mean_expert_logit = 0.0
    mean_accuracy = 0.0
    updates = self.amp_discriminator_updates_per_iteration
    for _ in range(updates):
      policy_state, policy_next_state = self.amp_replay_buffer.sample(batch_size)
      expert_state, expert_next_state = self.amp_loader.sample(batch_size)
      policy_state = self.amp_normalizer.normalize(policy_state)
      policy_next_state = self.amp_normalizer.normalize(policy_next_state)
      expert_state = self.amp_normalizer.normalize(expert_state)
      expert_next_state = self.amp_normalizer.normalize(expert_next_state)

      policy_logits = self.discriminator(policy_state, policy_next_state)
      expert_logits = self.discriminator(expert_state, expert_next_state)
      policy_loss = F.mse_loss(policy_logits, -torch.ones_like(policy_logits))
      expert_loss = F.mse_loss(expert_logits, torch.ones_like(expert_logits))
      grad_penalty = self.discriminator.compute_grad_penalty(
        expert_state, expert_next_state, coefficient=self.amp_grad_penalty_coef
      )
      disc_loss = 0.5 * (policy_loss + expert_loss) + grad_penalty

      self.discriminator_optimizer.zero_grad()
      disc_loss.backward()
      torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.max_grad_norm)
      self.discriminator_optimizer.step()

      with torch.no_grad():
        accuracy = (
          (expert_logits > 0).float().mean() + (policy_logits < 0).float().mean()
        ) * 0.5
      mean_disc_loss += disc_loss.item()
      mean_policy_loss += policy_loss.item()
      mean_expert_loss += expert_loss.item()
      mean_grad_penalty += grad_penalty.item()
      mean_policy_logit += policy_logits.mean().item()
      mean_expert_logit += expert_logits.mean().item()
      mean_accuracy += accuracy.item()

    self._clamp_action_std()
    loss_dict.update(
      {
        "amp_policy_loss": mean_policy_loss / updates,
        "amp_expert_loss": mean_expert_loss / updates,
        "amp_discriminator": mean_disc_loss / updates,
        "amp_grad_penalty": mean_grad_penalty / updates,
        "AMP/policy_logit": mean_policy_logit / updates,
        "AMP/expert_logit": mean_expert_logit / updates,
        "AMP/disc_accuracy": mean_accuracy / updates,
        "AMP/replay_buffer_size": float(self.amp_replay_buffer.size),
        "AMP/disc_learning_rate": self.discriminator_optimizer.param_groups[0]["lr"],
      }
    )
    self._reset_rollout_metrics()
    return loss_dict

  def _build_rollout_metrics(self) -> dict[str, float]:
    if self._rollout_step_count == 0:
      return {}
    count = self._rollout_step_count
    return {
      "AMP/task_reward": self._rollout_task_reward_sum / count,
      "AMP/style_reward": self._rollout_style_reward_sum / count,
      "AMP/mixed_reward": self._rollout_mixed_reward_sum / count,
      "AMP/rollout_logit": self._rollout_logit_sum / count,
    }

  def _reset_rollout_metrics(self) -> None:
    self._rollout_task_reward_sum = 0.0
    self._rollout_style_reward_sum = 0.0
    self._rollout_mixed_reward_sum = 0.0
    self._rollout_logit_sum = 0.0
    self._rollout_step_count = 0

  def _clamp_action_std(self) -> None:
    distribution = self.actor.distribution
    if distribution is None:
      return
    with torch.no_grad():
      if hasattr(distribution, "std_param"):
        distribution.std_param.clamp_(
          min=self.min_normalized_std,
          max=self.max_normalized_std,
        )
      elif hasattr(distribution, "log_std_param"):
        max_log_std = (
          torch.log(self.max_normalized_std)
          if self.max_normalized_std is not None
          else None
        )
        distribution.log_std_param.clamp_(
          min=torch.log(self.min_normalized_std),
          max=max_log_std,
        )

  def train_mode(self) -> None:
    super().train_mode()
    self.discriminator.train()

  def eval_mode(self) -> None:
    super().eval_mode()
    self.discriminator.eval()

  def save(self) -> dict:
    saved_dict = super().save()
    saved_dict.update(
      {
        "discriminator_state_dict": self.discriminator.state_dict(),
        "discriminator_optimizer_state_dict": self.discriminator_optimizer.state_dict(),
        "amp_normalizer_state_dict": self.amp_normalizer.state_dict(),
      }
    )
    return saved_dict

  def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
    load_iteration = super().load(loaded_dict, load_cfg, strict)
    load_amp = load_cfg is None or load_cfg.get("amp", True)
    if load_amp:
      self.discriminator.load_state_dict(
        loaded_dict["discriminator_state_dict"], strict=strict
      )
      self.amp_normalizer.load_state_dict(
        loaded_dict["amp_normalizer_state_dict"], strict=strict
      )
      if load_cfg is None or load_cfg.get("optimizer", True):
        self.discriminator_optimizer.load_state_dict(
          loaded_dict["discriminator_optimizer_state_dict"]
        )
        for param_group in self.discriminator_optimizer.param_groups:
          param_group["lr"] = self.amp_discriminator_learning_rate
    return load_iteration

  @staticmethod
  def construct_algorithm(
    obs: TensorDict, env: VecEnv, cfg: dict, device: str
  ) -> AMPPPO:
    if "amp" not in obs.keys():
      raise KeyError("AMP task requires an 'amp' observation group")
    robot = env.unwrapped.scene["robot"]
    amp_loader = AMPLoader(
      motion_path=cfg["amp_motion_files"],
      body_names=cfg["amp_body_names"],
      anchor_name=cfg["amp_anchor_name"],
      all_body_names=robot.body_names,
      device=device,
    )
    if amp_loader.observation_dim != obs["amp"].shape[-1]:
      raise ValueError(
        f"Expert AMP dim {amp_loader.observation_dim} != "
        f"environment AMP dim {obs['amp'].shape[-1]}"
      )
    alg_class: type[AMPPPO] = resolve_callable(
      cfg["algorithm"].pop("class_name")
    )
    actor_class: type[MLPModel] = resolve_callable(cfg["actor"].pop("class_name"))
    critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))
    cfg["obs_groups"] = resolve_obs_groups(
      obs, cfg["obs_groups"], ["actor", "critic"]
    )
    algorithm_cfg = resolve_rnd_config(
      cfg["algorithm"], obs, cfg["obs_groups"], env
    )
    algorithm_cfg = resolve_symmetry_config(algorithm_cfg, env)

    actor = actor_class(
      obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]
    ).to(device)
    print(f"Actor Model: {actor}")
    if algorithm_cfg.pop("share_cnn_encoders", None):
      cfg["critic"]["cnns"] = actor.cnns
    critic = critic_class(
      obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]
    ).to(device)
    print(f"Critic Model: {critic}")
    storage = RolloutStorage(
      "rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device
    )
    return alg_class(
      actor,
      critic,
      storage,
      amp_loader=amp_loader,
      amp_observation_dim=amp_loader.observation_dim,
      amp_reward_coef=cfg["amp_reward_coef"],
      amp_task_reward_lerp=cfg["amp_task_reward_lerp"],
      amp_discr_hidden_dims=list(cfg["amp_discr_hidden_dims"]),
      amp_replay_buffer_size=cfg["amp_num_preload_transitions"],
      amp_discriminator_learning_rate=cfg.get(
        "amp_discriminator_learning_rate", 1.0e-4
      ),
      amp_discriminator_updates_per_iteration=cfg.get(
        "amp_discriminator_updates_per_iteration", 4
      ),
      amp_grad_penalty_coef=cfg.get("amp_grad_penalty_coef", 10.0),
      min_normalized_std=list(cfg["min_normalized_std"]),
      max_normalized_std=(
        list(cfg["max_normalized_std"])
        if cfg.get("max_normalized_std") is not None
        else None
      ),
      device=device,
      **algorithm_cfg,
      multi_gpu_cfg=cfg["multi_gpu"],
    )
