"""RSL-RL vector environment adapter used by all local tasks."""

from collections.abc import Mapping

import torch
from .vec_env import VecEnv
from tensordict import TensorDict

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.utils.spaces import Space


def _flatten_observation_groups(obs: Mapping) -> dict[str, torch.Tensor]:
  """Convert nested term dictionaries to time-major flat group tensors.

  Standard mjlab emits each unflattened history term as ``(B, H, D)``.
  Concatenating terms on the last axis before flattening produces
  ``[t0(all terms), t1(all terms), ...]`` without patching mjlab itself.
  """
  result: dict[str, torch.Tensor] = {}
  for group_name, group_obs in obs.items():
    if isinstance(group_obs, Mapping):
      terms = list(group_obs.values())
      if not terms:
        raise ValueError(f"Observation group {group_name!r} has no terms")
      merged = torch.cat(terms, dim=-1)
      result[group_name] = merged.reshape(merged.shape[0], -1)
    else:
      result[group_name] = group_obs
  return result


class RslRlVecEnvWrapper(VecEnv):
  def __init__(
    self,
    env: ManagerBasedRlEnv,
    clip_actions: float | None = None,
  ):
    self.env = env
    self.clip_actions = clip_actions
    self.num_envs = self.unwrapped.num_envs
    self.device = torch.device(self.unwrapped.device)
    self.max_episode_length = self.unwrapped.max_episode_length
    self.num_actions = self.unwrapped.action_manager.total_action_dim
    self._modify_action_space()
    self.env.reset()

  @property
  def cfg(self) -> ManagerBasedRlEnvCfg:
    return self.unwrapped.cfg

  @property
  def render_mode(self) -> str | None:
    return self.env.render_mode

  @property
  def observation_space(self) -> Space:
    return self.env.observation_space

  @property
  def action_space(self) -> Space:
    return self.env.action_space

  @classmethod
  def class_name(cls) -> str:
    return cls.__name__

  @property
  def unwrapped(self) -> ManagerBasedRlEnv:
    return self.env.unwrapped

  @property
  def episode_length_buf(self) -> torch.Tensor:
    return self.unwrapped.episode_length_buf

  @episode_length_buf.setter
  def episode_length_buf(self, value: torch.Tensor) -> None:
    self.unwrapped.episode_length_buf = value

  def seed(self, seed: int = -1) -> int:
    return self.unwrapped.seed(seed)

  def _to_tensordict(self, obs: Mapping) -> TensorDict:
    return TensorDict(
      _flatten_observation_groups(obs),
      batch_size=[self.num_envs],
      device=self.device,
    )

  def get_observations(self) -> TensorDict:
    return self._to_tensordict(self.unwrapped.observation_manager.compute())

  def reset(self) -> tuple[TensorDict, dict]:
    obs_dict, extras = self.env.reset()
    return self._to_tensordict(obs_dict), extras

  def step(
    self, actions: torch.Tensor
  ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
    if self.clip_actions is not None:
      actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
    obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
    dones = (terminated | truncated).to(dtype=torch.long)
    if not self.cfg.is_finite_horizon:
      extras["time_outs"] = truncated
    return self._to_tensordict(obs_dict), rew, dones, extras

  def close(self) -> None:
    return self.env.close()

  def _modify_action_space(self) -> None:
    if self.clip_actions is None:
      return
    from mjlab.utils.spaces import Box, batch_space

    self.unwrapped.single_action_space = Box(
      shape=(self.num_actions,), low=-self.clip_actions, high=self.clip_actions
    )
    self.unwrapped.action_space = batch_space(
      self.unwrapped.single_action_space, self.num_envs
    )
