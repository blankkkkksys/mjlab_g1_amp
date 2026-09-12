"""Left-right symmetry mapping for the 29-DoF G1 DWL velocity task."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from src.tasks.amp_loco.g1_motion_constants import G1_JOINT_NAMES

_NUM_JOINTS = len(G1_JOINT_NAMES)
_ACTOR_DIM = 3 + 3 + 3 + 2 + 3 * _NUM_JOINTS
_CRITIC_DIM = _ACTOR_DIM + 3 + 2 + 2 + 2 + 6

_POLAR_SIGN = (1.0, -1.0, 1.0)
_AXIAL_SIGN = (-1.0, 1.0, -1.0)


def _paired_indices(names: tuple[str, ...]) -> tuple[int, ...]:
  index = {name: i for i, name in enumerate(names)}
  result = []
  for name in names:
    if name.startswith("left_"):
      mirror_name = "right_" + name[len("left_") :]
    elif name.startswith("right_"):
      mirror_name = "left_" + name[len("right_") :]
    else:
      mirror_name = name
    result.append(index[mirror_name])
  return tuple(result)


_JOINT_MIRROR_INDEX = _paired_indices(G1_JOINT_NAMES)
_JOINT_MIRROR_SIGN = tuple(
  -1.0 if ("roll" in name or "yaw" in name) else 1.0
  for name in G1_JOINT_NAMES
)


def _mirror_joints(values: torch.Tensor) -> torch.Tensor:
  index = torch.as_tensor(_JOINT_MIRROR_INDEX, device=values.device)
  sign = values.new_tensor(_JOINT_MIRROR_SIGN)
  return values.index_select(-1, index) * sign


def _mirror_feet(values: torch.Tensor, components: int = 1) -> torch.Tensor:
  shape = values.shape
  values = values.reshape(*shape[:-1], 2, components)
  index = torch.tensor((1, 0), device=values.device)
  return values.index_select(-2, index).reshape(shape)


def _mirror_actor(observation: torch.Tensor) -> torch.Tensor:
  if observation.shape[-1] != _ACTOR_DIM:
    raise ValueError(
      f"Expected G1 DWL actor dim {_ACTOR_DIM}, got {observation.shape[-1]}"
    )
  result = observation.clone()
  result[..., 0:3] *= observation.new_tensor(_AXIAL_SIGN)
  result[..., 3:6] *= observation.new_tensor(_POLAR_SIGN)
  result[..., 6:9] *= observation.new_tensor((1.0, -1.0, -1.0))
  # Left/right gait exchange is a half-cycle phase shift.
  result[..., 9:11] *= -1.0
  for start in (11, 11 + _NUM_JOINTS, 11 + 2 * _NUM_JOINTS):
    result[..., start : start + _NUM_JOINTS] = _mirror_joints(
      observation[..., start : start + _NUM_JOINTS]
    )
  return result


def _mirror_critic(observation: torch.Tensor) -> torch.Tensor:
  if observation.shape[-1] != _CRITIC_DIM:
    raise ValueError(
      f"Expected G1 DWL critic dim {_CRITIC_DIM}, got {observation.shape[-1]}"
    )
  result = observation.clone()
  result[..., :_ACTOR_DIM] = _mirror_actor(observation[..., :_ACTOR_DIM])
  offset = _ACTOR_DIM
  result[..., offset : offset + 3] *= observation.new_tensor(_POLAR_SIGN)
  offset += 3
  for width in (2, 2, 2):
    result[..., offset : offset + width] = _mirror_feet(
      observation[..., offset : offset + width]
    )
    offset += width
  forces = _mirror_feet(observation[..., offset : offset + 6], components=3)
  result[..., offset : offset + 6] = forces * observation.new_tensor(
    _POLAR_SIGN * 2
  )
  return result


def mirror_g1_dwl(
  obs: TensorDict | None,
  actions: torch.Tensor | None,
  env,
) -> tuple[TensorDict | None, torch.Tensor | None]:
  """Append a sagittal-plane mirrored copy after the original samples."""
  del env
  augmented_obs = None
  if obs is not None:
    mirrored = obs.clone()
    mirrored["actor"] = _mirror_actor(obs["actor"])
    if "critic" in obs.keys():
      mirrored["critic"] = _mirror_critic(obs["critic"])
    augmented_obs = torch.cat((obs, mirrored), dim=0)

  augmented_actions = None
  if actions is not None:
    if actions.shape[-1] != _NUM_JOINTS:
      raise ValueError(
        f"Expected {_NUM_JOINTS} G1 actions, got {actions.shape[-1]}"
      )
    augmented_actions = torch.cat((actions, _mirror_joints(actions)), dim=0)

  return augmented_obs, augmented_actions
