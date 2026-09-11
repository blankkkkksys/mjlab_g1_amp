"""Left-right symmetry augmentation for the 29-DoF Unitree G1 AMP task."""
# flake8: noqa

from __future__ import annotations

import torch
from tensordict import TensorDict

from src.tasks.amp_loco.config.g1.env_cfgs import G1_AMP_BODY_NAMES
from src.tasks.amp_loco.g1_motion_constants import G1_JOINT_NAMES

_NUM_JOINTS = len(G1_JOINT_NAMES)
_NUM_BODIES = len(G1_AMP_BODY_NAMES)
_ACTOR_FRAME_DIM = 3 + 3 + 3 + 3 * _NUM_JOINTS
_CRITIC_FRAME_DIM = _ACTOR_FRAME_DIM + 3 + 3 * _NUM_BODIES + 6 * _NUM_BODIES
_AMP_DIM = 15 * _NUM_BODIES


def _paired_indices(names: tuple[str, ...]) -> list[int]:
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
  return result


_JOINT_MIRROR_INDEX = _paired_indices(G1_JOINT_NAMES)
_BODY_MIRROR_INDEX = _paired_indices(G1_AMP_BODY_NAMES)

# Joint axes parallel to x or z change sign under sagittal-plane reflection.
_JOINT_MIRROR_SIGN = [
  -1.0 if ("roll" in name or "yaw" in name) else 1.0 for name in G1_JOINT_NAMES
]
_POLAR_SIGN = (1.0, -1.0, 1.0)
_AXIAL_SIGN = (-1.0, 1.0, -1.0)
_ROT6D_SIGN = (1.0, -1.0, -1.0, 1.0, 1.0, -1.0)


def _mirror_joints(values: torch.Tensor) -> torch.Tensor:
  index = torch.as_tensor(_JOINT_MIRROR_INDEX, device=values.device)
  sign = values.new_tensor(_JOINT_MIRROR_SIGN)
  return values.index_select(-1, index) * sign


def _mirror_bodies(
  values: torch.Tensor, components: int, signs: tuple[float, ...]
) -> torch.Tensor:
  shape = values.shape
  values = values.reshape(*shape[:-1], _NUM_BODIES, components)
  index = torch.as_tensor(_BODY_MIRROR_INDEX, device=values.device)
  sign = values.new_tensor(signs)
  return (values.index_select(-2, index) * sign).reshape(shape)


def _mirror_actor_frame(frame: torch.Tensor) -> torch.Tensor:
  if frame.shape[-1] != _ACTOR_FRAME_DIM:
    raise ValueError(
      f"Expected G1 AMP actor frame dim {_ACTOR_FRAME_DIM}, got {frame.shape[-1]}"
    )
  result = frame.clone()
  result[..., 0:3] *= frame.new_tensor(_AXIAL_SIGN)
  result[..., 3:6] *= frame.new_tensor(_POLAR_SIGN)
  result[..., 6:9] *= frame.new_tensor((1.0, -1.0, -1.0))
  for start in (9, 9 + _NUM_JOINTS, 9 + 2 * _NUM_JOINTS):
    result[..., start : start + _NUM_JOINTS] = _mirror_joints(
      frame[..., start : start + _NUM_JOINTS]
    )
  return result


def _mirror_actor(observation: torch.Tensor) -> torch.Tensor:
  if observation.shape[-1] % _ACTOR_FRAME_DIM != 0:
    raise ValueError(
      f"Actor observation dim {observation.shape[-1]} is not divisible by "
      f"G1 frame dim {_ACTOR_FRAME_DIM}"
    )
  frames = observation.reshape(
    *observation.shape[:-1], -1, _ACTOR_FRAME_DIM
  )
  return _mirror_actor_frame(frames).reshape(observation.shape)


def _mirror_critic(observation: torch.Tensor) -> torch.Tensor:
  if observation.shape[-1] % _CRITIC_FRAME_DIM != 0:
    raise ValueError(
      f"Critic observation dim {observation.shape[-1]} is not divisible by "
      f"G1 frame dim {_CRITIC_FRAME_DIM}"
    )
  frames = observation.reshape(
    *observation.shape[:-1], -1, _CRITIC_FRAME_DIM
  )
  result = frames.clone()
  result[..., :_ACTOR_FRAME_DIM] = _mirror_actor_frame(
    frames[..., :_ACTOR_FRAME_DIM]
  )
  offset = _ACTOR_FRAME_DIM
  result[..., offset : offset + 3] *= frames.new_tensor(_POLAR_SIGN)
  offset += 3
  result[..., offset : offset + 3 * _NUM_BODIES] = _mirror_bodies(
    frames[..., offset : offset + 3 * _NUM_BODIES], 3, _POLAR_SIGN
  )
  offset += 3 * _NUM_BODIES
  result[..., offset : offset + 6 * _NUM_BODIES] = _mirror_bodies(
    frames[..., offset : offset + 6 * _NUM_BODIES], 6, _ROT6D_SIGN
  )
  return result.reshape(observation.shape)


def _mirror_amp(observation: torch.Tensor) -> torch.Tensor:
  if observation.shape[-1] != _AMP_DIM:
    raise ValueError(
      f"Expected G1 AMP observation dim {_AMP_DIM}, got {observation.shape[-1]}"
    )
  result = observation.clone()
  offset = 0
  for components, signs in (
    (3, _POLAR_SIGN),
    (6, _ROT6D_SIGN),
    (3, _POLAR_SIGN),
    (3, _AXIAL_SIGN),
  ):
    width = components * _NUM_BODIES
    result[..., offset : offset + width] = _mirror_bodies(
      observation[..., offset : offset + width], components, signs
    )
    offset += width
  return result


def mirror_g1_amp(
  obs: TensorDict | None,
  actions: torch.Tensor | None,
  env,
) -> tuple[TensorDict | None, torch.Tensor | None]:
  """Append a sagittal-plane mirrored copy after each original batch."""
  del env
  augmented_obs = None
  if obs is not None:
    mirrored = obs.clone()
    mirrored["actor"] = _mirror_actor(obs["actor"])
    if "critic" in obs.keys():
      mirrored["critic"] = _mirror_critic(obs["critic"])
    if "amp" in obs.keys():
      mirrored["amp"] = _mirror_amp(obs["amp"])
    augmented_obs = torch.cat((obs, mirrored), dim=0)

  augmented_actions = None
  if actions is not None:
    if actions.shape[-1] != _NUM_JOINTS:
      raise ValueError(
        f"Expected {_NUM_JOINTS} G1 actions, got {actions.shape[-1]}"
      )
    augmented_actions = torch.cat((actions, _mirror_joints(actions)), dim=0)

  return augmented_obs, augmented_actions
