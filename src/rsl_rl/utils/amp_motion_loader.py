"""Expert motion loader for adversarial motion-prior training.

Derived from RSL-RL/AMP_mjlab implementations distributed under BSD-3-Clause.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

import mjlab.utils.lab_api.math as math_utils


def normalize_quat_wxyz(quat: np.ndarray) -> np.ndarray:
  quat64 = np.asarray(quat, dtype=np.float64)
  norm = np.linalg.norm(quat64, axis=-1, keepdims=True)
  normalized = quat64 / np.clip(norm, 1e-8, None)
  normalized = np.where(normalized[..., :1] < 0.0, -normalized, normalized)
  return normalized.astype(np.float32)


class AMPLoader:
  """Load prepared G1 motions and sample expert ``(state, next_state)`` pairs."""

  def __init__(
    self,
    motion_path: str,
    body_names: Sequence[str],
    anchor_name: str,
    all_body_names: Sequence[str],
    device: str,
  ) -> None:
    path = Path(motion_path)
    if path.is_file():
      files = [path]
    elif path.is_dir():
      files = sorted(
        file
        for file in path.rglob("*.npz")
        if not file.name.endswith("_M.npz")
      )
    else:
      raise FileNotFoundError(f"AMP motion path does not exist: {path}")
    if not files:
      raise FileNotFoundError(f"No AMP motion .npz files found in: {path}")

    all_names = list(all_body_names)
    missing = [name for name in (*body_names, anchor_name) if name not in all_names]
    if missing:
      raise ValueError(f"AMP body names not found in robot model: {missing}")
    body_indices = [all_names.index(name) for name in body_names]
    anchor_index = all_names.index(anchor_name)

    self._states: list[torch.Tensor] = []
    self.device = torch.device(device)
    for file in files:
      # Prepared clips contain a trusted object-array ``body_names`` metadata field.
      data = np.load(file, allow_pickle=True)
      positions = torch.as_tensor(
        data["body_pos_w"], dtype=torch.float32, device=self.device
      )
      quaternions = torch.as_tensor(
        normalize_quat_wxyz(data["body_quat_w"]),
        dtype=torch.float32,
        device=self.device,
      )
      linear_velocity = torch.as_tensor(
        data["body_lin_vel_w"], dtype=torch.float32, device=self.device
      )
      angular_velocity = torch.as_tensor(
        data["body_ang_vel_w"], dtype=torch.float32, device=self.device
      )
      if positions.shape[1] != len(all_names):
        raise ValueError(
          f"{file.name}: body count {positions.shape[1]} != "
          f"robot body count {len(all_names)}"
        )
      if "body_names" in data:
        stored_names = [str(name) for name in data["body_names"].tolist()]
        if stored_names != all_names:
          raise ValueError(f"{file.name}: stored body_names do not match robot order")

      body_pos = positions[:, body_indices]
      body_quat = quaternions[:, body_indices]
      anchor_pos = positions[:, anchor_index, None].expand_as(body_pos)
      anchor_quat = quaternions[:, anchor_index, None].expand_as(body_quat)
      pos_b, quat_b = math_utils.subtract_frame_transforms(
        anchor_pos, anchor_quat, body_pos, body_quat
      )
      ori_b = math_utils.matrix_from_quat(quat_b)[..., :, :2]
      lin_vel_b = math_utils.quat_apply_inverse(
        body_quat.reshape(-1, 4), linear_velocity[:, body_indices].reshape(-1, 3)
      ).reshape_as(body_pos)
      ang_vel_b = math_utils.quat_apply_inverse(
        body_quat.reshape(-1, 4), angular_velocity[:, body_indices].reshape(-1, 3)
      ).reshape_as(body_pos)
      state = torch.cat(
        (
          pos_b.flatten(1),
          ori_b.flatten(1),
          lin_vel_b.flatten(1),
          ang_vel_b.flatten(1),
        ),
        dim=-1,
      )
      self._states.append(state)

    self.observation_dim = self._states[0].shape[-1]
    if any(state.shape[-1] != self.observation_dim for state in self._states):
      raise ValueError("AMP clips have inconsistent observation dimensions")
    self.motion_names = [os.path.splitext(file.name)[0] for file in files]

  def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    clip_indices = torch.randint(
      len(self._states), (batch_size,), device=self.device
    )
    states = torch.empty(
      batch_size, self.observation_dim, device=self.device
    )
    next_states = torch.empty_like(states)
    for clip_index in clip_indices.unique():
      mask = clip_indices == clip_index
      clip = self._states[int(clip_index)]
      frame_indices = torch.randint(
        max(clip.shape[0] - 1, 1),
        (int(mask.sum()),),
        device=self.device,
      )
      states[mask] = clip[frame_indices]
      next_states[mask] = clip[
        torch.clamp(frame_indices + 1, max=clip.shape[0] - 1)
      ]
    return states, next_states
