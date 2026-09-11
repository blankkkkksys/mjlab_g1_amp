"""Depth camera with configurable noise and per-environment history."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from mjlab.sensor import CameraSensor, CameraSensorCfg, CameraSensorData

from .noise import DepthNoiseCfg, apply_depth_noise


@dataclass
class NoisyDepthCameraData(CameraSensorData):
  """Raw and processed outputs from :class:`NoisyDepthCamera`."""

  depth_noisy: torch.Tensor | None = None
  """Noisy depth image with shape ``[B, H, W, 1]``."""

  depth_history: torch.Tensor | None = None
  """Oldest-to-newest noisy frames with shape ``[B, T, H, W, 1]``."""


@dataclass
class NoisyDepthCameraCfg(CameraSensorCfg):
  """Configuration for a depth-only camera noise pipeline."""

  data_types: tuple[str, ...] = ("depth",)
  noise_pipeline: tuple[DepthNoiseCfg, ...] = field(default_factory=tuple)
  history_length: int = 1

  def __post_init__(self) -> None:
    super().__post_init__()
    if self.data_types != ("depth",):
      raise ValueError(
        "NoisyDepthCamera only supports data_types=('depth',), got "
        f"{self.data_types}"
      )
    if self.history_length < 1:
      raise ValueError(
        f"history_length must be at least 1, got {self.history_length}"
      )

  def build(self) -> NoisyDepthCamera:
    return NoisyDepthCamera(self)


class NoisyDepthCamera(CameraSensor):
  """mjlab camera sensor extended with noise and temporal history."""

  cfg: NoisyDepthCameraCfg

  def __init__(self, cfg: NoisyDepthCameraCfg) -> None:
    super().__init__(cfg)
    self._depth_history: torch.Tensor | None = None
    self._history_valid: torch.Tensor | None = None
    self._pending_reset_all = True
    self._pending_reset_ids: set[int] = set()

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    if env_ids is None or isinstance(env_ids, slice):
      self._pending_reset_all = True
      self._pending_reset_ids.clear()
      return
    self._pending_reset_ids.update(
      int(index) for index in env_ids.detach().cpu().flatten().tolist()
    )

  def _compute_data(self) -> NoisyDepthCameraData:
    camera_data = super()._compute_data()
    if camera_data.depth is None:
      raise RuntimeError("NoisyDepthCamera requires depth rendering")

    raw_depth = camera_data.depth.clone()
    noisy_depth = apply_depth_noise(raw_depth, self.cfg.noise_pipeline)
    depth_history = self._append_history(noisy_depth)
    return NoisyDepthCameraData(
      rgb=None,
      depth=raw_depth,
      depth_noisy=noisy_depth,
      depth_history=depth_history,
    )

  def _append_history(self, depth: torch.Tensor) -> torch.Tensor:
    """Append one frame, backfilling newly reset environments."""
    expected_shape = (self.cfg.height, self.cfg.width, 1)
    if tuple(depth.shape[1:]) != expected_shape:
      raise ValueError(
        f"Expected depth frames [B, {expected_shape}], got {tuple(depth.shape)}"
      )

    batch_size = depth.shape[0]
    history_shape = (
      batch_size,
      self.cfg.history_length,
      *expected_shape,
    )
    if (
      self._depth_history is None
      or tuple(self._depth_history.shape) != history_shape
      or self._depth_history.device != depth.device
    ):
      self._depth_history = torch.empty(
        history_shape, dtype=depth.dtype, device=depth.device
      )
      self._history_valid = torch.zeros(
        batch_size, dtype=torch.bool, device=depth.device
      )
      self._pending_reset_all = True

    assert self._history_valid is not None
    if self._pending_reset_all:
      self._history_valid[:] = False
    elif self._pending_reset_ids:
      reset_ids = torch.tensor(
        sorted(self._pending_reset_ids),
        dtype=torch.long,
        device=depth.device,
      )
      if torch.any((reset_ids < 0) | (reset_ids >= batch_size)):
        raise IndexError(
          f"Camera reset environment IDs out of range for batch {batch_size}"
        )
      self._history_valid[reset_ids] = False
    self._pending_reset_all = False
    self._pending_reset_ids.clear()

    valid = self._history_valid
    if torch.any(valid):
      self._depth_history[valid, :-1] = self._depth_history[valid, 1:].clone()
      self._depth_history[valid, -1] = depth[valid]
    if torch.any(~valid):
      self._depth_history[~valid] = depth[~valid].unsqueeze(1)
      self._history_valid[~valid] = True

    return self._depth_history.clone()
