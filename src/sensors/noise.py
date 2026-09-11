"""Composable depth-image noise models."""
# flake8: noqa

from __future__ import annotations

import abc
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(kw_only=True)
class DepthNoiseCfg(abc.ABC):
  """Base configuration for one depth-image transform."""

  @abc.abstractmethod
  def apply(self, depth: torch.Tensor) -> torch.Tensor:
    """Transform a ``[B, H, W, 1]`` depth tensor."""


@dataclass(kw_only=True)
class DepthRangeClipCfg(DepthNoiseCfg):
  """Replace non-finite values and clamp depth to the sensor range."""

  min_depth: float = 0.1
  max_depth: float = 5.0

  def __post_init__(self) -> None:
    if self.min_depth < 0.0 or self.max_depth <= self.min_depth:
      raise ValueError(
        "Depth range must satisfy 0 <= min_depth < max_depth, got "
        f"({self.min_depth}, {self.max_depth})"
      )

  def apply(self, depth: torch.Tensor) -> torch.Tensor:
    depth = torch.nan_to_num(
      depth,
      nan=self.max_depth,
      posinf=self.max_depth,
      neginf=self.min_depth,
    )
    return depth.clamp(self.min_depth, self.max_depth)


@dataclass(kw_only=True)
class DepthGaussianNoiseCfg(DepthNoiseCfg):
  """Add independent Gaussian range noise."""

  mean: float = 0.0
  std: float = 0.01

  def __post_init__(self) -> None:
    if self.std < 0.0:
      raise ValueError(f"Gaussian noise std must be non-negative, got {self.std}")

  def apply(self, depth: torch.Tensor) -> torch.Tensor:
    if self.std == 0.0:
      return depth.clone()
    return depth + torch.randn_like(depth) * self.std + self.mean


@dataclass(kw_only=True)
class DepthPixelDropoutCfg(DepthNoiseCfg):
  """Replace randomly selected pixels with a fixed invalid value."""

  probability: float = 0.005
  value: float = 0.0

  def __post_init__(self) -> None:
    if not 0.0 <= self.probability <= 1.0:
      raise ValueError(
        f"Pixel dropout probability must be in [0, 1], got {self.probability}"
      )

  def apply(self, depth: torch.Tensor) -> torch.Tensor:
    if self.probability == 0.0:
      return depth.clone()
    mask = torch.rand_like(depth) < self.probability
    return torch.where(mask, torch.full_like(depth, self.value), depth)


@dataclass(kw_only=True)
class DepthContourNoiseCfg(DepthNoiseCfg):
  """Invalidate pixels near large local depth discontinuities."""

  threshold: float = 0.15
  kernel_size: int = 3
  value: float = 0.0

  def __post_init__(self) -> None:
    if self.threshold < 0.0:
      raise ValueError(
        f"Contour threshold must be non-negative, got {self.threshold}"
      )
    if self.kernel_size < 1 or self.kernel_size % 2 == 0:
      raise ValueError("Contour kernel_size must be a positive odd integer")

  def apply(self, depth: torch.Tensor) -> torch.Tensor:
    if depth.ndim != 4 or depth.shape[-1] != 1:
      raise ValueError(
        "Depth contour noise expects [B, H, W, 1], got "
        f"{tuple(depth.shape)}"
      )
    image = depth.permute(0, 3, 1, 2)
    padding = self.kernel_size // 2
    local_max = F.max_pool2d(
      image, self.kernel_size, stride=1, padding=padding
    )
    local_min = -F.max_pool2d(
      -image, self.kernel_size, stride=1, padding=padding
    )
    contour = (local_max - local_min) > self.threshold
    result = torch.where(contour, torch.full_like(image, self.value), image)
    return result.permute(0, 2, 3, 1)


def apply_depth_noise(
  depth: torch.Tensor, pipeline: tuple[DepthNoiseCfg, ...]
) -> torch.Tensor:
  """Apply depth transforms in configuration order without mutating input."""
  result = depth.clone()
  for noise_cfg in pipeline:
    result = noise_cfg.apply(result)
  return result
