"""Torch-native running normalizer for AMP observations.

Derived from RSL-RL/AMP_mjlab implementations distributed under BSD-3-Clause.
"""

import torch
from torch import nn


class AmpNormalizer(nn.Module):
  def __init__(
    self,
    observation_dim: int,
    epsilon: float = 1e-4,
    clip: float = 10.0,
  ) -> None:
    super().__init__()
    self.epsilon = epsilon
    self.clip = clip
    self.register_buffer("mean", torch.zeros(observation_dim))
    self.register_buffer("var", torch.ones(observation_dim))
    self.register_buffer("count", torch.tensor(epsilon))

  @torch.no_grad()
  def update(self, samples: torch.Tensor) -> None:
    samples = samples.detach()
    batch_count = samples.shape[0]
    if batch_count == 0:
      return
    batch_mean = samples.mean(dim=0)
    batch_var = samples.var(dim=0, unbiased=False)
    delta = batch_mean - self.mean
    total = self.count + batch_count
    new_mean = self.mean + delta * batch_count / total
    m_a = self.var * self.count
    m_b = batch_var * batch_count
    m2 = m_a + m_b + delta.square() * self.count * batch_count / total
    self.mean.copy_(new_mean)
    self.var.copy_(m2 / total)
    self.count.copy_(total)

  def normalize(self, samples: torch.Tensor) -> torch.Tensor:
    normalized = (samples - self.mean) / torch.sqrt(self.var + self.epsilon)
    return torch.clamp(normalized, -self.clip, self.clip)
