"""Adversarial motion-prior discriminator.

Derived from RSL-RL/AMP_mjlab implementations distributed under BSD-3-Clause.
"""

import torch
from torch import autograd, nn

from src.rsl_rl.modules.amp_normalizer import AmpNormalizer


class Discriminator(nn.Module):
  def __init__(
    self,
    observation_dim: int,
    reward_coef: float,
    hidden_dims: list[int],
    task_reward_lerp: float,
  ) -> None:
    super().__init__()
    layers: list[nn.Module] = []
    input_dim = 2 * observation_dim
    for hidden_dim in hidden_dims:
      layers.extend((nn.Linear(input_dim, hidden_dim), nn.ReLU()))
      input_dim = hidden_dim
    self.trunk = nn.Sequential(*layers)
    self.head = nn.Linear(input_dim, 1)
    self.reward_coef = reward_coef
    self.task_reward_lerp = task_reward_lerp

  def forward(self, state: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
    return self.head(self.trunk(torch.cat((state, next_state), dim=-1)))

  def compute_grad_penalty(
    self,
    expert_state: torch.Tensor,
    expert_next_state: torch.Tensor,
    coefficient: float = 10.0,
  ) -> torch.Tensor:
    expert_data = torch.cat((expert_state, expert_next_state), dim=-1)
    expert_data.requires_grad_(True)
    logits = self.head(self.trunk(expert_data))
    gradient = autograd.grad(
      outputs=logits,
      inputs=expert_data,
      grad_outputs=torch.ones_like(logits),
      create_graph=True,
      retain_graph=True,
      only_inputs=True,
    )[0]
    return coefficient * gradient.norm(2, dim=1).square().mean()

  @torch.no_grad()
  def predict_style_reward(
    self,
    state: torch.Tensor,
    next_state: torch.Tensor,
    normalizer: AmpNormalizer,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Return style reward and raw logits for logging."""
    state = normalizer.normalize(state)
    next_state = normalizer.normalize(next_state)
    logits = self(state, next_state).squeeze(-1)
    style_reward = self.reward_coef * torch.clamp(
      1.0 - 0.25 * (logits - 1.0).square(), min=0.0
    )
    return style_reward, logits

  @torch.no_grad()
  def predict_reward(
    self,
    state: torch.Tensor,
    next_state: torch.Tensor,
    task_reward: torch.Tensor,
    normalizer: AmpNormalizer,
  ) -> torch.Tensor:
    style_reward, _ = self.predict_style_reward(state, next_state, normalizer)
    return (
      (1.0 - self.task_reward_lerp) * style_reward
      + self.task_reward_lerp * task_reward
    )
