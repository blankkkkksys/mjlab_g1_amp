"""Fixed-size replay buffer for AMP policy transitions.

Derived from RSL-RL/AMP_mjlab implementations distributed under BSD-3-Clause.
"""

import torch


class AmpReplayBuffer:
  def __init__(self, observation_dim: int, capacity: int, device: str) -> None:
    self.states = torch.zeros(capacity, observation_dim, device=device)
    self.next_states = torch.zeros_like(self.states)
    self.capacity = capacity
    self.write_index = 0
    self.size = 0

  @torch.no_grad()
  def insert(self, states: torch.Tensor, next_states: torch.Tensor) -> None:
    states = states.detach()
    next_states = next_states.detach()
    count = states.shape[0]
    if count >= self.capacity:
      self.states.copy_(states[-self.capacity :])
      self.next_states.copy_(next_states[-self.capacity :])
      self.write_index = 0
      self.size = self.capacity
      return
    first = min(count, self.capacity - self.write_index)
    self.states[self.write_index : self.write_index + first].copy_(states[:first])
    self.next_states[self.write_index : self.write_index + first].copy_(
      next_states[:first]
    )
    remaining = count - first
    if remaining:
      self.states[:remaining].copy_(states[first:])
      self.next_states[:remaining].copy_(next_states[first:])
    self.write_index = (self.write_index + count) % self.capacity
    self.size = min(self.capacity, self.size + count)

  def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    if self.size == 0:
      raise RuntimeError("Cannot sample an empty AMP replay buffer")
    indices = torch.randint(self.size, (batch_size,), device=self.states.device)
    return self.states[indices], self.next_states[indices]
