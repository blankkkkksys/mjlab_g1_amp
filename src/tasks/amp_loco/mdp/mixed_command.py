"""Velocity sampling with dedicated turning and forward-running examples."""
from dataclasses import dataclass

import torch
from mjlab.tasks.velocity.mdp.velocity_command import (
  UniformVelocityCommand, UniformVelocityCommandCfg,
)


class AmpVelocityCommand(UniformVelocityCommand):
  cfg: "AmpVelocityCommandCfg"

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)
    mode = torch.rand(len(env_ids), device=self.device)
    moving = ~self.is_standing_env[env_ids]
    turn = env_ids[moving & (mode < self.cfg.turn_fraction)]
    run = env_ids[moving & (mode >= self.cfg.turn_fraction)
                  & (mode < self.cfg.turn_fraction + self.cfg.run_fraction)]
    self.is_heading_env[turn] = False
    self.is_heading_env[run] = False
    self.vel_command_b[turn, :2] = 0.0
    magnitude = torch.empty(len(turn), device=self.device).uniform_(
      0.5, max(0.5, self.cfg.ranges.ang_vel_z[1]))
    sign = torch.where(torch.rand(len(turn), device=self.device) < 0.5, -1., 1.)
    self.vel_command_b[turn, 2] = magnitude * sign
    upper = self.cfg.ranges.lin_vel_x[1]
    self.vel_command_b[run, 0] = torch.empty(len(run), device=self.device).uniform_(
      min(self.cfg.running_min_speed, upper), upper)
    self.vel_command_b[run, 1] = 0.0
    self.vel_command_b[run, 2] = torch.empty(len(run), device=self.device).uniform_(-0.3, 0.3)


@dataclass(kw_only=True)
class AmpVelocityCommandCfg(UniformVelocityCommandCfg):
  turn_fraction: float = 0.2
  run_fraction: float = 0.3
  running_min_speed: float = 1.6

  def build(self, env):
    if self.init_velocity_prob != 0.0:
      raise ValueError("AMP mixed commands require init_velocity_prob=0")
    if not (0 <= self.turn_fraction <= 1 and 0 <= self.run_fraction <= 1
            and self.turn_fraction + self.run_fraction <= 1):
      raise ValueError("Command fractions must be nonnegative and sum to at most 1")
    return AmpVelocityCommand(self, env)
