from __future__ import annotations

import torch

from mjlab.managers.termination_manager import TerminationManager


class DelayedTerminationManager(TerminationManager):
    """Delay reset while selected environments remain in a failed state.

    This matches AMP_mjlab: the counter advances only while a termination
    condition is active and clears as soon as the robot recovers.
    """

    def __init__(
        self,
        base: TerminationManager,
        delay_env_mask: torch.Tensor,
        max_delay_steps: int,
    ) -> None:
        # Steal all internal state from the base manager (avoid re-init).
        self.__dict__.update(base.__dict__)
        self._delay_env_mask = delay_env_mask          # (num_envs,) bool
        self._delay_counters = torch.zeros_like(delay_env_mask, dtype=torch.long)
        self._max_delay_steps = max_delay_steps

    def reset(self, env_ids=None):
        extras = super().reset(env_ids)
        self._delay_counters[slice(None) if env_ids is None else env_ids] = 0
        return extras

    def compute(self) -> torch.Tensor:
        dones = super().compute()  # fills _truncated_buf, _terminated_buf

        if self._max_delay_steps <= 0:
            return dones

        delay_and_done = self._delay_env_mask & dones
        self._delay_counters[delay_and_done] += 1

        not_ready = delay_and_done & (self._delay_counters < self._max_delay_steps)
        self._terminated_buf[not_ready] = False

        ready = delay_and_done & (self._delay_counters >= self._max_delay_steps)
        self._delay_counters[ready] = 0

        self._delay_counters[self._delay_env_mask & ~dones] = 0

        return self._truncated_buf | self._terminated_buf
