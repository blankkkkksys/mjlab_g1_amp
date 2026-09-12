# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO with the Denoising World Model Learning objective."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from src.rsl_rl.algorithms.ppo import PPO
from src.rsl_rl.models.dwl_model import DWLModel
from src.rsl_rl.modules import HiddenState
from src.rsl_rl.storage import RolloutStorage
from src.rsl_rl.utils import resolve_callable


class DWLPPO(PPO):
    """PPO with privileged-state reconstruction from the actor latent."""

    actor: DWLModel

    def __init__(
        self,
        *args,
        reconstruction_loss_coef: float = 1.0,
        latent_l1_coef: float = 2.0e-3,
        symmetry_cfg: dict | None = None,
        **kwargs,
    ) -> None:
        # Standard PPO symmetry assumes feed-forward policies. DWL instead
        # maintains a second recurrent stream with its own hidden state.
        super().__init__(*args, symmetry_cfg=None, **kwargs)
        if not isinstance(self.actor, DWLModel):
            raise TypeError("DWLPPO requires its actor to be a DWLModel.")
        self.reconstruction_loss_coef = reconstruction_loss_coef
        self.latent_l1_coef = latent_l1_coef
        self.symmetric_hidden_state: HiddenState = None
        if symmetry_cfg is not None:
            symmetry_cfg = symmetry_cfg.copy()
            symmetry_cfg["data_augmentation_func"] = resolve_callable(
                symmetry_cfg["data_augmentation_func"]
            )
        self.dwl_symmetry = symmetry_cfg

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Collect original actions and advance the mirrored recurrent stream."""
        if self.dwl_symmetry is not None:
            augmentation_func = self.dwl_symmetry["data_augmentation_func"]
            augmented_obs, _ = augmentation_func(
                obs=obs,
                actions=None,
                env=self.dwl_symmetry["_env"],
            )
            if augmented_obs is None:
                raise RuntimeError("DWL symmetry function did not return observations.")
            num_envs = obs.batch_size[0]
            symmetric_obs = augmented_obs[num_envs:]
            self.transition.symmetric_observations = symmetric_obs
            self.transition.symmetric_hidden_state = self._clone_hidden_state(
                self.symmetric_hidden_state
            )
            _, self.symmetric_hidden_state = self.actor.act_with_hidden_state(
                symmetric_obs, self.symmetric_hidden_state
            )
        return super().act(obs)

    def process_env_step(
        self,
        obs: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict[str, torch.Tensor],
    ) -> None:
        """Store the dual-stream transition and reset both states on termination."""
        super().process_env_step(obs, rewards, dones, extras)
        self._reset_symmetric_hidden_state(dones)

    def compute_auxiliary_loss(
        self, batch: RolloutStorage.Batch
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Add DWL reconstruction and sparse-latent regularization to PPO."""
        reconstruction_loss, latent_l1_loss = self.actor.compute_dwl_losses(
            batch.observations,
            masks=batch.masks,
            hidden_state=batch.hidden_states[0],
        )
        total_loss = (
            self.reconstruction_loss_coef * reconstruction_loss
            + self.latent_l1_coef * latent_l1_loss
        )
        metrics = {
            "dwl_reconstruction": reconstruction_loss.detach(),
            "dwl_latent_l1": latent_l1_loss.detach(),
        }
        if self.dwl_symmetry is not None:
            if batch.symmetric_observations is None:
                raise RuntimeError("Mirrored observations are missing from DWL storage.")
            original_actions = self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
            )
            symmetric_actions = self.actor(
                batch.symmetric_observations,
                masks=batch.masks,
                hidden_state=batch.symmetric_hidden_state,
            )
            augmentation_func = self.dwl_symmetry["data_augmentation_func"]
            _, augmented_actions = augmentation_func(
                obs=None,
                actions=original_actions,
                env=self.dwl_symmetry["_env"],
            )
            if augmented_actions is None:
                raise RuntimeError("DWL symmetry function did not return actions.")
            mirrored_original_actions = augmented_actions[original_actions.shape[0] :]
            symmetry_loss = torch.nn.functional.mse_loss(
                symmetric_actions, mirrored_original_actions.detach()
            )
            total_loss = (
                total_loss
                + self.dwl_symmetry["mirror_loss_coeff"] * symmetry_loss
            )
            metrics["dwl_symmetry"] = symmetry_loss.detach()
        metrics["dwl_total"] = total_loss.detach()
        return total_loss, metrics

    @staticmethod
    def _clone_hidden_state(hidden_state: HiddenState) -> HiddenState:
        if isinstance(hidden_state, tuple):
            return tuple(state.detach().clone() for state in hidden_state)
        if hidden_state is not None:
            return hidden_state.detach().clone()
        return None

    def _reset_symmetric_hidden_state(self, dones: torch.Tensor) -> None:
        hidden_state = self.symmetric_hidden_state
        if hidden_state is None:
            return
        states = hidden_state if isinstance(hidden_state, tuple) else (hidden_state,)
        for state in states:
            state[..., dones == 1, :] = 0.0
