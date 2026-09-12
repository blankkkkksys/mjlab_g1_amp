# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Denoising World Model Learning policy model."""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn
from tensordict import TensorDict

from src.rsl_rl.models.rnn_model import RNNModel
from src.rsl_rl.modules import MLP, HiddenState
from src.rsl_rl.utils import unpad_trajectories


class DWLModel(RNNModel):
    """Recurrent policy whose sparse latent reconstructs privileged state."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (48,),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        num_embedding: int = 24,
        enc_hidden_dims: tuple[int, ...] | list[int] = (256,),
        dec_hidden_dims: tuple[int, ...] | list[int] = (64,),
        rnn_type: str = "lstm",
        rnn_hidden_dim: int = 256,
        rnn_num_layers: int = 1,
        decoder_obs_set: str = "critic",
    ) -> None:
        if not enc_hidden_dims or not dec_hidden_dims or not hidden_dims:
            raise ValueError(
                "DWL encoder, decoder, and actor hidden dimensions must be non-empty."
            )
        if decoder_obs_set not in obs_groups:
            raise ValueError(
                f"Decoder observation set '{decoder_obs_set}' is not configured."
            )

        super().__init__(
            obs=obs,
            obs_groups=obs_groups,
            obs_set=obs_set,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            obs_normalization=obs_normalization,
            distribution_cfg=distribution_cfg,
            rnn_type=rnn_type,
            rnn_hidden_dim=rnn_hidden_dim,
            rnn_num_layers=rnn_num_layers,
        )

        self.num_embedding = num_embedding
        self.decoder_obs_groups = tuple(obs_groups[decoder_obs_set])
        self.decoder_obs_dim = sum(
            obs[group].shape[-1] for group in self.decoder_obs_groups
        )

        policy_output_dim = (
            self.distribution.input_dim
            if self.distribution is not None
            else output_dim
        )
        embedding = MLP(
            rnn_hidden_dim,
            num_embedding,
            enc_hidden_dims,
            activation,
        )
        policy_head = MLP(
            num_embedding,
            policy_output_dim,
            hidden_dims,
            activation,
        )
        # Keeping the complete inference path in ``mlp`` lets the existing
        # exporters include the embedding but omit the training-only decoder.
        self.mlp = nn.Sequential(
            OrderedDict(
                (
                    ("embedding", embedding),
                    ("policy_head", policy_head),
                )
            )
        )
        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.mlp)

        self.decoder = MLP(
            num_embedding,
            self.decoder_obs_dim,
            dec_hidden_dims,
            activation,
        )

    @property
    def embedding(self) -> nn.Module:
        """Return the policy embedding network."""
        return self.mlp.embedding  # type: ignore[attr-defined, no-any-return]

    def compute_dwl_losses(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute privileged-state reconstruction and latent sparsity losses."""
        rnn_features = self.get_latent(obs, masks, hidden_state)
        latent = self.embedding(rnn_features)
        predicted_state = self.decoder(latent)

        target_state = torch.cat(
            [obs[group] for group in self.decoder_obs_groups], dim=-1
        )
        if masks is not None:
            target_state = unpad_trajectories(target_state, masks)

        reconstruction_loss = torch.nn.functional.mse_loss(
            predicted_state, target_state.detach()
        )
        latent_l1_loss = torch.mean(torch.sum(torch.abs(latent), dim=-1))
        return reconstruction_loss, latent_l1_loss

    def act_with_hidden_state(
        self,
        obs: TensorDict,
        hidden_state: HiddenState,
    ) -> tuple[torch.Tensor, HiddenState]:
        """Run one deterministic step with an explicitly managed RNN state."""
        obs_input = torch.cat([obs[group] for group in self.obs_groups], dim=-1)
        obs_input = self.obs_normalizer(obs_input)
        rnn_output, next_hidden_state = self.rnn.rnn(
            obs_input.unsqueeze(0), hidden_state
        )
        model_output = self.mlp(rnn_output.squeeze(0))
        if self.distribution is not None:
            actions = self.distribution.deterministic_output(model_output)
        else:
            actions = model_output
        return actions, next_hidden_state
