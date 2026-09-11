# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import copy

from src.rsl_rl.env import VecEnv


def resolve_symmetry_config(alg_cfg: dict, env: VecEnv) -> dict:
    """Resolve the symmetry configuration.

    Args:
        alg_cfg: Algorithm configuration dictionary.
        env: Environment object.

    Returns:
        The resolved algorithm configuration dictionary.
    """
    # Keep runtime-only environment references out of the persisted runner config.
    # The original config is later serialized to YAML and potentially sent to W&B.
    resolved_cfg = alg_cfg.copy()
    if alg_cfg.get("symmetry_cfg") is not None:
        symmetry_cfg = copy.copy(alg_cfg["symmetry_cfg"])
        symmetry_cfg["_env"] = env
        resolved_cfg["symmetry_cfg"] = symmetry_cfg
    else:
        resolved_cfg["symmetry_cfg"] = None
    return resolved_cfg
