#!/usr/bin/env python3
"""Regenerate G1 AMP deploy.yaml from the training environment config."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from mjlab.envs import ManagerBasedRlEnv
from mjlab.scene import Scene
from mjlab.sim.sim import Simulation, SimulationCfg
from src.tasks.amp_loco.config.g1.env_cfgs import g1_amp_flat_env_cfg

_DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "deploy/robots/g1/config/policy/amp/v0/params/deploy.yaml"
)


def _fmt_floats(values: list[float], decimals: int = 3) -> list[float]:
  return [round(float(v), decimals) for v in values]


def build_deploy_cfg() -> dict:
  cfg = g1_amp_flat_env_cfg()
  cfg.scene.num_envs = 1
  scene = Scene(cfg.scene, device="cpu")
  sim = Simulation(num_envs=1, cfg=SimulationCfg(), model=scene.compile(), device="cpu")
  scene.initialize(sim.mj_model, sim.model, sim.data)
  env = ManagerBasedRlEnv(cfg=cfg, scene=scene, sim=sim, device="cpu")
  robot = env.scene["robot"]
  joint_action = env.action_manager.get_term("joint_pos")
  scale = (
    joint_action._scale[0].tolist()
    if isinstance(joint_action._scale, torch.Tensor)
    else joint_action._scale
  )
  default_joint_pos = robot.data.default_joint_pos[0].tolist()
  joint_name_to_ctrl_id = {
    actuator.target.split("/")[-1]: actuator.id for actuator in robot.spec.actuators
  }
  ctrl_ids = [
    joint_name_to_ctrl_id[jname]
    for jname in robot.joint_names
    if jname in joint_name_to_ctrl_id
  ]
  stiffness = env.sim.mj_model.actuator_gainprm[ctrl_ids, 0].tolist()
  damping = (-env.sim.mj_model.actuator_biasprm[ctrl_ids, 2]).tolist()
  ones29 = [1.0] * 29

  return {
    "joint_ids_map": list(range(len(robot.joint_names))),
    "step_dt": cfg.sim.mujoco.timestep * cfg.decimation,
    "stiffness": _fmt_floats(stiffness, 1),
    "damping": _fmt_floats(damping, 1),
    "default_joint_pos": _fmt_floats(default_joint_pos, 2),
    "commands": {
      "base_velocity": {
        "ranges": {
          "lin_vel_x": [-1.5, 2.0],
          "lin_vel_y": [-1.0, 1.0],
          "ang_vel_z": [-1.0, 1.0],
          "heading": None,
        }
      }
    },
    "actions": {
      "JointPositionAction": {
        "clip": None,
        "joint_names": [".*"],
        "scale": _fmt_floats(scale, 3),
        "offset": _fmt_floats(default_joint_pos, 2),
        "joint_ids": None,
      }
    },
    "observations": {
      "use_gym_history": True,
      "base_ang_vel": {
        "params": {},
        "clip": None,
        "scale": [1.0, 1.0, 1.0],
        "history_length": cfg.observations["actor"].history_length,
      },
      "projected_gravity": {
        "params": {},
        "clip": None,
        "scale": [1.0, 1.0, 1.0],
        "history_length": cfg.observations["actor"].history_length,
      },
      "velocity_commands": {
        "params": {"command_name": "base_velocity"},
        "clip": None,
        "scale": [1.0, 1.0, 1.0],
        "history_length": cfg.observations["actor"].history_length,
      },
      "joint_pos_rel": {
        "params": {},
        "clip": None,
        "scale": ones29,
        "history_length": cfg.observations["actor"].history_length,
      },
      "joint_vel_rel": {
        "params": {},
        "clip": None,
        "scale": ones29,
        "history_length": cfg.observations["actor"].history_length,
      },
      "last_action": {
        "params": {},
        "clip": None,
        "scale": ones29,
        "history_length": cfg.observations["actor"].history_length,
      },
    },
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--output",
    type=Path,
    default=_DEFAULT_OUTPUT,
    help=f"Output deploy.yaml path (default: {_DEFAULT_OUTPUT})",
  )
  args = parser.parse_args()
  deploy_cfg = build_deploy_cfg()
  args.output.parent.mkdir(parents=True, exist_ok=True)
  header = (
    "# Auto-generated from scripts/export_amp_deploy_yaml.py\n"
    "# Unitree G1 AMP-Flat deploy config (aligned with Unitree-G1-AMP-Flat training).\n"
  )
  yaml_text = yaml.safe_dump(
    deploy_cfg,
    sort_keys=False,
    default_flow_style=None,
    width=120,
  )
  args.output.write_text(header + yaml_text, encoding="utf-8")
  print(f"Wrote {args.output}")


if __name__ == "__main__":
  main()
