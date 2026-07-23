"""Replay prepared AMP motion clips in MuJoCo viewer (kinematic playback)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import tyro

import mjlab
import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from src.rsl_rl.env.vecenv_wrapper import RslRlVecEnvWrapper


def _resolve_motion_path(motion_file: str | None, motion_dir: str) -> Path:
  if motion_file is not None:
    path = Path(motion_file).expanduser().resolve()
    if not path.exists():
      raise FileNotFoundError(f"Motion file not found: {path}")
    return path
  root = Path(motion_dir).expanduser().resolve()
  candidates = sorted(p for p in root.glob("*.npz") if not p.name.endswith("_M.npz"))
  if not candidates:
    raise FileNotFoundError(f"No motion .npz files found in {root}")
  return candidates[0]


class MotionReplayPolicy:
  """Apply prepared AMP clip frames directly to the simulation."""

  def __init__(self, env: ManagerBasedRlEnv, motion_path: Path, device: str) -> None:
    self._env = env
    self._robot = env.scene["robot"]
    self._sim = env.sim
    self._device = device
    data = np.load(motion_path, allow_pickle=True)
    if "joint_names" not in data:
      raise ValueError(f"{motion_path.name}: missing joint_names; regenerate motions")
    clip_names = [str(name) for name in data["joint_names"].tolist()]
    if clip_names != list(self._robot.joint_names):
      raise ValueError(f"{motion_path.name}: joint order does not match the G1 model")

    self._motion_name = motion_path.stem
    self._fps = int(np.asarray(data["fps"]).reshape(-1)[0])
    self._joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
    self._joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
    self._root_pos = torch.tensor(data["body_pos_w"][:, 0], dtype=torch.float32, device=device)
    self._root_quat = torch.tensor(data["body_quat_w"][:, 0], dtype=torch.float32, device=device)
    self._root_lin_vel = torch.tensor(
      data["body_lin_vel_w"][:, 0], dtype=torch.float32, device=device
    )
    self._root_ang_vel = torch.tensor(
      data["body_ang_vel_w"][:, 0], dtype=torch.float32, device=device
    )
    self._num_frames = self._joint_pos.shape[0]
    self._frame = 0
    self._action_dim = env.action_manager.total_action_dim
    print(
      f"[INFO] Playing {self._motion_name}: {self._num_frames} frames @ {self._fps} fps "
      f"({self._num_frames / self._fps:.1f}s)"
    )

  def reset(self) -> None:
    self._frame = 0
    self._apply_frame(self._frame)

  def advance(self) -> None:
    self._frame = (self._frame + 1) % self._num_frames
    self._apply_frame(self._frame)

  def _apply_frame(self, frame_idx: int) -> None:
    env_ids = torch.arange(self._env.num_envs, device=self._device)
    root_state = self._robot.data.default_root_state.clone()
    root_state[:, 0:3] = self._root_pos[frame_idx : frame_idx + 1]
    root_state[:, 3:7] = self._root_quat[frame_idx : frame_idx + 1]
    root_state[:, 7:10] = self._root_lin_vel[frame_idx : frame_idx + 1]
    root_state[:, 10:13] = self._root_ang_vel[frame_idx : frame_idx + 1]
    self._robot.write_root_state_to_sim(root_state, env_ids=env_ids)

    joint_pos = self._robot.data.default_joint_pos.clone()
    joint_vel = self._robot.data.default_joint_vel.clone()
    joint_pos[:, :] = self._joint_pos[frame_idx : frame_idx + 1]
    joint_vel[:, :] = self._joint_vel[frame_idx : frame_idx + 1]
    self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    self._sim.forward()
    self._env.scene.update(self._sim.mj_model.opt.timestep)

  def __call__(self, obs) -> torch.Tensor:
    del obs
    return torch.zeros(self._env.num_envs, self._action_dim, device=self._device)


class KinematicReplayEnv:
  """Viewer adapter that advances motion clips without physics stepping."""

  def __init__(self, vec_env: RslRlVecEnvWrapper, policy: MotionReplayPolicy) -> None:
    self._vec = vec_env
    self._policy = policy

  @property
  def unwrapped(self) -> ManagerBasedRlEnv:
    return self._vec.unwrapped

  @property
  def num_envs(self) -> int:
    return self._vec.num_envs

  @property
  def device(self) -> torch.device:
    return self._vec.device

  @property
  def cfg(self):
    return self._vec.cfg

  def get_observations(self):
    return self._vec.get_observations()

  def step(self, actions: torch.Tensor):
    del actions
    self._policy.advance()
    obs = self.get_observations()
    zeros = torch.zeros(self.num_envs, device=self.device)
    done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    return obs, zeros, done, {}

  def reset(self):
    obs, extras = self._vec.reset()
    self._policy.reset()
    return obs, extras

  def close(self) -> None:
    self._vec.close()


def main(
  motion_file: str | None = None,
  motion_dir: str = "src/assets/motions/g1/amp/WalkandRun",
  num_envs: int = 1,
  device: str | None = None,
  viewer: str = "auto",
) -> None:
  configure_torch_backends()
  device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  motion_path = _resolve_motion_path(motion_file, motion_dir)

  env_cfg = load_env_cfg("Unitree-G1-AMP-Flat", play=True)
  env_cfg.scene.num_envs = num_envs
  env_cfg.terminations = {}
  # Pure kinematic replay: do not randomize spawn from the motion loader.
  env_cfg.events.pop("reset_from_motion", None)
  env_cfg.events.pop("init_motion_loader", None)
  # Viser velocity-command GUI requires each axis max >= 0.1.
  twist_cmd = env_cfg.commands["twist"]
  if twist_cmd.ranges.lin_vel_y[1] < 0.1:
    twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)

  base_env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  vec_env = RslRlVecEnvWrapper(base_env, clip_actions=None)
  policy = MotionReplayPolicy(base_env, motion_path, device=device)
  policy.reset()
  env = KinematicReplayEnv(vec_env, policy)

  if viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved = "native" if has_display else "viser"
  else:
    resolved = viewer

  if resolved == "native":
    NativeMujocoViewer(env, policy).run()
  elif resolved == "viser":
    ViserPlayViewer(env, policy).run()
  else:
    raise ValueError(f"Unsupported viewer: {viewer}")

  env.close()


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
