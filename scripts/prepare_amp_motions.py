"""Prepare AMP motion clips from LAFAN CSV or stamp metadata on existing NPZ.

LAFAN retargeted CSV files use the same layout as ``scripts/csv_to_npz.py``:
root pose (xyz + xyzw quaternion) followed by 29 G1 joint positions.

Outputs are written to ``src/assets/motions/g1/amp/WalkandRun`` and
``src/assets/motions/g1/amp/Recovery`` with ``joint_names`` / ``body_names``
metadata for AMP loaders.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import tyro
from tqdm import tqdm

import mjlab
from mjlab.entity import Entity
from mjlab.scene import Scene
from mjlab.sim.sim import Simulation, SimulationCfg
from src.tasks.amp_loco.g1_motion_constants import G1_JOINT_NAMES
from src.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg


def _normalize_quat_wxyz(quat: np.ndarray) -> np.ndarray:
  q = np.asarray(quat, dtype=np.float64)
  norm = np.linalg.norm(q, axis=-1, keepdims=True)
  q = q / np.clip(norm, 1e-8, None)
  flip = q[..., :1] < 0.0
  q = np.where(flip, -q, q)
  return q.astype(np.float32)


def _build_robot(device: str) -> tuple[Scene, Simulation, Entity]:
  env_cfg = unitree_g1_flat_tracking_env_cfg()
  env_cfg.scene.num_envs = 1
  scene = Scene(env_cfg.scene, device=device)
  sim_cfg = SimulationCfg()
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=scene.compile(), device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)
  robot: Entity = scene["robot"]
  if list(robot.joint_names) != list(G1_JOINT_NAMES):
    raise ValueError("G1 joint order changed; update G1_JOINT_NAMES in this script")
  return scene, sim, robot


def stamp_npz_metadata(path: Path, robot: Entity) -> None:
  """Add joint/body name metadata to an existing AMP NPZ clip."""
  data = dict(np.load(path, allow_pickle=True))
  if data["joint_pos"].shape[1] != len(robot.joint_names):
    raise ValueError(
      f"{path.name}: joint count {data['joint_pos'].shape[1]} != "
      f"{len(robot.joint_names)}"
    )
  if data["body_pos_w"].shape[1] != len(robot.body_names):
    raise ValueError(
      f"{path.name}: body count {data['body_pos_w'].shape[1]} != "
      f"{len(robot.body_names)}"
    )
  data["joint_names"] = np.array(robot.joint_names, dtype=object)
  data["body_names"] = np.array(robot.body_names, dtype=object)
  if "body_quat_w" in data:
    data["body_quat_w"] = _normalize_quat_wxyz(data["body_quat_w"])
  np.savez(path, **data)


@torch.no_grad()
def convert_csv_to_npz(
  csv_path: Path,
  output_path: Path,
  device: str,
  input_fps: float = 30.0,
  output_fps: float = 50.0,
) -> None:
  """Convert one LAFAN CSV clip to AMP NPZ via kinematic replay in MuJoCo."""
  from scripts.csv_to_npz import MotionLoader, run_sim

  scene, sim, robot = _build_robot(device)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  run_sim(
    sim=sim,
    scene=scene,
    joint_names=list(G1_JOINT_NAMES),
    input_fps=input_fps,
    input_file=str(csv_path),
    output_fps=output_fps,
    output_path=str(output_path),
    render=False,
    line_range=None,
    renderer=None,
  )
  stamp_npz_metadata(output_path, robot)


def _convert_dir(
  input_dir: Path,
  output_dir: Path,
  device: str,
  input_fps: float,
  output_fps: float,
) -> None:
  csv_files = sorted(input_dir.glob("*.csv"))
  if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {input_dir}")
  output_dir.mkdir(parents=True, exist_ok=True)
  print(f"Converting {len(csv_files)} LAFAN CSV clips -> {output_dir}")
  for csv_path in csv_files:
    convert_csv_to_npz(
      csv_path,
      output_dir / f"{csv_path.stem}.npz",
      device=device,
      input_fps=input_fps,
      output_fps=output_fps,
    )


def _stamp_dir(npz_dir: Path, robot: Entity) -> None:
  files = sorted(p for p in npz_dir.glob("*.npz") if not p.name.endswith("_M.npz"))
  if not files:
    raise FileNotFoundError(f"No NPZ files found in {npz_dir}")
  print(f"Stamping metadata on {len(files)} clips in {npz_dir}")
  for path in tqdm(files, desc=npz_dir.name):
    stamp_npz_metadata(path, robot)


def _clean_amp_motion_dirs(walk_output_dir: Path, recovery_output_dir: Path) -> None:
  """Remove existing AMP NPZ clips before syncing from a reference repo."""
  removed = 0
  for directory in (walk_output_dir, recovery_output_dir):
    if not directory.is_dir():
      continue
    for path in directory.glob("*.npz"):
      path.unlink()
      removed += 1
  print(f"Cleaned {removed} existing AMP motion clips")


def _sync_from_ref(
  ref_repo: Path,
  walk_output_dir: Path,
  recovery_output_dir: Path,
) -> None:
  """Copy prepared NPZ clips from a reference AMP_mjlab checkout."""
  ref_amp = ref_repo / "src" / "assets" / "motions" / "g1" / "amp"
  if not ref_amp.is_dir():
    raise FileNotFoundError(f"Reference AMP motion root not found: {ref_amp}")

  import shutil

  _clean_amp_motion_dirs(walk_output_dir, recovery_output_dir)

  for subdir, output_dir in (
    ("WalkandRun", walk_output_dir),
    ("Recovery", recovery_output_dir),
  ):
    src_dir = ref_amp / subdir
    if not src_dir.is_dir():
      raise FileNotFoundError(f"Reference motion directory not found: {src_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in src_dir.glob("*.npz") if not p.name.endswith("_M.npz"))
    if not files:
      raise FileNotFoundError(f"No NPZ files found in {src_dir}")
    print(f"Syncing {len(files)} clips from {src_dir} -> {output_dir}")
    for src in files:
      shutil.copy2(src, output_dir / src.name)


def main(
  mode: str = "stamp",
  ref_repo: str = "~/projects/AMP_mjlab",
  walk_csv_dir: str = "src/assets/motions/g1/lafan/walk_run",
  recovery_csv_dir: str = "src/assets/motions/g1/lafan/recovery",
  walk_output_dir: str = "src/assets/motions/g1/amp/WalkandRun",
  recovery_output_dir: str = "src/assets/motions/g1/amp/Recovery",
  input_fps: float = 30.0,
  output_fps: float = 50.0,
  device: str = "cpu",
) -> None:
  """Prepare AMP motion datasets.

  Modes:
    stamp: add joint/body metadata to existing NPZ under output dirs
    sync: clean local AMP dirs, copy NPZ from a reference AMP_mjlab repo, then stamp metadata
    csv: convert LAFAN CSV directories to AMP NPZ (WalkandRun + Recovery)
  """
  if mode == "sync":
    _sync_from_ref(
      Path(ref_repo).expanduser().resolve(),
      Path(walk_output_dir),
      Path(recovery_output_dir),
    )
    _scene, _sim, robot = _build_robot(device)
    for directory in (Path(walk_output_dir), Path(recovery_output_dir)):
      _stamp_dir(directory, robot)
    print("Done.")
    return

  if mode == "csv":
    _convert_dir(Path(walk_csv_dir), Path(walk_output_dir), device, input_fps, output_fps)
    recovery_dir = Path(recovery_csv_dir)
    if recovery_dir.is_dir() and any(recovery_dir.glob("*.csv")):
      _convert_dir(recovery_dir, Path(recovery_output_dir), device, input_fps, output_fps)
    print("Done.")
    return

  if mode != "stamp":
    raise ValueError(f"Unsupported mode: {mode}")

  _scene, _sim, robot = _build_robot(device)
  for directory in (Path(walk_output_dir), Path(recovery_output_dir)):
    if directory.is_dir():
      _stamp_dir(directory, robot)
  print("Done.")


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
