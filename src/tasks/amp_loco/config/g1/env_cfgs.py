"""Unitree G1 AMP Locomotion environment configurations."""

import os

from src.assets.robots import (
  G1_ACTION_SCALE,
  get_g1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from src.sensors import (
  DepthContourNoiseCfg,
  DepthGaussianNoiseCfg,
  DepthPixelDropoutCfg,
  DepthRangeClipCfg,
  NoisyDepthCameraCfg,
)
from src.tasks.amp_loco.amp_env_cfg import make_amp_env_cfg

# Bodies tracked by AMP style reward and amp/critic body observations.
# Includes full leg chains plus shoulder_pitch for natural arm swing.
G1_AMP_BODY_NAMES = (
  "pelvis",
  "left_hip_pitch_link",
  "left_hip_roll_link",
  "left_hip_yaw_link",
  "left_knee_link",
  "left_ankle_pitch_link",
  "left_ankle_roll_link",
  "right_hip_pitch_link",
  "right_hip_roll_link",
  "right_hip_yaw_link",
  "right_knee_link",
  "right_ankle_pitch_link",
  "right_ankle_roll_link",
  "left_shoulder_pitch_link",
  "left_shoulder_roll_link",
  "left_elbow_link",
  "left_wrist_yaw_link",
  "right_shoulder_pitch_link",
  "right_shoulder_roll_link",
  "right_elbow_link",
  "right_wrist_yaw_link",
)


def add_g1_amp_depth_camera(
  cfg: ManagerBasedRlEnvCfg,
  *,
  width: int = 64,
  height: int = 48,
  history_length: int = 3,
) -> NoisyDepthCameraCfg:
  """Attach an optional forward-facing noisy depth camera to the G1 torso."""
  camera_cfg = NoisyDepthCameraCfg(
    name="depth_camera",
    parent_body="robot/torso_link",
    pos=(0.08, 0.0, 0.32),
    # MuJoCo cameras look along local -Z; this maps forward to torso +X
    # and image-up to torso +Z.
    quat=(0.5, 0.5, -0.5, -0.5),
    fovy=75.0,
    width=width,
    height=height,
    use_textures=False,
    use_shadows=False,
    enabled_geom_groups=(0, 1, 2),
    clone_data=True,
    history_length=history_length,
    noise_pipeline=(
      DepthRangeClipCfg(min_depth=0.1, max_depth=5.0),
      DepthContourNoiseCfg(threshold=0.15, kernel_size=3, value=0.0),
      DepthGaussianNoiseCfg(std=0.01),
      DepthPixelDropoutCfg(probability=0.005, value=0.0),
    ),
  )
  sensors = tuple(
    sensor
    for sensor in (cfg.scene.sensors or ())
    if sensor.name != camera_cfg.name
  )
  cfg.scene.sensors = sensors + (camera_cfg,)
  return camera_cfg


def g1_amp_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 rough terrain velocity configuration."""
  cfg = make_amp_env_cfg()

  # Keep CCD high enough for stability but avoid Warp OOM from excessive EPA buffers.
  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 48

  cfg.scene.entities = {"robot": get_g1_robot_cfg()}

  # Set raycast sensor frame to G1 pelvis.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = "pelvis"

  site_names = ("left_foot", "right_foot")
  geom_names = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
  )
  body_names = G1_AMP_BODY_NAMES
  anchor_name = "torso_link"
  root_name = "pelvis"

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )

  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_ACTION_SCALE

  cfg.viewer.body_name = "torso_link"

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.15

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  # Motion reset from LAFAN-derived AMP clips (see scripts/prepare_amp_motions.py).
  cfg.events["init_motion_loader"].params["delay_reset_env_ratio"] = 0.0
  cfg.events["init_motion_loader"].params["max_delay_steps"] = 0

  _motion_base = os.path.abspath(
    os.path.join(
      os.path.dirname(__file__),
      "..",
      "..",
      "..",
      "..",
      "assets",
      "motions",
      "g1",
      "amp",
    )
  )
  _motion_dir = os.path.join(_motion_base, "WalkandRun")
  _recovery_dir = os.path.join(_motion_base, "Recovery")

  cfg.events["init_motion_loader"].params["motion_dir"] = _motion_dir
  cfg.events["init_motion_loader"].params["recovery_dir"] = _recovery_dir
  cfg.events["reset_from_motion"].params["motion_dir"] = _motion_dir

  cfg.rewards["track_anchor_linear_velocity"].params["anchor_cfg"].body_names = (anchor_name,)
  cfg.rewards["track_anchor_angular_velocity"].params["anchor_cfg"].body_names = (anchor_name,)
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-0.1,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )
  cfg.rewards["body_ang_vel_xy_l2"].params["body_cfg"].body_names = (root_name,)
  cfg.rewards["body_orientation_l2"].params["asset_cfg"].body_names = (anchor_name,)
  cfg.rewards["pose"].params["std_standing"] = {
    r".*hip.*": 0.05,
    r".*knee.*": 0.05,
    r".*ankle.*": 0.05,
    r"waist.*": 0.05,
    r".*shoulder.*": 0.08,
    r".*elbow.*": 0.08,
    r".*wrist.*": 0.10,
  }
  cfg.rewards["pose"].params["std_walking"] = {
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.15,
    r".*hip_yaw.*": 0.15,
    r".*knee.*": 0.5,
    r".*ankle_pitch.*": 0.15,
    r".*ankle_roll.*": 0.1,
    r".*waist_yaw.*": 0.15,
    r".*waist_roll.*": 0.1,
    r".*waist_pitch.*": 0.1,
    # Looser arm std: let AMP style drive swing instead of pulling to default.
    r".*shoulder_pitch.*": 0.35,
    r".*shoulder_roll.*": 0.35,
    r".*shoulder_yaw.*": 0.30,
    r".*elbow.*": 0.35,
    r".*wrist.*": 0.30,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.25,
    r".*hip_yaw.*": 0.25,
    r".*knee.*": 0.5,
    r".*ankle_pitch.*": 0.25,
    r".*ankle_roll.*": 0.1,
    r".*waist_yaw.*": 0.25,
    r".*waist_roll.*": 0.1,
    r".*waist_pitch.*": 0.1,
    r".*shoulder_pitch.*": 0.45,
    r".*shoulder_roll.*": 0.40,
    r".*shoulder_yaw.*": 0.35,
    r".*elbow.*": 0.40,
    r".*wrist.*": 0.35,
  }
  cfg.rewards["stand_still"].params["asset_cfg"] = SceneEntityCfg(
    "robot",
    joint_names=(r".*hip.*", r".*knee.*", r".*ankle.*", r"waist.*"),
  )
  cfg.rewards["action_rate_l2"] = RewardTermCfg(
    func=mdp.action_rate_l2,
    weight=-0.005,
  )

  cfg.observations["critic"].terms["body_pos_b"].params["anchor_cfg"].body_names = (anchor_name,)
  cfg.observations["critic"].terms["body_pos_b"].params["body_cfg"].body_names = body_names
 
  cfg.observations["critic"].terms["body_ori_b"].params["anchor_cfg"].body_names = (anchor_name,)
  cfg.observations["critic"].terms["body_ori_b"].params["body_cfg"].body_names = body_names

  cfg.observations["amp"].terms["body_pos_b"].params["anchor_cfg"].body_names = (anchor_name,)
  cfg.observations["amp"].terms["body_pos_b"].params["body_cfg"].body_names = body_names

  cfg.observations["amp"].terms["body_ori_b"].params["anchor_cfg"].body_names = (anchor_name,)
  cfg.observations["amp"].terms["body_ori_b"].params["body_cfg"].body_names = body_names

  cfg.observations["amp"].terms["body_lin_vel_b"].params["anchor_cfg"].body_names = (anchor_name,)
  cfg.observations["amp"].terms["body_lin_vel_b"].params["body_cfg"].body_names = body_names

  cfg.observations["amp"].terms["body_ang_vel_b"].params["anchor_cfg"].body_names = (anchor_name,)
  cfg.observations["amp"].terms["body_ang_vel_b"].params["body_cfg"].body_names = body_names

  

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    # Match training reset distribution: policy is trained from motion frames, not HOME.
    # Disable domain randomization for deterministic deployment-style evaluation.
    cfg.events.pop("foot_friction", None)
    cfg.events.pop("encoder_bias", None)
    cfg.events.pop("base_com", None)
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def g1_amp_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain velocity configuration."""
  cfg = g1_amp_rough_env_cfg(play=play)

  cfg.sim.njmax = 640
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 256
  cfg.sim.nconmax = None

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove raycast sensor and height scan (no terrain to scan).
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  # Disable terrain curriculum (not present in play mode since rough clears all).
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    # Match post-curriculum training command ranges (iter > 5000).
    twist_cmd.ranges.lin_vel_x = (-1.5, 2.0)
    twist_cmd.ranges.lin_vel_y = (-0.0, 0.0)
    twist_cmd.ranges.ang_vel_z = (-1.0, 1.0)

  return cfg
