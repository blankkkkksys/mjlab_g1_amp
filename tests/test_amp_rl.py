import numpy as np
import torch

from src.rsl_rl.modules.amp_discriminator import Discriminator
from src.rsl_rl.modules.amp_normalizer import AmpNormalizer
from src.rsl_rl.storage.amp_replay_buffer import AmpReplayBuffer
from src.rsl_rl.utils.amp_motion_loader import normalize_quat_wxyz
from src.rsl_rl.env.vecenv_wrapper import _flatten_observation_groups
from src.tasks.amp_loco.g1_motion_constants import G1_JOINT_NAMES
from scripts.prepare_amp_motions import stamp_npz_metadata


def test_quaternion_normalization_is_wxyz_and_canonical() -> None:
  quaternions = np.array([[-2.0, 0.0, 0.0, 0.0], [1.0, 2.0, 3.0, 4.0]])
  normalized = normalize_quat_wxyz(quaternions)
  np.testing.assert_allclose(np.linalg.norm(normalized, axis=-1), 1.0)
  assert np.all(normalized[:, 0] >= 0.0)
  np.testing.assert_allclose(normalized[0], [1.0, 0.0, 0.0, 0.0])


def test_g1_joint_names_match_amp_loader_fallback() -> None:
  assert len(G1_JOINT_NAMES) == 29


def test_stamp_npz_metadata_adds_joint_and_body_names(tmp_path) -> None:
  import numpy as np
  from mjlab.entity import Entity
  from mjlab.scene import Scene
  from mjlab.sim.sim import Simulation, SimulationCfg
  from src.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg

  path = tmp_path / "clip.npz"
  np.savez(
    path,
    fps=np.array([50]),
    joint_pos=np.zeros((10, 29), dtype=np.float32),
    joint_vel=np.zeros((10, 29), dtype=np.float32),
    body_pos_w=np.zeros((10, 30, 3), dtype=np.float32),
    body_quat_w=np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (10, 30, 1)),
    body_lin_vel_w=np.zeros((10, 30, 3), dtype=np.float32),
    body_ang_vel_w=np.zeros((10, 30, 3), dtype=np.float32),
  )
  env_cfg = unitree_g1_flat_tracking_env_cfg()
  env_cfg.scene.num_envs = 1
  scene = Scene(env_cfg.scene, device="cpu")
  sim = Simulation(num_envs=1, cfg=SimulationCfg(), model=scene.compile(), device="cpu")
  scene.initialize(sim.mj_model, sim.model, sim.data)
  robot: Entity = scene["robot"]
  assert list(robot.joint_names) == list(G1_JOINT_NAMES)
  stamp_npz_metadata(path, robot)
  data = np.load(path, allow_pickle=True)
  assert [str(n) for n in data["joint_names"].tolist()] == list(robot.joint_names)
  assert [str(n) for n in data["body_names"].tolist()] == list(robot.body_names)


def test_time_major_observation_flattening() -> None:
  observation = {
    "actor": {
      "a": torch.tensor([[[1.0], [2.0]]]),
      "b": torch.tensor([[[10.0, 11.0], [20.0, 21.0]]]),
    },
    "amp": torch.tensor([[3.0, 4.0]]),
  }
  flattened = _flatten_observation_groups(observation)
  torch.testing.assert_close(
    flattened["actor"], torch.tensor([[1.0, 10.0, 11.0, 2.0, 20.0, 21.0]])
  )
  torch.testing.assert_close(flattened["amp"], observation["amp"])


def test_g1_amp_symmetry_is_involutive_and_augments_batch() -> None:
  from tensordict import TensorDict

  from src.tasks.amp_loco.config.g1.symmetry import mirror_g1_amp

  batch_size = 3
  actor = torch.randn(batch_size, 4 * 96)
  critic = torch.randn(batch_size, 4 * 288)
  amp = torch.randn(batch_size, 315)
  actions = torch.randn(batch_size, 29)
  obs = TensorDict(
    {"actor": actor, "critic": critic, "amp": amp},
    batch_size=[batch_size],
  )

  augmented_obs, augmented_actions = mirror_g1_amp(obs, actions, env=None)
  assert augmented_obs is not None
  assert augmented_actions is not None
  assert augmented_obs.batch_size == torch.Size([2 * batch_size])
  assert augmented_actions.shape == (2 * batch_size, 29)
  torch.testing.assert_close(augmented_obs["actor"][:batch_size], actor)
  torch.testing.assert_close(augmented_actions[:batch_size], actions)

  mirrored_obs = augmented_obs[batch_size:]
  mirrored_actions = augmented_actions[batch_size:]
  twice_obs, twice_actions = mirror_g1_amp(
    mirrored_obs, mirrored_actions, env=None
  )
  assert twice_obs is not None
  assert twice_actions is not None
  torch.testing.assert_close(twice_obs["actor"][batch_size:], actor)
  torch.testing.assert_close(twice_obs["critic"][batch_size:], critic)
  torch.testing.assert_close(twice_obs["amp"][batch_size:], amp)
  torch.testing.assert_close(twice_actions[batch_size:], actions)


def test_g1_amp_tracking_and_symmetry_configuration() -> None:
  from src.tasks.amp_loco.amp_env_cfg import make_amp_env_cfg
  from src.tasks.amp_loco.config.g1.rl_cfg import g1_amp_ppo_runner_cfg

  env_cfg = make_amp_env_cfg()
  linear_reward = env_cfg.rewards["track_anchor_linear_velocity"]
  assert linear_reward.weight == 2.0
  assert linear_reward.params["std"] == 0.5

  runner_cfg = g1_amp_ppo_runner_cfg()
  assert runner_cfg.amp_reward_coef == 0.05
  assert runner_cfg.amp_task_reward_lerp == 0.75
  assert runner_cfg.algorithm.symmetry_cfg is not None
  assert runner_cfg.algorithm.symmetry_cfg["use_data_augmentation"] is True
  assert runner_cfg.algorithm.symmetry_cfg["use_mirror_loss"] is True


def test_symmetry_resolution_does_not_pollute_serializable_config() -> None:
  import yaml

  from src.rsl_rl.extensions.symmetry import resolve_symmetry_config

  class RuntimeEnv:
    def __reduce_ex__(self, _protocol):
      raise TypeError("runtime environment must not be serialized")

  original = {
    "symmetry_cfg": {
      "use_data_augmentation": True,
      "use_mirror_loss": True,
      "mirror_loss_coeff": 0.1,
      "data_augmentation_func": "module:function",
    }
  }
  runtime_env = RuntimeEnv()
  resolved = resolve_symmetry_config(original, runtime_env)

  assert "_env" not in original["symmetry_cfg"]
  assert resolved["symmetry_cfg"]["_env"] is runtime_env
  yaml.dump(original)


def test_replay_buffer_wraps_and_samples() -> None:
  buffer = AmpReplayBuffer(observation_dim=2, capacity=3, device="cpu")
  states = torch.arange(10, dtype=torch.float32).reshape(5, 2)
  buffer.insert(states, states + 1)
  assert buffer.size == 3
  assert buffer.states.shape == (3, 2)
  sampled, sampled_next = buffer.sample(8)
  assert sampled.shape == sampled_next.shape == (8, 2)


def test_normalizer_updates_and_stays_finite() -> None:
  normalizer = AmpNormalizer(2)
  samples = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
  normalizer.update(samples)
  normalized = normalizer.normalize(samples)
  assert torch.isfinite(normalized).all()
  torch.testing.assert_close(normalized.mean(dim=0), torch.zeros(2), atol=2e-4, rtol=0)


def test_discriminator_reward_and_gradient_penalty() -> None:
  normalizer = AmpNormalizer(3)
  discriminator = Discriminator(3, 0.1, [8, 4], 0.75)
  state = torch.randn(5, 3)
  next_state = torch.randn(5, 3)
  style_reward, logits = discriminator.predict_style_reward(
    state, next_state, normalizer
  )
  reward = discriminator.predict_reward(
    state, next_state, torch.ones(5), normalizer
  )
  penalty = discriminator.compute_grad_penalty(state, next_state)
  assert style_reward.shape == (5,)
  assert logits.shape == (5,)
  assert reward.shape == (5,)
  assert torch.isfinite(style_reward).all()
  assert torch.isfinite(reward).all()
  assert penalty.ndim == 0 and torch.isfinite(penalty)


def test_amp_ppo_rollout_metrics() -> None:
  from src.rsl_rl.algorithms.amp_ppo import AMPPPO

  alg = object.__new__(AMPPPO)
  alg._rollout_task_reward_sum = 10.0
  alg._rollout_style_reward_sum = 4.0
  alg._rollout_mixed_reward_sum = 8.0
  alg._rollout_logit_sum = 2.0
  alg._rollout_step_count = 5
  metrics = AMPPPO._build_rollout_metrics(alg)
  assert metrics["AMP/task_reward"] == 2.0
  assert metrics["AMP/style_reward"] == 0.8
  assert metrics["AMP/mixed_reward"] == 1.6
  assert metrics["AMP/rollout_logit"] == 0.4
  AMPPPO._reset_rollout_metrics(alg)
  assert alg._rollout_step_count == 0


def test_g1_amp_play_cfg_matches_training_reset() -> None:
  from src.tasks.amp_loco.config.g1.env_cfgs import (
    G1_AMP_BODY_NAMES,
    g1_amp_flat_env_cfg,
  )

  cfg = g1_amp_flat_env_cfg(play=True)
  assert "reset_from_motion" in cfg.events
  assert "init_motion_loader" in cfg.events
  assert cfg.events["init_motion_loader"].params["recovery_dir"].endswith("Recovery")
  assert cfg.events["init_motion_loader"].params["motion_dir"].endswith("WalkandRun")
  assert "foot_friction" not in cfg.events
  assert len(G1_AMP_BODY_NAMES) == 21
  assert len(G1_AMP_BODY_NAMES) * 15 == 315


def test_g1_amp_deploy_yaml_matches_training_obs() -> None:
  from pathlib import Path

  import yaml

  from src.tasks.amp_loco.config.g1.env_cfgs import g1_amp_flat_env_cfg

  deploy_path = (
    Path(__file__).resolve().parents[1]
    / "deploy/robots/g1/config/policy/amp/v0/params/deploy.yaml"
  )
  deploy_cfg = yaml.safe_load(deploy_path.read_text(encoding="utf-8"))
  train_cfg = g1_amp_flat_env_cfg()

  obs_cfg = deploy_cfg["observations"]
  assert obs_cfg["use_gym_history"] is True
  assert obs_cfg["base_ang_vel"]["history_length"] == train_cfg.observations["actor"].history_length
  assert "gait_phase" not in obs_cfg
  assert len(deploy_cfg["actions"]["JointPositionAction"]["scale"]) == 29
  assert deploy_cfg["commands"]["base_velocity"]["ranges"]["lin_vel_x"] == [-1.5, 2.0]
