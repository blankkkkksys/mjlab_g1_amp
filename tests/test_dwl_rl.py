import torch
from tensordict import TensorDict

from src.rsl_rl.models.dwl_model import DWLModel


def _make_dwl_model(batch_size: int = 4) -> tuple[DWLModel, TensorDict]:
    obs = TensorDict(
        {
            "actor": torch.randn(batch_size, 10),
            "critic": torch.randn(batch_size, 18),
        },
        batch_size=[batch_size],
    )
    model = DWLModel(
        obs=obs,
        obs_groups={"actor": ["actor"], "critic": ["critic"]},
        obs_set="actor",
        output_dim=6,
        hidden_dims=(12,),
        activation="elu",
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
        num_embedding=5,
        enc_hidden_dims=(9,),
        dec_hidden_dims=(7,),
        rnn_hidden_dim=8,
    )
    return model, obs


def test_dwl_model_output_losses_and_gradients() -> None:
    model, obs = _make_dwl_model()

    actions = model(obs, stochastic_output=True)
    assert actions.shape == (4, 6)
    assert model.decoder_obs_dim == 18

    model.reset()
    reconstruction_loss, latent_l1_loss = model.compute_dwl_losses(obs)
    total_loss = reconstruction_loss + 2.0e-3 * latent_l1_loss
    assert torch.isfinite(total_loss)
    total_loss.backward()

    assert any(param.grad is not None for param in model.embedding.parameters())
    assert any(param.grad is not None for param in model.decoder.parameters())


def test_dwl_model_resets_done_lstm_state() -> None:
    model, obs = _make_dwl_model()
    model(obs)
    hidden_state = model.get_hidden_state()
    assert isinstance(hidden_state, tuple)

    model.reset(torch.tensor([False, True, False, False]))
    reset_hidden_state = model.get_hidden_state()
    assert isinstance(reset_hidden_state, tuple)
    for state in reset_hidden_state:
        torch.testing.assert_close(state[:, 1], torch.zeros_like(state[:, 1]))


def test_g1_dwl_task_configuration() -> None:
    import src.tasks  # noqa: F401
    from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg

    task_id = "Unitree-G1-DWL-Flat"
    assert task_id in list_tasks()

    env_cfg = load_env_cfg(task_id)
    runner_cfg = load_rl_cfg(task_id)
    assert env_cfg.scene.terrain is not None
    assert env_cfg.scene.terrain.terrain_type == "plane"
    assert "height_scan" not in env_cfg.observations["actor"].terms
    assert "base_lin_vel" not in env_cfg.observations["actor"].terms
    assert "base_lin_vel" in env_cfg.observations["critic"].terms

    assert runner_cfg.experiment_name == "g1_dwl_locomotion"
    assert runner_cfg.actor.class_name == "DWLModel"
    assert runner_cfg.actor.num_embedding == 24
    assert runner_cfg.actor.rnn_hidden_dim == 256
    assert runner_cfg.algorithm.class_name == "DWLPPO"
    assert runner_cfg.algorithm.reconstruction_loss_coef == 1.0
    assert runner_cfg.algorithm.latent_l1_coef == 2.0e-3
    assert runner_cfg.algorithm.symmetry_cfg is not None
    assert runner_cfg.algorithm.symmetry_cfg["use_mirror_loss"] is True


def test_g1_dwl_symmetry_is_involutive() -> None:
    from src.tasks.dwl.config.g1.symmetry import mirror_g1_dwl

    batch_size = 3
    actor = torch.randn(batch_size, 98)
    critic = torch.randn(batch_size, 113)
    actions = torch.randn(batch_size, 29)
    obs = TensorDict(
        {"actor": actor, "critic": critic},
        batch_size=[batch_size],
    )

    augmented_obs, augmented_actions = mirror_g1_dwl(obs, actions, env=None)
    assert augmented_obs is not None
    assert augmented_actions is not None
    mirrored_obs = augmented_obs[batch_size:]
    mirrored_actions = augmented_actions[batch_size:]
    twice_obs, twice_actions = mirror_g1_dwl(
        mirrored_obs, mirrored_actions, env=None
    )
    assert twice_obs is not None
    assert twice_actions is not None
    torch.testing.assert_close(twice_obs["actor"][batch_size:], actor)
    torch.testing.assert_close(twice_obs["critic"][batch_size:], critic)
    torch.testing.assert_close(twice_actions[batch_size:], actions)


def test_rollout_storage_keeps_symmetric_trajectory_state() -> None:
    from src.rsl_rl.storage import RolloutStorage

    num_steps = 2
    num_envs = 2
    obs = TensorDict(
        {
            "actor": torch.zeros(num_envs, 3),
            "critic": torch.zeros(num_envs, 5),
        },
        batch_size=[num_envs],
    )
    storage = RolloutStorage(
        "rl", num_envs, num_steps, obs, actions_shape=[2], device="cpu"
    )

    for step in range(num_steps):
        transition = RolloutStorage.Transition()
        transition.observations = obs.apply(
            lambda value: torch.full_like(value, float(step))
        )
        transition.symmetric_observations = obs.apply(
            lambda value: torch.full_like(value, float(step + 10))
        )
        transition.actions = torch.zeros(num_envs, 2)
        transition.rewards = torch.zeros(num_envs)
        transition.dones = torch.zeros(num_envs, dtype=torch.bool)
        transition.values = torch.zeros(num_envs, 1)
        transition.actions_log_prob = torch.zeros(num_envs)
        transition.distribution_params = (
            torch.zeros(num_envs, 2),
            torch.ones(num_envs, 2),
        )
        transition.hidden_states = (
            (
                torch.full((1, num_envs, 4), float(step)),
                torch.full((1, num_envs, 4), float(step)),
            ),
            None,
        )
        transition.symmetric_hidden_state = (
            torch.full((1, num_envs, 4), float(step + 20)),
            torch.full((1, num_envs, 4), float(step + 30)),
        )
        storage.add_transition(transition)

    batch = next(storage.recurrent_mini_batch_generator(1, 1))
    assert batch.symmetric_observations is not None
    assert isinstance(batch.symmetric_hidden_state, list)
    torch.testing.assert_close(
        batch.symmetric_observations["actor"][0],
        torch.full((num_envs, 3), 10.0),
    )
    torch.testing.assert_close(
        batch.symmetric_hidden_state[0],
        torch.full((1, num_envs, 4), 20.0),
    )


def test_g1_amp_dwl_task_combines_both_algorithms() -> None:
    import src.tasks  # noqa: F401
    from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg

    from src.rsl_rl.algorithms.amp_dwl_ppo import AMPDWLPPO
    from src.rsl_rl.algorithms.amp_ppo import AMPPPO
    from src.rsl_rl.algorithms.dwl_ppo import DWLPPO

    task_id = "Unitree-G1-AMP-DWL-Flat"
    assert task_id in list_tasks()
    assert issubclass(AMPDWLPPO, DWLPPO)
    assert issubclass(AMPDWLPPO, AMPPPO)

    env_cfg = load_env_cfg(task_id)
    runner_cfg = load_rl_cfg(task_id)
    assert env_cfg.observations["actor"].history_length == 1
    assert env_cfg.observations["critic"].history_length == 1
    assert "amp" in env_cfg.observations
    assert runner_cfg.actor.class_name == "DWLModel"
    assert runner_cfg.algorithm.class_name.endswith(":AMPDWLPPO")
    assert runner_cfg.algorithm.symmetry_cfg is not None
    assert runner_cfg.algorithm.symmetry_cfg["mirror_loss_coeff"] == 2.0
    assert runner_cfg.algorithm.entropy_coef == 0.001
    assert runner_cfg.actor.distribution_cfg["init_std"] == 0.8
    assert runner_cfg.amp_task_reward_lerp == 0.75
    assert runner_cfg.amp_reward_coef == 0.05
    assert runner_cfg.amp_discriminator_updates_per_iteration == 1
    assert runner_cfg.max_normalized_std == [1.0] * 29


def test_amp_action_std_is_clamped_to_configured_range() -> None:
    from types import SimpleNamespace

    from src.rsl_rl.algorithms.amp_ppo import AMPPPO
    from src.rsl_rl.modules.distribution import GaussianDistribution

    distribution = GaussianDistribution(3, init_std=1.5)
    alg = object.__new__(AMPPPO)
    alg.actor = SimpleNamespace(distribution=distribution)
    alg.min_normalized_std = torch.tensor([0.05, 0.05, 0.05])
    alg.max_normalized_std = torch.tensor([1.0, 1.0, 1.0])

    AMPPPO._clamp_action_std(alg)

    torch.testing.assert_close(distribution.std_param, torch.ones(3))
