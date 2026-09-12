"""Joint AMP-DWL training configuration for Unitree G1."""

from src.rsl_rl import (
  RslRlDwlAlgorithmCfg,
  RslRlDwlModelCfg,
  RslRlModelCfg,
)
from src.tasks.amp_loco.config.g1.rl_cfg import (
  RslRlAmpRunnerCfg,
  g1_amp_ppo_runner_cfg,
)


def g1_amp_dwl_ppo_runner_cfg() -> RslRlAmpRunnerCfg:
    """Create the G1 runner combining AMP rewards with a recurrent DWL actor."""
    cfg = g1_amp_ppo_runner_cfg()
    cfg.actor = RslRlDwlModelCfg(
        hidden_dims=(48,),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 0.8,
            "std_type": "scalar",
        },
        num_embedding=24,
        enc_hidden_dims=(256,),
        dec_hidden_dims=(64,),
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
        decoder_obs_set="critic",
    )
    cfg.critic = RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    )
    cfg.algorithm = RslRlDwlAlgorithmCfg(
        reconstruction_loss_coef=1.0,
        latent_l1_coef=2.0e-3,
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg={
            "use_data_augmentation": False,
            "use_mirror_loss": True,
            "data_augmentation_func": (
                "src.tasks.amp_loco.config.g1.symmetry:mirror_g1_amp"
            ),
            "mirror_loss_coeff": 2.0,
        },
        class_name="src.rsl_rl.algorithms.amp_dwl_ppo:AMPDWLPPO",
    )
    cfg.experiment_name = "g1_amp_dwl_locomotion"
    cfg.max_iterations = 20000
    cfg.amp_discriminator_updates_per_iteration = 1
    cfg.max_normalized_std = [1.0] * 29
    return cfg
