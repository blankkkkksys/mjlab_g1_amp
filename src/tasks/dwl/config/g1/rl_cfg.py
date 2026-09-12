"""DWL training configuration for the Unitree G1 flat velocity task."""

from src.rsl_rl import (
  RslRlDwlAlgorithmCfg,
  RslRlDwlModelCfg,
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
)


def g1_dwl_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create the recurrent DWL-PPO runner configuration."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlDwlModelCfg(
      hidden_dims=(48,),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
      num_embedding=24,
      enc_hidden_dims=(256,),
      dec_hidden_dims=(64,),
      rnn_type="lstm",
      rnn_hidden_dim=256,
      rnn_num_layers=1,
      decoder_obs_set="critic",
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlDwlAlgorithmCfg(
      reconstruction_loss_coef=1.0,
      latent_l1_coef=2.0e-3,
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
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
          "src.tasks.dwl.config.g1.symmetry:mirror_g1_dwl"
        ),
        "mirror_loss_coeff": 1.0,
      },
    ),
    experiment_name="g1_dwl_locomotion",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=20000,
  )
