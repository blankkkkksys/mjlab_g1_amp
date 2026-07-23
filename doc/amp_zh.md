# G1 AMP 任务说明

本仓库在本地统一 RL 层 `src/rsl_rl/` 中实现 AMP（Adversarial Motion
Priors）。Velocity、Tracking 和 AMP 共用标准 `rsl-rl-lib==5.0.1`，
无需切换依赖、安装 fork 或修改 mjlab 的 site-packages。

实现参考 [ccrpRepo/AMP_mjlab](https://github.com/ccrpRepo/AMP_mjlab)，
AMP 派生组件沿用 BSD-3-Clause 许可说明。

## 1. 任务

- `Unitree-G1-AMP-Flat`
- `Unitree-G1-AMP-Rough`

```bash
env -u PYTHONPATH python scripts/list_envs.py --keyword AMP
```

## 2. 统一 RL 架构

```text
src/rsl_rl/
├── config.py                   # 所有任务共用的 RSL-RL 配置
├── algorithms/
│   ├── ppo.py                  # baseline PPO
│   └── amp_ppo.py              # AMPPPO
├── runners/
│   ├── on_policy_runner.py     # baseline runner
│   └── amp_runner.py           # AMP runner（reward/判别器日志 + ONNX）
├── env/
│   ├── vec_env.py
│   └── vecenv_wrapper.py       # TensorDict + time-major 历史观测
├── modules/
│   ├── amp_discriminator.py
│   └── amp_normalizer.py
├── storage/
│   ├── rollout_storage.py
│   └── amp_replay_buffer.py
├── utils/
│   ├── exporter_utils.py       # ONNX metadata
│   └── amp_motion_loader.py    # 专家 (s, s_next)
├── models/                     # baseline MLP/CNN/RNN
├── extensions/                 # RND、symmetry 等
└── ...

src/tasks/amp_loco/              # AMP 环境、MDP、配置和任务 runner
```

`AMPPPO` 直接扩展标准 RSL-RL 5.x PPO/MLPModel/TensorDict API。AMP 判别器、
normalizer 和 optimizer 状态随标准 actor/critic 一同写入 checkpoint。

训练时会记录以下指标（TensorBoard/W&B 与控制台）：

| 指标 | 含义 |
|------|------|
| `AMP/task_reward` | rollout 原始任务 reward |
| `AMP/style_reward` | 判别器给出的 style reward |
| `AMP/mixed_reward` | 混合后送入 PPO 的 reward |
| `AMP/rollout_logit` | rollout 阶段判别器 logit 均值 |
| `AMP/policy_logit` / `AMP/expert_logit` | 判别器更新时的 logit |
| `AMP/disc_accuracy` | 专家>0 且策略<0 的分类准确率 |
| `Loss/amp_policy_loss` / `Loss/amp_expert_loss` | 判别器 MSE 分项 |
| `Loss/amp_discriminator` / `Loss/amp_grad_penalty` | 总损失与梯度惩罚 |
| `AMP/replay_buffer_size` | 当前 replay 占用量 |

Actor/Critic 的四帧历史由标准 observation manager 保留为各 term 的
`(time, feature)`，随后在本地 `RslRlVecEnvWrapper` 中按 time-major 顺序
拼接，因此不需要 `history_ordering` 补丁。

## 3. 动作数据与四元数

原始数据：`src/assets/motions/walk_run_g1/`

- 四元数约定为 **wxyz**
- 原始 body 数量为 37，与当前 G1 的 30 bodies 不一致
- 默认跳过 `*_M.npz`

首次使用或机器人模型变化后运行：

```bash
env -u PYTHONPATH python scripts/prepare_amp_motions.py
```

输出目录：

```text
src/assets/motions/g1/amp/walk_run_g1/
```

当前输出包含 24 条 clip。预处理及 AMP loader 均会归一化 wxyz 四元数；
预处理脚本还会将原始数据的左右交错关节顺序显式重排为当前 G1 MJCF
顺序，并写入 `joint_names`。loader 会同时检查 `body_names`、`joint_names`
顺序及数量，避免错误关节映射被用于重置或判别器专家数据。

## 4. 训练

与 Velocity 任务使用**完全相同**的启动方式（先清掉 Isaac Lab 的 PYTHONPATH 污染）：

```bash
conda activate unitree_rl_mjlab   # 若使用 conda
env -u PYTHONPATH python scripts/train.py Unitree-G1-AMP-Flat \
  --env.scene.num-envs=4096 \
  --agent.logger=tensorboard    # 或先 wandb login 后用 --agent.logger=wandb
```

等价写法（与 velocity 一致）：

```bash
env -u PYTHONPATH python scripts/train.py Unitree-G1-Flat \
  --env.scene.num-envs=4096 \
  --agent.logger=tensorboard
```

只需把任务名换成 `Unitree-G1-AMP-Flat` 或 `Unitree-G1-AMP-Rough`。

日志路径：

```text
logs/rsl_rl/g1_amp_locomotion/<timestamp>/
```

## 5. Play

```bash
env -u PYTHONPATH python scripts/play.py Unitree-G1-AMP-Flat \
  --checkpoint-file=logs/rsl_rl/g1_amp_locomotion/<timestamp>/model_<iter>.pt \
  --num-envs=1 \
  --viewer=auto
```

## 6. 验证

```bash
python -m pytest tests/test_amp_rl.py -q
```

常见问题：

- `No AMP motion .npz files found`：先运行 motion 预处理脚本。
- `body count ... != robot body count`：旧数据未按当前 G1 模型重导出。
- `stored body_names do not match`：动作文件 body 顺序与当前模型不同。
- `tyro` 导入异常：确认已 `env -u PYTHONPATH`，与 velocity 训练环境一致。
- GPU 报错：不要用 `env -u PYTHONPATH` 却调用 conda 自带的 CPU 版 `python`；请与 velocity 一样使用项目 `.venv` 里的 Python（`which python` 应指向 `.venv/bin/python`）。
