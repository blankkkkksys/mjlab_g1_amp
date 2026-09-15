# G1 AMP 任务说明

## 当前恢复训练配置

G1 AMP 默认为 70% 的并行环境启用恢复初始化和延迟终止，恢复窗口为
350 个控制步（50 Hz 下约 7 秒的连续终止条件）。恢复到正常状态会清零计数，
episode 超时仍会终止；显式 reset 也会清零计数。环境数较小时，恢复环境数
按 `int(num_envs * 0.7)` 取整，例如单环境 play 默认没有恢复环境；专门测试
起身时可将 `init_motion_loader` 的 `delay_reset_env_ratio` 设置为 1.0。

判别器从 `src/assets/motions/g1/amp/` 递归加载 WalkandRun 和 Recovery。
专家采样按 clip 均匀进行，专家恢复样本比例与 70% 的环境比例相互独立。
当前已有 17 条 WalkandRun 和 1 条 Recovery，因此恢复 clip 的采样概率约为 1/18。

恢复窗口内暂停默认关节姿态、静止姿态和躯干倾斜约束，原有速度跟踪项也
按延迟掩码暂停；`track_root_height` 使用连续高度奖励（weight=2，std=0.35），
站起后仍发放，避免越过恢复阈值时奖励骤降。高度相对环境原点计算。
Recovery reset 的 80% 采样优先来自根部高度低于 0.6 m 的帧，其余从全部帧采样；
若不存在低位帧则回退至全部帧。对称数据增强和 mirror loss 保持开启，权重 0.1。
判别器使用独立 Adam、5e-5 学习率，每轮 2 次更新；风格系数 0.1，task lerp 为 0.6。

## 转向与跑步采样

非静止指令中 20% 专门采样原地转向，30% 专门采样前进跑步，剩余使用普通速度采样。
专门分组关闭 heading 覆盖；跑步横向速度为零，偏航速度为 ±0.3 rad/s。
前向速度上限依次为 1.8、2.4、3.0 m/s（环境步数 0、1500×24、3000×24）；
跑步组下限为 1.6 m/s。课程按环境步数推进，恢复 checkpoint 不代表课程计数自动恢复。
偏航奖励只跟踪 yaw，std=0.75、weight=2；roll/pitch 稳定性使用独立奖励。
这些范围允许训练跑步，但实际步态与达速能力需训练验证。

以下历史预处理说明中的 `walk_run_g1` 路径不代表当前配置；当前预处理脚本
默认 `--mode stamp`，从 CSV 转换需使用 `--mode csv`。

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
