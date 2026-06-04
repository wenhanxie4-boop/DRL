# AoI-aware Hierarchical PPO

这是不加 Transformer 的第一版 AoI-aware hierarchical PPO。
低层仍然使用原来的普通连续 PPO，高层先使用可解释的 AoI 启发式调度器。

## 文件

- `PPO_HRL_AoI_main.py`: 训练入口，保存路径和 TensorBoard 风格与原来的 `PPO_continuous_main.py` 保持一致。
- `env/env_uav_hrl_aoi.py`: AoI-aware hierarchical UAV 环境。
- `ppo_continuous.py`, `replaybuffer.py`, `normalization.py`: 从原项目复制的普通 PPO 代码。
- `plot_ppo_trajectory.py`: 从原项目复制的轨迹绘图脚本。
- `env/common_functions.py`: 从原项目复制的能耗和通信速率函数。
- `data_train/`: 用户数据文件。

## 主要思路

高层调度器每隔 `option_interval` 步选择一个目标用户，打分考虑：

- 剩余数据量；
- AoI；
- 爆仓风险；
- 与 UAV 的距离。

低层 PPO 不再直接观察全部 50 个用户，而是观察 12 维紧凑状态：

- UAV 归一化坐标：2 维；
- 当前高层目标用户状态：6 维；
- 禁飞区相对信息：3 维；
- 高层目标阶段信息：1 维。

## 运行

```bash
python PPO_HRL_AoI_main.py
```

可尝试不同高层目标保持步数：

```bash
python PPO_HRL_AoI_main.py --option_interval 5
python PPO_HRL_AoI_main.py --option_interval 15
python PPO_HRL_AoI_main.py --seed 20
```

训练结果保存到 `results/PPO/`：

- `best_env_时间.pkl`
- `best_model_时间.pth`

TensorBoard 日志保存到 `runs/时间/`。

画轨迹时把训练生成的 pkl 路径填进去：

```bash
python plot_ppo_trajectory.py --path ./results/PPO/best_env_时间.pkl
```
