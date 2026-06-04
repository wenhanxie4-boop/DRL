# Learnable AoI-HRL PPO

This version trains two policies:

- high-level discrete PPO: chooses one target user from top-k candidates;
- low-level continuous PPO: controls UAV movement with `dx, dy`.

The TensorBoard metrics are kept aligned with `6.AoI-Hierarchical-PPO` so the
two versions can be compared directly.

## Run

```bash
python PPO_Learnable_HRL_AoI_main.py
```

Default high-level settings:

- `option_interval=15`
- `high_top_k=3`
- high-level option reward:

```text
0.01 * option_collected_data
- 0.0005 * option_dropped_data
- 1.0 * average_option_aoi_cost
- 0.2 * target_switch_penalty
```

Useful variants:

```bash
python PPO_Learnable_HRL_AoI_main.py --option_interval 10
python PPO_Learnable_HRL_AoI_main.py --high_top_k 5
python PPO_Learnable_HRL_AoI_main.py --high_top_k 8
```

Results follow the original project style:

- TensorBoard logs: `runs/time/`
- best environment: `results/PPO/best_env_time.pkl`
- low-level actor: `results/PPO/best_model_time.pth`
- high-level actor: `results/PPO/best_high_model_time.pth`

Trajectory plot:

```bash
python plot_ppo_trajectory.py --path ./results/PPO/best_env_time.pkl
```
