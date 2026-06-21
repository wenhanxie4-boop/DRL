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

- `option_interval=10`
- `high_top_k=3`
- `high_prior_beta=4.0`; set it to `0.0` to recover the pure learnable high-level policy
- `high_residual_scale=0.25`; the high-level actor can only make a small bounded residual correction to the heuristic rank prior
- candidate ranking uses the same AoI/data/overflow/distance weights as the heuristic HRL baseline
- options use the stable fixed-interval behavior from the best learnable baseline
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
python PPO_Learnable_HRL_AoI_main.py --high_prior_beta 0.0
python PPO_Learnable_HRL_AoI_main.py --high_residual_scale 1.0
```

Additional high-level diagnostics:

- `High_Level/Entropy`
- `High_Level/Repeat_Target_Rate`
- `High_Level/Action_Rank_{k}_Rate`
- `High_Level/Action_Rank_{k}_Prob`
- `High_Level/Option_Data_By_Rank_{k}`
- `High_Level/Option_AoI_By_Rank_{k}`

Results follow the original project style:

- TensorBoard logs: `runs/time/`
- best environment: `results/PPO/best_env_time.pkl`
- low-level actor: `results/PPO/best_model_time.pth`
- high-level actor: `results/PPO/best_high_model_time.pth`

Trajectory plot:

```bash
python plot_ppo_trajectory.py --path ./results/PPO/best_env_time.pkl
```
