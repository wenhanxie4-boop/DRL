# PPO-continuous
This is a concise Pytorch implementation of PPO on continuous action space with 10 tricks.<br />

## 10 tricks
Trick 1—Advantage Normalization.<br />
Trick 2—State Normalization.<br />
Trick 3 & Trick 4—— Reward Normalization & Reward Scaling.<br />
Trick 5—Policy Entropy.<br />
Trick 6—Learning Rate Decay.<br />
Trick 7—Gradient clip.<br />
Trick 8—Orthogonal Initialization.<br />
Trick 9—Adam Optimizer Epsilon Parameter.<br />
Trick10—Tanh Activation Function.<br />

## How to use my code?
You can dircetly run 'PPO_continuous_main.py' in your own IDE.<br />

## Trainning environments
You can set the 'env_index' in the codes to change the environments. Here, we train our code in 4 environments.<br />
env_index=0 represent 'BipedalWalker-v3'<br />
env_index=1 represent 'HalfCheetah-v2'<br />
env_index=2 represent 'Hopper-v2'<br />
env_index=3 represent 'Walker2d-v2'<br />

## Trainning result
![image](https://github.com/Lizhi-sjtu/DRL-code-pytorch/blob/main/5.PPO-continuous/training_result.png)

## Tutorial
If you can read Chinese, you can get more information from this blog.https://zhuanlan.zhihu.com/p/512327050

## Single-level GAT/Pointer PPO

Run:

```bash
python PPO_GAT_main.py
```

This version has no high-level policy, target selector, or option mechanism.
The PPO actor and critic directly process all user nodes and output the UAV's
continuous movement.

- `ENV/env_uav_gat_hrl.py`: global-state AoI environment without heuristic target decisions.
- `ppo_gat_continuous.py`: graph-attention/pointer encoder embedded in continuous PPO.
- `PPO_GAT_main.py`: single-level training and evaluation entry.

Example:

```bash
python PPO_GAT_main.py --gat_hidden_dim 128 --gat_heads 4 --gat_layers 2
```

TensorBoard logs are written to `runs/*_GAT_PPO`. Best actor and critic
checkpoints are saved to `results/GAT_PPO`.

### Current experiment environment

The GAT and MLP baseline now use the same environment definition:

- real per-user queue buffers; uploaded data immediately frees capacity;
- heterogeneous arrival rates, burst probabilities, and buffer capacities;
- randomized user observation order at every episode reset;
- normalized data, drop, energy, AoI, and guide reward components;
- efficiency metrics including energy per step, data per energy, drop rate,
  and full-episode completion rate.

The observation contains 5 UAV/NFZ features and 9 features per user. Models
trained with the previous 305-dimensional observation are not compatible with
this version and both comparison methods should be retrained from scratch.

For the current main comparison:

- user features use a fixed user-id order;
- entering the NFZ restores the previous UAV position, applies the NFZ
  penalty, and does not terminate the episode;
- the MLP baseline uses hidden dimensions 256 and 128 instead of 64 and 64.
