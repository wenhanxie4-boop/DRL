# AoI-HRL 可学习高层 PPO 迭代总结

记录日期：2026-06-16

项目目录：

`D:\workspace\DRL_code_pytorch\DRL-code-pytorch-main\PPO\7.AoI-Learnable-Hierarchical-PPO`

## 1. 当前研究主线

本项目尝试在 UAV AoI-aware 数据采集任务中，将原本启发式的高层目标选择改为可学习的高层策略。整体结构是层次强化学习：

- 低层 PPO：控制 UAV 连续运动。
- 高层策略：每隔若干步选择一个服务目标或候选目标 rank。
- 高层候选集合：根据用户 AoI、剩余数据量、溢出风险、距离等构造 top-k 候选用户。

经过多轮实验后，目前比较明确的结论是：

**纯高层 PPO 从零学习目标选择效果不稳定，整体不如启发式高层。当前最好的版本不是纯 PPO，而是 heuristic-prior residual PPO，即启发式先验引导下的小幅 PPO 残差修正。**

## 2. 关键实验版本与结论

### 2.1 启发式高层基线

结果目录：

`env_uav_hrl_aoi结果`

主要特点：

- 高层由启发式规则选择目标。
- 启发式评分综合剩余数据、AoI、溢出风险和距离。
- 轨迹覆盖范围较大，能够形成比较有效的全局服务路径。
- 采集数据量和 reward 长期是最强 baseline。
- 缺点是 NFZ violation 风险较高，安全性波动明显。

大致表现：

- Data collected 约 `1.6e6~1.7e6`。
- Reward 约 `1.5e4~1.7e4`。
- Active mean AoI 可降到 `150~170` 区间，但波动较大。
- NFZ violation 明显高于当前 1604 版本。

### 2.2 结果2：较好的可学习高层基线

结果目录：

`env_uav_learnable_hrl_aoi结果2`

这是早期最有希望的 learnable high-level 版本。

当时的关键设置：

- `top_k=3`
- `option_interval=15`
- 高层使用 option-level reward

高层 reward 近似为：

```text
0.01 * option_collected_data
- 0.0005 * option_dropped_data
- 1.0 * average_option_aoi_cost
- 0.2 * high_switch_penalty
```

大致表现：

- Data collected 约 `1.3e6~1.4e6`。
- Reward 约 `1.2e4~1.4e4`。
- Active mean AoI 多在 `180` 以上。
- 轨迹比失败版本更合理，但仍明显弱于启发式高层。

结论：

结果2可以作为“纯 learnable high-level PPO”较好的参考基线，但它没有稳定超过启发式高层。

### 2.3 边际 reward 尝试

曾尝试把高层 reward 改成 option 前后状态差，即：

```text
数据采集增益
+ AoI 下降增益
+ 溢出风险下降
- 能耗
- NFZ 风险
- 无效切换
```

预期是让高层关注“服务是否有效”，而不是简单“是否到达目标”。

实际结果较差：

- Data collected 降到约 `5.5e5`。
- Reward 约 `4.8k`。
- Active mean AoI 约 `198`。
- UAV 轨迹变得局部、保守。

主要原因判断：

- AoI 和风险在任务中天然随时间增长，简单使用 `end - start` 的边际差容易长期为负。
- 高层 reward 变得过于苛刻，导致策略倾向保守。
- 到达目标和有效服务之间的 credit assignment 仍然困难。

结论：

直接使用粗糙边际 reward 不适合作为当前主线。

### 2.4 target-service reward 与 option termination 尝试

尝试过：

- 强化低层 target-service 奖励。
- 加入 no-progress option termination。
- 缩短无效 option。

结果并不好，尤其是 `0610` 版本：

- 高层决策次数变多。
- no-progress termination 过多。
- option 被过度碎片化。
- Data collected 约 `6.8e5~7.0e5`。
- Reward 约 `3.5k`。
- Active mean AoI 约 `198`。

主要原因判断：

- termination 太激进，破坏了低层完成服务目标的连续性。
- 高层频繁重新选目标，反而让 UAV 路径失去稳定方向。
- 对当前任务而言，稳定 fixed-interval option 比复杂 termination 更可靠。

结论：

当前阶段不建议继续走激进 option termination 路线。

### 2.5 1604 当前最好版本：heuristic-prior residual PPO

结果目录：

`env_uav_learnable_hrl_aoi结果1604`

这是目前最好的 learnable high-level 版本。

关键修改：

- 将高层候选评分 `_candidate_score` 对齐启发式高层。
- 将 `option_interval` 从 15 改为 10。
- 高层 PPO 不再完全从零学习，而是在启发式 rank prior 上做小幅残差修正。
- 设置：

```text
high_prior_beta = 4.0
high_residual_scale = 0.25
high_top_k = 3
option_interval = 10
```

高层 logits 形式可理解为：

```text
logits = heuristic_rank_prior + bounded_PPO_residual
```

其中 PPO residual 被 `tanh` 和 `high_residual_scale` 限制，不能大幅偏离启发式排序。

大致表现：

- Data collected 约 `1.55e6~1.65e6`。
- Reward 约 `1.5e4~1.6e4`。
- Active mean AoI 约 `170~178`。
- 轨迹覆盖范围明显改善，接近启发式高层。
- NFZ violation 基本接近 0，仅后期有极小尖峰。

高层诊断：

- `Action_Rank_0_Prob` 约 `0.975`。
- `Action_Rank_0_Rate` 多在 `0.95~0.99`。
- Rank1/Rank2 仅少量参与。

结论：

1604 效果确实不错，但它本质上不是纯 PPO 高层，而是“启发式为主、PPO 小幅修正”的方法。它不能被表述为“纯 PPO 高层战胜启发式高层”，更适合表述为“启发式先验引导的残差高层策略”。

## 3. 当前代码主要改动文件

当前主线相关改动集中在：

- `PPO_Learnable_HRL_AoI_main.py`
- `ppo_discrete_high.py`
- `env/env_uav_learnable_hrl_aoi.py`
- `README.md`

主要改动点：

### 3.1 `env/env_uav_learnable_hrl_aoi.py`

将候选用户评分对齐启发式高层：

```text
score =
    1.0 * norm_data
  + 1.2 * norm_aoi
  + 1.5 * overflow_risk
  - 0.8 * norm_dist
```

如果用户是当前 target user，则额外加 `0.15`，鼓励目标保持，避免无效频繁切换。

### 3.2 `ppo_discrete_high.py`

高层 actor 改成 bounded residual policy：

```text
residual_logits = high_residual_scale * tanh(actor_output)
logits = residual_logits + high_prior_beta * rank_prior
```

其中 rank prior 对 rank0、rank1、rank2 施加递减偏置，使策略默认更偏向启发式排序靠前的候选。

### 3.3 `PPO_Learnable_HRL_AoI_main.py`

增加高层诊断指标：

- `High_Level/Entropy`
- `High_Level/Repeat_Target_Rate`
- `High_Level/Action_Rank_{k}_Rate`
- `High_Level/Action_Rank_{k}_Prob`
- `High_Level/Option_Data_By_Rank_{k}`
- `High_Level/Option_AoI_By_Rank_{k}`

默认参数改为：

```text
option_interval = 10
high_top_k = 3
high_prior_beta = 4.0
high_residual_scale = 0.25
```

## 4. 对“PPO 高层是否不如启发式”的判断

目前可以比较明确地说：

**在当前 AoI-UAV 场景、当前状态设计、当前奖励设计和训练预算下，纯高层 PPO 的效果没有启发式高层好。**

原因包括：

- 高层决策稀疏，一个 episode 中高层 transition 数量少，PPO 样本效率不足。
- 高层动作虽然只是 top-k rank，但不同 rank 背后对应的用户集合会随状态变化，动作语义非平稳。
- 目标选择具有很强的任务结构，启发式能直接利用 AoI、数据量、溢出风险、距离等先验。
- 纯 PPO 很容易学到局部路径或保守策略。
- reward credit assignment 困难，低层是否真正服务成功会影响高层 reward 的有效性。

因此，当前不建议继续强行证明“纯 PPO 高层优于启发式高层”。

## 5. 当前论文创新性判断

目前已有的创新点雏形是：

**面向 AoI-aware UAV 数据采集的启发式先验残差高层 PPO。**

可以表述为：

> 纯学习式高层策略在稀疏 option 决策下训练不稳定，因此引入由 AoI、剩余数据、溢出风险和距离构成的启发式先验，将高层动作空间约束在 top-k 候选目标排序附近，再由 PPO 学习有界残差修正，从而兼顾启发式稳定性和强化学习自适应能力。

这个点有一定论文价值，尤其适合当前实验事实：

- 纯 PPO 高层不稳定。
- 启发式高层很强。
- heuristic-prior residual PPO 能恢复接近启发式的性能。
- 1604 在安全性上可能优于启发式高层，NFZ violation 明显更低。

但也存在风险：

- 如果 PPO residual 贡献很小，方法容易被认为只是“启发式策略套 PPO 外壳”。
- 1604 的 rank0 选择率很高，说明策略大部分时候仍按启发式行动。
- 要支撑论文创新，需要消融证明 prior 和 residual 都有作用。

## 6. 下一步最重要的实验

为了判断 1604 是否足够作为论文主算法，需要做消融实验。

建议优先级如下：

### 6.1 同一 7 号环境下的纯 rank0 启发式

目的：

验证 1604 的提升到底来自“候选评分/启发式先验”，还是来自 PPO residual。

方式：

- 强制高层每次选择 rank0。
- 或设置 `high_residual_scale=0`，让 PPO residual 完全不起作用。

如果 1604 明显优于纯 rank0：

- PPO residual 有贡献。
- 当前论文创新性更强。

如果 1604 与纯 rank0 几乎一样：

- 当前方法主要依赖启发式。
- 后续需要增加新的机制来增强创新。

### 6.2 纯 PPO 高层对照

设置：

```text
high_prior_beta = 0.0
```

目的：

证明没有启发式 prior 时，高层 PPO 学习困难。

### 6.3 prior 强度消融

建议测试：

```text
high_prior_beta = 1.0, 2.0, 4.0, 6.0
```

目的：

说明 prior 太弱时 PPO 不稳定，prior 太强时接近纯启发式，存在合理折中。

### 6.4 residual 范围消融

建议测试：

```text
high_residual_scale = 0.0, 0.25, 0.5, 1.0
```

目的：

说明残差自由度过大可能破坏启发式稳定性，过小则学习贡献不足。

## 7. 如果创新性仍不够，下一步可加的机制

最推荐的下一步创新方向：

**自适应 heuristic-prior residual PPO。**

当前形式：

```text
logits = fixed_prior_weight * heuristic_prior
       + fixed_residual_scale * PPO_residual
```

可改为：

```text
logits = state_dependent_prior_weight * heuristic_prior
       + state_dependent_residual_weight * PPO_residual
```

含义：

- 状态简单时，更信任启发式。
- AoI、数据溢出、安全风险、距离冲突更复杂时，允许 PPO 做更大修正。

可以作为论文中的更强算法点：

**Adaptive Heuristic-Prior Residual PPO for AoI-Aware UAV Data Collection**

这样比固定 `high_prior_beta=4.0` 和 `high_residual_scale=0.25` 更有创新性。

## 8. 当前建议

短期不要继续盲目更换高层算法。当前最稳妥路线是：

1. 固定 1604 作为当前主线版本。
2. 先做纯 rank0 启发式、纯 PPO、prior beta、residual scale 消融。
3. 如果 residual 确实有贡献，就围绕 heuristic-prior residual PPO 写论文。
4. 如果 residual 贡献很弱，就加入自适应 prior/residual 门控机制。
5. 若自适应机制仍无优势，再考虑将高层 PPO 替换为 Dueling Double DQN 等更适合稀疏高层决策的 off-policy 方法。

当前最重要的论文叙事应从：

```text
可学习高层 PPO 超过启发式高层
```

调整为：

```text
纯高层 PPO 在稀疏 AoI-UAV option 决策中训练困难；
本文提出启发式先验引导的残差高层策略，
在保留启发式稳定性的同时，通过学习式残差改善动态适应性和安全性。
```

