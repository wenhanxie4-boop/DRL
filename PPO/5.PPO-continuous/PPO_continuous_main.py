import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import argparse
from normalization import Normalization, RewardScaling
from replaybuffer import ReplayBuffer
from ppo_continuous_CA import PPO_continuous

import os
import copy
import pickle
from datetime import datetime

# 2026.04.03修改
from env.env_uav import EnvCore


def evaluate_policy(args, env, agent, state_norm):
    times = 3
    evaluate_reward = 0
    for _ in range(times):
        s, _ = env.reset()
        if args.use_state_norm:
            s = state_norm(s, update=False)

        terminated = False
        truncated = False
        episode_reward = 0

        while not (terminated or truncated):
            a = agent.evaluate(s)
            if args.policy_dist == "Beta":
                action = 2 * (a - 0.5) * args.max_action
            else:
                action = a

            s_, r, terminated, truncated, _ = env.step(action)

            if args.use_state_norm:
                s_ = state_norm(s_, update=False)
            episode_reward += r
            s = s_
        evaluate_reward += episode_reward

    return evaluate_reward / times


def main(args, env_name, number, seed):
    # 实例化环境
    env = EnvCore(users_path="./data_train/users_50_new.txt")
    env_evaluate = EnvCore(users_path="./data_train/users_50_new.txt")

    # Set random seed
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env_evaluate.reset(seed=seed)
    env_evaluate.action_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    args.max_action = float(env.action_space.high[0])
    args.max_episode_steps = env.T

    # --- 新增：为了适应未来的多无人机扩展，获取智能体数量（目前为1） ---
    num_agent = getattr(env, 'num_agent', 1)

    # --- 新增：设置保存路径和日期命名 ---
    timestamp = datetime.now().strftime("%m%d%H%M")
    base_save_dir = os.path.join('.', 'results', 'PPO')
    os.makedirs(base_save_dir, exist_ok=True)

    # 2. 精简保存文件名，直接使用时间戳
    env_save_path = os.path.join(base_save_dir, f'best_env_{timestamp}.pkl')
    model_save_path = os.path.join(base_save_dir, f'best_model_{timestamp}.pth')

    best_episode_reward = -float('inf')  # 用于打擂台，记录历史最高分
    # --------------------------------

    evaluate_num = 0
    evaluate_rewards = []
    total_steps = 0
    total_episodes = 0  # 用于 TensorBoard 横坐标的回合计数器

    replay_buffer = ReplayBuffer(args)
    agent = PPO_continuous(args)

    # 3. 精简 TensorBoard 日志目录，直接以时间戳作为文件夹名
    writer = SummaryWriter(log_dir=f'runs/{timestamp}')

    state_norm = Normalization(shape=args.state_dim)
    if args.use_reward_norm:
        reward_norm = Normalization(shape=1)
    elif args.use_reward_scaling:
        reward_scaling = RewardScaling(shape=1, gamma=args.gamma)

    while total_steps < args.max_train_steps:
        s, _ = env.reset()
        if args.use_state_norm:
            s = state_norm(s)
        if args.use_reward_scaling:
            reward_scaling.reset()

        # --- 新增：每个 Episode 的数据统计变量 (兼容多智能体) ---
        ep_total_reward = 0
        ep_total_data = 0
        ep_total_energy = 0
        ep_nfz_count = 0

        # 使用列表来记录每个智能体的数据
        ep_agent_rewards = [0.0] * num_agent
        ep_agent_data = [0.0] * num_agent
        ep_agent_energy = [0.0] * num_agent
        # --------------------------------------

        terminated = False
        truncated = False
        episode_steps = 0

        while not (terminated or truncated):
            episode_steps += 1
            a, a_logprob = agent.choose_action(s)
            if args.policy_dist == "Beta":
                action = 2 * (a - 0.5) * args.max_action
            else:
                action = a

            # 执行动作，获取 info 字典中的环境信息
            s_, r, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # --- 修改：累加统计数据，做向下兼容处理 ---
            # 为了同时兼容现在的单智能体(标量)和未来的多智能体(列表/数组)
            r_list = r if isinstance(r, (list, np.ndarray)) else [r]
            data_list = info["collected_data"] if isinstance(info["collected_data"], (list, np.ndarray)) else [
                info["collected_data"]]
            energy_list = info["energy"] if isinstance(info["energy"], (list, np.ndarray)) else [info["energy"]]

            ep_total_reward += np.sum(r_list)
            ep_total_data += np.sum(data_list)
            ep_total_energy += np.sum(energy_list)
            if info.get("in_nfz", False):
                ep_nfz_count += 1

            # 用 for 循环单独记录每个智能体的信息
            for i in range(min(num_agent, len(r_list))):
                ep_agent_rewards[i] += r_list[i]
                ep_agent_data[i] += data_list[i]
                ep_agent_energy[i] += energy_list[i]
            # ----------------------------------------

            if args.use_state_norm:
                s_ = state_norm(s_)

            # 记录用于训练的奖励（处理过的奖励）
            r_for_train = r
            if args.use_reward_norm:
                r_for_train = reward_norm(r_for_train)
            elif args.use_reward_scaling:
                r_for_train = reward_scaling(r_for_train)

            if terminated and episode_steps != args.max_episode_steps:
                dw = True
            else:
                dw = False

            replay_buffer.store(s, a, a_logprob, r_for_train, s_, dw, done)
            s = s_
            total_steps += 1

            # 网络更新
            if replay_buffer.count == args.batch_size:
                agent.update(replay_buffer, total_steps)
                replay_buffer.count = 0

            # 定期测试
            if total_steps % args.evaluate_freq == 0:
                evaluate_num += 1
                evaluate_reward = evaluate_policy(args, env_evaluate, agent, state_norm)
                evaluate_rewards.append(evaluate_reward)
                writer.add_scalar('Evaluate/Reward_vs_Step', evaluate_rewards[-1], global_step=total_steps)

        # ================== Episode 结束，写入 TensorBoard ==================
        total_episodes += 1

        # 1. 全局数据
        writer.add_scalar('Episode_Total/Reward', ep_total_reward, global_step=total_episodes)
        writer.add_scalar('Episode_Total/Data_Collected', ep_total_data, global_step=total_episodes)
        writer.add_scalar('Episode_Total/Energy_Consumption', ep_total_energy, global_step=total_episodes)
        writer.add_scalar('Episode_Total/NFZ_Violation_Steps', ep_nfz_count, global_step=total_episodes)

        # 2. 用 for 循环分发到每个智能体的面板
        for i in range(num_agent):
            writer.add_scalar(f'Agent_{i}/Reward', ep_agent_rewards[i], global_step=total_episodes)
            writer.add_scalar(f'Agent_{i}/Data_Collected', ep_agent_data[i], global_step=total_episodes)
            writer.add_scalar(f'Agent_{i}/Energy_Consumption', ep_agent_energy[i], global_step=total_episodes)

        # ================== 打擂台：保存最佳环境与模型 ==================
        if ep_total_reward > best_episode_reward:
            best_episode_reward = ep_total_reward

            # 冻结并保存环境快照
            best_episode_envs = copy.deepcopy(env)
            with open(env_save_path, "wb") as f:
                pickle.dump(best_episode_envs, f)

            # 同步保存大脑 (Actor网络权重)
            torch.save(agent.actor.state_dict(), model_save_path)

            print(
                f"🌟 [New Record] Episode {total_episodes}: Reward={best_episode_reward:.2f} | Envs & Model Saved to ./results/PPO/")

        # 打印进度监控
        if total_episodes % 10 == 0:
            print(
                f"Episode: {total_episodes} | Total Steps: {total_steps} | Data: {ep_total_data:.1f} | Energy: {ep_total_energy:.2e}")

    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for PPO-continuous")
    # ... (参数保持不变)
    parser.add_argument("--max_train_steps", type=int, default=int(1e6), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=5e3,
                        help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--policy_dist", type=str, default="Gaussian", help="Beta or Gaussian")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--hidden_width", type=int, default=64, help="The number of neurons in hidden layers")
    parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=True, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")

    args = parser.parse_args()

    main(args, env_name="UAV_Env_v1", number=1, seed=10)