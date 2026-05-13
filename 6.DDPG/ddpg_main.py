import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import os
import pickle
import argparse
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

# 引入自定义无人机环境
from env.env_uav import EnvCore


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_width, max_action):
        super(Actor, self).__init__()
        self.max_action = max_action
        self.l1 = nn.Linear(state_dim, hidden_width)
        self.l2 = nn.Linear(hidden_width, hidden_width)
        self.l3 = nn.Linear(hidden_width, action_dim)

    def forward(self, s):
        s = F.relu(self.l1(s))
        s = F.relu(self.l2(s))
        a = self.max_action * torch.tanh(self.l3(s))  # [-max,max]
        return a


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_width):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, hidden_width)
        self.l2 = nn.Linear(hidden_width, hidden_width)
        self.l3 = nn.Linear(hidden_width, 1)

    def forward(self, s, a):
        q = F.relu(self.l1(torch.cat([s, a], 1)))
        q = F.relu(self.l2(q))
        q = self.l3(q)
        return q


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim):
        self.max_size = int(1e6)
        self.count = 0
        self.size = 0
        self.s = np.zeros((self.max_size, state_dim))
        self.a = np.zeros((self.max_size, action_dim))
        self.r = np.zeros((self.max_size, 1))
        self.s_ = np.zeros((self.max_size, state_dim))
        self.dw = np.zeros((self.max_size, 1))

    def store(self, s, a, r, s_, dw):
        self.s[self.count] = s
        self.a[self.count] = a
        self.r[self.count] = r
        self.s_[self.count] = s_
        self.dw[self.count] = dw
        self.count = (self.count + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        index = np.random.choice(self.size, size=batch_size)
        batch_s = torch.tensor(self.s[index], dtype=torch.float)
        batch_a = torch.tensor(self.a[index], dtype=torch.float)
        batch_r = torch.tensor(self.r[index], dtype=torch.float)
        batch_s_ = torch.tensor(self.s_[index], dtype=torch.float)
        batch_dw = torch.tensor(self.dw[index], dtype=torch.float)

        return batch_s, batch_a, batch_r, batch_s_, batch_dw


class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action, args):
        self.max_action = max_action
        self.hidden_width = args.hidden_width
        self.batch_size = args.batch_size
        self.GAMMA = args.gamma
        self.TAU = args.tau
        self.lr = args.lr

        self.actor = Actor(state_dim, action_dim, self.hidden_width, max_action)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = Critic(state_dim, action_dim, self.hidden_width)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

        self.MseLoss = nn.MSELoss()

    def choose_action(self, s):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        a = self.actor(s).data.numpy().flatten()
        return a

    def learn(self, relay_buffer):
        batch_s, batch_a, batch_r, batch_s_, batch_dw = relay_buffer.sample(self.batch_size)

        # Compute the target Q
        with torch.no_grad():
            Q_ = self.critic_target(batch_s_, self.actor_target(batch_s_))
            target_Q = batch_r + self.GAMMA * (1 - batch_dw) * Q_

        # Compute the current Q and the critic loss
        current_Q = self.critic(batch_s, batch_a)
        critic_loss = self.MseLoss(target_Q, current_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Freeze critic networks so you don't waste computational effort
        for params in self.critic.parameters():
            params.requires_grad = False

        # Compute the actor loss
        actor_loss = -self.critic(batch_s, self.actor(batch_s)).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Unfreeze critic networks
        for params in self.critic.parameters():
            params.requires_grad = True

        # Softly update the target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)


def evaluate_policy(env, agent):
    times = 3
    evaluate_reward = 0
    for _ in range(times):
        s, _ = env.reset()
        terminated = False
        truncated = False
        episode_reward = 0
        while not (terminated or truncated):
            a = agent.choose_action(s)
            s_, r, terminated, truncated, _ = env.step(a)
            episode_reward += r
            s = s_
        evaluate_reward += episode_reward

    return evaluate_reward / times


if __name__ == '__main__':
    # ================= 0. Argparse 参数解析 =================
    parser = argparse.ArgumentParser("Hyperparameters Setting for DDPG")
    parser.add_argument("--max_train_steps", type=int, default=int(1e6), help="Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=5e3,
                        help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--seed", type=int, default=10, help="Random seed")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--hidden_width", type=int, default=256, help="The number of neurons in hidden layers")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate of actor and critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.005, help="Softly update the target network")
    parser.add_argument("--exploration_noise", type=float, default=0.1,
                        help="The std of Gaussian noise for exploration (as ratio of max_action)")

    args = parser.parse_args()

    # ================= 1. 初始化环境与参数 =================
    env = EnvCore(users_path="./data_train/users_50_new.txt")
    env_evaluate = EnvCore(users_path="./data_train/users_50_new.txt")

    seed = args.seed
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env_evaluate.reset(seed=seed)
    env_evaluate.action_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    max_episode_steps = env.T

    num_agent = getattr(env, 'num_agent', 1)

    print(f"Algorithm=DDPG | env=UAV_Env (EnvCore) | Seed={seed} | Max Steps={args.max_train_steps}")
    print(f"state_dim={state_dim} | action_dim={action_dim} | max_action={max_action}")

    # ================= 2. 日志与保存路径设置 =================
    timestamp = datetime.now().strftime("%m%d%H%M")
    base_save_dir = os.path.join('.', 'results', 'DDPG')
    os.makedirs(base_save_dir, exist_ok=True)

    env_save_path = os.path.join(base_save_dir, f'best_env_{timestamp}.pkl')
    model_save_path = os.path.join(base_save_dir, f'best_model_{timestamp}.pth')

    best_episode_reward = -float('inf')

    agent = DDPG(state_dim, action_dim, max_action, args)
    replay_buffer = ReplayBuffer(state_dim, action_dim)

    writer = SummaryWriter(log_dir=f'runs/DDPG_{timestamp}')

    # ================= 3. 训练主循环 =================
    max_train_steps = args.max_train_steps
    evaluate_freq = args.evaluate_freq
    noise_std = args.exploration_noise * max_action
    random_steps = 25e3
    evaluate_num = 0
    evaluate_rewards = []
    total_steps = 0
    total_episodes = 0

    while total_steps < max_train_steps:
        s, _ = env.reset()
        terminated = False
        truncated = False
        episode_steps = 0

        ep_total_reward = 0
        ep_total_data = 0
        ep_total_energy = 0
        ep_nfz_count = 0

        ep_agent_rewards = [0.0] * num_agent
        ep_agent_data = [0.0] * num_agent
        ep_agent_energy = [0.0] * num_agent

        while not (terminated or truncated):
            episode_steps += 1
            if total_steps < random_steps:
                a = env.action_space.sample()
            else:
                a = agent.choose_action(s)
                a = (a + np.random.normal(0, noise_std, size=action_dim)).clip(-max_action, max_action)

            s_, r, terminated, truncated, info = env.step(a)

            # 数据指标提取
            r_list = r if isinstance(r, (list, np.ndarray)) else [r]
            data_list = info["collected_data"] if isinstance(info["collected_data"], (list, np.ndarray)) else [
                info["collected_data"]]
            energy_list = info["energy"] if isinstance(info["energy"], (list, np.ndarray)) else [info["energy"]]

            ep_total_reward += np.sum(r_list)
            ep_total_data += np.sum(data_list)
            ep_total_energy += np.sum(energy_list)
            if info.get("in_nfz", False):
                ep_nfz_count += 1

            for i in range(min(num_agent, len(r_list))):
                ep_agent_rewards[i] += r_list[i]
                ep_agent_data[i] += data_list[i]
                ep_agent_energy[i] += energy_list[i]

            if terminated and episode_steps != max_episode_steps:
                dw = True
            else:
                dw = False

            replay_buffer.store(s, a, r, s_, dw)
            s = s_
            total_steps += 1

            # DDPG 更新网络 (对齐 TD3 的单步更新逻辑)
            if total_steps >= random_steps:
                agent.learn(replay_buffer)

            # 策略评估
            if total_steps % evaluate_freq == 0 and total_steps > 0:
                evaluate_num += 1
                evaluate_reward = evaluate_policy(env_evaluate, agent)
                evaluate_rewards.append(evaluate_reward)
                writer.add_scalar('Evaluate/Reward_vs_Step', evaluate_reward, global_step=total_steps)

        total_episodes += 1

        # TensorBoard 记录
        writer.add_scalar('Episode_Total/Reward', ep_total_reward, global_step=total_episodes)
        writer.add_scalar('Episode_Total/Data_Collected', ep_total_data, global_step=total_episodes)
        writer.add_scalar('Episode_Total/Energy_Consumption', ep_total_energy, global_step=total_episodes)
        writer.add_scalar('Episode_Total/NFZ_Violation_Steps', ep_nfz_count, global_step=total_episodes)

        for i in range(num_agent):
            writer.add_scalar(f'Agent_{i}/Reward', ep_agent_rewards[i], global_step=total_episodes)
            writer.add_scalar(f'Agent_{i}/Data_Collected', ep_agent_data[i], global_step=total_episodes)
            writer.add_scalar(f'Agent_{i}/Energy_Consumption', ep_agent_energy[i], global_step=total_episodes)

        # 保存最佳模型
        if ep_total_reward > best_episode_reward and total_steps >= random_steps:
            best_episode_reward = ep_total_reward

            best_episode_envs = copy.deepcopy(env)
            with open(env_save_path, "wb") as f:
                pickle.dump(best_episode_envs, f)

            torch.save(agent.actor.state_dict(), model_save_path)
            print(
                f"🌟 [New Record] Episode {total_episodes}: Reward={best_episode_reward:.2f} | Envs & Model Saved to ./results/DDPG/")

        if total_episodes % 10 == 0:
            print(
                f"Episode: {total_episodes} | Total Steps: {total_steps} | Data: {ep_total_data:.1f} | Energy: {ep_total_energy:.2e}")

    writer.close()