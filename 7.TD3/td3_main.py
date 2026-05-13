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

# 0407修改
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
        a = self.max_action * torch.tanh(self.l3(s))
        return a


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_width):
        super(Critic, self).__init__()
        # Q1
        self.l1 = nn.Linear(state_dim + action_dim, hidden_width)
        self.l2 = nn.Linear(hidden_width, hidden_width)
        self.l3 = nn.Linear(hidden_width, 1)
        # Q2
        self.l4 = nn.Linear(state_dim + action_dim, hidden_width)
        self.l5 = nn.Linear(hidden_width, hidden_width)
        self.l6 = nn.Linear(hidden_width, 1)

    def forward(self, s, a):
        s_a = torch.cat([s, a], 1)
        q1 = F.relu(self.l1(s_a))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(s_a))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)

        return q1, q2

    def Q1(self, s, a):
        s_a = torch.cat([s, a], 1)
        q1 = F.relu(self.l1(s_a))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


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


class TD3(object):
    def __init__(self, state_dim, action_dim, max_action, args):
        self.max_action = max_action
        self.hidden_width = args.hidden_width
        self.batch_size = args.batch_size
        self.GAMMA = args.gamma
        self.TAU = args.tau
        self.lr = args.lr
        self.policy_noise = args.policy_noise * max_action
        self.noise_clip = args.noise_clip * max_action
        self.policy_freq = args.policy_freq

        self.actor_pointer = 0

        self.actor = Actor(state_dim, action_dim, self.hidden_width, max_action)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = Critic(state_dim, action_dim, self.hidden_width)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

    def choose_action(self, s, deterministic=True):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        a = self.actor(s).data.numpy().flatten()
        return a

    def learn(self, relay_buffer):
        self.actor_pointer += 1
        batch_s, batch_a, batch_r, batch_s_, batch_dw = relay_buffer.sample(self.batch_size)

        with torch.no_grad():
            noise = (torch.randn_like(batch_a) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(batch_s_) + noise).clamp(-self.max_action, self.max_action)

            target_Q1, target_Q2 = self.critic_target(batch_s_, next_action)
            target_Q = batch_r + self.GAMMA * (1 - batch_dw) * torch.min(target_Q1, target_Q2)

        current_Q1, current_Q2 = self.critic(batch_s, batch_a)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.actor_pointer % self.policy_freq == 0:
            for params in self.critic.parameters():
                params.requires_grad = False

            actor_loss = -self.critic.Q1(batch_s, self.actor(batch_s)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for params in self.critic.parameters():
                params.requires_grad = True

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
            a = agent.choose_action(s, deterministic=True)
            s_, r, terminated, truncated, _ = env.step(a)
            episode_reward += r
            s = s_
        evaluate_reward += episode_reward

    return evaluate_reward / times


if __name__ == '__main__':
    # ================= 0. Argparse 参数解析 =================
    parser = argparse.ArgumentParser("Hyperparameters Setting for TD3-continuous")
    parser.add_argument("--max_train_steps", type=int, default=int(1e6), help="Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=5e3,
                        help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--seed", type=int, default=10, help="Random seed")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--hidden_width", type=int, default=256, help="The number of neurons in hidden layers")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate of actor and critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.005, help="Softly update the target network")
    parser.add_argument("--policy_noise", type=float, default=0.2,
                        help="The noise for target policy smoothing (as ratio of max_action)")
    parser.add_argument("--noise_clip", type=float, default=0.5, help="Clip the noise (as ratio of max_action)")
    parser.add_argument("--policy_freq", type=int, default=2, help="The frequency of policy updates")

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

    print(f"env=UAV_Env_v1 (EnvCore) | Seed={seed} | Max Steps={args.max_train_steps}")
    print(f"state_dim={state_dim} | action_dim={action_dim} | max_action={max_action}")

    # ================= 2. 日志与保存路径设置 =================
    timestamp = datetime.now().strftime("%m%d%H%M")
    base_save_dir = os.path.join('.', 'results', 'TD3')
    os.makedirs(base_save_dir, exist_ok=True)

    env_save_path = os.path.join(base_save_dir, f'best_env_{timestamp}.pkl')
    model_save_path = os.path.join(base_save_dir, f'best_model_{timestamp}.pth')

    best_episode_reward = -float('inf')

    agent = TD3(state_dim, action_dim, max_action, args)
    replay_buffer = ReplayBuffer(state_dim, action_dim)

    writer = SummaryWriter(log_dir=f'runs/TD3_{timestamp}')

    # ================= 3. 训练主循环 =================
    max_train_steps = args.max_train_steps
    evaluate_freq = args.evaluate_freq
    noise_std = 0.1 * max_action
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

            if total_steps >= random_steps:
                agent.learn(replay_buffer)

            if total_steps % evaluate_freq == 0 and total_steps > 0:
                evaluate_num += 1
                evaluate_reward = evaluate_policy(env_evaluate, agent)
                evaluate_rewards.append(evaluate_reward)
                writer.add_scalar('Evaluate/Reward_vs_Step', evaluate_reward, global_step=total_steps)

        total_episodes += 1

        writer.add_scalar('Episode_Total/Reward', ep_total_reward, global_step=total_episodes)
        writer.add_scalar('Episode_Total/Data_Collected', ep_total_data, global_step=total_episodes)
        writer.add_scalar('Episode_Total/Energy_Consumption', ep_total_energy, global_step=total_episodes)
        writer.add_scalar('Episode_Total/NFZ_Violation_Steps', ep_nfz_count, global_step=total_episodes)

        for i in range(num_agent):
            writer.add_scalar(f'Agent_{i}/Reward', ep_agent_rewards[i], global_step=total_episodes)
            writer.add_scalar(f'Agent_{i}/Data_Collected', ep_agent_data[i], global_step=total_episodes)
            writer.add_scalar(f'Agent_{i}/Energy_Consumption', ep_agent_energy[i], global_step=total_episodes)

        if ep_total_reward > best_episode_reward and total_steps >= random_steps:
            best_episode_reward = ep_total_reward

            best_episode_envs = copy.deepcopy(env)
            with open(env_save_path, "wb") as f:
                pickle.dump(best_episode_envs, f)

            torch.save(agent.actor.state_dict(), model_save_path)
            print(
                f"🌟 [New Record] Episode {total_episodes}: Reward={best_episode_reward:.2f} | Envs & Model Saved to ./results/TD3/")

        if total_episodes % 10 == 0:
            print(
                f"Episode: {total_episodes} | Total Steps: {total_steps} | Data: {ep_total_data:.1f} | Energy: {ep_total_energy:.2e}")

    writer.close()