import argparse
import copy
import os
import pickle
from datetime import datetime

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from env.env_uav_hrl_aoi import EnvCore
from normalization import Normalization, RewardScaling
from ppo_continuous import PPO_continuous
from replaybuffer import ReplayBuffer


def evaluate_policy(args, env, agent, state_norm):
    times = 3
    evaluate_reward = 0.0
    evaluate_data = 0.0
    evaluate_aoi = 0.0

    for _ in range(times):
        s, _ = env.reset()
        if args.use_state_norm:
            s = state_norm(s, update=False)

        terminated = False
        truncated = False
        episode_reward = 0.0
        episode_data = 0.0
        last_mean_aoi = 0.0

        while not (terminated or truncated):
            a = agent.evaluate(s)
            if args.policy_dist == "Beta":
                action = 2 * (a - 0.5) * args.max_action
            else:
                action = a

            s_, r, terminated, truncated, info = env.step(action)
            if args.use_state_norm:
                s_ = state_norm(s_, update=False)

            episode_reward += r
            episode_data += info.get("collected_data", 0.0)
            last_mean_aoi = info.get("active_mean_aoi", 0.0)
            s = s_

        evaluate_reward += episode_reward
        evaluate_data += episode_data
        evaluate_aoi += last_mean_aoi

    return evaluate_reward / times, evaluate_data / times, evaluate_aoi / times


def main(args, seed):
    env = EnvCore(
        users_path=args.users_path,
        option_interval=args.option_interval,
    )
    env_evaluate = EnvCore(
        users_path=args.users_path,
        option_interval=args.option_interval,
    )

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

    timestamp = datetime.now().strftime("%m%d%H%M")
    base_save_dir = os.path.join(".", "results", "PPO")
    os.makedirs(base_save_dir, exist_ok=True)
    env_save_path = os.path.join(base_save_dir, f"best_env_{timestamp}.pkl")
    model_save_path = os.path.join(base_save_dir, f"best_model_{timestamp}.pth")

    replay_buffer = ReplayBuffer(args)
    agent = PPO_continuous(args)
    writer = SummaryWriter(log_dir=f"runs/{timestamp}")

    state_norm = Normalization(shape=args.state_dim)
    if args.use_reward_norm:
        reward_norm = Normalization(shape=1)
    elif args.use_reward_scaling:
        reward_scaling = RewardScaling(shape=1, gamma=args.gamma)

    best_episode_reward = -float("inf")
    total_steps = 0
    total_episodes = 0
    num_agent = getattr(env, "num_agent", 1)

    while total_steps < args.max_train_steps:
        s, _ = env.reset()
        if args.use_state_norm:
            s = state_norm(s)
        if args.use_reward_scaling:
            reward_scaling.reset()

        ep_reward = 0.0
        ep_data = 0.0
        ep_energy = 0.0
        ep_drop = 0.0
        ep_nfz_count = 0
        ep_aoi_cost = 0.0
        ep_active_mean_aoi = 0.0
        ep_active_users = 0
        ep_reward_data = 0.0
        ep_reward_distance = 0.0
        ep_penalty_drop = 0.0
        ep_penalty_aoi = 0.0
        ep_penalty_energy = 0.0
        ep_penalty_nfz = 0.0
        ep_penalty_step = 0.0
        ep_agent_rewards = [0.0] * num_agent
        ep_agent_data = [0.0] * num_agent
        ep_agent_energy = [0.0] * num_agent
        episode_steps = 0

        terminated = False
        truncated = False

        while not (terminated or truncated):
            episode_steps += 1
            a, a_logprob = agent.choose_action(s)
            if args.policy_dist == "Beta":
                action = 2 * (a - 0.5) * args.max_action
            else:
                action = a

            s_, r, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            ep_reward += r
            ep_data += info.get("collected_data", 0.0)
            ep_energy += info.get("energy", 0.0)
            ep_drop += info.get("dropped_data", 0.0)
            ep_aoi_cost += info.get("aoi_cost", 0.0)
            ep_active_mean_aoi = info.get("active_mean_aoi", 0.0)
            ep_active_users = info.get("active_users", 0)
            ep_reward_data += info.get("reward_data", 0.0)
            ep_reward_distance += info.get("reward_distance", 0.0)
            ep_penalty_drop += info.get("penalty_drop", 0.0)
            ep_penalty_aoi += info.get("penalty_aoi", 0.0)
            ep_penalty_energy += info.get("penalty_energy", 0.0)
            ep_penalty_nfz += info.get("penalty_nfz", 0.0)
            ep_penalty_step += info.get("penalty_step", 0.0)
            if info.get("in_nfz", False):
                ep_nfz_count += 1

            r_list = r if isinstance(r, (list, np.ndarray)) else [r]
            data_list = info["collected_data"] if isinstance(info["collected_data"], (list, np.ndarray)) else [
                info["collected_data"]]
            energy_list = info["energy"] if isinstance(info["energy"], (list, np.ndarray)) else [info["energy"]]
            for i in range(min(num_agent, len(r_list))):
                ep_agent_rewards[i] += r_list[i]
                ep_agent_data[i] += data_list[i]
                ep_agent_energy[i] += energy_list[i]

            if args.use_state_norm:
                s_ = state_norm(s_)

            r_for_train = r
            if args.use_reward_norm:
                r_for_train = reward_norm(r_for_train)
            elif args.use_reward_scaling:
                r_for_train = reward_scaling(r_for_train)

            dw = bool(terminated and episode_steps != args.max_episode_steps)
            replay_buffer.store(s, a, a_logprob, r_for_train, s_, dw, done)
            s = s_
            total_steps += 1

            if replay_buffer.count == args.batch_size:
                agent.update(replay_buffer, total_steps)
                replay_buffer.count = 0

            if total_steps % args.evaluate_freq == 0:
                eval_reward, eval_data, eval_aoi = evaluate_policy(args, env_evaluate, agent, state_norm)
                writer.add_scalar("Evaluate/Reward_vs_Step", eval_reward, global_step=total_steps)
                writer.add_scalar("Evaluate/Data_Collected_vs_Step", eval_data, global_step=total_steps)
                writer.add_scalar("Evaluate/Active_Mean_AoI_vs_Step", eval_aoi, global_step=total_steps)
                writer.add_scalar("Mission/Evaluate_Data_Collected", eval_data, global_step=total_steps)
                writer.add_scalar("AoI/Evaluate_Active_Mean_AoI", eval_aoi, global_step=total_steps)

        total_episodes += 1

        writer.add_scalar("Episode_Total/Reward", ep_reward, global_step=total_episodes)
        writer.add_scalar("Episode_Total/Data_Collected", ep_data, global_step=total_episodes)
        writer.add_scalar("Episode_Total/Energy_Consumption", ep_energy, global_step=total_episodes)
        writer.add_scalar("Episode_Total/NFZ_Violation_Steps", ep_nfz_count, global_step=total_episodes)

        for i in range(num_agent):
            writer.add_scalar(f"Agent_{i}/Reward", ep_agent_rewards[i], global_step=total_episodes)
            writer.add_scalar(f"Agent_{i}/Data_Collected", ep_agent_data[i], global_step=total_episodes)
            writer.add_scalar(f"Agent_{i}/Energy_Consumption", ep_agent_energy[i], global_step=total_episodes)

        writer.add_scalar("Episode_Total/Dropped_Data", ep_drop, global_step=total_episodes)
        writer.add_scalar("Episode_Total/AoI_Cost", ep_aoi_cost, global_step=total_episodes)
        writer.add_scalar("Episode_Total/Active_Mean_AoI_Final", ep_active_mean_aoi, global_step=total_episodes)
        writer.add_scalar("Episode_Total/Target_Switch_Count", env.target_switch_count, global_step=total_episodes)

        writer.add_scalar("Mission/Data_Collected", ep_data, global_step=total_episodes)
        writer.add_scalar("Mission/Dropped_Data", ep_drop, global_step=total_episodes)
        writer.add_scalar("Mission/Target_Switch_Count", env.target_switch_count, global_step=total_episodes)
        writer.add_scalar("Mission/Active_Users_Final", ep_active_users, global_step=total_episodes)

        writer.add_scalar("AoI/Cost", ep_aoi_cost, global_step=total_episodes)
        writer.add_scalar("AoI/Active_Mean_Final", ep_active_mean_aoi, global_step=total_episodes)

        writer.add_scalar("Safety_Cost/NFZ_Violation_Steps", ep_nfz_count, global_step=total_episodes)
        writer.add_scalar("Safety_Cost/Energy_Consumption", ep_energy, global_step=total_episodes)

        writer.add_scalar("Reward_Breakdown/Data_Reward", ep_reward_data, global_step=total_episodes)
        writer.add_scalar("Reward_Breakdown/Distance_Reward", ep_reward_distance, global_step=total_episodes)
        writer.add_scalar("Reward_Breakdown/Drop_Penalty", ep_penalty_drop, global_step=total_episodes)
        writer.add_scalar("Reward_Breakdown/AoI_Penalty", ep_penalty_aoi, global_step=total_episodes)
        writer.add_scalar("Reward_Breakdown/Energy_Penalty", ep_penalty_energy, global_step=total_episodes)
        writer.add_scalar("Reward_Breakdown/NFZ_Penalty", ep_penalty_nfz, global_step=total_episodes)
        writer.add_scalar("Reward_Breakdown/Step_Penalty", ep_penalty_step, global_step=total_episodes)

        if ep_reward > best_episode_reward:
            best_episode_reward = ep_reward
            with open(env_save_path, "wb") as f:
                pickle.dump(copy.deepcopy(env), f)
            torch.save(agent.actor.state_dict(), model_save_path)
            print(
                f"[New Record] Episode {total_episodes}: Reward={best_episode_reward:.2f} "
                f"| Envs & Model Saved to ./results/PPO/"
            )

        if total_episodes % 10 == 0:
            print(
                f"Episode: {total_episodes} | Total Steps: {total_steps} "
                f"| Data: {ep_data:.1f} | Energy: {ep_energy:.2e}"
            )

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("AoI-aware Hierarchical PPO for UAV data collection")
    parser.add_argument("--users_path", type=str, default="./data_train/users_50_v2.txt")
    parser.add_argument("--option_interval", type=int, default=10, help="High-level target holding steps")
    parser.add_argument("--seed", type=int, default=10)

    parser.add_argument("--max_train_steps", type=int, default=int(1e6))
    parser.add_argument("--evaluate_freq", type=float, default=5e3)
    parser.add_argument("--policy_dist", type=str, default="Gaussian", help="Beta or Gaussian")
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--mini_batch_size", type=int, default=64)
    parser.add_argument("--hidden_width", type=int, default=64)
    parser.add_argument("--lr_a", type=float, default=3e-4)
    parser.add_argument("--lr_c", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lamda", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--K_epochs", type=int, default=10)
    parser.add_argument("--use_adv_norm", type=bool, default=True)
    parser.add_argument("--use_state_norm", type=bool, default=True)
    parser.add_argument("--use_reward_norm", type=bool, default=False)
    parser.add_argument("--use_reward_scaling", type=bool, default=True)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--use_lr_decay", type=bool, default=True)
    parser.add_argument("--use_grad_clip", type=bool, default=True)
    parser.add_argument("--use_orthogonal_init", type=bool, default=True)
    parser.add_argument("--set_adam_eps", type=bool, default=True)
    parser.add_argument("--use_tanh", type=bool, default=True)

    args = parser.parse_args()
    main(args, seed=args.seed)
