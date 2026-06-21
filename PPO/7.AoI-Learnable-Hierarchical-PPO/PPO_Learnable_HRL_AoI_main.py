import argparse
import copy
import os
import pickle
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from env.env_uav_learnable_hrl_aoi import EnvCore
from high_replaybuffer import HighReplayBuffer
from normalization import Normalization, RewardScaling
from ppo_continuous import PPO_continuous
from ppo_discrete_high import PPO_discrete_high
from replaybuffer import ReplayBuffer


def choose_high_option(args, env, high_agent, high_state_norm, high_state_raw=None, evaluate=False):
    if high_state_raw is None:
        high_state_raw = env.get_high_level_state()
    if args.use_state_norm:
        high_state = high_state_norm(high_state_raw, update=not evaluate)
    else:
        high_state = high_state_raw

    if evaluate:
        high_action = high_agent.evaluate(high_state)
        high_logprob = 0.0
    else:
        high_action, high_logprob = high_agent.choose_action(high_state)

    selected_user_id = env.set_high_level_action(high_action)
    return high_state, high_action, high_logprob, selected_user_id


def evaluate_policy(args, env, low_agent, high_agent, low_state_norm, high_state_norm):
    times = 3
    evaluate_reward = 0.0
    evaluate_data = 0.0
    evaluate_aoi = 0.0

    for _ in range(times):
        env.reset()
        choose_high_option(args, env, high_agent, high_state_norm, evaluate=True)
        s = env.get_current_state()
        if args.use_state_norm:
            s = low_state_norm(s, update=False)

        terminated = False
        truncated = False
        episode_reward = 0.0
        episode_data = 0.0
        last_mean_aoi = 0.0

        while not (terminated or truncated):
            a = low_agent.evaluate(s)
            if args.policy_dist == "Beta":
                action = 2 * (a - 0.5) * args.max_action
            else:
                action = a

            _, r, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if info.get("option_done", False) and not done:
                next_high_raw = env.get_high_level_state()
                choose_high_option(args, env, high_agent, high_state_norm, high_state_raw=next_high_raw, evaluate=True)

            s_ = env.get_current_state()
            if args.use_state_norm:
                s_ = low_state_norm(s_, update=False)

            episode_reward += r
            episode_data += info.get("collected_data", 0.0)
            last_mean_aoi = info.get("active_mean_aoi", 0.0)
            s = s_

        evaluate_reward += episode_reward
        evaluate_data += episode_data
        evaluate_aoi += last_mean_aoi

    return evaluate_reward / times, evaluate_data / times, evaluate_aoi / times


def build_high_args(args, env):
    return SimpleNamespace(
        high_state_dim=env.high_state_dim,
        high_action_dim=env.high_action_dim,
        high_batch_size=args.high_batch_size,
        high_mini_batch_size=args.high_mini_batch_size,
        high_hidden_width=args.high_hidden_width,
        high_lr_a=args.high_lr_a,
        high_lr_c=args.high_lr_c,
        high_gamma=args.high_gamma,
        high_entropy_coef=args.high_entropy_coef,
        max_train_steps=args.max_train_steps,
        lamda=args.lamda,
        epsilon=args.epsilon,
        K_epochs=args.K_epochs,
        use_adv_norm=args.use_adv_norm,
        use_lr_decay=args.use_lr_decay,
        use_grad_clip=args.use_grad_clip,
        use_orthogonal_init=args.use_orthogonal_init,
        set_adam_eps=args.set_adam_eps,
        use_tanh=args.use_tanh,
        high_prior_beta=args.high_prior_beta,
        high_residual_scale=args.high_residual_scale,
    )


def update_high_decision_stats(high_agent, high_state, action, stats):
    probs, entropy = high_agent.action_diagnostics(high_state)
    action = int(action)
    stats["counts"][action] += 1
    stats["prob_sums"] += probs
    stats["entropy_sum"] += entropy


def main(args, seed):
    env = EnvCore(
        users_path=args.users_path,
        option_interval=args.option_interval,
        high_top_k=args.high_top_k,
    )
    env_evaluate = EnvCore(
        users_path=args.users_path,
        option_interval=args.option_interval,
        high_top_k=args.high_top_k,
    )

    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.high_action_space.seed(seed)
    env_evaluate.reset(seed=seed)
    env_evaluate.action_space.seed(seed)
    env_evaluate.high_action_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    args.max_action = float(env.action_space.high[0])
    args.max_episode_steps = env.T
    args.high_state_dim = env.high_state_dim
    args.high_action_dim = env.high_action_dim

    timestamp = datetime.now().strftime("%m%d%H%M")
    base_save_dir = os.path.join(".", "results", "PPO")
    os.makedirs(base_save_dir, exist_ok=True)
    env_save_path = os.path.join(base_save_dir, f"best_env_{timestamp}.pkl")
    low_model_save_path = os.path.join(base_save_dir, f"best_model_{timestamp}.pth")
    high_model_save_path = os.path.join(base_save_dir, f"best_high_model_{timestamp}.pth")

    low_buffer = ReplayBuffer(args)
    low_agent = PPO_continuous(args)

    high_args = build_high_args(args, env)
    high_buffer = HighReplayBuffer(high_args)
    high_agent = PPO_discrete_high(high_args)

    writer = SummaryWriter(log_dir=f"runs/{timestamp}")
    low_state_norm = Normalization(shape=args.state_dim)
    high_state_norm = Normalization(shape=args.high_state_dim)

    if args.use_reward_norm:
        low_reward_norm = Normalization(shape=1)
        high_reward_norm = Normalization(shape=1)
    elif args.use_reward_scaling:
        low_reward_scaling = RewardScaling(shape=1, gamma=args.gamma)
        high_reward_scaling = RewardScaling(shape=1, gamma=args.high_gamma)

    best_episode_reward = -float("inf")
    total_steps = 0
    total_episodes = 0
    num_agent = getattr(env, "num_agent", 1)

    while total_steps < args.max_train_steps:
        env.reset()
        if args.use_reward_scaling:
            low_reward_scaling.reset()
            high_reward_scaling.reset()

        current_high_s, current_high_a, current_high_logprob, current_selected_user_id = choose_high_option(
            args, env, high_agent, high_state_norm, evaluate=False
        )
        option_collected_data = 0.0
        option_dropped_data = 0.0
        option_aoi_cost = 0.0
        option_steps = 0

        s = env.get_current_state()
        if args.use_state_norm:
            s = low_state_norm(s)

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
        ep_high_decisions = 1
        ep_high_reward = 0.0
        ep_high_option_data = 0.0
        ep_high_option_drop = 0.0
        ep_high_option_aoi = 0.0
        ep_high_stats = {
            "counts": np.zeros(args.high_action_dim, dtype=np.float32),
            "prob_sums": np.zeros(args.high_action_dim, dtype=np.float32),
            "option_counts": np.zeros(args.high_action_dim, dtype=np.float32),
            "option_data": np.zeros(args.high_action_dim, dtype=np.float32),
            "option_aoi": np.zeros(args.high_action_dim, dtype=np.float32),
            "entropy_sum": 0.0,
            "repeat_targets": 0.0,
            "repeat_checks": 0.0,
        }
        update_high_decision_stats(high_agent, current_high_s, current_high_a, ep_high_stats)
        ep_agent_rewards = [0.0] * num_agent
        ep_agent_data = [0.0] * num_agent
        ep_agent_energy = [0.0] * num_agent

        terminated = False
        truncated = False
        episode_steps = 0

        while not (terminated or truncated):
            episode_steps += 1
            option_steps += 1
            a, a_logprob = low_agent.choose_action(s)
            if args.policy_dist == "Beta":
                action = 2 * (a - 0.5) * args.max_action
            else:
                action = a

            _, r, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            option_collected_data += info.get("collected_data", 0.0)
            option_dropped_data += info.get("dropped_data", 0.0)
            option_aoi_cost += info.get("aoi_cost", 0.0)

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

            option_done = info.get("option_done", False) or done
            if option_done:
                next_high_raw = env.get_high_level_state()
                if args.use_state_norm:
                    next_high_s = high_state_norm(next_high_raw)
                else:
                    next_high_s = next_high_raw

                avg_option_aoi_cost = option_aoi_cost / max(option_steps, 1)
                high_reward_raw = (
                    args.high_w_data * option_collected_data
                    - args.high_w_drop * option_dropped_data
                    - args.high_w_aoi * avg_option_aoi_cost
                    - args.high_switch_penalty
                )
                ep_high_reward += high_reward_raw
                ep_high_option_data += option_collected_data
                ep_high_option_drop += option_dropped_data
                ep_high_option_aoi += avg_option_aoi_cost
                current_high_rank = int(current_high_a)
                ep_high_stats["option_counts"][current_high_rank] += 1
                ep_high_stats["option_data"][current_high_rank] += option_collected_data
                ep_high_stats["option_aoi"][current_high_rank] += avg_option_aoi_cost

                high_r_for_train = high_reward_raw
                if args.use_reward_norm:
                    high_r_for_train = high_reward_norm(high_r_for_train)
                elif args.use_reward_scaling:
                    high_r_for_train = high_reward_scaling(high_r_for_train)

                high_dw = bool(terminated and episode_steps != args.max_episode_steps)
                high_buffer.store(
                    current_high_s,
                    current_high_a,
                    current_high_logprob,
                    high_r_for_train,
                    next_high_s,
                    high_dw,
                    done,
                )

                if high_buffer.count == args.high_batch_size:
                    high_agent.update(high_buffer, total_steps)
                    high_buffer.count = 0

                option_collected_data = 0.0
                option_dropped_data = 0.0
                option_aoi_cost = 0.0
                option_steps = 0

                if not done:
                    current_high_s = next_high_s
                    current_high_a, current_high_logprob = high_agent.choose_action(current_high_s)
                    previous_selected_user_id = current_selected_user_id
                    current_selected_user_id = env.set_high_level_action(current_high_a)
                    ep_high_stats["repeat_checks"] += 1.0
                    ep_high_stats["repeat_targets"] += float(
                        current_selected_user_id == previous_selected_user_id
                    )
                    update_high_decision_stats(high_agent, current_high_s, current_high_a, ep_high_stats)
                    ep_high_decisions += 1

            s_ = env.get_current_state()
            if args.use_state_norm:
                s_ = low_state_norm(s_)

            r_for_train = r
            if args.use_reward_norm:
                r_for_train = low_reward_norm(r_for_train)
            elif args.use_reward_scaling:
                r_for_train = low_reward_scaling(r_for_train)

            dw = bool(terminated and episode_steps != args.max_episode_steps)
            low_buffer.store(s, a, a_logprob, r_for_train, s_, dw, done)
            s = s_
            total_steps += 1

            if low_buffer.count == args.batch_size:
                low_agent.update(low_buffer, total_steps)
                low_buffer.count = 0

            if total_steps % args.evaluate_freq == 0:
                eval_reward, eval_data, eval_aoi = evaluate_policy(
                    args, env_evaluate, low_agent, high_agent, low_state_norm, high_state_norm
                )
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

        writer.add_scalar("High_Level/Decisions", ep_high_decisions, global_step=total_episodes)
        writer.add_scalar("High_Level/Buffer_Count", high_buffer.count, global_step=total_episodes)
        writer.add_scalar("High_Level/Option_Reward", ep_high_reward, global_step=total_episodes)
        writer.add_scalar("High_Level/Option_Data_Collected", ep_high_option_data, global_step=total_episodes)
        writer.add_scalar("High_Level/Option_Dropped_Data", ep_high_option_drop, global_step=total_episodes)
        writer.add_scalar("High_Level/Option_AoI_Cost", ep_high_option_aoi, global_step=total_episodes)
        writer.add_scalar(
            "High_Level/Entropy",
            ep_high_stats["entropy_sum"] / max(ep_high_decisions, 1),
            global_step=total_episodes,
        )
        writer.add_scalar(
            "High_Level/Repeat_Target_Rate",
            ep_high_stats["repeat_targets"] / max(ep_high_stats["repeat_checks"], 1.0),
            global_step=total_episodes,
        )
        for rank in range(args.high_action_dim):
            action_count = ep_high_stats["counts"][rank]
            option_count = ep_high_stats["option_counts"][rank]
            writer.add_scalar(
                f"High_Level/Action_Rank_{rank}_Rate",
                action_count / max(ep_high_decisions, 1),
                global_step=total_episodes,
            )
            writer.add_scalar(
                f"High_Level/Action_Rank_{rank}_Prob",
                ep_high_stats["prob_sums"][rank] / max(ep_high_decisions, 1),
                global_step=total_episodes,
            )
            writer.add_scalar(
                f"High_Level/Option_Data_By_Rank_{rank}",
                ep_high_stats["option_data"][rank] / max(option_count, 1.0),
                global_step=total_episodes,
            )
            writer.add_scalar(
                f"High_Level/Option_AoI_By_Rank_{rank}",
                ep_high_stats["option_aoi"][rank] / max(option_count, 1.0),
                global_step=total_episodes,
            )

        if ep_reward > best_episode_reward:
            best_episode_reward = ep_reward
            with open(env_save_path, "wb") as f:
                pickle.dump(copy.deepcopy(env), f)
            torch.save(low_agent.actor.state_dict(), low_model_save_path)
            torch.save(high_agent.actor.state_dict(), high_model_save_path)
            print(
                f"[New Record] Episode {total_episodes}: Reward={best_episode_reward:.2f} "
                f"| Envs & Models Saved to ./results/PPO/"
            )

        if total_episodes % 10 == 0:
            print(
                f"Episode: {total_episodes} | Total Steps: {total_steps} "
                f"| Data: {ep_data:.1f} | Energy: {ep_energy:.2e} | High Decisions: {ep_high_decisions}"
            )

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Learnable AoI-HRL PPO for UAV data collection")
    parser.add_argument("--users_path", type=str, default="./data_train/users_50_v2.txt")
    parser.add_argument("--option_interval", type=int, default=10)
    parser.add_argument("--high_top_k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--high_w_data", type=float, default=0.01)
    parser.add_argument("--high_w_drop", type=float, default=0.0005)
    parser.add_argument("--high_w_aoi", type=float, default=1.0)
    parser.add_argument("--high_switch_penalty", type=float, default=0.2)

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

    parser.add_argument("--high_batch_size", type=int, default=256)
    parser.add_argument("--high_mini_batch_size", type=int, default=32)
    parser.add_argument("--high_hidden_width", type=int, default=64)
    parser.add_argument("--high_lr_a", type=float, default=3e-4)
    parser.add_argument("--high_lr_c", type=float, default=3e-4)
    parser.add_argument("--high_gamma", type=float, default=0.99)
    parser.add_argument("--high_entropy_coef", type=float, default=0.02)
    parser.add_argument("--high_prior_beta", type=float, default=4.0)
    parser.add_argument("--high_residual_scale", type=float, default=0.25)

    args = parser.parse_args()
    main(args, seed=args.seed)
