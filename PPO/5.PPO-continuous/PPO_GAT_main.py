import argparse
import copy
import os
import pickle
from datetime import datetime

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from ENV.env_uav_gat_hrl import EnvCore
from normalization import Normalization, RewardScaling
from ppo_gat_continuous import PPO_GAT
from replaybuffer import ReplayBuffer


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("true", "1", "yes"):
        return True
    if value.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def attention_statistics(attention, env):
    attention = np.asarray(attention, dtype=np.float64)
    entropy = -np.sum(attention * np.log(attention + 1e-12))
    normalized_entropy = entropy / np.log(len(attention))
    active_mask = env.get_observation_active_mask().astype(np.float64)
    return {
        "entropy": float(normalized_entropy),
        "max_weight": float(np.max(attention)),
        "active_mass": float(np.sum(attention * active_mask)),
    }


def evaluate_policy(args, env, agent, state_norm):
    episode_rewards = []
    episode_data = []
    episode_generated_data = []
    episode_dropped_data = []
    episode_energy = []
    episode_steps = []
    episode_nfz = []
    episode_aoi = []
    episode_completed = []

    for evaluate_index in range(args.evaluate_times):
        state, _ = env.reset(seed=args.seed + 1000 + evaluate_index)
        if args.use_state_norm:
            state = state_norm(state, update=False)

        total_reward = 0.0
        total_data = 0.0
        total_generated_data = 0.0
        total_dropped_data = 0.0
        total_energy = 0.0
        steps = 0
        nfz_steps = 0
        final_aoi = 0.0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = agent.evaluate(state)
            if args.policy_dist == "Beta":
                action = 2.0 * (action - 0.5) * args.max_action

            next_state, reward, terminated, truncated, info = env.step(action)
            if args.use_state_norm:
                next_state = state_norm(next_state, update=False)

            state = next_state
            steps += 1
            total_reward += reward
            total_data += info["collected_data"]
            total_generated_data += info["generated_data"]
            total_dropped_data += info["dropped_data"]
            total_energy += info["energy"]
            nfz_steps += int(info["in_nfz"])
            final_aoi = info["active_mean_aoi"]

        episode_rewards.append(total_reward)
        episode_data.append(total_data)
        episode_generated_data.append(total_generated_data)
        episode_dropped_data.append(total_dropped_data)
        episode_energy.append(total_energy)
        episode_steps.append(steps)
        episode_nfz.append(nfz_steps)
        episode_aoi.append(final_aoi)
        episode_completed.append(float(truncated and not terminated))

    return {
        "reward": float(np.mean(episode_rewards)),
        "data": float(np.mean(episode_data)),
        "drop_rate": float(
            np.mean(
                np.asarray(episode_dropped_data)
                / (np.asarray(episode_generated_data) + 1e-9)
            )
        ),
        "energy_per_step": float(
            np.mean(
                np.asarray(episode_energy)
                / np.maximum(np.asarray(episode_steps), 1)
            )
        ),
        "data_per_energy": float(
            np.mean(
                np.asarray(episode_data)
                / (np.asarray(episode_energy) + 1e-9)
            )
        ),
        "completion_rate": float(np.mean(episode_completed)),
        "nfz": float(np.mean(episode_nfz)),
        "aoi": float(np.mean(episode_aoi)),
    }


def main(args):
    env = EnvCore(users_path=args.users_path)
    evaluate_env = EnvCore(users_path=args.users_path)

    env.reset(seed=args.seed)
    evaluate_env.reset(seed=args.seed)
    env.action_space.seed(args.seed)
    evaluate_env.action_space.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    args.max_action = float(env.action_space.high[0])
    args.max_episode_steps = env.T
    args.num_user = env.num_user
    args.uav_feature_dim = env.uav_feature_dim
    args.user_feature_dim = env.user_feature_dim

    replay_buffer = ReplayBuffer(args)
    agent = PPO_GAT(args)
    state_norm = Normalization(shape=args.state_dim)

    if args.use_reward_norm:
        reward_norm = Normalization(shape=1)
    elif args.use_reward_scaling:
        reward_scaling = RewardScaling(shape=1, gamma=args.gamma)

    timestamp = datetime.now().strftime("%m%d%H%M")
    run_name = f"{timestamp}_GAT_PPO"
    writer = SummaryWriter(log_dir=os.path.join("runs", run_name))

    save_dir = os.path.join(".", "results", "GAT_PPO")
    os.makedirs(save_dir, exist_ok=True)
    env_save_path = os.path.join(save_dir, f"best_env_{run_name}.pkl")
    actor_save_path = os.path.join(save_dir, f"best_actor_{run_name}.pth")
    critic_save_path = os.path.join(save_dir, f"best_critic_{run_name}.pth")

    total_steps = 0
    total_episodes = 0
    best_episode_reward = -float("inf")

    while total_steps < args.max_train_steps:
        state, _ = env.reset()
        if args.use_state_norm:
            state = state_norm(state)
        if args.use_reward_scaling:
            reward_scaling.reset()

        total_episodes += 1
        episode_steps = 0
        episode_reward = 0.0
        episode_data = 0.0
        episode_generated_data = 0.0
        episode_dropped_data = 0.0
        episode_energy = 0.0
        episode_nfz = 0
        episode_guide_reward = 0.0
        final_aoi = 0.0
        attention_entropy = []
        attention_max_weight = []
        attention_active_mass = []

        terminated = False
        truncated = False

        while (
            not (terminated or truncated)
            and total_steps < args.max_train_steps
        ):
            episode_steps += 1

            attention = agent.get_attention(state)
            attention_stats = attention_statistics(attention, env)
            attention_entropy.append(attention_stats["entropy"])
            attention_max_weight.append(attention_stats["max_weight"])
            attention_active_mass.append(attention_stats["active_mass"])

            action, action_logprob = agent.choose_action(state)
            if args.policy_dist == "Beta":
                env_action = 2.0 * (action - 0.5) * args.max_action
            else:
                env_action = action

            next_state, reward, terminated, truncated, info = env.step(
                env_action
            )
            done = terminated or truncated

            episode_reward += reward
            episode_data += info["collected_data"]
            episode_generated_data += info["generated_data"]
            episode_dropped_data += info["dropped_data"]
            episode_energy += info["energy"]
            episode_nfz += int(info["in_nfz"])
            episode_guide_reward += info["guide_distance_reward"]
            final_aoi = info["active_mean_aoi"]

            if args.use_state_norm:
                next_state = state_norm(next_state)

            training_reward = reward
            if args.use_reward_norm:
                training_reward = reward_norm(training_reward)
            elif args.use_reward_scaling:
                training_reward = reward_scaling(training_reward)

            replay_buffer.store(
                state,
                action,
                action_logprob,
                training_reward,
                next_state,
                terminated,
                done,
            )
            state = next_state
            total_steps += 1

            if replay_buffer.count == args.batch_size:
                agent.update(replay_buffer, total_steps)
                replay_buffer.count = 0

            if total_steps % args.evaluate_freq == 0:
                result = evaluate_policy(
                    args,
                    evaluate_env,
                    agent,
                    state_norm,
                )
                writer.add_scalar(
                    "Evaluate/Reward_vs_Step",
                    result["reward"],
                    total_steps,
                )
                writer.add_scalar(
                    "Evaluate/Data_Collected_vs_Step",
                    result["data"],
                    total_steps,
                )
                writer.add_scalar(
                    "Evaluate/NFZ_Violation_vs_Step",
                    result["nfz"],
                    total_steps,
                )
                writer.add_scalar(
                    "Evaluate/Active_Mean_AoI_vs_Step",
                    result["aoi"],
                    total_steps,
                )
                writer.add_scalar(
                    "Evaluate/Drop_Rate_vs_Step",
                    result["drop_rate"],
                    total_steps,
                )
                writer.add_scalar(
                    "Evaluate/Energy_Per_Step_vs_Step",
                    result["energy_per_step"],
                    total_steps,
                )
                writer.add_scalar(
                    "Evaluate/Data_Per_Energy_vs_Step",
                    result["data_per_energy"],
                    total_steps,
                )
                writer.add_scalar(
                    "Evaluate/Completion_Rate_vs_Step",
                    result["completion_rate"],
                    total_steps,
                )

        writer.add_scalar(
            "Episode_Total/Reward",
            episode_reward,
            total_episodes,
        )
        writer.add_scalar(
            "Episode_Total/Data_Collected",
            episode_data,
            total_episodes,
        )
        writer.add_scalar(
            "Episode_Total/Dropped_Data",
            episode_dropped_data,
            total_episodes,
        )
        writer.add_scalar(
            "Episode_Total/Generated_Data",
            episode_generated_data,
            total_episodes,
        )
        writer.add_scalar(
            "Episode_Total/Energy_Consumption",
            episode_energy,
            total_episodes,
        )
        writer.add_scalar(
            "Episode_Total/NFZ_Violation_Steps",
            episode_nfz,
            total_episodes,
        )
        writer.add_scalar(
            "Episode_Total/Active_Mean_AoI",
            final_aoi,
            total_episodes,
        )
        writer.add_scalar(
            "Episode_Total/Guide_Distance_Reward",
            episode_guide_reward,
            total_episodes,
        )
        writer.add_scalar(
            "Efficiency/Energy_Per_Step",
            episode_energy / max(episode_steps, 1),
            total_episodes,
        )
        writer.add_scalar(
            "Efficiency/Data_Per_Energy",
            episode_data / (episode_energy + 1e-9),
            total_episodes,
        )
        writer.add_scalar(
            "Efficiency/Drop_Rate",
            episode_dropped_data / (episode_generated_data + 1e-9),
            total_episodes,
        )
        writer.add_scalar(
            "Efficiency/Completed_Full_Episode",
            float(truncated and not terminated),
            total_episodes,
        )
        writer.add_scalar(
            "Attention/Normalized_Entropy",
            float(np.mean(attention_entropy)),
            total_episodes,
        )
        writer.add_scalar(
            "Attention/Max_User_Weight",
            float(np.mean(attention_max_weight)),
            total_episodes,
        )
        writer.add_scalar(
            "Attention/Active_User_Mass",
            float(np.mean(attention_active_mass)),
            total_episodes,
        )
        if args.policy_dist == "Gaussian":
            action_std = agent.actor.get_action_std()
            writer.add_scalar(
                "Policy/Action_Std_X",
                float(action_std[0]),
                total_episodes,
            )
            writer.add_scalar(
                "Policy/Action_Std_Y",
                float(action_std[1]),
                total_episodes,
            )

        if episode_reward > best_episode_reward:
            best_episode_reward = episode_reward
            with open(env_save_path, "wb") as file:
                pickle.dump(copy.deepcopy(env), file)
            torch.save(agent.actor.state_dict(), actor_save_path)
            torch.save(agent.critic.state_dict(), critic_save_path)
            print(
                f"[New Record] Episode {total_episodes}: "
                f"Reward={episode_reward:.2f}, Data={episode_data:.1f}"
            )

        if total_episodes % 10 == 0:
            print(
                f"Episode: {total_episodes} | Steps: {total_steps} | "
                f"Reward: {episode_reward:.1f} | Data: {episode_data:.1f} | "
                f"AoI: {final_aoi:.1f} | NFZ: {episode_nfz}"
            )

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Single-level Graph Attention PPO for UAV data collection"
    )
    parser.add_argument(
        "--users_path",
        type=str,
        default="./ENV/users_50_v2.txt",
    )
    parser.add_argument("--max_train_steps", type=int, default=int(1e6))
    parser.add_argument("--evaluate_freq", type=int, default=5000)
    parser.add_argument("--evaluate_times", type=int, default=3)
    parser.add_argument(
        "--policy_dist",
        type=str,
        default="Gaussian",
        choices=["Gaussian", "Beta"],
    )
    parser.add_argument("--init_action_std", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--mini_batch_size", type=int, default=64)
    parser.add_argument("--lr_a", type=float, default=3e-4)
    parser.add_argument("--lr_c", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lamda", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--K_epochs", type=int, default=10)
    parser.add_argument("--entropy_coef", type=float, default=0.01)

    parser.add_argument("--gat_hidden_dim", type=int, default=128)
    parser.add_argument("--gat_heads", type=int, default=4)
    parser.add_argument("--gat_layers", type=int, default=2)

    parser.add_argument("--use_adv_norm", type=str_to_bool, default=True)
    parser.add_argument("--use_state_norm", type=str_to_bool, default=True)
    parser.add_argument("--use_reward_norm", type=str_to_bool, default=False)
    parser.add_argument("--use_reward_scaling", type=str_to_bool, default=True)
    parser.add_argument("--use_lr_decay", type=str_to_bool, default=True)
    parser.add_argument("--use_grad_clip", type=str_to_bool, default=True)
    parser.add_argument(
        "--use_orthogonal_init",
        type=str_to_bool,
        default=True,
    )
    parser.add_argument("--set_adam_eps", type=str_to_bool, default=True)
    parser.add_argument("--seed", type=int, default=10)

    main(parser.parse_args())
