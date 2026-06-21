import copy

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from . import common_functions


class UAV(object):
    def __init__(self, position):
        self.position = position
        self.trajectory = []

    def update_position(self, dx, dy):
        self.position[0] += dx
        self.position[1] += dy
        self.position[2] = 100.0
        return self.position


class User(object):
    def __init__(self, position, amount_data, radius, user_id):
        self.position = position
        self.initial_amount_data = amount_data
        self.amount_data = amount_data
        self.radius = radius
        self.user_id = user_id
        self.total_transmitted_data = 0.0
        self.is_active = True
        self.aoi = 0.0
        self.served_times = 0


class EnvCore(gym.Env):
    """
    Learnable AoI-HRL environment.

    High-level policy: discrete action, chooses one target from top-k candidates.
    Low-level policy: continuous PPO action, controls dx and dy.
    """

    def __init__(
            self,
            length=500,
            width=500,
            num_user=50,
            UAV_fixed_z=100,
            delta_t=1,
            users_path="./data_train/users_50_v2.txt",
            option_interval=10,
            high_top_k=5):
        super(EnvCore, self).__init__()

        self.length = length
        self.width = width
        self.max_x = length
        self.min_x = 0
        self.max_y = width
        self.min_y = 0
        self.num_user = num_user
        self.UAV_fixed_z = UAV_fixed_z
        self.delta_t = delta_t
        self.users_path = users_path
        self.T = 200
        self.t = 0

        self.nfz_center = np.array([length / 2, width / 2], dtype=np.float32)
        self.nfz_radius = 50.0

        self.action_space = spaces.Box(
            low=np.array([-20.0, -20.0], dtype=np.float32),
            high=np.array([20.0, 20.0], dtype=np.float32),
            dtype=np.float32
        )

        self.obs_dim = 12
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)

        self.high_top_k = high_top_k
        self.high_user_dim = 6
        self.high_state_dim = 2 + 3 + self.high_top_k * self.high_user_dim
        self.high_action_dim = self.high_top_k
        self.high_action_space = spaces.Discrete(self.high_action_dim)

        self.min_data_increase = 0
        self.max_data_increase = 500
        self.max_capacity = 60000.0
        self.wake_up_threshold = 25000.0
        self.sleep_threshold = 100.0
        self.max_theoretical_energy = self.estimate_max_energy()

        self.w_data = 0.01
        self.w_drop = 0.0005
        self.w_dist = 0.30
        self.w_energy = 1.0
        self.w_aoi = 0.50
        self.step_penalty = 0.5
        self.nfz_penalty = 500.0

        self.option_interval = option_interval
        self.steps_since_target_switch = 0
        self.target_user_id = -1
        self.prev_target_dist = 0.0
        self.target_switch_count = 0
        self.selected_target_history = []
        self.high_level_candidates = []

        self.initialize_users()
        self.uav = None

    def initialize_users(self):
        self.Users = []
        try:
            with open(self.users_path, "r") as f:
                for i, line in enumerate(f):
                    if not line.strip():
                        continue
                    arr = line.split()
                    user = User(
                        position=np.array([float(arr[0]), float(arr[1]), 0.0], dtype=np.float32),
                        amount_data=min(float(arr[2]), self.max_capacity),
                        radius=50.0,
                        user_id=i
                    )
                    self.Users.append(user)
                    if len(self.Users) >= self.num_user:
                        break
        except FileNotFoundError:
            raise ValueError(f"User data file not found: {self.users_path}")

    def estimate_max_energy(self):
        return common_functions.calculate_uav_energy_consumption(20.0, 20.0, 0, self.delta_t)

    def _remaining_data(self, user):
        return max(user.amount_data - user.total_transmitted_data, 0.0)

    def _overflow_risk(self, user):
        rem_capacity = max(self.max_capacity - user.amount_data, 0.0)
        avg_increase = 0.5 * (self.min_data_increase + self.max_data_increase)
        if avg_increase <= 0:
            return 0.0
        tto_steps = rem_capacity / avg_increase
        return 1.0 - min(tto_steps / self.T, 1.0)

    def _candidate_score(self, user):
        if not user.is_active or self._remaining_data(user) <= 0:
            return -1e9

        uav_pos = self.uav.position[:2]
        dist = np.linalg.norm(uav_pos - user.position[:2])
        norm_dist = min(dist / np.sqrt(self.length ** 2 + self.width ** 2), 1.0)
        norm_data = min(self._remaining_data(user) / self.max_capacity, 1.0)
        norm_aoi = min(user.aoi / self.T, 1.0)
        overflow_risk = self._overflow_risk(user)
        score = (
            1.0 * norm_data
            + 1.2 * norm_aoi
            + 1.5 * overflow_risk
            - 0.8 * norm_dist
        )
        if user.user_id == self.target_user_id:
            score += 0.15
        return score

    def _get_high_level_candidates(self):
        candidates = [u for u in self.Users if u.is_active and self._remaining_data(u) > 0]
        candidates.sort(key=self._candidate_score, reverse=True)
        candidates = candidates[:self.high_top_k]
        while len(candidates) < self.high_top_k:
            candidates.append(None)
        self.high_level_candidates = candidates
        return candidates

    def _user_feature(self, user):
        if user is None:
            return [0.0] * self.high_user_dim

        uav_pos = self.uav.position[:2]
        rel_x = (user.position[0] - uav_pos[0]) / self.length
        rel_y = (user.position[1] - uav_pos[1]) / self.width
        norm_data = min(self._remaining_data(user) / self.max_capacity, 1.0)
        norm_aoi = min(user.aoi / self.T, 1.0)
        overflow_risk = self._overflow_risk(user)
        active_flag = 1.0 if user.is_active else 0.0
        return [rel_x, rel_y, norm_data, norm_aoi, overflow_risk, active_flag]

    def get_high_level_state(self):
        obs = []
        uav_pos = self.uav.position[:2]
        obs.extend([
            (uav_pos[0] / self.length) * 2 - 1.0,
            (uav_pos[1] / self.width) * 2 - 1.0,
        ])

        rel_nfz_x = (self.nfz_center[0] - uav_pos[0]) / self.length
        rel_nfz_y = (self.nfz_center[1] - uav_pos[1]) / self.width
        dist_to_nfz = np.linalg.norm(self.nfz_center - uav_pos)
        dist_to_nfz_edge = (dist_to_nfz - self.nfz_radius) / self.length
        obs.extend([rel_nfz_x, rel_nfz_y, dist_to_nfz_edge])

        for user in self._get_high_level_candidates():
            obs.extend(self._user_feature(user))

        return np.array(obs, dtype=np.float32)

    def set_high_level_action(self, action_index):
        if not self.high_level_candidates:
            self._get_high_level_candidates()

        action_index = int(np.clip(action_index, 0, self.high_top_k - 1))
        selected_user = self.high_level_candidates[action_index]
        old_target_id = self.target_user_id

        if selected_user is None:
            valid_candidates = [u for u in self.high_level_candidates if u is not None]
            selected_user = valid_candidates[0] if valid_candidates else None

        if selected_user is None:
            self.target_user_id = -1
            self.prev_target_dist = 0.0
            selected_id = -1
        else:
            self.target_user_id = selected_user.user_id
            self.prev_target_dist = np.linalg.norm(self.uav.position[:2] - selected_user.position[:2])
            selected_id = selected_user.user_id

        self.steps_since_target_switch = 0
        self.selected_target_history.append(selected_id)
        if old_target_id != self.target_user_id:
            self.target_switch_count += 1
        return selected_id

    def high_level_decision_due(self):
        target_user = next((u for u in self.Users if u.user_id == self.target_user_id), None)
        target_invalid = target_user is None or not target_user.is_active or self._remaining_data(target_user) <= 0
        return self.steps_since_target_switch >= self.option_interval or target_invalid

    def _update_user_active_flags(self):
        for user in self.Users:
            rem_data = self._remaining_data(user)
            if user.is_active and rem_data < self.sleep_threshold:
                user.is_active = False
            elif not user.is_active and rem_data > self.wake_up_threshold:
                user.is_active = True

    def update_user_data(self):
        dropped_data_this_step = 0.0
        for user in self.Users:
            data_increase = np.random.randint(self.min_data_increase, self.max_data_increase)
            new_amount = user.amount_data + data_increase

            if new_amount > self.max_capacity:
                dropped_data_this_step += new_amount - self.max_capacity
                user.amount_data = self.max_capacity
            else:
                user.amount_data = new_amount
        return dropped_data_this_step

    def get_current_state(self):
        obs = []
        uav_pos = self.uav.position[:2]
        obs.extend([
            (uav_pos[0] / self.length) * 2 - 1.0,
            (uav_pos[1] / self.width) * 2 - 1.0,
        ])

        target_user = next((u for u in self.Users if u.user_id == self.target_user_id), None)
        if target_user is not None:
            rel_x = (target_user.position[0] - uav_pos[0]) / self.length
            rel_y = (target_user.position[1] - uav_pos[1]) / self.width
            norm_data = min(self._remaining_data(target_user) / self.max_capacity, 1.0)
            norm_aoi = min(target_user.aoi / self.T, 1.0)
            overflow_risk = self._overflow_risk(target_user)
            active_flag = 1.0 if target_user.is_active else 0.0
            obs.extend([rel_x, rel_y, norm_data, norm_aoi, overflow_risk, active_flag])
        else:
            obs.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        rel_nfz_x = (self.nfz_center[0] - uav_pos[0]) / self.length
        rel_nfz_y = (self.nfz_center[1] - uav_pos[1]) / self.width
        dist_to_nfz = np.linalg.norm(self.nfz_center - uav_pos)
        dist_to_nfz_edge = (dist_to_nfz - self.nfz_radius) / self.length
        obs.extend([rel_nfz_x, rel_nfz_y, dist_to_nfz_edge])

        phase = 1.0 - min(self.steps_since_target_switch / max(self.option_interval, 1), 1.0)
        obs.append(phase)
        return np.array(obs, dtype=np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.t = 0

        init_pos = [(self.max_x - self.min_x) * 0.1, (self.max_y - self.min_y) * 0.1, self.UAV_fixed_z]
        self.uav = UAV(position=np.array(init_pos, dtype=np.float32))
        self.uav.trajectory.append(init_pos.copy())

        for user in self.Users:
            user.amount_data = user.initial_amount_data
            user.total_transmitted_data = 0.0
            user.aoi = 0.0
            user.served_times = 0
            user.is_active = self._remaining_data(user) > self.wake_up_threshold

        self.target_user_id = -1
        self.prev_target_dist = 0.0
        self.steps_since_target_switch = self.option_interval
        self.target_switch_count = 0
        self.selected_target_history = []
        self.high_level_candidates = []
        return self.get_current_state(), {}

    def step(self, action):
        dx, dy = action[0], action[1]
        self.uav.update_position(dx, dy)
        p = self.uav.position
        p[0] = np.clip(p[0], self.min_x, self.max_x)
        p[1] = np.clip(p[1], self.min_y, self.max_y)
        self.uav.trajectory.append(copy.deepcopy(self.uav.position))

        energy = common_functions.calculate_uav_energy_consumption(dx, dy, 0, self.delta_t)
        normalized_energy = energy / (self.max_theoretical_energy + 1e-9)

        collected_data_this_step = 0.0
        tx_power = 0.1
        users_in_range = []
        served_user_ids = set()

        for user in self.Users:
            dist_2d = np.linalg.norm(self.uav.position[:2] - user.position[:2])
            if dist_2d <= user.radius and user.is_active and self._remaining_data(user) > 0:
                users_in_range.append(user)

        if users_in_range:
            band_fraction = 1.0 / len(users_in_range)
            for user in users_in_range:
                rate = common_functions.calculate_rate_device_UAV(user, self.uav, band_fraction, tx_power)
                uploaded = rate * self.delta_t
                actual_upload = min(uploaded, self._remaining_data(user))
                user.total_transmitted_data += actual_upload
                collected_data_this_step += actual_upload
                if actual_upload > 0:
                    served_user_ids.add(user.user_id)
                    user.served_times += 1

        for user in self.Users:
            user.aoi = 0.0 if user.user_id in served_user_ids else user.aoi + 1.0

        self._update_user_active_flags()

        reward = 0.0
        data_reward = 0.0
        distance_reward = 0.0
        drop_penalty = 0.0
        aoi_penalty = 0.0
        nfz_penalty = 0.0
        energy_penalty = 0.0
        step_penalty = self.step_penalty

        if collected_data_this_step > 0:
            data_reward = self.w_data * collected_data_this_step
            reward += data_reward
        else:
            target_user = next((u for u in self.Users if u.user_id == self.target_user_id), None)
            if target_user and target_user.is_active:
                curr_target_dist = np.linalg.norm(self.uav.position[:2] - target_user.position[:2])
                if curr_target_dist > target_user.radius:
                    distance_improvement = max(self.prev_target_dist - curr_target_dist, 0.0)
                    distance_reward = self.w_dist * distance_improvement
                    reward += distance_reward

        dropped_data = self.update_user_data()
        if dropped_data > 0:
            drop_penalty = dropped_data * self.w_drop
            reward -= drop_penalty

        aoi_cost = 0.0
        active_users = [u for u in self.Users if u.is_active]
        if active_users:
            urgencies = [
                min(u.aoi / self.T, 1.0) * min(self._remaining_data(u) / self.max_capacity, 1.0)
                for u in active_users
            ]
            aoi_cost = float(np.mean(urgencies))
            aoi_penalty = self.w_aoi * aoi_cost
            reward -= aoi_penalty

        dist_to_nfz = np.linalg.norm(self.uav.position[:2] - self.nfz_center)
        in_nfz = dist_to_nfz < self.nfz_radius
        if in_nfz:
            nfz_penalty = self.nfz_penalty
            reward -= nfz_penalty
            done = True
        else:
            done = self.t >= self.T

        energy_penalty = self.w_energy * normalized_energy
        reward -= energy_penalty
        reward -= step_penalty

        self.steps_since_target_switch += 1
        self._update_user_active_flags()

        current_target = next((u for u in self.Users if u.user_id == self.target_user_id), None)
        if current_target is not None:
            self.prev_target_dist = np.linalg.norm(self.uav.position[:2] - current_target.position[:2])

        self.t += 1
        truncated = False
        obs = self.get_current_state()
        info = {
            "collected_data": collected_data_this_step,
            "dropped_data": dropped_data,
            "energy": energy,
            "in_nfz": in_nfz,
            "target_user_id": self.target_user_id,
            "target_switch_count": self.target_switch_count,
            "mean_aoi": float(np.mean([u.aoi for u in self.Users])),
            "active_mean_aoi": float(np.mean([u.aoi for u in active_users])) if active_users else 0.0,
            "aoi_cost": aoi_cost,
            "active_users": len(active_users),
            "option_done": self.high_level_decision_due() or done,
            "reward_data": data_reward,
            "reward_distance": distance_reward,
            "penalty_drop": drop_penalty,
            "penalty_aoi": aoi_penalty,
            "penalty_energy": energy_penalty,
            "penalty_nfz": nfz_penalty,
            "penalty_step": step_penalty,
        }
        return obs, float(reward), done, truncated, info
