import copy

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from . import common_functions


class UAV:
    def __init__(self, position):
        self.position = position
        self.trajectory = []

    def update_position(self, dx, dy):
        self.position[0] += dx
        self.position[1] += dy
        self.position[2] = 100.0


class User:
    def __init__(
        self,
        position,
        initial_data,
        radius,
        user_id,
        mean_arrival,
        burst_probability,
        burst_low,
        burst_high,
        capacity,
    ):
        self.position = position
        self.initial_queue_data = min(initial_data, capacity)
        self.queue_data = self.initial_queue_data
        self.amount_data = self.queue_data
        self.radius = radius
        self.user_id = user_id
        self.mean_arrival = mean_arrival
        self.burst_probability = burst_probability
        self.burst_low = burst_low
        self.burst_high = burst_high
        self.capacity = capacity
        self.total_transmitted_data = 0.0
        self.is_active = True
        self.aoi = 1.0


class EnvCore(gym.Env):
    """Single-level AoI-aware UAV data collection environment."""

    def __init__(
        self,
        length=500,
        width=500,
        num_user=50,
        UAV_fixed_z=100,
        delta_t=1,
        users_path="./ENV/users_50_v2.txt",
    ):
        super().__init__()
        self.length = length
        self.width = width
        self.num_user = num_user
        self.UAV_fixed_z = UAV_fixed_z
        self.delta_t = delta_t
        self.users_path = users_path
        self.T = 200
        self.t = 0
        self.rng = np.random.default_rng()

        self.min_x = 0.0
        self.max_x = float(length)
        self.min_y = 0.0
        self.max_y = float(width)
        self.nfz_center = np.array([length / 2, width / 2], dtype=np.float32)
        self.nfz_radius = 50.0

        self.action_space = spaces.Box(
            low=np.array([-20.0, -20.0], dtype=np.float32),
            high=np.array([20.0, 20.0], dtype=np.float32),
            dtype=np.float32,
        )

        # UAV/NFZ: 5 dimensions.
        # Per user: relative x/y, queue ratio, time-to-overflow, active flag,
        # AoI, mean arrival rate, burst risk, and capacity.
        self.uav_feature_dim = 5
        self.user_feature_dim = 9
        self.obs_dim = self.uav_feature_dim + self.num_user * self.user_feature_dim
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

        self.max_capacity = 75000.0
        self.max_mean_arrival = 260.0
        self.max_burst_probability = 0.012
        self.wake_up_ratio = 0.4
        self.sleep_threshold = 100.0
        self.max_theoretical_energy = self.estimate_max_energy()

        # All reward components are normalized before weighting.
        self.data_scale = 10000.0
        self.drop_scale = 10000.0
        self.w_data = 2.0
        self.w_drop = 6.0
        self.w_energy = 0.5
        self.w_aoi = 2.0
        self.w_guide = 0.1
        self.step_penalty = 0.05
        self.nfz_penalty = 20.0

        # This guide is only a dense reward teacher. It never selects actions
        # and is not included in the observation.
        self.guide_user_id = -1
        self.prev_guide_distance = 0.0

        self.Users = []
        self.observation_user_ids = np.arange(self.num_user, dtype=np.int64)
        self.uav = None
        self.initialize_users()

    @staticmethod
    def _traffic_profile(user_id):
        group = user_id % 10
        if group < 2:
            return 210.0, 0.012, 8000, 15000, 45000.0
        if group < 7:
            return 120.0, 0.005, 5000, 10000, 60000.0
        return 55.0, 0.002, 2000, 5000, 75000.0

    def initialize_users(self):
        try:
            with open(self.users_path, "r") as file:
                for i, line in enumerate(file):
                    if not line.strip():
                        continue
                    values = line.split()
                    profile = self._traffic_profile(i)
                    self.Users.append(
                        User(
                            position=np.array(
                                [float(values[0]), float(values[1]), 0.0],
                                dtype=np.float32,
                            ),
                            initial_data=float(values[2]),
                            radius=50.0,
                            user_id=i,
                            mean_arrival=profile[0],
                            burst_probability=profile[1],
                            burst_low=profile[2],
                            burst_high=profile[3],
                            capacity=profile[4],
                        )
                    )
                    if len(self.Users) >= self.num_user:
                        break
        except FileNotFoundError as exc:
            raise ValueError(f"用户数据文件未找到: {self.users_path}") from exc

        if len(self.Users) != self.num_user:
            raise ValueError(
                f"用户数量不匹配: 期望 {self.num_user}, 实际读取 {len(self.Users)}"
            )

    def estimate_max_energy(self):
        return common_functions.calculate_uav_energy_consumption(
            20.0, 20.0, 0.0, self.delta_t
        )

    @staticmethod
    def _remaining_data(user):
        return user.queue_data

    def _active_mean_aoi(self):
        values = [user.aoi for user in self.Users if user.is_active]
        return float(np.mean(values)) if values else 0.0

    def _guide_candidates(self):
        candidates = []
        uav_pos = self.uav.position[:2]
        for user in self.Users:
            distance = np.linalg.norm(uav_pos - user.position[:2])
            score = -1.0
            if user.is_active and user.queue_data > 0.0:
                urgency = user.queue_data / user.capacity
                score = urgency * user.capacity / (distance + 80.0)
            candidates.append((score, distance, user))
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates

    def _guide_user(self):
        if self.guide_user_id < 0:
            return None
        return self.Users[self.guide_user_id]

    def _reset_guide(self):
        candidates = self._guide_candidates()
        if candidates and candidates[0][0] > 0.0:
            self.guide_user_id = candidates[0][2].user_id
            self.prev_guide_distance = float(candidates[0][1])
        else:
            self.guide_user_id = -1
            self.prev_guide_distance = 0.0

    def _update_guide(self):
        candidates = self._guide_candidates()
        current = self._guide_user()
        if current is None or not current.is_active:
            self._reset_guide()
            return

        current_score = next(
            (
                score
                for score, _, user in candidates
                if user.user_id == current.user_id
            ),
            -1.0,
        )
        if candidates and candidates[0][0] > current_score * 1.2:
            self.guide_user_id = candidates[0][2].user_id
            self.prev_guide_distance = float(candidates[0][1])
        else:
            self.prev_guide_distance = float(
                np.linalg.norm(self.uav.position[:2] - current.position[:2])
            )

    def get_observation_active_mask(self):
        return np.array(
            [
                1.0 if self.Users[user_id].is_active else 0.0
                for user_id in self.observation_user_ids
            ],
            dtype=np.float32,
        )

    def get_current_state(self):
        uav_pos = self.uav.position[:2]
        obs = [
            (uav_pos[0] / self.length) * 2.0 - 1.0,
            (uav_pos[1] / self.width) * 2.0 - 1.0,
            (self.nfz_center[0] - uav_pos[0]) / self.length,
            (self.nfz_center[1] - uav_pos[1]) / self.width,
            (
                np.linalg.norm(self.nfz_center - uav_pos) - self.nfz_radius
            )
            / self.length,
        ]

        for user_id in self.observation_user_ids:
            user = self.Users[user_id]
            remaining_capacity = user.capacity - user.queue_data
            time_to_overflow = remaining_capacity / max(user.mean_arrival, 1.0)
            obs.extend(
                [
                    (user.position[0] - uav_pos[0]) / self.length,
                    (user.position[1] - uav_pos[1]) / self.width,
                    min(user.queue_data / user.capacity, 1.0),
                    min(time_to_overflow / self.T, 1.0),
                    1.0 if user.is_active else 0.0,
                    min(user.aoi / self.T, 1.0),
                    user.mean_arrival / self.max_mean_arrival,
                    user.burst_probability / self.max_burst_probability,
                    user.capacity / self.max_capacity,
                ]
            )
        return np.asarray(obs, dtype=np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        initial_position = [
            (self.max_x - self.min_x) * 0.1,
            (self.max_y - self.min_y) * 0.1,
            self.UAV_fixed_z,
        ]
        self.uav = UAV(np.array(initial_position, dtype=np.float32))
        self.uav.trajectory.append(initial_position.copy())

        for user in self.Users:
            user.queue_data = user.initial_queue_data
            user.amount_data = user.queue_data
            user.total_transmitted_data = 0.0
            user.aoi = 1.0
            user.is_active = user.queue_data > user.capacity * self.wake_up_ratio

        self.observation_user_ids = np.arange(
            self.num_user,
            dtype=np.int64,
        )
        self._reset_guide()
        return self.get_current_state(), {}

    def _generate_user_data(self):
        generated_data = 0.0
        dropped_data = 0.0
        for user in self.Users:
            arrival = float(self.rng.poisson(user.mean_arrival))
            if self.rng.random() < user.burst_probability:
                arrival += float(
                    self.rng.integers(user.burst_low, user.burst_high)
                )
            generated_data += arrival
            accepted = min(arrival, user.capacity - user.queue_data)
            user.queue_data += accepted
            user.amount_data = user.queue_data
            dropped_data += arrival - accepted
        return generated_data, dropped_data

    def _collect_data(self):
        users_in_range = []
        for user in self.Users:
            distance = np.linalg.norm(self.uav.position[:2] - user.position[:2])
            if (
                distance <= user.radius
                and user.is_active
                and user.queue_data > 0.0
            ):
                users_in_range.append(user)

        collected_data = 0.0
        if users_in_range:
            band_fraction = 1.0 / len(users_in_range)
            for user in users_in_range:
                rate = common_functions.calculate_rate_device_UAV(
                    user, self.uav, band_fraction, 0.1
                )
                uploaded = min(rate * self.delta_t, user.queue_data)
                user.queue_data -= uploaded
                user.amount_data = user.queue_data
                user.total_transmitted_data += uploaded
                user.aoi = 1.0
                collected_data += uploaded
        return collected_data

    def _update_activity(self):
        for user in self.Users:
            if user.is_active and user.queue_data < self.sleep_threshold:
                user.is_active = False
            elif (
                not user.is_active
                and user.queue_data > user.capacity * self.wake_up_ratio
            ):
                user.is_active = True

    def step(self, action):
        dx, dy = float(action[0]), float(action[1])
        previous_position = self.uav.position.copy()
        self.uav.update_position(dx, dy)
        self.uav.position[0] = np.clip(
            self.uav.position[0], self.min_x, self.max_x
        )
        self.uav.position[1] = np.clip(
            self.uav.position[1], self.min_y, self.max_y
        )

        in_nfz = (
            np.linalg.norm(self.uav.position[:2] - self.nfz_center)
            < self.nfz_radius
        )
        if in_nfz:
            self.uav.position = previous_position

        self.uav.trajectory.append(copy.deepcopy(self.uav.position))

        energy = common_functions.calculate_uav_energy_consumption(
            dx, dy, 0.0, self.delta_t
        )
        normalized_energy = energy / (self.max_theoretical_energy + 1e-9)

        guide_improvement = 0.0
        guide_user = self._guide_user()
        if guide_user is not None and guide_user.is_active:
            current_distance = float(
                np.linalg.norm(self.uav.position[:2] - guide_user.position[:2])
            )
            if current_distance > guide_user.radius:
                guide_improvement = max(
                    self.prev_guide_distance - current_distance, 0.0
                )

        for user in self.Users:
            user.aoi = min(user.aoi + 1.0, float(self.T))

        collected_data = self._collect_data()
        generated_data, dropped_data = self._generate_user_data()
        self._update_activity()

        active_mean_aoi = self._active_mean_aoi()
        normalized_aoi = active_mean_aoi / self.T
        normalized_data = collected_data / self.data_scale
        normalized_drop = dropped_data / self.drop_scale
        normalized_guide = min(guide_improvement / 20.0, 1.0)
        guide_reward = (
            self.w_guide * normalized_guide
            if collected_data <= 0.0
            else 0.0
        )

        reward = (
            self.w_data * normalized_data
            - self.w_drop * normalized_drop
            - self.w_energy * normalized_energy
            - self.w_aoi * normalized_aoi
            + guide_reward
            - self.step_penalty
        )

        if in_nfz:
            reward -= self.nfz_penalty

        self._update_guide()
        self.t += 1
        terminated = False
        truncated = self.t >= self.T

        info = {
            "collected_data": collected_data,
            "generated_data": generated_data,
            "dropped_data": dropped_data,
            "energy": energy,
            "in_nfz": in_nfz,
            "active_mean_aoi": active_mean_aoi,
            "guide_distance_reward": guide_reward,
        }
        return self.get_current_state(), float(reward), terminated, truncated, info
