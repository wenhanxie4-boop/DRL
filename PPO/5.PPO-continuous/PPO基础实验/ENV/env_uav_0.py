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
    def __init__(self, position, amount_data, radius, user_id):
        self.position = position
        self.initial_amount_data = amount_data
        self.amount_data = amount_data
        self.radius = radius
        self.user_id = user_id
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
        # Each user: relative x/y, data ratio, time-to-overflow, active flag, AoI.
        self.uav_feature_dim = 5
        self.user_feature_dim = 6
        self.obs_dim = self.uav_feature_dim + self.num_user * self.user_feature_dim
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

        self.min_data_increase = 0
        self.max_data_increase = 200
        self.max_capacity = 60000.0
        self.wake_up_threshold = 25000.0
        self.sleep_threshold = 100.0
        self.max_theoretical_energy = self.estimate_max_energy()

        self.w_data = 0.01
        self.w_drop = 0.01
        self.w_dist = 0.05
        self.w_energy = 1.0
        self.w_aoi = 2.0
        self.step_penalty = 0.5
        self.nfz_penalty = 500.0

        # The guide is used only for dense reward shaping. It never selects
        # the UAV action and is not included in the observation.
        self.guide_user_id = -1
        self.prev_guide_distance = 0.0

        self.Users = []
        self.uav = None
        self.initialize_users()

    def initialize_users(self):
        try:
            with open(self.users_path, "r") as file:
                for i, line in enumerate(file):
                    if not line.strip():
                        continue
                    values = line.split()
                    user = User(
                        position=np.array(
                            [float(values[0]), float(values[1]), 0.0],
                            dtype=np.float32,
                        ),
                        amount_data=min(float(values[2]), self.max_capacity),
                        radius=50.0,
                        user_id=i,
                    )
                    self.Users.append(user)
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

    def _remaining_data(self, user):
        return max(user.amount_data - user.total_transmitted_data, 0.0)

    def _active_mean_aoi(self):
        active_aoi = [user.aoi for user in self.Users if user.is_active]
        return float(np.mean(active_aoi)) if active_aoi else 0.0

    def _guide_candidates(self):
        candidates = []
        uav_pos = self.uav.position[:2]
        for user in self.Users:
            remaining_data = self._remaining_data(user)
            distance = np.linalg.norm(uav_pos - user.position[:2])
            score = -1.0
            if user.is_active and remaining_data > 0.0:
                score = remaining_data / (distance + 80.0)
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
                np.linalg.norm(
                    self.uav.position[:2] - current.position[:2]
                )
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

        avg_data_increase = max(
            (self.min_data_increase + self.max_data_increase) / 2.0,
            1.0,
        )
        for user in self.Users:
            remaining_data = self._remaining_data(user)
            remaining_capacity = self.max_capacity - remaining_data
            time_to_overflow = remaining_capacity / avg_data_increase

            obs.extend(
                [
                    (user.position[0] - uav_pos[0]) / self.length,
                    (user.position[1] - uav_pos[1]) / self.width,
                    min(remaining_data / self.max_capacity, 1.0),
                    min(time_to_overflow / self.T, 1.0),
                    1.0 if user.is_active else 0.0,
                    min(user.aoi / self.T, 1.0),
                ]
            )

        return np.asarray(obs, dtype=np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.t = 0

        initial_position = [
            (self.max_x - self.min_x) * 0.1,
            (self.max_y - self.min_y) * 0.1,
            self.UAV_fixed_z,
        ]
        self.uav = UAV(np.array(initial_position, dtype=np.float32))
        self.uav.trajectory.append(initial_position.copy())

        for user in self.Users:
            user.amount_data = user.initial_amount_data
            user.total_transmitted_data = 0.0
            user.aoi = 1.0
            user.is_active = self._remaining_data(user) > self.wake_up_threshold

        self._reset_guide()
        return self.get_current_state(), {}

    def _update_user_data(self):
        dropped_data = 0.0
        for user in self.Users:
            data_increase = np.random.randint(
                self.min_data_increase,
                self.max_data_increase,
            )
            if user.is_active and np.random.rand() < 0.005:
                data_increase += np.random.randint(5000, 10000)

            new_amount = user.amount_data + data_increase
            if new_amount > self.max_capacity:
                dropped_data += new_amount - self.max_capacity
                user.amount_data = self.max_capacity
            else:
                user.amount_data = new_amount
        return dropped_data

    def _collect_data(self):
        users_in_range = []
        for user in self.Users:
            distance = np.linalg.norm(
                self.uav.position[:2] - user.position[:2]
            )
            if (
                distance <= user.radius
                and user.is_active
                and self._remaining_data(user) > 0.0
            ):
                users_in_range.append(user)

        collected_data = 0.0
        if users_in_range:
            band_fraction = 1.0 / len(users_in_range)
            for user in users_in_range:
                rate = common_functions.calculate_rate_device_UAV(
                    user,
                    self.uav,
                    band_fraction,
                    0.1,
                )
                uploaded_data = min(
                    rate * self.delta_t,
                    self._remaining_data(user),
                )
                user.total_transmitted_data += uploaded_data
                user.aoi = 1.0
                collected_data += uploaded_data
        return collected_data

    def _update_activity(self):
        for user in self.Users:
            remaining_data = self._remaining_data(user)
            if user.is_active and remaining_data < self.sleep_threshold:
                user.is_active = False
            elif not user.is_active and remaining_data > self.wake_up_threshold:
                user.is_active = True

    def step(self, action):
        dx, dy = float(action[0]), float(action[1])
        self.uav.update_position(dx, dy)
        self.uav.position[0] = np.clip(
            self.uav.position[0], self.min_x, self.max_x
        )
        self.uav.position[1] = np.clip(
            self.uav.position[1], self.min_y, self.max_y
        )
        self.uav.trajectory.append(copy.deepcopy(self.uav.position))

        energy = common_functions.calculate_uav_energy_consumption(
            dx, dy, 0.0, self.delta_t
        )
        normalized_energy = energy / (self.max_theoretical_energy + 1e-9)

        guide_improvement = 0.0
        guide_user = self._guide_user()
        if guide_user is not None and guide_user.is_active:
            current_distance = float(
                np.linalg.norm(
                    self.uav.position[:2] - guide_user.position[:2]
                )
            )
            if current_distance > guide_user.radius:
                guide_improvement = max(
                    self.prev_guide_distance - current_distance,
                    0.0,
                )

        for user in self.Users:
            user.aoi = min(user.aoi + 1.0, float(self.T))

        collected_data = self._collect_data()
        self._update_activity()
        dropped_data = self._update_user_data()

        active_mean_aoi = self._active_mean_aoi()
        normalized_aoi = active_mean_aoi / self.T
        distance_reward = (
            self.w_dist * guide_improvement
            if collected_data <= 0.0
            else 0.0
        )

        reward = (
            self.w_data * collected_data
            + distance_reward
            - self.w_drop * dropped_data
            - self.w_energy * normalized_energy
            - self.w_aoi * normalized_aoi
            - self.step_penalty
        )

        distance_to_nfz = np.linalg.norm(
            self.uav.position[:2] - self.nfz_center
        )
        in_nfz = distance_to_nfz < self.nfz_radius
        if in_nfz:
            reward -= self.nfz_penalty

        self._update_guide()
        self.t += 1
        terminated = in_nfz
        truncated = self.t >= self.T

        info = {
            "collected_data": collected_data,
            "dropped_data": dropped_data,
            "energy": energy,
            "in_nfz": in_nfz,
            "active_mean_aoi": active_mean_aoi,
            "guide_distance_reward": distance_reward,
        }
        return (
            self.get_current_state(),
            float(reward),
            terminated,
            truncated,
            info,
        )
