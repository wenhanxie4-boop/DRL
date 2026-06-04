# Hybrid CA environment:
#   basic heuristic observation: 14 dims
#   global user observation: num_user * 5 dims
# Total shape with 50 users: 264 dims.

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import copy
from . import common_functions


# 带有唤醒/休眠机制、防爆仓溢出惩罚、防状态抖动粘性槽位的 IoT 突发数据收集模型

class UAV(object):
    def __init__(self, position):
        self.position = position
        self.trajectory = []

    def update_position(self, dx, dy):
        self.position[0] += dx
        self.position[1] += dy
        self.position[2] = 100.0  # Z轴高度固定 100m
        return self.position


class User(object):
    def __init__(self, position, amount_data, radius, user_id):
        self.position = position

        # 初始数据量，用于每个 episode reset
        self.initial_amount_data = amount_data

        self.amount_data = amount_data
        self.radius = radius
        self.user_id = user_id
        self.total_transmitted_data = 0.0

        # 状态标签，True表示可采集，False表示休眠中
        self.is_active = True
        self.max_capacity = 60000.0


class EnvCore(gym.Env):  # 用户数据
    def __init__(self, length=500, width=500, num_user=50, UAV_fixed_z=100, delta_t=1,
                 users_path="./data_train/users_50_v2.txt"):
        super(EnvCore, self).__init__()

        # ---------------------- 1. 基础环境参数 -----------------------------#
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

        # 【修改点】：回合步长延长，给无人机充足的时间巡逻
        self.T = 200
        self.t = 0

        # ---------------------- 2. 禁飞区 (NFZ) 参数 -----------------------------#
        self.nfz_center = np.array([length / 2, width / 2], dtype=np.float32)
        self.nfz_radius = 50.0

        # ---------------------- 3. 动作空间与状态空间定义 -----------------------------#
        self.action_space = spaces.Box(
            low=np.array([-20.0, -20.0], dtype=np.float32),
            high=np.array([20.0, 20.0], dtype=np.float32),
            dtype=np.float32
        )

        # ---------状态空间维度：无人机坐标(2) + 锁定目标一号位(3) + 粘性副槽位二三号位(6) + 禁飞区(3)
        self.obs_dim = 14 + self.num_user * 5
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)

        # ---------------------- 4. 动态数据与阈值参数 -----------------------------#
        # 【修改点】：大幅降低自然增长率，配合你修改的初始数据，防止开局大面积连坐爆仓
        self.min_data_increase = 0
        self.max_data_increase = 200
        self.max_capacity = 60000.0  # 容量上限
        self.wake_up_threshold = 25000.0  # 唤醒阈值
        self.sleep_threshold = 100.0  # 休眠阈值
        self.max_theoretical_energy = self.estimate_max_energy()

        # ---------------------- 5. 奖励权重设计 (打破习得性无助) -----------------------------#
        self.w_data = 0.01  # 收集数据奖励
        self.w_drop = 0.01  # 爆仓惩罚
        self.w_dist = 0.05  # 距离引导奖励
        self.w_energy = 1  # 能耗惩罚
        self.step_penalty = 0.5  # 原地悬停惩罚
        self.nfz_penalty = 500.0  # 禁飞区惩罚

        # --- 目标锁定与槽位状态追踪器 ---
        self.target_user_id = -1
        self.prev_target_dist = 0.0  # 上一时刻无人机与主目标用户之间的平面距离
        self.secondary_slots = [-1, -1]  # 【新增】：目标二号和三号位的粘性记录器

        self.initialize_users()
        self.uav = None

    def initialize_users(self):
        self.Users = []
        try:
            with open(self.users_path, 'r') as f:
                for i, line in enumerate(f):
                    if not line.strip(): continue
                    arr = line.split()
                    user = User(
                        position=np.array([float(arr[0]), float(arr[1]), 0.0], dtype=np.float32),
                        amount_data=float(arr[2]),
                        radius=50.0,  # 强制缩小通信半径，逼迫移动
                        user_id=i
                    )
                    user.amount_data = min(user.amount_data, self.max_capacity)
                    self.Users.append(user)
                    if len(self.Users) >= self.num_user: break
        except FileNotFoundError:
            raise ValueError(f"用户数据文件未找到，请检查路径: {self.users_path}")

    def estimate_max_energy(self):
        max_dx, max_dy = 20.0, 20.0
        return common_functions.calculate_uav_energy_consumption(max_dx, max_dy, 0, self.delta_t)

    def update_user_data(self):
        """更新所有设备的数据，并返回爆仓丢弃的总数据量"""
        dropped_data_this_step = 0.0
        for user in self.Users:
            # 日常缓慢增长 (0~200)
            data_increase = np.random.randint(self.min_data_increase, self.max_data_increase)

            # 【温和版修改】：0.5% 极低概率触发 5000~10000 的中等突发
            if user.is_active and np.random.rand() < 0.005:
                data_increase += np.random.randint(5000, 10000)

            new_amount = user.amount_data + data_increase

            if new_amount > self.max_capacity:
                dropped_data_this_step += (new_amount - self.max_capacity)
                user.amount_data = self.max_capacity
            else:
                user.amount_data = new_amount
        return dropped_data_this_step

    def _get_user_scores(self):
        """计算性价比得分，休眠设备直接出局 (-1)"""
        user_info = []
        uav_pos = self.uav.position[:2]
        for user in self.Users:
            rem_data = user.amount_data - user.total_transmitted_data
            dist = np.linalg.norm(uav_pos - user.position[:2])  # 距离是按直线距离计算的，但是有禁飞区，待完善-----

            if user.is_active and rem_data > 0:  # 处于激活状态才可以
                score = rem_data / (dist + 80.0)  # 常数的大小决定了无人机是近视眼还是远视眼
            else:
                score = -1.0
            user_info.append({'user': user, 'dist': dist, 'rem_data': rem_data, 'score': score})
        user_info.sort(key=lambda x: x['score'], reverse=True)
        return user_info

    def get_current_state(self):
        """重构后的 14 维状态空间：彻底消除排列抖动"""

        obs = []  # ----------观测空间---------
        uav_pos = self.uav.position[:2]

        # 1. 无人机坐标 (2维)
        norm_uav_x = (uav_pos[0] / self.length) * 2 - 1.0
        norm_uav_y = (uav_pos[1] / self.width) * 2 - 1.0
        obs.extend([norm_uav_x, norm_uav_y])

        # 2. 当前锁定的一号位主目标 (3维)
        target_user = next((u for u in self.Users if u.user_id == self.target_user_id), None)
        if target_user and target_user.is_active:
            rel_x = (target_user.position[0] - uav_pos[0]) / self.length
            rel_y = (target_user.position[1] - uav_pos[1]) / self.width
            rem_data = target_user.amount_data - target_user.total_transmitted_data
            norm_data = min(rem_data / self.max_capacity, 1.0)
            obs.extend([rel_x, rel_y, norm_data])
        else:
            obs.extend([0.0, 0.0, 0.0])

        # ---------- 3 & 4. 二三槽位粘性锁定逻辑 (防排列抖动) ---------- #
        active_pool = {u.user_id: u for u in self.Users if u.is_active}  # 构建一个活跃用户的字典

        # 避免主目标霸占副槽位  排除掉主目标
        if target_user and target_user.is_active and target_user.user_id in active_pool:
            del active_pool[target_user.user_id]

        # 步骤 A：踢出失效的占坑者 (已经休眠，或者被提拔为主目标了)
        for i in range(2):
            if self.secondary_slots[i] not in active_pool:
                self.secondary_slots[i] = -1

        # 步骤 B：如果有更优秀的竞争者，抢占槽位 (距离迟滞 30 米)
        for i in range(2):
            if self.secondary_slots[i] != -1:
                curr_u = active_pool[self.secondary_slots[i]]
                curr_dist = np.linalg.norm(uav_pos - curr_u.position[:2])

                candidates = [u for u in active_pool.values() if u.user_id not in self.secondary_slots]
                if candidates:
                    best_cand = min(candidates, key=lambda u: np.linalg.norm(uav_pos - u.position[:2]))
                    best_dist = np.linalg.norm(uav_pos - best_cand.position[:2])

                    # 迟滞判定：比现任近 30 米以上才能踢走
                    if best_dist < curr_dist - 30.0:
                        self.secondary_slots[i] = best_cand.user_id

        # 步骤 C：填补空缺的副槽位
        for i in range(2):
            if self.secondary_slots[i] == -1:
                candidates = [u for u in active_pool.values() if u.user_id not in self.secondary_slots]
                if candidates:
                    best_cand = min(candidates, key=lambda u: np.linalg.norm(uav_pos - u.position[:2]))
                    self.secondary_slots[i] = best_cand.user_id

        # 将二三号槽位填入观测空间 (6维)
        for i in range(2):
            uid = self.secondary_slots[i]
            if uid != -1:
                u = active_pool[uid]
                rel_x = (u.position[0] - uav_pos[0]) / self.length
                rel_y = (u.position[1] - uav_pos[1]) / self.width
                rem_data = u.amount_data - u.total_transmitted_data
                norm_data = min(rem_data / self.max_capacity, 1.0)
                obs.extend([rel_x, rel_y, norm_data])
            else:
                obs.extend([0.0, 0.0, 0.0])

        # 5. 禁飞区信息 (3维)
        rel_nfz_x = (self.nfz_center[0] - uav_pos[0]) / self.length
        rel_nfz_y = (self.nfz_center[1] - uav_pos[1]) / self.width
        dist_to_nfz = np.linalg.norm(self.nfz_center - uav_pos)
        dist_to_nfz_edge = (dist_to_nfz - self.nfz_radius) / self.length
        obs.extend([rel_nfz_x, rel_nfz_y, dist_to_nfz_edge])

        # 6. Global user state for the cross-attention branch.
        avg_increase = 137.5
        for user in self.Users:
            rel_x = (user.position[0] - uav_pos[0]) / self.length
            rel_y = (user.position[1] - uav_pos[1]) / self.width

            rem_data = user.amount_data - user.total_transmitted_data
            rem_data = max(rem_data, 0.0)
            norm_data = min(rem_data / self.max_capacity, 1.0)

            if rem_data > 0:
                rem_capacity = self.max_capacity - rem_data
                tto_steps = rem_capacity / avg_increase
                tto_norm = min(tto_steps / self.T, 1.0)
            else:
                tto_norm = 1.0

            active_flag = 1.0 if user.is_active else 0.0
            obs.extend([rel_x, rel_y, norm_data, tto_norm, active_flag])

        return np.array(obs, dtype=np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.t = 0

        init_pos = [(self.max_x - self.min_x) * 0.1, (self.max_y - self.min_y) * 0.1, self.UAV_fixed_z]
        self.uav = UAV(position=np.array(init_pos, dtype=np.float32))
        self.uav.trajectory.append(init_pos.copy())

        for user in self.Users:
            # 重置为初始数据量
            user.amount_data = user.initial_amount_data

            user.total_transmitted_data = 0.0
            rem_data = user.amount_data - user.total_transmitted_data

            if rem_data > self.wake_up_threshold:
                user.is_active = True
            else:
                user.is_active = False

        top_users = self._get_user_scores()
        if top_users and top_users[0]['score'] > 0:
            self.target_user_id = top_users[0]['user'].user_id
            self.prev_target_dist = top_users[0]['dist']
        else:
            self.target_user_id = -1
            self.prev_target_dist = 0.0

        # 回合重置时，清空粘性槽位
        self.secondary_slots = [-1, -1]

        obs = self.get_current_state()
        return obs, {}

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

        # 【采集判定】
        for user in self.Users:
            dist_2d = np.linalg.norm(self.uav.position[:2] - user.position[:2])
            rem_data = user.amount_data - user.total_transmitted_data
            if dist_2d <= user.radius and user.is_active and rem_data > 0:
                users_in_range.append(user)

        if len(users_in_range) > 0:
            band_fraction = 1.0 / len(users_in_range)
            for user in users_in_range:
                rate = common_functions.calculate_rate_device_UAV(user, self.uav, band_fraction, tx_power)
                uploaded = rate * self.delta_t
                remaining = user.amount_data - user.total_transmitted_data
                actual_upload = min(uploaded, remaining)

                user.total_transmitted_data += actual_upload
                collected_data_this_step += actual_upload

        # 【状态机更新：休眠与唤醒】
        for user in self.Users:
            rem_data = user.amount_data - user.total_transmitted_data
            if user.is_active and rem_data < self.sleep_threshold:
                user.is_active = False
            elif not user.is_active and rem_data > self.wake_up_threshold:
                user.is_active = True

        # 【奖励计算与引导】
        reward = 0.0
        if collected_data_this_step > 0:
            reward += self.w_data * collected_data_this_step
        else:
            target_user = next((u for u in self.Users if u.user_id == self.target_user_id), None)
            if target_user and target_user.is_active:
                curr_target_dist_2d = np.linalg.norm(self.uav.position[:2] - target_user.position[:2])
                if curr_target_dist_2d > 50.0:
                    distance_improvement = self.prev_target_dist - curr_target_dist_2d
                    # 【神级优化】：消灭绕路惩罚！
                    # 如果为了躲避禁飞区导致距离变远 (负数)，强制归零，不予惩罚。
                    # 它依然会承受基础的耗电和步数惩罚，所以不会无意义乱飞，但敢于绕过障碍物了！
                    distance_improvement = max(0.0, distance_improvement)

                    reward += self.w_dist * distance_improvement
            else:
                reward -= self.step_penalty

        # 【施加爆仓重罚】
        dropped_data = self.update_user_data()
        if dropped_data > 0:
            reward -= (dropped_data * self.w_drop)

        # 【禁飞区与时间判定】
        dist_to_nfz = np.linalg.norm(self.uav.position[:2] - self.nfz_center)
        if dist_to_nfz < self.nfz_radius:
            reward -= self.nfz_penalty
            done = True
        else:
            done = self.t >= self.T

        reward -= self.w_energy * normalized_energy
        reward -= self.step_penalty

        # 【平滑的一号位目标锁定逻辑：防止频繁易主】
        top_users = self._get_user_scores()
        current_target = next((u for u in self.Users if u.user_id == self.target_user_id), None)

        need_switch = False
        if current_target is None or not current_target.is_active:
            need_switch = True
        elif top_users and top_users[0]['score'] > 0:
            curr_score = next((x['score'] for x in top_users if x['user'].user_id == self.target_user_id), 0)
            # 【优化 3】：降低固执阈值到 1.2，让它看到高 20% 收益的目标就果断去“舔包”
            if top_users[0]['score'] > curr_score * 1.2:
                need_switch = True

        if need_switch and top_users and top_users[0]['score'] > 0:
            self.target_user_id = top_users[0]['user'].user_id
            self.prev_target_dist = top_users[0]['dist']
        elif not need_switch and current_target is not None:
            self.prev_target_dist = np.linalg.norm(self.uav.position[:2] - current_target.position[:2])
        else:
            self.target_user_id = -1

        self.t += 1
        truncated = False

        obs = self.get_current_state()
        info = {
            "collected_data": collected_data_this_step,
            "dropped_data": dropped_data,
            "energy": energy,
            "in_nfz": dist_to_nfz < self.nfz_radius
        }

        return obs, float(reward), done, truncated, info
