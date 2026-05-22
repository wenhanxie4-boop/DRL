import numpy as np
import gymnasium as gym
from gymnasium import spaces
import copy
from . import common_functions


# 融合交叉注意力cross-attention前置特征提取的 IoT 突发数据收集模型 (全局视野版)

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
        self.amount_data = amount_data
        self.radius = radius
        self.user_id = user_id
        self.total_transmitted_data = 0.0

        # 状态标签，True表示可采集，False表示休眠中
        self.is_active = True
        self.max_capacity = 60000.0


class EnvCore(gym.Env):
    def __init__(self, length=500, width=500, num_user=50, UAV_fixed_z=100, delta_t=1,
                 users_path="./data_train/users_50_new.txt"):  # 注意这里的路径，根据你的实际情况调整
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

        # 【核心修改点】：重构观测空间维度
        # 无人机与禁飞区特征 (5维) + 50个设备特征 (每设备4维：相对X, 相对Y, 数据占比, 预估爆仓时间)
        self.obs_dim = 5 + self.num_user * 4
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)

        # ---------------------- 4. 动态数据与阈值参数 -----------------------------#
        # 【修改点】：大幅降低自然增长率，配合你修改的初始数据，防止开局大面积连坐爆仓
        self.min_data_increase = 0
        self.max_data_increase = 200
        self.max_capacity = 60000.0
        self.wake_up_threshold = 25000.0
        self.sleep_threshold = 100.0
        self.max_theoretical_energy = self.estimate_max_energy()

        # ---------------------- 5. 奖励权重设计 (打破习得性无助) -----------------------------#
        self.w_data = 0.01
        # 【核心修改 1】：把爆仓惩罚从 0.0005 暴增到 0.01！漏掉1M数据等于白收集1M数据
        self.w_drop = 0.002
        self.w_dist = 0.30
        self.w_energy = 1
        self.step_penalty = 0.5
        self.nfz_penalty = 500.0

        # 保留主目标追踪器：它现在【不放入观测状态中】，仅仅作为后台“奖励整形老师”，引导无人机前期探索
        self.target_user_id = -1
        self.prev_target_dist = 0.0

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
                        radius=50.0,
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
            # 日常缓慢增长
            data_increase = np.random.randint(self.min_data_increase, self.max_data_increase)

            # 【温和版修改】：同步为 0.5% 极低概率触发 5000~10000 的中等突发
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
            """
            恢复为基础的距离驱动型向导。
            让奖励整形回归“提供局部密集引导”的本职工作，
            全局的爆仓危机调度，交由 Actor 神经网络里的 Cross-Attention 去自己领悟！
            """
            user_info = []
            uav_pos = self.uav.position[:2]
            for user in self.Users:
                rem_data = user.amount_data - user.total_transmitted_data
                dist = np.linalg.norm(uav_pos - user.position[:2])

                if user.is_active and rem_data > 0:
                    # 恢复为单纯的“数据量/距离”近视眼打分
                    score = rem_data / (dist + 80.0)
                else:
                    score = -1.0
                user_info.append({'user': user, 'dist': dist, 'rem_data': rem_data, 'score': score})

            user_info.sort(key=lambda x: x['score'], reverse=True)
            return user_info

    def get_current_state(self):
        """提供给交叉注意力机制的全局 205 维视野"""
        obs = []
        uav_pos = self.uav.position[:2]

        # 1. 提取无人机自身与禁飞区状态 (5 维)
        norm_uav_x = (uav_pos[0] / self.length) * 2 - 1.0
        norm_uav_y = (uav_pos[1] / self.width) * 2 - 1.0
        rel_nfz_x = (self.nfz_center[0] - uav_pos[0]) / self.length
        rel_nfz_y = (self.nfz_center[1] - uav_pos[1]) / self.width
        dist_to_nfz_edge = (np.linalg.norm(self.nfz_center - uav_pos) - self.nfz_radius) / self.length
        obs.extend([norm_uav_x, norm_uav_y, rel_nfz_x, rel_nfz_y, dist_to_nfz_edge])

        # 2. 提取所有 50 个用户的状态
        # 【温和版修改】：严格匹配新的数学期望 (137.5)
        avg_increase = 137.5

        for user in self.Users:
            rel_x = (user.position[0] - uav_pos[0]) / self.length
            rel_y = (user.position[1] - uav_pos[1]) / self.width
            rem_data = user.amount_data - user.total_transmitted_data
            norm_data = min(rem_data / self.max_capacity, 1.0)

            # 预估爆仓时间 TTO (Time-to-Overflow)
            if user.is_active and rem_data > 0:
                rem_capacity = self.max_capacity - rem_data
                tto_steps = rem_capacity / avg_increase
                # 越接近 0 越危险，大于 1 代表本回合内绝对安全
                tto_norm = min(tto_steps / self.T, 1.0)
            else:
                tto_norm = 1.0
                norm_data = 0.0

            obs.extend([rel_x, rel_y, norm_data, tto_norm])

        return np.array(obs, dtype=np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.t = 0

        init_pos = [(self.max_x - self.min_x) * 0.1, (self.max_y - self.min_y) * 0.1, self.UAV_fixed_z]
        self.uav = UAV(position=np.array(init_pos, dtype=np.float32))
        self.uav.trajectory.append(init_pos.copy())

        for user in self.Users:
            user.total_transmitted_data = 0
            rem_data = user.amount_data - user.total_transmitted_data

            if rem_data > self.wake_up_threshold:
                user.is_active = True
            else:
                user.is_active = False

        # 初始化后台引导目标
        top_users = self._get_user_scores()
        if top_users and top_users[0]['score'] > 0:
            self.target_user_id = top_users[0]['user'].user_id
            self.prev_target_dist = top_users[0]['dist']
        else:
            self.target_user_id = -1
            self.prev_target_dist = 0.0

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
            # 这里利用后台保留的 heuristic 目标给注意力机制提供早期探索向导
            target_user = next((u for u in self.Users if u.user_id == self.target_user_id), None)
            if target_user and target_user.is_active:
                curr_target_dist_2d = np.linalg.norm(self.uav.position[:2] - target_user.position[:2])
                if curr_target_dist_2d > 50.0:
                    distance_improvement = self.prev_target_dist - curr_target_dist_2d
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

        # 【更新后台向导目标】
        top_users = self._get_user_scores()
        current_target = next((u for u in self.Users if u.user_id == self.target_user_id), None)

        need_switch = False
        if current_target is None or not current_target.is_active:
            need_switch = True
        elif top_users and top_users[0]['score'] > 0:
            curr_score = next((x['score'] for x in top_users if x['user'].user_id == self.target_user_id), 0)
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