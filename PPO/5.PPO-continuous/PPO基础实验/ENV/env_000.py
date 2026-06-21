import numpy as np
import gymnasium as gym
from gymnasium import spaces
import copy
from . import common_functions

#0403：暂时取消数据增长的版本

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


class EnvCore(gym.Env):
    def __init__(self, length=500, width=500, num_user=50, UAV_fixed_z=100, delta_t=1,
                 users_path="/home/xiewenhan_25/Project/DRL_code_pytorch/DRL-code-pytorch-main/5.PPO-continuous/data_train/users_50_new.txt"):
        # 这里的 super 初始化的是 gym.Env，使其成为标准强化学习环境，与旧代码无关
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
        self.T = 100
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

        # 状态空间维度：无人机坐标(2) + 3个设备的(相对x, 相对y, 剩余数据量)(9) + 禁飞区(相对x, 相对y, 边缘距离)(3) = 14维
        self.obs_dim = 14
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)

        # ---------------------- 4. 动态数据与物理极限参数 -----------------------------#
        self.min_data_increase = 1500
        self.max_data_increase = 2000
        self.max_theoretical_energy = self.estimate_max_energy()

        # ---------------------- 5. 奖励权重设计 -----------------------------#
        self.w_data = 0.002  # 数据采集权重
        self.w_dist = 0.6  # 距离引导权重
        self.w_energy = 15.0  # 能耗惩罚权重
        self.step_penalty = 0.5  # 防原地横跳的步数惩罚

        # 【修改点】将禁飞区惩罚加大到 500，用极大的负奖励代替“物理回弹”防止穿透
        self.nfz_penalty = 500.0

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
                    # Z坐标保留0.0，供信道增益函数计算高度差使用
                    user = User(
                        position=np.array([float(arr[0]), float(arr[1]), 0.0], dtype=np.float32),
                        amount_data=float(arr[2]),
                        radius=150.0,
                        user_id=i
                    )
                    self.Users.append(user)
                    if len(self.Users) >= self.num_user: break
        except FileNotFoundError:
            raise ValueError(f"用户数据文件未找到，请检查路径: {self.users_path}")

    def estimate_max_energy(self):
        max_dx, max_dy = 20.0, 20.0
        return common_functions.calculate_uav_energy_consumption(max_dx, max_dy, 0, self.delta_t)

    def update_user_data(self):
        for user in self.Users:
            #暂时取消数据增长
            #   data_increase = np.random.randint(self.min_data_increase, self.max_data_increase)
            #   user.amount_data += data_increase
            user.amount_data += 0

    def _get_user_scores(self):
        user_info = []
        uav_pos = self.uav.position[:2]
        for user in self.Users:
            rem_data = user.amount_data - user.total_transmitted_data
            dist = np.linalg.norm(uav_pos - user.position[:2])

            # 【核心修改】：加入常数 150，消除距离致盲
            score = rem_data / (dist + 150.0) if rem_data > 0 else -1.0

            user_info.append({'user': user, 'dist': dist, 'rem_data': rem_data, 'score': score})
        user_info.sort(key=lambda x: x['score'], reverse=True)
        return user_info

    def get_current_state(self):
        obs = []
        uav_pos = self.uav.position[:2]

        norm_uav_x = (uav_pos[0] / self.length) * 2 - 1.0
        norm_uav_y = (uav_pos[1] / self.width) * 2 - 1.0
        obs.extend([norm_uav_x, norm_uav_y])

        user_info = self._get_user_scores()
        top_3 = user_info[:3]

        for info in top_3:
            if info['rem_data'] > 0:
                rel_x = (info['user'].position[0] - uav_pos[0]) / self.length
                rel_y = (info['user'].position[1] - uav_pos[1]) / self.width
                norm_data = min(info['rem_data'] / 50000.0, 1.0)
                obs.extend([rel_x, rel_y, norm_data])
            else:
                obs.extend([0.0, 0.0, 0.0])

        rel_nfz_x = (self.nfz_center[0] - uav_pos[0]) / self.length
        rel_nfz_y = (self.nfz_center[1] - uav_pos[1]) / self.width
        dist_to_nfz = np.linalg.norm(self.nfz_center - uav_pos)
        dist_to_nfz_edge = (dist_to_nfz - self.nfz_radius) / self.length
        obs.extend([rel_nfz_x, rel_nfz_y, dist_to_nfz_edge])

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

        top_users = self._get_user_scores()
        self.target_user_id = top_users[0]['user'].user_id
        self.prev_target_dist = top_users[0]['dist']

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

        for user in self.Users:
            # 【修改点】数据采集范围判定：统一改为计算 2D 距离
            dist_2d = np.linalg.norm(self.uav.position[:2] - user.position[:2])
            if dist_2d <= user.radius and (user.amount_data - user.total_transmitted_data) > 0:
                users_in_range.append(user)

        if len(users_in_range) > 0:
            band_fraction = 1.0 / len(users_in_range)
            for user in users_in_range:
                # 这里的 user.position 仍然带有 Z=0.0，传给 common_functions 依然能正确计算路径损耗
                rate = common_functions.calculate_rate_device_UAV(user, self.uav, band_fraction, tx_power)
                uploaded = rate * self.delta_t
                remaining = user.amount_data - user.total_transmitted_data
                actual_upload = min(uploaded, remaining)

                user.total_transmitted_data += actual_upload
                collected_data_this_step += actual_upload

        reward = 0.0

        if collected_data_this_step > 0:
            reward += self.w_data * collected_data_this_step
        else:
            target_user = next((u for u in self.Users if u.user_id == self.target_user_id), None)
            if target_user and (target_user.amount_data - target_user.total_transmitted_data) > 0:
                # 【统一】引导奖励判定：使用 2D 距离
                curr_target_dist_2d = np.linalg.norm(self.uav.position[:2] - target_user.position[:2])
                if curr_target_dist_2d > 150.0:
                    distance_improvement = self.prev_target_dist - curr_target_dist_2d
                    reward += self.w_dist * distance_improvement
            else:
                reward -= self.step_penalty

        dist_to_nfz = np.linalg.norm(self.uav.position[:2] - self.nfz_center)
        if dist_to_nfz < self.nfz_radius:
            reward -= self.nfz_penalty  # 触发 500 的重罚

        reward -= self.w_energy * normalized_energy
        reward -= self.step_penalty

        top_users = self._get_user_scores()
        if top_users[0]['score'] > 0:
            self.target_user_id = top_users[0]['user'].user_id
            # 记录新的 2D 目标距离
            self.prev_target_dist = top_users[0]['dist']
        else:
            self.target_user_id = -1

        self.update_user_data()
        self.t += 1
        done = self.t >= self.T
        truncated = False

        obs = self.get_current_state()
        info = {
            "collected_data": collected_data_this_step,
            "energy": energy,
            "in_nfz": dist_to_nfz < self.nfz_radius
        }

        return obs, float(reward), done, truncated, info