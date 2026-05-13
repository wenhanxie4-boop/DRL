import torch
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torch.nn as nn
from torch.distributions import Beta, Normal

# 交叉注意力 cross-attention 版本
# Trick 8: orthogonal initialization
def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


# =====================================================================
# 【核心创新模块】：交叉注意力特征提取器 (Cross-Attention Extractor)
# =====================================================================
class AttentionExtractor(nn.Module):
    def __init__(self, embed_dim=32, num_heads=4):
        super(AttentionExtractor, self).__init__()
        # 1. 身份转换器 (Linear层)
        # UAV状态有 5 维，User状态有 4 维
        self.q_linear = nn.Linear(5, embed_dim)
        self.k_linear = nn.Linear(4, embed_dim)
        self.v_linear = nn.Linear(4, embed_dim)

        # 2. PyTorch 官方交叉注意力模块 (batch_first=True 方便处理 (Batch, Seq, Feature) 格式)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True)

    def forward(self, obs):
        # obs 形状: (Batch, 205)
        uav_state = obs[:, :5]  # 无人机与禁飞区状态 (Batch, 5)
        users_state = obs[:, 5:]  # 50个设备的状态 (Batch, 200)

        # 将 200 维的设备状态折叠成 50 个独立的实体，每个实体 4 维
        users_state = users_state.view(-1, 50, 4)  # (Batch, 50, 4)

        # 生成 Q, K, V
        # Q 增加一个维度代表只有 1 个查询主体(无人机) -> (Batch, 1, embed_dim)
        Q = self.q_linear(uav_state).unsqueeze(1)
        K = self.k_linear(users_state)  # (Batch, 50, embed_dim)
        V = self.v_linear(users_state)  # (Batch, 50, embed_dim)

        # 计算交叉注意力 (attn_out 形状: Batch, 1, embed_dim)
        attn_out, attn_weights = self.attention(query=Q, key=K, value=V)

        # 去掉中间多余的维度 -> (Batch, embed_dim)
        attn_out = attn_out.squeeze(1)

        # 将原始无人机状态与注意力提炼出的核心目标特征拼接
        combined_features = torch.cat([uav_state, attn_out], dim=-1)  # (Batch, 5 + embed_dim) = (Batch, 37)
        return combined_features


# =====================================================================
# 重构的 Actor (高斯分布) - 结合了交叉注意力
# =====================================================================
class Actor_Gaussian(nn.Module):
    def __init__(self, args):
        super(Actor_Gaussian, self).__init__()
        self.max_action = args.max_action

        # 植入注意力提取器
        self.extractor = AttentionExtractor(embed_dim=32, num_heads=4)

        # 注意力提取后，输出维度是 37 (5维UAV + 32维注意力特征)
        extracted_dim = 37
        self.fc1 = nn.Linear(extracted_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.mean_layer = nn.Linear(args.hidden_width, args.action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, args.action_dim))
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]

        if args.use_orthogonal_init:
            print("------use_orthogonal_init (Actor)------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.mean_layer, gain=0.01)

    def forward(self, s):
        # 第一步：经过交叉注意力“聚光灯”筛选信息
        features = self.extractor(s)
        # 第二步：常规的多层感知机进行动作推理
        x = self.activate_func(self.fc1(features))
        x = self.activate_func(self.fc2(x))
        mean = self.max_action * torch.tanh(self.mean_layer(x))
        return mean

    def get_dist(self, s):
        mean = self.forward(s)
        log_std = self.log_std.expand_as(mean)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        return dist


# =====================================================================
# 重构的 Critic - 同样结合交叉注意力，使其打分更准确
# =====================================================================
class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()

        # 植入注意力提取器
        self.extractor = AttentionExtractor(embed_dim=32, num_heads=4)

        extracted_dim = 37
        self.fc1 = nn.Linear(extracted_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.fc3 = nn.Linear(args.hidden_width, 1)
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]

        if args.use_orthogonal_init:
            print("------use_orthogonal_init (Critic)------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, s):
        # 第一步：经过交叉注意力提取全局局势
        features = self.extractor(s)
        # 第二步：对当前局势进行价值打分
        x = self.activate_func(self.fc1(features))
        x = self.activate_func(self.fc2(x))
        v_s = self.fc3(x)
        return v_s


# （注：原有的 Actor_Beta 如果不使用，可以暂时不改，因为你在主函数默认用的是 Gaussian）
class Actor_Beta(nn.Module):
    pass  # 省略，当前使用 Gaussian


class PPO_continuous():
    def __init__(self, args):
        self.policy_dist = args.policy_dist
        self.max_action = args.max_action
        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.max_train_steps = args.max_train_steps
        self.lr_a = args.lr_a  # Learning rate of actor
        self.lr_c = args.lr_c  # Learning rate of critic
        self.gamma = args.gamma  # Discount factor
        self.lamda = args.lamda  # GAE parameter
        self.epsilon = args.epsilon  # PPO clip parameter
        self.K_epochs = args.K_epochs  # PPO parameter
        self.entropy_coef = args.entropy_coef  # Entropy coefficient
        self.set_adam_eps = args.set_adam_eps
        self.use_grad_clip = args.use_grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.use_adv_norm = args.use_adv_norm

        # 默认使用 Gaussian
        self.actor = Actor_Gaussian(args)
        self.critic = Critic(args)

        if self.set_adam_eps:  # Trick 9: set Adam epsilon=1e-5
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a, eps=1e-5)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c, eps=1e-5)
        else:
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)

    def evaluate(self, s):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        a = self.actor(s).detach().numpy().flatten()
        return a

    def choose_action(self, s):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        with torch.no_grad():
            dist = self.actor.get_dist(s)
            a = dist.sample()
            a = torch.clamp(a, -self.max_action, self.max_action)  # [-max,max]
            a_logprob = dist.log_prob(a)
        return a.numpy().flatten(), a_logprob.numpy().flatten()

    def update(self, replay_buffer, total_steps):
        s, a, a_logprob, r, s_, dw, done = replay_buffer.numpy_to_tensor()

        adv = []
        gae = 0
        with torch.no_grad():
            vs = self.critic(s)
            vs_ = self.critic(s_)
            deltas = r + self.gamma * (1.0 - dw) * vs_ - vs
            for delta, d in zip(reversed(deltas.flatten().numpy()), reversed(done.flatten().numpy())):
                gae = delta + self.gamma * self.lamda * gae * (1.0 - d)
                adv.insert(0, gae)
            adv = torch.tensor(adv, dtype=torch.float).view(-1, 1)
            v_target = adv + vs
            if self.use_adv_norm:
                adv = ((adv - adv.mean()) / (adv.std() + 1e-5))

        for _ in range(self.K_epochs):
            for index in BatchSampler(SubsetRandomSampler(range(self.batch_size)), self.mini_batch_size, False):
                dist_now = self.actor.get_dist(s[index])
                dist_entropy = dist_now.entropy().sum(1, keepdim=True)
                a_logprob_now = dist_now.log_prob(a[index])
                ratios = torch.exp(a_logprob_now.sum(1, keepdim=True) - a_logprob[index].sum(1, keepdim=True))

                surr1 = ratios * adv[index]
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy

                self.optimizer_actor.zero_grad()
                actor_loss.mean().backward()
                if self.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.optimizer_actor.step()

                v_s = self.critic(s[index])
                critic_loss = F.mse_loss(v_target[index], v_s)

                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                if self.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer_critic.step()

        if self.use_lr_decay:
            self.lr_decay(total_steps)

    def lr_decay(self, total_steps):
        lr_a_now = self.lr_a * (1 - total_steps / self.max_train_steps)
        lr_c_now = self.lr_c * (1 - total_steps / self.max_train_steps)
        for p in self.optimizer_actor.param_groups:
            p['lr'] = lr_a_now
        for p in self.optimizer_critic.param_groups:
            p['lr'] = lr_c_now