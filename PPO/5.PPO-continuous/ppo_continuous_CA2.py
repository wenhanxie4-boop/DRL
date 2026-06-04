import torch
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torch.nn as nn
from torch.distributions import Normal


def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class HybridAttentionExtractor(nn.Module):
    """Feature extractor for 14-dim heuristic state + global user attention."""

    def __init__(self, state_dim, num_users=50, user_dim=5, embed_dim=32, num_heads=4):
        super(HybridAttentionExtractor, self).__init__()
        self.basic_dim = 14
        self.num_users = num_users
        self.user_dim = user_dim

        expected_dim = self.basic_dim + self.num_users * self.user_dim
        if state_dim != expected_dim:
            raise ValueError(
                f"CA2 expects state_dim={expected_dim} "
                f"(14 + {self.num_users} * {self.user_dim}), got {state_dim}"
            )

        self.basic_mlp = nn.Sequential(
            nn.Linear(self.basic_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
        )
        self.user_encoder = nn.Sequential(
            nn.Linear(self.user_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
        )
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True)

        self.output_dim = embed_dim * 2

    def forward(self, obs):
        basic_state = obs[:, :self.basic_dim]
        users_state = obs[:, self.basic_dim:].view(-1, self.num_users, self.user_dim)

        basic_feature = self.basic_mlp(basic_state)
        user_feature = self.user_encoder(users_state)

        query = self.q_linear(basic_feature).unsqueeze(1)
        attn_out, _ = self.attention(query=query, key=user_feature, value=user_feature)
        attn_feature = attn_out.squeeze(1)

        return torch.cat([basic_feature, attn_feature], dim=-1)


class Actor_Gaussian(nn.Module):
    def __init__(self, args):
        super(Actor_Gaussian, self).__init__()
        self.max_action = args.max_action

        num_users = (args.state_dim - 14) // 5
        self.extractor = HybridAttentionExtractor(
            state_dim=args.state_dim,
            num_users=num_users,
            user_dim=5,
            embed_dim=32,
            num_heads=4,
        )

        self.fc1 = nn.Linear(self.extractor.output_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.mean_layer = nn.Linear(args.hidden_width, args.action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, args.action_dim))
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]

        if args.use_orthogonal_init:
            print("------use_orthogonal_init (CA2 Actor)------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.mean_layer, gain=0.01)

    def forward(self, s):
        features = self.extractor(s)
        x = self.activate_func(self.fc1(features))
        x = self.activate_func(self.fc2(x))
        mean = self.max_action * torch.tanh(self.mean_layer(x))
        return mean

    def get_dist(self, s):
        mean = self.forward(s)
        log_std = self.log_std.expand_as(mean)
        std = torch.exp(log_std)
        return Normal(mean, std)


class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()

        num_users = (args.state_dim - 14) // 5
        self.extractor = HybridAttentionExtractor(
            state_dim=args.state_dim,
            num_users=num_users,
            user_dim=5,
            embed_dim=32,
            num_heads=4,
        )

        self.fc1 = nn.Linear(self.extractor.output_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.fc3 = nn.Linear(args.hidden_width, 1)
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]

        if args.use_orthogonal_init:
            print("------use_orthogonal_init (CA2 Critic)------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, s):
        features = self.extractor(s)
        x = self.activate_func(self.fc1(features))
        x = self.activate_func(self.fc2(x))
        return self.fc3(x)


class Actor_Beta(nn.Module):
    pass


class PPO_continuous():
    def __init__(self, args):
        self.policy_dist = args.policy_dist
        self.max_action = args.max_action
        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.max_train_steps = args.max_train_steps
        self.lr_a = args.lr_a
        self.lr_c = args.lr_c
        self.gamma = args.gamma
        self.lamda = args.lamda
        self.epsilon = args.epsilon
        self.K_epochs = args.K_epochs
        self.entropy_coef = args.entropy_coef
        self.set_adam_eps = args.set_adam_eps
        self.use_grad_clip = args.use_grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.use_adv_norm = args.use_adv_norm

        self.actor = Actor_Gaussian(args)
        self.critic = Critic(args)

        if self.set_adam_eps:
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
            a = torch.clamp(a, -self.max_action, self.max_action)
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
