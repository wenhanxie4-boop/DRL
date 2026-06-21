import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta, Normal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0.0)


class MultiHeadGraphAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, nodes):
        batch_size, num_nodes, hidden_dim = nodes.shape

        def split_heads(x):
            return x.view(
                batch_size,
                num_nodes,
                self.num_heads,
                self.head_dim,
            ).transpose(1, 2)

        query = split_heads(self.query(nodes))
        key = split_heads(self.key(nodes))
        value = split_heads(self.value(nodes))

        scores = torch.matmul(query, key.transpose(-1, -2))
        scores = scores / math.sqrt(self.head_dim)
        attention = torch.softmax(scores, dim=-1)
        messages = torch.matmul(attention, value)
        messages = messages.transpose(1, 2).contiguous().view(
            batch_size,
            num_nodes,
            hidden_dim,
        )
        return self.norm(nodes + self.output(messages))


class GraphPointerEncoder(nn.Module):
    """Encodes all users and learns soft pointer weights over user nodes."""

    def __init__(self, args):
        super().__init__()
        self.num_user = args.num_user
        self.uav_feature_dim = args.uav_feature_dim
        self.user_feature_dim = args.user_feature_dim
        hidden_dim = args.gat_hidden_dim

        self.user_encoder = nn.Sequential(
            nn.Linear(self.user_feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.uav_encoder = nn.Sequential(
            nn.Linear(self.uav_feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gat_layers = nn.ModuleList(
            MultiHeadGraphAttention(hidden_dim, args.gat_heads)
            for _ in range(args.gat_layers)
        )
        self.pointer_query = nn.Linear(hidden_dim, hidden_dim)
        self.pointer_key = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.Tanh(),
        )

    def forward(self, state):
        uav_state = state[:, : self.uav_feature_dim]
        user_state = state[:, self.uav_feature_dim :].reshape(
            -1,
            self.num_user,
            self.user_feature_dim,
        )

        uav_embedding = self.uav_encoder(uav_state)
        user_embedding = self.user_encoder(user_state)
        for gat_layer in self.gat_layers:
            user_embedding = gat_layer(user_embedding)

        query = self.pointer_query(uav_embedding).unsqueeze(1)
        keys = self.pointer_key(user_embedding)
        pointer_logits = (query * keys).sum(dim=-1) / math.sqrt(keys.size(-1))
        pointer_weights = torch.softmax(pointer_logits, dim=-1)

        pointer_context = torch.sum(
            pointer_weights.unsqueeze(-1) * user_embedding,
            dim=1,
        )
        global_context = user_embedding.mean(dim=1)
        feature = self.output(
            torch.cat(
                [uav_embedding, pointer_context, global_context],
                dim=-1,
            )
        )
        return feature, pointer_weights


class ActorGaussian(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.max_action = args.max_action
        self.encoder = GraphPointerEncoder(args)
        self.mean_layer = nn.Linear(args.gat_hidden_dim, args.action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, args.action_dim))

        if args.use_orthogonal_init:
            self.apply(self._initialize)
            orthogonal_init(self.mean_layer, gain=0.01)

    @staticmethod
    def _initialize(module):
        if isinstance(module, nn.Linear):
            orthogonal_init(module)

    def forward(self, state, return_attention=False):
        feature, attention = self.encoder(state)
        mean = self.max_action * torch.tanh(self.mean_layer(feature))
        if return_attention:
            return mean, attention
        return mean

    def get_dist(self, state):
        mean = self.forward(state)
        std = torch.exp(self.log_std.expand_as(mean))
        return Normal(mean, std)


class ActorBeta(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder = GraphPointerEncoder(args)
        self.alpha_layer = nn.Linear(args.gat_hidden_dim, args.action_dim)
        self.beta_layer = nn.Linear(args.gat_hidden_dim, args.action_dim)

        if args.use_orthogonal_init:
            self.apply(self._initialize)
            orthogonal_init(self.alpha_layer, gain=0.01)
            orthogonal_init(self.beta_layer, gain=0.01)

    @staticmethod
    def _initialize(module):
        if isinstance(module, nn.Linear):
            orthogonal_init(module)

    def forward(self, state, return_attention=False):
        feature, attention = self.encoder(state)
        alpha = F.softplus(self.alpha_layer(feature)) + 1.0
        beta = F.softplus(self.beta_layer(feature)) + 1.0
        if return_attention:
            return alpha, beta, attention
        return alpha, beta

    def get_dist(self, state):
        alpha, beta = self.forward(state)
        return Beta(alpha, beta)

    def mean(self, state):
        alpha, beta = self.forward(state)
        return alpha / (alpha + beta)


class Critic(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder = GraphPointerEncoder(args)
        self.value_layer = nn.Linear(args.gat_hidden_dim, 1)

        if args.use_orthogonal_init:
            self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, nn.Linear):
            orthogonal_init(module)

    def forward(self, state):
        feature, _ = self.encoder(state)
        return self.value_layer(feature)


class PPO_GAT:
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
        self.use_grad_clip = args.use_grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.use_adv_norm = args.use_adv_norm

        if self.policy_dist == "Beta":
            self.actor = ActorBeta(args)
        else:
            self.actor = ActorGaussian(args)
        self.critic = Critic(args)

        adam_eps = 1e-5 if args.set_adam_eps else 1e-8
        self.optimizer_actor = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.lr_a,
            eps=adam_eps,
        )
        self.optimizer_critic = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.lr_c,
            eps=adam_eps,
        )

    def evaluate(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            if self.policy_dist == "Beta":
                action = self.actor.mean(state)
            else:
                action = self.actor(state)
        return action.numpy().flatten()

    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            distribution = self.actor.get_dist(state)
            action = distribution.sample()
            if self.policy_dist == "Gaussian":
                action = torch.clamp(
                    action,
                    -self.max_action,
                    self.max_action,
                )
            log_probability = distribution.log_prob(action)
        return (
            action.numpy().flatten(),
            log_probability.numpy().flatten(),
        )

    def get_attention(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            if self.policy_dist == "Beta":
                _, _, attention = self.actor(state, return_attention=True)
            else:
                _, attention = self.actor(state, return_attention=True)
        return attention.numpy().flatten()

    def update(self, replay_buffer, total_steps):
        state, action, old_logprob, reward, next_state, dw, done = (
            replay_buffer.numpy_to_tensor()
        )

        with torch.no_grad():
            value = self.critic(state)
            next_value = self.critic(next_state)
            deltas = reward + self.gamma * (1.0 - dw) * next_value - value

            advantages = []
            gae = 0.0
            for delta, terminal in zip(
                reversed(deltas.flatten().numpy()),
                reversed(done.flatten().numpy()),
            ):
                gae = (
                    delta
                    + self.gamma
                    * self.lamda
                    * gae
                    * (1.0 - terminal)
                )
                advantages.insert(0, gae)

            advantage = torch.tensor(
                advantages,
                dtype=torch.float32,
            ).view(-1, 1)
            value_target = advantage + value
            if self.use_adv_norm:
                advantage = (
                    advantage - advantage.mean()
                ) / (advantage.std() + 1e-5)

        for _ in range(self.K_epochs):
            sampler = BatchSampler(
                SubsetRandomSampler(range(self.batch_size)),
                self.mini_batch_size,
                False,
            )
            for index in sampler:
                distribution = self.actor.get_dist(state[index])
                entropy = distribution.entropy().sum(1, keepdim=True)
                current_logprob = distribution.log_prob(action[index])
                probability_ratio = torch.exp(
                    current_logprob.sum(1, keepdim=True)
                    - old_logprob[index].sum(1, keepdim=True)
                )

                surrogate_1 = probability_ratio * advantage[index]
                surrogate_2 = torch.clamp(
                    probability_ratio,
                    1.0 - self.epsilon,
                    1.0 + self.epsilon,
                ) * advantage[index]
                actor_loss = (
                    -torch.min(surrogate_1, surrogate_2)
                    - self.entropy_coef * entropy
                ).mean()

                self.optimizer_actor.zero_grad()
                actor_loss.backward()
                if self.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(
                        self.actor.parameters(),
                        0.5,
                    )
                self.optimizer_actor.step()

                current_value = self.critic(state[index])
                critic_loss = F.mse_loss(
                    value_target[index],
                    current_value,
                )
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                if self.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(
                        self.critic.parameters(),
                        0.5,
                    )
                self.optimizer_critic.step()

        if self.use_lr_decay:
            self.lr_decay(total_steps)

    def lr_decay(self, total_steps):
        actor_lr = self.lr_a * (1.0 - total_steps / self.max_train_steps)
        critic_lr = self.lr_c * (1.0 - total_steps / self.max_train_steps)
        for parameter_group in self.optimizer_actor.param_groups:
            parameter_group["lr"] = actor_lr
        for parameter_group in self.optimizer_critic.param_groups:
            parameter_group["lr"] = critic_lr
