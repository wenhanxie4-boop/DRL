import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class ActorDiscrete(nn.Module):
    def __init__(self, args):
        super(ActorDiscrete, self).__init__()
        self.fc1 = nn.Linear(args.high_state_dim, args.high_hidden_width)
        self.fc2 = nn.Linear(args.high_hidden_width, args.high_hidden_width)
        self.logits_layer = nn.Linear(args.high_hidden_width, args.high_action_dim)
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]

        if args.use_orthogonal_init:
            print("------use_orthogonal_init (High Actor)------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.logits_layer, gain=0.01)

    def forward(self, s):
        x = self.activate_func(self.fc1(s))
        x = self.activate_func(self.fc2(x))
        return self.logits_layer(x)

    def get_dist(self, s):
        logits = self.forward(s)
        return Categorical(logits=logits)


class CriticDiscrete(nn.Module):
    def __init__(self, args):
        super(CriticDiscrete, self).__init__()
        self.fc1 = nn.Linear(args.high_state_dim, args.high_hidden_width)
        self.fc2 = nn.Linear(args.high_hidden_width, args.high_hidden_width)
        self.fc3 = nn.Linear(args.high_hidden_width, 1)
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]

        if args.use_orthogonal_init:
            print("------use_orthogonal_init (High Critic)------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, s):
        x = self.activate_func(self.fc1(s))
        x = self.activate_func(self.fc2(x))
        return self.fc3(x)


class PPO_discrete_high:
    def __init__(self, args):
        self.batch_size = args.high_batch_size
        self.mini_batch_size = args.high_mini_batch_size
        self.max_train_steps = args.max_train_steps
        self.lr_a = args.high_lr_a
        self.lr_c = args.high_lr_c
        self.gamma = args.high_gamma
        self.lamda = args.lamda
        self.epsilon = args.epsilon
        self.K_epochs = args.K_epochs
        self.entropy_coef = args.high_entropy_coef
        self.use_grad_clip = args.use_grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.use_adv_norm = args.use_adv_norm

        self.actor = ActorDiscrete(args)
        self.critic = CriticDiscrete(args)

        if args.set_adam_eps:
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a, eps=1e-5)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c, eps=1e-5)
        else:
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)

    def evaluate(self, s):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        with torch.no_grad():
            logits = self.actor(s)
            return torch.argmax(logits, dim=-1).item()

    def choose_action(self, s):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        with torch.no_grad():
            dist = self.actor.get_dist(s)
            a = dist.sample()
            a_logprob = dist.log_prob(a)
        return a.item(), a_logprob.item()

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
                adv = (adv - adv.mean()) / (adv.std() + 1e-5)

        for _ in range(self.K_epochs):
            for index in BatchSampler(SubsetRandomSampler(range(self.batch_size)), self.mini_batch_size, False):
                dist_now = self.actor.get_dist(s[index])
                dist_entropy = dist_now.entropy().view(-1, 1)
                a_logprob_now = dist_now.log_prob(a[index].squeeze(-1)).view(-1, 1)
                ratios = torch.exp(a_logprob_now - a_logprob[index])

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
            p["lr"] = lr_a_now
        for p in self.optimizer_critic.param_groups:
            p["lr"] = lr_c_now
