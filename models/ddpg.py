# models/ddpg.py
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Actor network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action=1.0):
        super(Actor, self).__init__()
        self.max_action = max_action
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.max_action * self.net(x)

# Critic network (Q)
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, s, a):
        x = torch.cat([s, a], dim=1)
        return self.net(x)

class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action=1.0,
                 actor_lr=1e-4, critic_lr=1e-3, gamma=0.99, tau=1e-3, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.tau = tau
        self.action_dim = action_dim
        self.max_action = max_action

    def select_action(self, state, noise=0.0):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy().flatten()
        self.actor.train()
        if noise > 1e-6:
            action = action + np.random.normal(0, noise, size=self.action_dim)
        return np.clip(action, -self.max_action, self.max_action)

    def train(self, replay_buffer, batch_size=64):
        if replay_buffer.size() < batch_size:
            return None

        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones.astype(float), dtype=torch.float32).unsqueeze(1).to(self.device)

        # Critic update
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            y = rewards + self.gamma * (1 - dones) * target_q

        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, y)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update (policy gradient)
        pred_actions = self.actor(states)
        actor_loss = -self.critic(states, pred_actions).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)
        return {'actor_loss': actor_loss.item(), 'critic_loss': critic_loss.item()}

    def _soft_update(self, source, target):
        for param_src, param_tgt in zip(source.parameters(), target.parameters()):
            param_tgt.data.copy_(self.tau * param_src.data + (1.0 - self.tau) * param_tgt.data)

    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }, path)

    def load(self, path):
        data = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(data['actor'])
        self.critic.load_state_dict(data['critic'])
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
