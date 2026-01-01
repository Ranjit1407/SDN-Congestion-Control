#!/usr/bin/env python3
# agents/decision/dqn_agent.py
# Minimal DQN for per-port decisions based on a small state vector.

import math
import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

@dataclass
class DQNConfig:
    state_dim: int = 4          # e.g., [tx_bps_norm, rx_bps_norm, dpid_norm, port_norm]
    n_actions: int = 4          # 0:noop, 1:meter, 2:drop, 3:reroute
    hidden: int = 64
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    buffer_size: int = 10000
    start_epsilon: float = 1.0
    end_epsilon: float = 0.05
    epsilon_decay: int = 20000  # steps
    target_update_every: int = 1000

class MLP(nn.Module):
    def __init__(self, state_dim, hidden, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions)
        )
    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, cfg: DQNConfig):
        self.cfg = cfg
        self.q = MLP(cfg.state_dim, cfg.hidden, cfg.n_actions)
        self.target = MLP(cfg.state_dim, cfg.hidden, cfg.n_actions)
        self.target.load_state_dict(self.q.state_dict())
        self.opt = optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.buffer = deque(maxlen=cfg.buffer_size)
        self.step_count = 0
        self.epsilon = cfg.start_epsilon
        self.device = torch.device('cpu')
        self.q.to(self.device)
        self.target.to(self.device)

    def act(self, state_np, explore=True):
        """
        state_np: np.array shape (state_dim,)
        returns: int action in [0..n_actions-1]
        """
        self.step_count += 1
        if explore:
            self.epsilon = self.cfg.end_epsilon + (self.cfg.start_epsilon - self.cfg.end_epsilon) * \
                           math.exp(-1.0 * self.step_count / self.cfg.epsilon_decay)
            if random.random() < self.epsilon:
                return random.randrange(self.cfg.n_actions)

        with torch.no_grad():
            s = torch.tensor(state_np, dtype=torch.float32, device=self.device).unsqueeze(0)
            qv = self.q(s)
            a = int(torch.argmax(qv, dim=1).item())
            return a

    def remember(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))

    def train_step(self):
        if len(self.buffer) < self.cfg.batch_size:
            return 0.0
        batch = random.sample(self.buffer, self.cfg.batch_size)
        s, a, r, s2, d = zip(*batch)
        s = torch.tensor(np.stack(s), dtype=torch.float32, device=self.device)
        a = torch.tensor(a, dtype=torch.int64, device=self.device).unsqueeze(1)
        r = torch.tensor(r, dtype=torch.float32, device=self.device).unsqueeze(1)
        s2 = torch.tensor(np.stack(s2), dtype=torch.float32, device=self.device)
        d = torch.tensor(d, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_sa = self.q(s).gather(1, a)
        with torch.no_grad():
            q_s2_max = self.target(s2).max(1, keepdim=True)[0]
            y = r + (1.0 - d) * self.cfg.gamma * q_s2_max

        loss = self.loss_fn(q_sa, y)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 5.0)
        self.opt.step()

        if self.step_count % self.cfg.target_update_every == 0:
            self.target.load_state_dict(self.q.state_dict())

        return float(loss.item())

    def save(self, path):
        torch.save({
            'model': self.q.state_dict(),
            'cfg': self.cfg.__dict__,
            'step_count': self.step_count,
            'epsilon': self.epsilon
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.q.load_state_dict(ckpt['model'])
        self.target.load_state_dict(self.q.state_dict())
        self.step_count = ckpt.get('step_count', 0)
        self.epsilon = ckpt.get('epsilon', self.cfg.start_epsilon)
