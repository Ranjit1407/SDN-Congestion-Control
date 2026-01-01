#!/usr/bin/env python3
# agents/multi/agent_manager.py
#
# Thin wrapper that:
# - normalizes observations,
# - routes (dpid, port_no) to a shared DQN agent,
# - returns an action id.

import os
import sys
import numpy as np

# Ensure project root on sys.path so "agents.decision.dqn_agent" imports work
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agents.decision.dqn_agent import DQNAgent, DQNConfig

class AgentManager:
    def __init__(self, model_dir=None, n_actions=4):
        self.n_actions = n_actions
        self.cfg = DQNConfig(state_dim=4, n_actions=n_actions)
        self.agent = DQNAgent(self.cfg)
        self.model_dir = model_dir or os.path.join(PROJECT_ROOT, 'agents', 'models')
        os.makedirs(self.model_dir, exist_ok=True)
        self.model_path = os.path.join(self.model_dir, 'dqn_port.pt')

    def load_models(self):
        if os.path.exists(self.model_path):
            try:
                self.agent.load(self.model_path)
                print(f"[AgentManager] Loaded model from {self.model_path}")
            except Exception as e:
                print(f"[AgentManager] Failed to load model: {e}")

    def save_models(self):
        try:
            self.agent.save(self.model_path)
            print(f"[AgentManager] Saved model to {self.model_path}")
        except Exception as e:
            print(f"[AgentManager] Failed to save model: {e}")

    def _normalize(self, dpid, port_no, tx_bps, rx_bps):
        # Simple normalization assuming 100 Mbps scale (adjust to your links)
        SCALE = 100e6
        tx_n = max(0.0, min(1.0, tx_bps / SCALE))
        rx_n = max(0.0, min(1.0, rx_bps / SCALE))
        dpid_n = min(1.0, float(dpid) / 100.0)
        port_n = min(1.0, float(port_no) / 100.0)
        return np.array([tx_n, rx_n, dpid_n, port_n], dtype=np.float32)

    def act(self, dpid, port_no, tx_bps, rx_bps, explore=False):
        state = self._normalize(dpid, port_no, tx_bps, rx_bps)
        action = self.agent.act(state, explore=explore)
        return action
