#!/usr/bin/env python3
"""
training/train_dqn.py

Offline DQN trainer for the port-level agent.
Reads data/features.csv and learns a simple policy:
- state = [tx_bps_norm, rx_bps_norm, dpid_norm, port_norm]
- reward = - rx_bps_norm (penalize high ingress load)
  with an extra penalty when rx_bps_norm > 0.8 (congestion)

Outputs: agents/models/dqn_port.pt which the controller auto-loads.
"""

import os
import sys
import csv
import math
import time
import random
from collections import defaultdict

import numpy as np
import torch

# ----- resolve project root -----
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))  # ~/Documents/project1
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agents.decision.dqn_agent import DQNAgent, DQNConfig

FEATURES_CSV = os.path.join(PROJECT_ROOT, "data", "features.csv")
MODEL_DIR    = os.path.join(PROJECT_ROOT, "agents", "models")
MODEL_PATH   = os.path.join(MODEL_DIR, "dqn_port.pt")

# Assume 100 Mbps normalization (adjust if you like)
BPS_SCALE = 100e6

def load_features(path):
    """
    Expect header: timestamp, dpid, port_no, tx_bps, rx_bps, tx_pps, rx_pps
    Returns list of dict rows with floats/ints as appropriate.
    """
    rows = []
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                rows.append({
                    "timestamp": float(row["timestamp"]),
                    "dpid": int(row["dpid"]),
                    "port_no": int(row["port_no"]),
                    "tx_bps": float(row["tx_bps"]),
                    "rx_bps": float(row["rx_bps"]),
                    "tx_pps": float(row["tx_pps"]),
                    "rx_pps": float(row["rx_pps"]),
                })
            except Exception:
                continue
    return rows

def group_by_port(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["dpid"], r["port_no"])].append(r)
    # sort each group by time
    for k in groups:
        groups[k].sort(key=lambda x: x["timestamp"])
    return groups

def norm_state(dpid, port_no, tx_bps, rx_bps):
    tx_n = max(0.0, min(1.0, tx_bps / BPS_SCALE))
    rx_n = max(0.0, min(1.0, rx_bps / BPS_SCALE))
    dpid_n = min(1.0, dpid / 100.0)
    port_n = min(1.0, port_no / 100.0)
    return np.array([tx_n, rx_n, dpid_n, port_n], dtype=np.float32)

def reward_from_state(state_vec):
    """
    Simple shaping:
      base = - rx_bps_norm
      penalty if rx_bps_norm > 0.8
    """
    rx_n = float(state_vec[1])
    rew = -rx_n
    if rx_n > 0.8:
        rew -= 0.5 * (rx_n - 0.8) / 0.2  # extra penalty up to -0.5
    return rew

def make_transitions(groups):
    """
    Build (s, a, r, s2, done) tuples.
    We don't have ground-truth actions offline; treat action as 0 (noop) for buffer population,
    since DQN update uses max_a' Q(s2,a') for target and learns from rewards.
    """
    transitions = []
    for (dpid, port), seq in groups.items():
        if len(seq) < 2:
            continue
        for i in range(len(seq) - 1):
            cur = seq[i]
            nxt = seq[i + 1]
            s  = norm_state(dpid, port, cur["tx_bps"], cur["rx_bps"])
            s2 = norm_state(dpid, port, nxt["tx_bps"], nxt["rx_bps"])
            r  = reward_from_state(s2)  # reward based on *next* state
            a  = 0  # placeholder (noop)
            done = 0.0 if (i + 1) < (len(seq) - 1) else 1.0
            transitions.append((s, a, r, s2, done))
    return transitions

def train_offline(transitions, epochs=5, steps_per_epoch=5000, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    os.makedirs(MODEL_DIR, exist_ok=True)
    cfg = DQNConfig(
        state_dim=4,
        n_actions=4,
        hidden=64,
        gamma=0.99,
        lr=1e-3,
        batch_size=64,
        buffer_size=50000,
        start_epsilon=0.2,  # lower explore for offline
        end_epsilon=0.05,
        epsilon_decay=20000,
        target_update_every=1000
    )
    agent = DQNAgent(cfg)

    # Pre-fill buffer with offline transitions
    for (s, a, r, s2, done) in transitions:
        agent.remember(s, a, r, s2, done)

    # Offline training loop
    for ep in range(1, epochs + 1):
        losses = []
        for step in range(steps_per_epoch):
            loss = agent.train_step()
            if loss:
                losses.append(loss)
        avg_loss = (sum(losses) / len(losses)) if losses else 0.0
        print(f"[epoch {ep}/{epochs}] avg_loss={avg_loss:.6f} buffer={len(agent.buffer)}")
        # refresh target
        agent.target.load_state_dict(agent.q.state_dict())

    # Save model
    agent.save(MODEL_PATH)
    print(f"[+] Saved model to {MODEL_PATH}")

def main():
    if not os.path.exists(FEATURES_CSV):
        print(f"[!] features file missing: {FEATURES_CSV}")
        return
    rows = load_features(FEATURES_CSV)
    if not rows:
        print(f"[!] no rows in {FEATURES_CSV}")
        return
    groups = group_by_port(rows)
    transitions = make_transitions(groups)
    if not transitions:
        print("[!] no transitions built (need >=2 samples per (dpid,port))")
        return
    print(f"[i] transitions: {len(transitions)} across {len(groups)} ports")
    train_offline(transitions, epochs=5, steps_per_epoch=5000, seed=42)

if __name__ == "__main__":
    main()

