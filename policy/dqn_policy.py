"""
DQN policy for racecar_gym.

Shared between:
  - train_dqn.py  (preprocessing / action mapping)
  - eval_dqn.py   (policy inference)
  - server-client client (pass with --policy policy/dqn_policy.py)

Usage (server-client):
    python -m server_client.client --agent-id A --policy policy/dqn_policy.py
"""
from __future__ import annotations

import os
from typing import Dict

import numpy as np

from policy.base_policy import Policy

# ── Action discretization ──────────────────────────────────────────────────────
MOTOR_BINS    = [-1.0, 0.0, 1.0]
STEERING_BINS = list(np.linspace(-1.0, 1.0, 7))  # 7 values: -1.0, -0.666..., 0.0, 0.666..., 1.0
ACTION_MAP    = [(m, s) for m in MOTOR_BINS for s in STEERING_BINS]  # 21 actions
N_ACTIONS     = len(ACTION_MAP)  # 21

# ── Observation normalization ──────────────────────────────────────────────────
LIDAR_MAX  = 15.25
POSE_SCALE = np.array([100., 100., 3., np.pi, np.pi, np.pi], dtype=np.float32)
VEL_SCALE  = np.array([14., 14., 14., 6., 6., 6.],           dtype=np.float32)
ACC_CLIP   = 10.0
TIME_LIMIT = 120.0
OBS_DIM    = 270 + 6 + 6 + 6 + 1   # 289  (lidar_down + pose + vel + acc + time)


def preprocess_obs(obs: Dict[str, np.ndarray]) -> np.ndarray:
    """Convert raw observation dict → normalized float32 vector of shape (OBS_DIM,)."""
    lidar = obs['lidar'][::4].astype(np.float32) / LIDAR_MAX
    pose  = obs['pose'].astype(np.float32) / POSE_SCALE
    vel   = obs['velocity'].astype(np.float32) / VEL_SCALE
    acc   = (np.clip(obs['acceleration'], -ACC_CLIP, ACC_CLIP).astype(np.float32) / ACC_CLIP)
    time  = np.array([float(obs['time']) / TIME_LIMIT], dtype=np.float32)
    return np.concatenate([lidar, pose, vel, acc, time])


def int_to_action(idx: int) -> Dict[str, np.ndarray]:
    """Convert discrete action index (0-20) → environment action dict."""
    motor, steering = ACTION_MAP[idx]
    return {
        'motor':    np.array([motor],    dtype=np.float32),
        'steering': np.array([steering], dtype=np.float32),
    }


# ── Policy class (server-client compatible) ────────────────────────────────────

class DQNPolicy(Policy):
    """Trained DQN policy."""

    def __init__(self, model_path: str = 'checkpoints/dqn/model.zip'):
        from stable_baselines3 import DQN
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"DQN model not found at '{model_path}'.")
        self.model = DQN.load(model_path, device='cpu')

    def act(self, observation: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        flat = preprocess_obs(observation)
        action_idx, _ = self.model.predict(flat, deterministic=True)
        action = int_to_action(int(action_idx))
        return action


def make_policy() -> Policy:
    """Entry point required by server_client.client."""
    return DQNPolicy()
