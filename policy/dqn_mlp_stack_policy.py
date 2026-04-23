"""
DQN MLP-Stack Policy — LiDAR 4-frame stack + velocity + acceleration.

Observation layout (4332 dims):
  [0    : 4320]  4 stacked LiDAR frames (4 × 1080 rays), each normalised to [-1, 1]
                 oldest → newest (left to right)
  [4320 : 4326]  Velocity (linear 3 + angular 3),     normalised to [-1, 1]
  [4326 : 4332]  Acceleration (linear 3 + angular 3), normalised to [-1, 1]

Architecture (BaseFeaturesExtractor):
  LiDAR-stack branch  (4320 → 512 → 256, ReLU)
  State branch        (12   → 64,        ReLU)
  Merge               (320  → features, no compression)

SB3 integration:
  Use "MlpPolicy" + policy_kwargs={
      "features_extractor_class": LiDARStackMLPExtractor,
      "features_extractor_kwargs": {"features_dim": 320},
  }

Policy file contract
--------------------
Expose a top-level function::

    def make_policy() -> Policy

The model path is resolved in this order:
  1. Environment variable DQN_STACK_MODEL_PATH
  2. Argument passed directly to DQNStackPolicy(model_path=...)
  3. Default: checkpoints/dqn_stack/final_model.zip
"""

import os
from collections import deque
from typing import Dict, Optional

import numpy as np
import torch as th
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from policy.base_policy import Policy

# ── Discrete action table ──────────────────────────────────────────────────────
# Identical to dqn_mlp_policy / dqn_cnn_policy (shared action space)
_ACTION_TABLE = np.array(
    [
        [-1.0,  0.0],   # 0  brake + straight
        [ 1.0, -1.0],   # 1  forward + left
        [ 1.0,  0.0],   # 2  forward + straight
        [ 1.0,  1.0],   # 3  forward + right
    ],
    dtype=np.float32,
)  # shape: (4, 2) — columns: [motor, steering]

# ── Observation constants ──────────────────────────────────────────────────────
_STACK_FRAMES = 4
_LIDAR_DIM    = 1080
_STATE_DIM    = 12     # vel(6) + acc(6)
_OBS_DIM      = _STACK_FRAMES * _LIDAR_DIM + _STATE_DIM  # 4332

# ── Normalisation constants ────────────────────────────────────────────────────
_LIDAR_MIN   = np.float32(0.25)
_LIDAR_RANGE = np.float32(15.0 - 0.25)   # 14.75 m  → mapped to [-1, 1]

_VEL_LINEAR_MAX  = np.float32(14.0)   # m/s
_VEL_ANGULAR_MAX = np.float32(6.0)    # rad/s

# Acceleration sensor has no inherent bound (racecar.yml: linear_bounds=[inf,inf,inf]).
# Values below are derived from typical racecar dynamics and used for clipping.
_ACC_LINEAR_MAX  = np.float32(50.0)   # m/s²
_ACC_ANGULAR_MAX = np.float32(30.0)   # rad/s²


# ── Single-frame preprocessing ─────────────────────────────────────────────────

def _preprocess_single_obs(obs: Dict[str, np.ndarray]) -> np.ndarray:
    """Preprocess one observation dict into a 1092-dim float32 array in [-1, 1].

    Layout:
      [0    : 1080]  LiDAR (1080 rays)
      [1080 : 1086]  Velocity (linear 3 + angular 3)
      [1086 : 1092]  Acceleration (linear 3 + angular 3)
    """
    # LiDAR
    lidar_raw = np.clip(
        obs["lidar"], _LIDAR_MIN, _LIDAR_MIN + _LIDAR_RANGE
    ).astype(np.float32)
    lidar = (lidar_raw - _LIDAR_MIN) / _LIDAR_RANGE * 2.0 - 1.0

    # Velocity
    vel_raw = np.asarray(obs.get("velocity", np.zeros(6)), dtype=np.float32).reshape(-1)
    linear_vel  = np.clip(vel_raw[:3], -_VEL_LINEAR_MAX,  _VEL_LINEAR_MAX)  / _VEL_LINEAR_MAX
    angular_vel = np.clip(vel_raw[3:], -_VEL_ANGULAR_MAX, _VEL_ANGULAR_MAX) / _VEL_ANGULAR_MAX

    # Acceleration
    acc_raw = np.asarray(obs.get("acceleration", np.zeros(6)), dtype=np.float32).reshape(-1)
    linear_acc  = np.clip(acc_raw[:3], -_ACC_LINEAR_MAX,  _ACC_LINEAR_MAX)  / _ACC_LINEAR_MAX
    angular_acc = np.clip(acc_raw[3:], -_ACC_ANGULAR_MAX, _ACC_ANGULAR_MAX) / _ACC_ANGULAR_MAX

    return np.concatenate([
        lidar,
        linear_vel.astype(np.float32),
        angular_vel.astype(np.float32),
        linear_acc.astype(np.float32),
        angular_acc.astype(np.float32),
    ])  # shape: (1092,)


def _discrete_to_action(action_idx: int) -> Dict[str, np.ndarray]:
    """Convert a discrete action index to the {motor, steering} dict expected by the env."""
    motor, steering = _ACTION_TABLE[int(action_idx)]
    return {
        "motor":    np.array([motor],    dtype=np.float32),
        "steering": np.array([steering], dtype=np.float32),
    }


# ── Feature extractor ──────────────────────────────────────────────────────────

class LiDARStackMLPExtractor(BaseFeaturesExtractor):
    """SB3 feature extractor: dual MLP branches for stacked LiDAR and state.

    Input layout  (4332 dims):
      [0    : 4320]  stacked LiDAR  (4 × 1080)
      [4320 : 4332]  state          (vel 6 + acc 6)

    Architecture:
      LiDAR branch : 4320 → Linear(512) → ReLU → Linear(256) → ReLU
      State branch :   12 → Linear(64)  → ReLU
      Merge        :  320 → output features (no compression)
    """

    def __init__(self, observation_space: gym.Space, features_dim: int = 320):
        super().__init__(observation_space, features_dim)
        self._lidar_stack_dim = _STACK_FRAMES * _LIDAR_DIM  # 4320
        self._state_dim       = _STATE_DIM                  # 12

        self._lidar_branch = nn.Sequential(
            nn.Linear(self._lidar_stack_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        self._state_branch = nn.Sequential(
            nn.Linear(self._state_dim, 64),
            nn.ReLU(),
        )

        self._merge = nn.Identity()

    def forward(self, observations: th.Tensor) -> th.Tensor:
        lidar_stack = observations[:, : self._lidar_stack_dim]   # (B, 4320)
        state       = observations[:, self._lidar_stack_dim :]   # (B, 12)

        lidar_feat = self._lidar_branch(lidar_stack)
        state_feat = self._state_branch(state)

        return self._merge(th.cat([lidar_feat, state_feat], dim=1))


# ── Inference policy ───────────────────────────────────────────────────────────

class DQNStackPolicy(Policy):
    """Wraps a trained SB3 DQN model (with LiDARStackMLPExtractor) for inference.

    Maintains a per-instance LiDAR frame buffer (deque of length _STACK_FRAMES).
    Call ``reset()`` at the start of every episode to clear the buffer.

    Args:
        model_path: Path to the SB3 DQN .zip checkpoint.
                    Falls back to env var DQN_STACK_MODEL_PATH, then
                    'checkpoints/dqn_stack/final_model.zip'.
    """

    def __init__(self, model_path: Optional[str] = None):
        from stable_baselines3 import DQN

        if model_path is None:
            model_path = os.environ.get(
                "DQN_STACK_MODEL_PATH",
                os.path.join("checkpoints", "dqn_stack", "final_model.zip"),
            )
        self._model = DQN.load(
            model_path,
            custom_objects={"features_extractor_class": LiDARStackMLPExtractor},
        )
        self._model.policy.set_training_mode(False)
        self._lidar_buffer: Optional[deque] = None

    def reset(self) -> None:
        """Clear the LiDAR frame buffer. Must be called at episode start."""
        self._lidar_buffer = None

    def act(self, observation: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Compute a discrete action from the current sensor observation.

        Args:
            observation: sensor dict from the environment (lidar, velocity, acceleration).

        Returns:
            {'motor': np.ndarray([±1], float32),
             'steering': np.ndarray([−1|0|1], float32)}
        """
        single_obs = _preprocess_single_obs(observation)
        lidar = single_obs[:_LIDAR_DIM]
        state = single_obs[_LIDAR_DIM:]

        if self._lidar_buffer is None:
            # Initialise by repeating the first frame
            self._lidar_buffer = deque(
                [lidar.copy() for _ in range(_STACK_FRAMES)],
                maxlen=_STACK_FRAMES,
            )
        else:
            self._lidar_buffer.append(lidar.copy())

        flat_obs = np.concatenate([*self._lidar_buffer, state])
        action_idx, _ = self._model.predict(flat_obs, deterministic=True)
        return _discrete_to_action(action_idx)


def make_policy() -> DQNStackPolicy:
    """Factory function required by the policy file contract."""
    return DQNStackPolicy()
