"""
DQN CNN Policy — wraps SB3 DQN with a LiDAR CNN feature extractor.

Observation layout (1086 dims):
  [0:1080]    Full 1080-ray LiDAR, normalised to [-1, 1]
  [1080:1086] Velocity (linear 3 + angular 3), normalised to [-1, 1]

The LiDAR branch uses Conv2d with H=1, which is functionally identical
to Conv1d but avoids a separate Conv1d path. Velocity is processed by a
small FC branch, then concatenated before the final projection.

SB3 integration:
  Use "MlpPolicy" + policy_kwargs={"features_extractor_class": LiDARCNNExtractor,
                                   "features_extractor_kwargs": {"features_dim": 256}}
  This is SB3's official mechanism for custom observation encoders.

Policy file contract
--------------------
Expose a top-level function::

    def make_policy() -> Policy

The model path is resolved in this order:
  1. Environment variable DQN_CNN_MODEL_PATH
  2. Argument passed directly to DQNCNNPolicy(model_path=...)
  3. Default: checkpoints/dqn_cnn/final_model.zip
"""

import os
from typing import Dict

import numpy as np
import torch as th
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from policy.base_policy import Policy

# ── Discrete action table ──────────────────────────────────────────────────────
# 4 actions: motor ∈ {-1, +1}  ×  steering ∈ {-1, 0, +1} (motor -1 for only brake)
# Index layout:
#   0: motor=-1, steering= 0   (brake + straight)
#   1: motor=+1, steering=-1   (forward + left)
#   2: motor=+1, steering= 0   (forward + straight)
#   3: motor=+1, steering=+1   (forward + right)
_ACTION_TABLE = np.array(
    [
        [-1.0,  0.0],   # 0
        [ 1.0, -1.0],   # 1
        [ 1.0,  0.0],   # 2
        [ 1.0,  1.0],   # 3
    ],
    dtype=np.float32,
)  # shape: (4, 2)  — columns: [motor, steering]

# ── Observation constants ──────────────────────────────────────────────────────
_OBS_DIM = 1086  # 1080 full LiDAR + 6 velocity

# ── Normalization constants ────────────────────────────────────────────────────
# LiDAR: sensor range [0.25, 15.0] m  (racecar.yml: min_range=0.25, range=15.0)
_LIDAR_MIN = np.float32(0.25)
_LIDAR_RANGE = np.float32(15.0 - 0.25)   # 14.75 → mapped to [-1, 1]

# Velocity: from racecar.yml max_linear_velocity
_VEL_LINEAR_MAX = np.float32(14.0)        # m/s
_VEL_ANGULAR_MAX = np.float32(6.0)        # rad/s


def _preprocess_obs(obs: Dict[str, np.ndarray]) -> np.ndarray:
    """Flatten and normalise observations into a 1-D float32 array in [-1, 1].

    Observation layout (1086 dims):
      [0:1080]    Full 1080-ray LiDAR, normalised to [-1, 1]
                  min_range=0.25 m → -1,  max_range=15.0 m → +1
      [1080:1083] Linear velocity (vx, vy, vz), body-frame, normalised to [-1, 1]
      [1083:1086] Angular velocity (wx, wy, wz), body-frame, normalised to [-1, 1]
    """
    lidar_raw = np.clip(
        obs["lidar"], _LIDAR_MIN, _LIDAR_MIN + _LIDAR_RANGE
    ).astype(np.float32)
    lidar = (lidar_raw - _LIDAR_MIN) / _LIDAR_RANGE * 2.0 - 1.0

    vel_raw = np.asarray(obs.get("velocity", np.zeros(6)), dtype=np.float32).reshape(-1)
    linear = np.clip(vel_raw[:3], -_VEL_LINEAR_MAX, _VEL_LINEAR_MAX) / _VEL_LINEAR_MAX
    angular = np.clip(vel_raw[3:6], -_VEL_ANGULAR_MAX, _VEL_ANGULAR_MAX) / _VEL_ANGULAR_MAX

    return np.concatenate([lidar, linear.astype(np.float32), angular.astype(np.float32)])


def _discrete_to_action(action_idx: int) -> Dict[str, np.ndarray]:
    """Convert a discrete action index to the {motor, steering} dict expected by the env."""
    motor, steering = _ACTION_TABLE[int(action_idx)]
    return {
        "motor": np.array([motor], dtype=np.float32),
        "steering": np.array([steering], dtype=np.float32),
    }


class LiDARCNNExtractor(BaseFeaturesExtractor):
    """SB3 feature extractor: Conv2d(H=1) on LiDAR + FC on velocity.

    Registered via policy_kwargs so SB3's MlpPolicy uses it as the observation
    encoder before applying the DQN Q-network head.

    Architecture
    ------------
    LiDAR branch  (1080 → 128):
        reshape (B, 1, 1, 1080)
        → Conv2d(1→32,  k=(1,8), s=(1,4)) → ReLU
        → Conv2d(32→64, k=(1,4), s=(1,2)) → ReLU
        → Conv2d(64→64, k=(1,3), s=(1,1)) → ReLU
        → Flatten → Linear(cnn_out→128)   → ReLU

    Velocity branch  (6 → 32):
        Linear(6→32) → ReLU

    Merge:
        Concat(128 + 32 = 160) → Linear(160→features_dim) → ReLU
    """

    def __init__(self, observation_space: gym.Space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        self._lidar_dim = 1080
        self._vel_dim = 6

        self._lidar_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 8), stride=(1, 4)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(1, 4), stride=(1, 2)),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(1, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute flattened CNN output size dynamically to avoid hardcoding
        with th.no_grad():
            dummy = th.zeros(1, 1, 1, self._lidar_dim)
            cnn_flat_dim = self._lidar_cnn(dummy).shape[1]

        self._lidar_fc = nn.Sequential(
            nn.Linear(cnn_flat_dim, 128),
            nn.ReLU(),
        )

        self._vel_fc = nn.Sequential(
            nn.Linear(self._vel_dim, 32),
            nn.ReLU(),
        )

        self._merge = nn.Sequential(
            nn.Linear(128 + 32, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        lidar = observations[:, : self._lidar_dim]
        vel = observations[:, self._lidar_dim :]

        # (B, 1080) → (B, 1, 1, 1080) for Conv2d with kernel=(1, k)
        lidar_feat = self._lidar_fc(self._lidar_cnn(lidar.unsqueeze(1).unsqueeze(2)))
        vel_feat = self._vel_fc(vel)

        return self._merge(th.cat([lidar_feat, vel_feat], dim=1))


class DQNCNNPolicy(Policy):
    """Wraps a trained SB3 DQN model (with LiDARCNNExtractor) for inference.

    Args:
        model_path: Path to the SB3 DQN .zip checkpoint.
                    Falls back to env var DQN_CNN_MODEL_PATH, then
                    'checkpoints/dqn_cnn/final_model.zip'.
    """

    def __init__(self, model_path: str = None):
        from stable_baselines3 import DQN

        if model_path is None:
            model_path = os.environ.get(
                "DQN_CNN_MODEL_PATH",
                os.path.join("checkpoints", "dqn_cnn", "final_model.zip"),
            )
        self._model = DQN.load(
            model_path,
            custom_objects={"features_extractor_class": LiDARCNNExtractor},
        )
        self._model.policy.set_training_mode(False)

    def act(self, observation: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Compute a discrete action from the current sensor observation.

        Args:
            observation: sensor dict from the environment (lidar, velocity).

        Returns:
            {'motor': np.ndarray([±1], float32),
             'steering': np.ndarray([−1|0|1], float32)}
        """
        flat_obs = _preprocess_obs(observation)
        action_idx, _ = self._model.predict(flat_obs, deterministic=True)
        return _discrete_to_action(action_idx)


def make_policy() -> DQNCNNPolicy:
    """Factory function required by the policy file contract."""
    return DQNCNNPolicy()
