"""
DQN Policy — wraps a trained Stable Baselines3 DQN model for use in racecar_gym.

Policy file contract
--------------------
Expose a top-level function::

    def make_policy() -> Policy

The model path is resolved in this order:
  1. Environment variable DQN_MODEL_PATH
  2. Argument passed directly to DQNPolicy(model_path=...)
  3. Default: checkpoints/dqn/final_model.zip
"""

import os
from typing import Dict

import numpy as np

from policy.base_policy import Policy

# ── Discrete action table ──────────────────────────────────────────────────────
# 4 actions: motor ∈ {-1, +1}  ×  steering ∈ {-1, 0, +1} (motor -1 for only break {-1, 0})
# Index layout:
#   1: motor=-1, steering= 0   (reverse + straight)
#   2: motor=+1, steering=-1   (forward + left)
#   3: motor=+1, steering= 0   (forward + straight)
#   4: motor=+1, steering=+1   (forward + right)
_ACTION_TABLE = np.array(
    [
        [-1.0,  0.0],   # 1
        [ 1.0, -1.0],   # 2
        [ 1.0,  0.0],   # 3
        [ 1.0,  1.0],   # 4
    ],
    dtype=np.float32,
)  # shape: (4, 2)  — columns: [motor, steering]

# ── Observation preprocessing ──────────────────────────────────────────────────
# LiDAR: 1080 rays → 216 rays (every 5th ray, same angular coverage)
_LIDAR_STRIDE = 5
_LIDAR_OUT = 1080 // _LIDAR_STRIDE  # 216

_OBS_DIM = _LIDAR_OUT + 6  # 222: lidar(216) + velocity(6)

# ── Normalization constants ────────────────────────────────────────────────────
# LiDAR: sensor range [0.25, 15.0] m  (racecar.yml: min_range=0.25, range=15.0)
_LIDAR_MIN = np.float32(0.25)
_LIDAR_RANGE = np.float32(15.0 - 0.25)   # 14.75 → mapped to [-1, 1]

# Velocity: from racecar.yml max_linear_velocity
_VEL_LINEAR_MAX = np.float32(14.0)        # m/s
_VEL_ANGULAR_MAX = np.float32(6.0)        # rad/s (upper bound for angular rates)


def _preprocess_obs(obs: Dict[str, np.ndarray]) -> np.ndarray:
    """Flatten and normalise sensor observations into a 1-D numpy array in [-1, 1].

    Observation layout (222 dims total):
      [0:_LIDAR_OUT]    lidar, downsampled (stride=5), normalised to [-1, 1]
                        min_range=0.25 m → -1,  max_range=15.0 m → +1
      [_LIDAR_OUT: _LIDAR_OUT+3]  linear velocity (vx, vy, vz), body-frame, normalised to [-1,1]
      [_LIDAR_OUT+3: _LIDAR_OUT+6] angular velocity (wx, wy, wz), body-frame, normalised to [-1,1]

    Notes:
      - All outputs are float32 and in [-1, 1].
      - Linear components are clipped to ±_VEL_LINEAR_MAX, angular to ±_VEL_ANGULAR_MAX.
    """
    # LiDAR: clip to [min_range, max_range] → min-max → [-1, 1]
    lidar_raw = np.clip(
        obs["lidar"][::_LIDAR_STRIDE], _LIDAR_MIN, _LIDAR_MIN + _LIDAR_RANGE
    ).astype(np.float32)
    lidar = (lidar_raw - _LIDAR_MIN) / _LIDAR_RANGE * 2.0 - 1.0

    # Velocity: body-frame linear (3) + angular (3)
    vel_raw = np.asarray(obs.get("velocity", np.zeros(6)), dtype=np.float32).reshape(-1)
    # Linear components
    linear = np.clip(vel_raw[:3], -_VEL_LINEAR_MAX, _VEL_LINEAR_MAX) / _VEL_LINEAR_MAX
    # Angular components
    angular = np.clip(vel_raw[3:6], -_VEL_ANGULAR_MAX, _VEL_ANGULAR_MAX) / _VEL_ANGULAR_MAX

    linear = linear.astype(np.float32)
    angular = angular.astype(np.float32)

    vel_norm = np.concatenate([linear, angular])  # shape (6,)

    return np.concatenate([lidar, vel_norm])


def _discrete_to_action(action_idx: int) -> Dict[str, np.ndarray]:
    """Convert a discrete action index to the {motor, steering} dict expected by the env."""
    motor, steering = _ACTION_TABLE[int(action_idx)]
    return {
        "motor": np.array([motor], dtype=np.float32),
        "steering": np.array([steering], dtype=np.float32),
    }


class DQNPolicy(Policy):
    """Wraps a trained SB3 DQN model for inference inside racecar_gym.

    Args:
        model_path: Path to the SB3 DQN .zip checkpoint.
                    Falls back to env var DQN_MODEL_PATH, then
                    'checkpoints/dqn/final_model.zip'.
    """

    def __init__(self, model_path: str = None):
        from stable_baselines3 import DQN  # imported here to keep the file importable without SB3

        if model_path is None:
            model_path = os.environ.get(
                "DQN_MODEL_PATH",
                os.path.join("checkpoints", "dqn", "final_model.zip"),
            )
        self._model = DQN.load(model_path)
        self._model.policy.set_training_mode(False)

    def act(self, observation: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Compute a discrete action from the current sensor observation.

        Args:
            observation: sensor dict from the environment (lidar, pose,
                         velocity, acceleration).

        Returns:
            {'motor': np.ndarray([±1], float32),
             'steering': np.ndarray([−1|0|1], float32)}
        """
        flat_obs = _preprocess_obs(observation)
        action_idx, _ = self._model.predict(flat_obs, deterministic=True)
        return _discrete_to_action(action_idx)


def make_policy() -> DQNPolicy:
    """Factory function required by the policy file contract."""
    return DQNPolicy()
