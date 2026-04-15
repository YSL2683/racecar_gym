from abc import ABC, abstractmethod
from typing import Dict
import numpy as np

# ── Policy base class ─────────────────────────────────────────────────────────

class Policy(ABC):
    """Subclass this and implement ``act()`` to plug in your trained policy."""

    @abstractmethod
    def act(self, observation: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Compute an action from the current observation.

        Args:
            observation: sensor readings keyed by sensor name.
                Available sensors (defined in models/vehicles/racecar/racecar.yml).
                Only sensors listed in the scenario YAML will be present.
                  'lidar'         — np.ndarray, shape (1080,), values in [0.25, 15.25] metres
                  'pose'          — np.ndarray, shape (6,)  [x, y, z, roll, pitch, yaw]
                  'velocity'      — np.ndarray, shape (6,)  [vx, vy, vz, wx, wy, wz]
                  'acceleration'  — np.ndarray, shape (6,)  [ax, ay, az, α_x, α_y, α_z]
                  'rgb_camera'    — np.ndarray, shape (128, 128, 3), dtype uint8, range [0, 255]
                  'hd_camera'     — np.ndarray, shape (240, 320, 3), dtype uint8, range [0, 255]
                  'low_res_camera'— np.ndarray, shape (64,  64,  3), dtype uint8, range [0, 255]
                  'time'          — np.float32

        Returns:
            action dict.  All values must be in [-1, 1].
                'motor'    — np.ndarray, shape (1,)  throttle (negative = reverse)
                'steering' — np.ndarray, shape (1,)  steering angle
        """