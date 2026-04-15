"""
Evaluation client: loads a trained policy and connects to the race server.

Usage:
  # With your own policy file:
  python -m server_client.client --agent-id A --policy my_policy.py

  # Quick test with the built-in random policy:
  python server_client/client.py --agent-id A

Policy file contract
--------------------
Your policy file must expose a top-level function::

    def make_policy() -> Policy:
        ...

where Policy is any object with an ``act(observation) -> action`` method.

Example policy file (e.g. my_policy.py):

    import torch
    from server_client.client import Policy

    class PPOPolicy(Policy):
        def __init__(self):
            self.model = torch.load('checkpoints/policy.pt')
            self.model.eval()

        def act(self, obs):
            import numpy as np
            lidar = obs['lidar']
            # ... pre-process, run model, post-process ...
            return {
                'motor':    np.array([0.5],  dtype=np.float32),
                'steering': np.array([-0.1], dtype=np.float32),
            }

    def make_policy():
        return PPOPolicy()
"""

import argparse
import importlib.util
import socket
import sys
import os
from abc import ABC, abstractmethod
from typing import Dict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server_client.utils import send_msg, recv_msg


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


class RandomPolicy(Policy):
    """Uniformly random actions — useful for testing the connection."""

    def act(self, observation: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return {
            'motor':    np.random.uniform(-1, 1, size=(1,)).astype(np.float32),
            'steering': np.random.uniform(-1, 1, size=(1,)).astype(np.float32),
        }


# ── Client ────────────────────────────────────────────────────────────────────

def run_client(agent_id: str, policy: Policy, host: str, port: int) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))
    print(f"Connected to {host}:{port}  as agent '{agent_id}'")

    # Handshake: tell the server which agent we are.
    send_msg(sock, {'agent_id': agent_id})

    ep_reward = 0.0
    total_reward = 0.0
    total_episodes = 0

    try:
        while True:
            msg = recv_msg(sock)
            if msg is None:
                print("Connection closed by server.")
                break

            if msg['type'] == 'close':
                print(
                    f"\nSession ended.  Episodes: {total_episodes}  |"
                    f"  Total reward: {total_reward:.2f}"
                )
                break

            elif msg['type'] == 'reset':
                ep_reward = 0.0
                episode = msg['episode'] + 1
                obs = msg['obs']
                print(f"── Episode {episode} started ──")

                action = policy.act(obs)
                send_msg(sock, {'action': action})

            elif msg['type'] == 'step':
                obs    = msg['obs']
                reward = msg['reward']
                done   = msg['done']
                ep_reward += reward

                if done:
                    total_reward += ep_reward
                    total_episodes += 1
                    print(f"  Episode done.  Reward: {ep_reward:.2f}")
                    # Do NOT send an action; wait for 'reset' or 'close'.
                else:
                    action = policy.act(obs)
                    send_msg(sock, {'action': action})

    finally:
        sock.close()


# ── Entry point ───────────────────────────────────────────────────────────────

def _load_policy_from_file(path: str) -> Policy:
    spec = importlib.util.spec_from_file_location('user_policy', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, 'make_policy'):
        raise AttributeError(
            f"'{path}' must define a top-level `make_policy()` function."
        )
    return module.make_policy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Racecar Gym evaluation client')
    parser.add_argument('--agent-id', type=str, required=True,
                        help="Agent ID matching the scenario YAML (e.g. 'A' or 'B')")
    parser.add_argument('--host', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=5555)
    parser.add_argument('--policy', type=str, default=None,
                        help='Path to a Python file exporting make_policy(). '
                             'Defaults to RandomPolicy if omitted.')
    args = parser.parse_args()

    if args.policy:
        policy = _load_policy_from_file(args.policy)
        print(f"Loaded policy from '{args.policy}'")
    else:
        print("No --policy provided — using RandomPolicy.")
        policy = RandomPolicy()

    run_client(
        agent_id=args.agent_id,
        policy=policy,
        host=args.host,
        port=args.port,
    )
