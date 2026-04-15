"""
Evaluation client: loads a trained policy and connects to the race server.

Usage:
  # With your own policy file:
  python -m server_client.client --agent-id A --policy my_policy.py

  # Quick test with the built-in random policy:
  python server_client/client.py --agent-id A
"""

import argparse
import importlib.util
import socket
import sys
import os
from typing import Dict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server_client.utils import send_msg, recv_msg
from policy.base_policy import Policy


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
