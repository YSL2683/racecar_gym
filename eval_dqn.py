"""Evaluate a trained DQN checkpoint locally (no server required).

Usage:
    python eval_dqn.py --checkpoint checkpoints/dqn/20260417_195354/checkpoint_000100
    python eval_dqn.py --checkpoint checkpoints/dqn/20260417_195354/checkpoint_000100 \\
                       --scenario scenarios/eval_austria.yml \\
                       --episodes 5 \\
                       --render-mode human

Note:
    Pass the specific checkpoint directory (e.g. .../checkpoint_000100), not the run
    directory.  The script uses Policy.from_checkpoint() so no extra PyBullet physics
    server is spawned by RLlib workers — only the single eval environment connects.
"""
import argparse
import glob
import os
import sys
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ.setdefault("PYTHONWARNINGS", "ignore::DeprecationWarning")

import ray
from ray.rllib.policy.policy import Policy
from ray.tune.registry import register_env

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from policy.dqn_policy import RacecarDQNEnv


def _resolve_checkpoint(path: str) -> str:
    """Accept either a run dir or a specific checkpoint dir."""
    if os.path.exists(os.path.join(path, "rllib_checkpoint.json")):
        return path
    subdirs = sorted(glob.glob(os.path.join(path, "checkpoint_*")))
    if subdirs:
        resolved = subdirs[-1]
        print(f"Auto-selected checkpoint: {resolved}")
        return resolved
    raise ValueError(f"No valid checkpoint found at: {path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate DQN policy on racecar env')
    parser.add_argument('--checkpoint',  required=True,
                        help='Path to RLlib checkpoint directory (run dir or specific checkpoint)')
    parser.add_argument('--scenario',    default='scenarios/eval_austria.yml',
                        help='Scenario YAML path')
    parser.add_argument('--episodes',    type=int, default=5,
                        help='Number of evaluation episodes')
    parser.add_argument('--render-mode', default='rgb_array_follow',
                        choices=['human', 'rgb_array_follow',
                                 'rgb_array_birds_eye'],
                        help='Render mode (human opens PyBullet GUI)')
    args = parser.parse_args()

    checkpoint_dir = _resolve_checkpoint(args.checkpoint)
    policy_dir = os.path.join(checkpoint_dir, "policies", "shared")
    if not os.path.isdir(policy_dir):
        raise FileNotFoundError(
            f"Policy directory not found: {policy_dir}\n"
            "Make sure --checkpoint points to a valid RLlib checkpoint."
        )

    ray.init(ignore_reinit_error=True)
    register_env('racecar_dqn', lambda cfg: RacecarDQNEnv(cfg))

    # Load only policy weights — no RLlib workers/environments are created,
    # so only ONE PyBullet physics server (the eval env below) will exist.
    policy = Policy.from_checkpoint(policy_dir)

    env = RacecarDQNEnv({
        'scenario':    args.scenario,
        'render_mode': args.render_mode,
    })

    for ep in range(args.episodes):
        obs, _  = env.reset()
        done    = False
        totals  = {aid: 0.0 for aid in env._agents}

        while not done:
            actions = {
                aid: int(policy.compute_single_action(obs[aid], explore=False)[0])
                for aid in env._agents
            }
            obs, rewards, terminateds, truncateds, _ = env.step(actions)
            for aid in env._agents:
                totals[aid] += rewards[aid]
            done = terminateds.get('__all__', False) or truncateds.get('__all__', False)

        reward_str = '  '.join(f'{aid}={totals[aid]:.1f}' for aid in env._agents)
        print(f'Episode {ep + 1}/{args.episodes}: {reward_str}')

    env._env.close()
    ray.shutdown()


if __name__ == '__main__':
    main()
