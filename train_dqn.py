"""
train_dqn.py — Train a DQN agent on a 2-agent racecar_gym scenario.

Usage
-----
    python train_dqn.py [options]

All hyperparameters and paths are configurable via CLI arguments.
Run `python train_dqn.py --help` for the full list.

tensorboard logging
-----
To visualize training progress in TensorBoard, run the following command after starting training:

    tensorboard --logdir outputs/
    ex. tensorboard --logdir outputs/20260421_114936/logs

Then open the provided URL (e.g., http://localhost:6006) in a web browser to see training curves and metrics.
"""

import argparse
import os
from datetime import datetime

import gymnasium
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback

import racecar_gym.envs.gym_api  # registers gymnasium env IDs
from policy.dqn_policy import _ACTION_TABLE, _OBS_DIM, _preprocess_obs, _discrete_to_action


# ── Wrappers ───────────────────────────────────────────────────────────────────

class MultiAgentSharedPolicyWrapper(gymnasium.Env):
    """Adapts a MultiAgentRaceEnv to a single-agent gymnasium.Env for SB3.

    How it works:
    - The underlying env has N agents (A, B, …).
    - At each step, the same discrete action index is applied to ALL agents.
      This implements a shared (parameter-tied) policy for self-play training.
    - SB3 only sees agent A's observation and reward, so training is driven
      by agent A's experience while B implicitly follows the same policy.
    - Episode terminates when ANY agent's done flag becomes True.

    Observation space: flat Box(111,) — lidar(108) + yaw sin/cos(2) + linear_speed(1), normalised to [-1, 1]
    Action space:      Discrete(6)
    """

    def __init__(self, scenario: str, render_mode: str = 'rgb_array_follow', reset_mode: str = 'grid'):
        super().__init__()
        kwargs = {'scenario': scenario, 'render_mode': render_mode}
        self._env = gymnasium.make("MultiAgentRaceEnv-v0", **kwargs)
        self.action_space = gymnasium.spaces.Discrete(len(_ACTION_TABLE))
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(_OBS_DIM,),
            dtype=np.float32,
        )
        self._agent_ids = list(self._env.observation_space.spaces.keys())
        # default reset mode used when reset() is called without options
        self._reset_mode = reset_mode

    def reset(self, *, seed=None, options=None):
        obs_dict, info_dict = self._env.reset(
            seed=seed, options=options or {"mode": "grid"}
        )
        primary_id = self._agent_ids[0]
        return _preprocess_obs(obs_dict[primary_id]), info_dict.get(primary_id, {})

    def step(self, action):
        agent_action = _discrete_to_action(action)
        multi_action = {aid: agent_action for aid in self._agent_ids}

        obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict = (
            self._env.step(multi_action)
        )

        primary_id = self._agent_ids[0]
        obs = _preprocess_obs(obs_dict[primary_id])
        reward = float(reward_dict[primary_id])

        # terminated: MultiAgentRaceEnv returns a per-agent dones dict → aggregate with any()
        terminated = any(terminated_dict.values())
        # truncated: MultiAgentRaceEnv returns a single bool (always False), not a dict
        truncated = bool(truncated_dict)

        return obs, reward, terminated, truncated, info_dict.get(primary_id, {})

    def close(self):
        self._env.close()


# ── Argument parsing ───────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DQN on racecar_gym (shared-policy 2-agent Austria)"
    )

    # ── Paths ──
    parser.add_argument(
        "--scenario", default=os.path.join("scenarios", "train_austria.yml"),
        help="Path to the scenario YAML file (default: scenarios/train_austria.yml)")
    parser.add_argument(
        "--output_dir", default="outputs",
        help="Base output directory; a timestamped subfolder is created inside (default: outputs/)")

    # ── Training duration ──
    parser.add_argument(
        "--total_timesteps", type=int, default=3_000_000,
        help="Total environment steps to train for (default: 3_000_000)")
    parser.add_argument(
        "--render-mode", type=str, default='rgb_array_follow',
        choices=['rgb_array_follow', 'rgb_array', 'human'],
        help="Render mode for the environment (default: 'rgb_array_follow')")
    parser.add_argument(
        "--reset-mode", type=str, default='grid',
        choices=['grid', 'random', 'random_ball'],
        help="Reset mode passed to env.reset(options={'mode':...}) (default: 'grid')")

    # ── DQN hyperparameters ──
    parser.add_argument("--learning_rate", type=float, default=1e-4,
        help="Optimizer learning rate (default: 1e-4)")
    parser.add_argument("--buffer_size", type=int, default=500_000,
        help="Replay buffer size (default: 500_000)")
    parser.add_argument("--learning_starts", type=int, default=10_000,
        help="Steps before learning begins (default: 10_000)")
    parser.add_argument("--batch_size", type=int, default=64,
        help="Mini-batch size for gradient updates (default: 64)")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="Discount factor (default: 0.99)")
    parser.add_argument("--target_update_interval", type=int, default=1_000,
        help="Steps between target network updates (default: 1_000)")
    parser.add_argument("--exploration_fraction", type=float, default=0.80,
        help="Fraction of training over which epsilon decays (default: 0.80)")
    parser.add_argument("--exploration_final_eps", type=float, default=0.10,
        help="Final epsilon for exploration (default: 0.10)")

    return parser.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Create a timestamped run directory: outputs/YYYYMMDD_HHMMSS/
    run_dir = os.path.join(args.output_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    save_path = os.path.join(run_dir, "checkpoints")
    log_path = os.path.join(run_dir, "logs")
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    print(f"[train_dqn] scenario      : {args.scenario}")
    print(f"[train_dqn] total_timesteps: {args.total_timesteps:,}")
    print(f"[train_dqn] run_dir        : {run_dir}")
    print(f"[train_dqn] save_path      : {save_path}")
    print(f"[train_dqn] log_path       : {log_path}")

    # Build the wrapped env
    env = MultiAgentSharedPolicyWrapper(scenario=args.scenario, render_mode=args.render_mode, reset_mode=args.reset_mode)

    # Checkpoint callback: save every (total_timesteps // 10) steps
    checkpoint_freq = max(1, args.total_timesteps // 10)
    checkpoint_cb = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=save_path,
        name_prefix="dqn_racecar",
        verbose=1,
    )

    # Build the DQN model
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        gamma=args.gamma,
        target_update_interval=args.target_update_interval,
        exploration_fraction=args.exploration_fraction,
        exploration_final_eps=args.exploration_final_eps,
        verbose=1,
        tensorboard_log=log_path,
    )

    # Train
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=checkpoint_cb,
        progress_bar=True,
    )

    # Save final model
    final_path = os.path.join(save_path, "final_model")
    model.save(final_path)
    print(f"\n[train_dqn] Training complete. Final model saved to {final_path}.zip")

    env.close()


if __name__ == "__main__":
    main()
