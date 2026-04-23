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
from collections import deque
from datetime import datetime

import gymnasium
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback

import racecar_gym.envs.gym_api  # registers gymnasium env IDs
from policy.dqn_mlp_policy import _ACTION_TABLE, _discrete_to_action


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

    Observation space: flat Box(obs_dim,), normalised to [-1, 1]
    Action space:      Discrete(len(_ACTION_TABLE))
    """

    def __init__(
        self,
        scenario: str,
        render_mode: str = 'rgb_array_follow',
        reset_mode: str = 'grid',
        preprocess_fn=None,
        obs_dim: int = None,
        lidar_stack_frames: int = 1,
        lidar_dim: int = 0,
    ):
        super().__init__()
        kwargs = {'scenario': scenario, 'render_mode': render_mode}
        self._env = gymnasium.make("MultiAgentRaceEnv-v0", **kwargs)
        self.action_space = gymnasium.spaces.Discrete(len(_ACTION_TABLE))
        self.observation_space = gymnasium.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self._agent_ids = list(self._env.observation_space.spaces.keys())
        self._reset_mode = reset_mode
        self._preprocess_fn = preprocess_fn
        self._lidar_stack_frames = lidar_stack_frames
        self._lidar_dim = lidar_dim
        self._lidar_buffer = None

    def reset(self, *, seed=None, options=None):
        self._lidar_buffer = None   # clear frame buffer on episode reset
        obs_dict, info_dict = self._env.reset(
            seed=seed, options=options or {"mode": self._reset_mode}
        )
        primary_id = self._agent_ids[0]
        single_obs = self._preprocess_fn(obs_dict[primary_id])
        return self._build_stacked_obs(single_obs), info_dict.get(primary_id, {})

    def step(self, action):
        agent_action = _discrete_to_action(action)
        multi_action = {aid: agent_action for aid in self._agent_ids}

        obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict = (
            self._env.step(multi_action)
        )

        primary_id = self._agent_ids[0]
        single_obs = self._preprocess_fn(obs_dict[primary_id])
        obs = self._build_stacked_obs(single_obs)
        reward = float(reward_dict[primary_id])

        # terminated: MultiAgentRaceEnv returns a per-agent dones dict → aggregate with any()
        terminated = any(terminated_dict.values())
        # truncated: MultiAgentRaceEnv returns a single bool (always False), not a dict
        truncated = bool(truncated_dict)

        return obs, reward, terminated, truncated, info_dict.get(primary_id, {})

    def close(self):
        self._env.close()

    def _build_stacked_obs(self, single_obs: np.ndarray) -> np.ndarray:
        """Return obs with LiDAR frames stacked (no-op when lidar_stack_frames <= 1)."""
        if self._lidar_stack_frames <= 1:
            return single_obs
        lidar = single_obs[: self._lidar_dim]
        state = single_obs[self._lidar_dim :]
        if self._lidar_buffer is None:
            # Initialise by repeating the first frame
            self._lidar_buffer = deque(
                [lidar.copy() for _ in range(self._lidar_stack_frames)],
                maxlen=self._lidar_stack_frames,
            )
        else:
            self._lidar_buffer.append(lidar.copy())
        return np.concatenate([*self._lidar_buffer, state])


# ── Argument parsing ───────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DQN on racecar_gym (shared-policy 2-agent Austria)"
    )

    # ── Policy ──
    parser.add_argument(
        "--policy", choices=["mlp", "mlp_stack"], default="mlp",
        help="Policy type: mlp (MLP + downsampled lidar) or mlp_stack (MLP + 4-frame stacked full lidar) "
             "(default: mlp)")
    parser.add_argument(
        "--lidar_stack_frames", type=int, default=4,
        help="Number of LiDAR frames to stack for mlp_stack policy (default: 4)")

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
    print(f"[train_dqn] policy        : {args.policy}")
    print(f"[train_dqn] total_timesteps: {args.total_timesteps:,}")
    print(f"[train_dqn] run_dir        : {run_dir}")
    print(f"[train_dqn] save_path      : {save_path}")
    print(f"[train_dqn] log_path       : {log_path}")

    # Select preprocessing function and obs_dim based on policy type
    if args.policy == "mlp":
        from policy.dqn_mlp_policy import _preprocess_obs, _OBS_DIM
        preprocess_fn = _preprocess_obs
        obs_dim = _OBS_DIM
        policy_kwargs = {}
        lidar_stack_frames = 1
        lidar_dim = 0
    else:  # mlp_stack
        from policy.dqn_mlp_stack_policy import (
            _preprocess_single_obs, _LIDAR_DIM, _STATE_DIM, LiDARStackMLPExtractor, _DEFAULT_STACK_FRAMES
        )
        preprocess_fn = _preprocess_single_obs
        lidar_stack_frames = args.lidar_stack_frames
        obs_dim = lidar_stack_frames * _LIDAR_DIM + _STATE_DIM
        policy_kwargs = {
            "features_extractor_class": LiDARStackMLPExtractor,
            "features_extractor_kwargs": {"features_dim": 320, "stack_frames": lidar_stack_frames},
        }
        lidar_dim = _LIDAR_DIM

    # Build the wrapped env
    env = MultiAgentSharedPolicyWrapper(
        scenario=args.scenario,
        render_mode=args.render_mode,
        reset_mode=args.reset_mode,
        preprocess_fn=preprocess_fn,
        obs_dim=obs_dim,
        lidar_stack_frames=lidar_stack_frames,
        lidar_dim=lidar_dim,
    )

    # Checkpoint callback: save every (total_timesteps // 10) steps
    checkpoint_freq = max(1, args.total_timesteps // 10)
    checkpoint_cb = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=save_path,
        name_prefix="dqn_racecar",
        verbose=1,
    )

    # Build the DQN model
    # Device selection: prefer GPU if available
    try:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except Exception:
        device = "cpu"
    print(f"[train_dqn] device        : {device}")

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
        device=device,
        **({"policy_kwargs": policy_kwargs} if policy_kwargs else {}),
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
