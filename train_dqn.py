"""
Train a shared DQN policy on MultiAgentRaceEnv (2 agents, parameter sharing).

Both agents experience the same track simultaneously; their transitions are
stored in a shared replay buffer so a single Q-network learns from both.

Usage:
    python train_dqn.py
    python train_dqn.py --timesteps 1000000 --save-path checkpoints/dqn/ --seed 0
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import gymnasium
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecEnv

from racecar_gym.envs.gym_api import MultiAgentRaceEnv
from policy.dqn_policy import N_ACTIONS, OBS_DIM, int_to_action, preprocess_obs


class MultiAgentDQNVecEnv(VecEnv):
    """Wraps MultiAgentRaceEnv (2 agents) as an SB3 VecEnv with n_envs=2.
    Both agents share a single Q-network (parameter sharing).
    """

    def __init__(
        self,
        scenario_path: str,
        render_mode: str = 'rgb_array_follow',
        reset_mode: str = 'grid',
    ) -> None:
        self._env = MultiAgentRaceEnv(scenario=scenario_path, render_mode=render_mode)
        self._agent_ids: List[str] = list(self._env.scenario.agents.keys())
        assert len(self._agent_ids) == 2, (
            f"Scenario must contain exactly 2 agents, got {self._agent_ids}"
        )
        self._reset_mode = reset_mode
        self._pending_actions: Optional[np.ndarray] = None

        obs_space = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)
        act_space = gymnasium.spaces.Discrete(N_ACTIONS)
        super().__init__(num_envs=2, observation_space=obs_space, action_space=act_space)
        self._ep_rewards = np.zeros(self.num_envs, dtype=np.float32)
        self._ep_lengths = np.zeros(self.num_envs, dtype=int)

    def reset(self) -> np.ndarray:
        obs_dict, _ = self._env.reset(options={'mode': self._reset_mode})
        return np.stack([preprocess_obs(obs_dict[aid]) for aid in self._agent_ids])

    def step_async(self, actions: np.ndarray) -> None:
        self._pending_actions = actions

    def step_wait(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        assert self._pending_actions is not None, "Call step_async before step_wait."

        action_dict = {
            self._agent_ids[i]: int_to_action(int(self._pending_actions[i]))
            for i in range(2)
        }
        obs_dict, rewards_dict, dones_dict, _, _ = self._env.step(action_dict)

        # episode_done = any(dones_dict.values())
        episode_done = all(dones_dict.values())  # only end episode when both agents are done

        obs     = np.stack([preprocess_obs(obs_dict[aid]) for aid in self._agent_ids])
        rewards = np.array([rewards_dict[aid] for aid in self._agent_ids], dtype=np.float32)
        self._ep_rewards += rewards
        self._ep_lengths += 1
        dones   = np.array([episode_done, episode_done], dtype=bool)
        infos: List[Dict] = [{} for _ in range(2)]

        if episode_done:
            for i, aid in enumerate(self._agent_ids):
                infos[i]['terminal_observation'] = preprocess_obs(obs_dict[aid])
                infos[i]['episode'] = {'r': float(self._ep_rewards[i]), 'l': int(self._ep_lengths[i])}
                print(f"[Episode] agent={aid} return={self._ep_rewards[i]:.2f} length={self._ep_lengths[i]}")
            self._ep_rewards[:] = 0.0
            self._ep_lengths[:] = 0
            reset_obs_dict, _ = self._env.reset(options={'mode': self._reset_mode})
            obs = np.stack([preprocess_obs(reset_obs_dict[aid]) for aid in self._agent_ids])

        self._pending_actions = None
        return obs, rewards, dones, infos

    def close(self) -> None:
        self._env.close()

    def get_attr(self, attr_name: str, indices=None) -> List[Any]:
        n = self.num_envs if indices is None else len(indices)
        return [getattr(self._env, attr_name, None)] * n

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        setattr(self._env, attr_name, value)

    def env_method(
        self,
        method_name: str,
        *method_args: Any,
        indices=None,
        **method_kwargs: Any,
    ) -> List[Any]:
        result = getattr(self._env, method_name)(*method_args, **method_kwargs)
        n = self.num_envs if indices is None else len(indices)
        return [result] * n

    def env_is_wrapped(self, wrapper_class: Any, indices=None) -> List[bool]:
        n = self.num_envs if indices is None else len(indices)
        return [False] * n

    def get_images(self) -> Sequence[Optional[np.ndarray]]:
        return [None, None]

    def seed(self, seed: Optional[int] = None) -> List[Optional[int]]:
        return [None, None]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario',    default='scenarios/train_austria.yml')
    parser.add_argument('--timesteps',   type=int, default=int(5e6))
    parser.add_argument('--save-path',   default='checkpoints/dqn/')
    parser.add_argument('--seed',        type=int, default=42)
    parser.add_argument('--render-mode', default='rgb_array_follow',
                        choices=['human', 'rgb_array_follow', 'rgb_array_birds_eye'])
    parser.add_argument('--reset-mode',  default='grid',
                        choices=['grid', 'random', 'random_bidirectional'])
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)

    # Create a timestamped run directory inside the save path (e.g. checkpoints/dqn/20260416-103000)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(args.save_path, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    train_env = MultiAgentDQNVecEnv(args.scenario, render_mode=args.render_mode, reset_mode=args.reset_mode)

    model = DQN(
        policy='MlpPolicy',
        env=train_env,
        learning_rate=1e-4,
        buffer_size=100_000,
        learning_starts=10_000,
        batch_size=64,
        gamma=0.99,
        tau=1.0,
        train_freq=4,
        target_update_interval=1_000,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        exploration_fraction=0.1,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        seed=args.seed,
        tensorboard_log=os.path.join(run_dir, 'tb_logs'),
    )

    ckpt_cb = CheckpointCallback(
        save_freq=args.timesteps // 10,
        save_path=os.path.join(run_dir, 'checkpoints'),
        name_prefix='dqn_racecar',
    )

    model.learn(
        total_timesteps=args.timesteps,
        callback=ckpt_cb,
        progress_bar=True,
    )

    final_path = os.path.join(run_dir, 'model')
    model.save(final_path)
    print(f"\nModel saved → {final_path}.zip")

    train_env.close()


if __name__ == '__main__':
    main()
