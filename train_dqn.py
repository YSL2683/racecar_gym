"""Train DQN on the racecar multi-agent environment.

Usage:
    python train_dqn.py
    python train_dqn.py --scenario scenarios/train_austria.yml \\
                        --iterations 200 \\
                        --checkpoint-dir checkpoints/dqn

Prerequisites (run once in the rc_1 conda environment):
    pip install "ray[rllib]==2.8.0" torch
"""
import argparse
import os
import sys
import datetime

import numpy as np
import gymnasium
import ray
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.policy.policy import PolicySpec
from ray.tune.registry import register_env
from tqdm import trange

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from policy.dqn_policy import RacecarDQNEnv, OBS_DIM, N_ACTIONS


def main():
    parser = argparse.ArgumentParser(description='Train DQN on racecar multi-agent env')
    parser.add_argument('--scenario',       default='scenarios/train_austria.yml',
                        help='Scenario YAML path')
    parser.add_argument('--render-mode',   default='rgb_array_follow', 
                        choices=['human', 'rgb_array_follow', 'rgb_array_birds_eye'],
                        help='Render mode for the environment')
    parser.add_argument('--iterations',     type=int, default=1000,
                        help='Number of training iterations')
    parser.add_argument('--checkpoint-dir', default='checkpoints/dqn',
                        help='Directory to save checkpoints')
    parser.add_argument('--save-freq',      type=int, default=100,
                        help='Save checkpoint every N iterations (0 = disable periodic saves)')
    args = parser.parse_args()

    ray.init(ignore_reinit_error=True)
    register_env('racecar_dqn', lambda cfg: RacecarDQNEnv(cfg))

    obs_space = gymnasium.spaces.Box(
        low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
    )
    act_space = gymnasium.spaces.Discrete(N_ACTIONS)

    config = (
        DQNConfig()
        .environment(
            'racecar_dqn',
            env_config={
                'scenario':    args.scenario,
                'render_mode': args.render_mode,
            },
        )
        .exploration(exploration_config={
            'type': 'EpsilonGreedy',
            'initial_epsilon': 1.0,
            'final_epsilon': 0.02,
            'epsilon_timesteps': int(args.iterations * 0.8),  # anneal over 80% of training
        })
        .framework('torch')
        .multi_agent(
            policies={
                'shared': PolicySpec(
                    observation_space=obs_space,
                    action_space=act_space,
                )
            },
            policy_mapping_fn=lambda agent_id, *a, **k: 'shared',
        )
        .rollouts(num_rollout_workers=0)
    )

    algo = config.build()
    # create a run-specific subdirectory named by start time
    start_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(args.checkpoint_dir, start_time)
    os.makedirs(run_dir, exist_ok=True)

    save_freq = getattr(args, 'save_freq', 10)

    with trange(args.iterations, desc="Training", unit="it") as pbar:
        for i in pbar:
            result = algo.train()
            reward = result.get('episode_reward_mean', None)
            length = result.get('episode_len_mean',    None)
            fmt_r  = f'{reward:.2f}' if isinstance(reward, float) else str(reward)
            fmt_l  = f'{length:.1f}' if isinstance(length, float) else str(length)
            pbar.set_postfix({'reward': fmt_r, 'len': fmt_l})
            if save_freq and save_freq > 0 and (i + 1) % save_freq == 0:
                ckpt = algo.save(run_dir)
                pbar.write(f"Saved checkpoint at iteration {i+1} → {ckpt}")

    final_ckpt = algo.save(run_dir)
    print(f'\nFinal checkpoint saved → {final_ckpt}')
    algo.stop()
    ray.shutdown()


if __name__ == '__main__':
    main()
