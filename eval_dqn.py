"""
Evaluate a trained DQN policy on MultiAgentRaceEnv (2 agents).

Usage:
    python eval_dqn.py --model checkpoints/dqn/<time_stamp>/model.zip
    python eval_dqn.py --model checkpoints/dqn/<time_stamp>/best_model.zip --render-mode human --episodes 10
"""
from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List

import numpy as np
from stable_baselines3 import DQN

from racecar_gym.envs.gym_api import MultiAgentRaceEnv
from policy.dqn_policy import int_to_action, preprocess_obs


def run_eval(
    model_path: str,
    scenario_path: str,
    n_episodes: int,
    render_mode: str,
    reset_mode: str,
    output_csv: str,
) -> None:
    model = DQN.load(model_path, device='cpu')
    env   = MultiAgentRaceEnv(scenario=scenario_path, render_mode=render_mode)
    agent_ids: List[str] = list(env.scenario.agents.keys())

    results: List[Dict] = []

    for ep in range(n_episodes):
        obs_dict, _ = env.reset(options={'mode': reset_mode})
        ep_rewards   = {aid: 0.0 for aid in agent_ids}
        ep_done      = False

        while not ep_done:
            actions = {}
            for aid in agent_ids:
                flat = preprocess_obs(obs_dict[aid])
                idx, _ = model.predict(flat, deterministic=True)
                actions[aid] = int_to_action(int(idx))

            obs_dict, rewards_dict, dones_dict, _, state_dict = env.step(actions)

            for aid in agent_ids:
                ep_rewards[aid] += rewards_dict[aid]

            ep_done = any(dones_dict.values())

        # ── Collect final metrics from world state ─────────────────────────────
        row: Dict = {'episode': ep + 1}
        for aid in agent_ids:
            s        = state_dict.get(aid, {})
            laps     = s.get('lap', 0)
            progress = float(s.get('progress', 0.0))
            wall_col = int(s.get('wall_collision', False))
            opp_col  = len(s.get('opponent_collisions', []))
            score    = float(laps) + progress

            row[f'reward_{aid}']   = round(ep_rewards[aid], 3)
            row[f'laps_{aid}']     = laps
            row[f'progress_{aid}'] = round(progress, 4)
            row[f'wall_col_{aid}'] = wall_col
            row[f'opp_col_{aid}']  = opp_col
            row[f'score_{aid}']    = round(score, 4)

        row['total_score'] = round(sum(row[f'score_{aid}'] for aid in agent_ids), 4)
        results.append(row)

        detail = '  |  '.join(
            f"{aid}: score={row[f'score_{aid}']:.3f} "
            f"(laps={row[f'laps_{aid}']}, "
            f"prog={row[f'progress_{aid}']:.3f}, "
            f"w_col={row[f'wall_col_{aid}']}, "
            f"o_col={row[f'opp_col_{aid}']})"
            for aid in agent_ids
        )
        print(f"Ep {ep + 1:3d}  |  {detail}")

    env.close()

    # ── Summary ────────────────────────────────────────────────────────────────
    print('\n── Summary ──────────────────────────────')
    for aid in agent_ids:
        scores = [r[f'score_{aid}'] for r in results]
        laps   = [r[f'laps_{aid}']  for r in results]
        print(
            f"  Agent {aid}:  score = {np.mean(scores):.3f} ± {np.std(scores):.3f}"
            f"  |  avg laps = {np.mean(laps):.2f}"
        )

    # ── CSV output ─────────────────────────────────────────────────────────────
    # if results:
    #     os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    #     fieldnames = list(results[0].keys())
    #     with open(output_csv, 'w', newline='') as f:
    #         writer = csv.DictWriter(f, fieldnames=fieldnames)
    #         writer.writeheader()
    #         writer.writerows(results)
    #     print(f"\nResults saved → {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',       default='checkpoints/dqn/model.zip')
    parser.add_argument('--scenario',    default='scenarios/eval_austria.yml')
    parser.add_argument('--episodes',    type=int, default=1)
    parser.add_argument('--render-mode', default='human',
                        choices=['human', 'rgb_array_follow', 'rgb_array_birds_eye'])
    parser.add_argument('--reset-mode',  default='grid',
                        choices=['grid', 'random', 'random_bidirectional'])
    parser.add_argument('--output-csv',  default='eval_results.csv')
    args = parser.parse_args()

    run_eval(
        model_path=args.model,
        scenario_path=args.scenario,
        n_episodes=args.episodes,
        render_mode=args.render_mode,
        reset_mode=args.reset_mode,
        output_csv=args.output_csv,
    )


if __name__ == '__main__':
    main()
