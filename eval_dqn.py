"""
eval_dqn.py — Evaluate a trained DQN agent in a 2-agent racecar_gym scenario.

Usage
-----
    python eval_dqn.py [options]

    # Example:
    python eval_dqn.py --model_path checkpoints/dqn/final_model.zip --episodes 5 --render_mode human

Run `python eval_dqn.py --help` for the full option list.
"""

import argparse
import os
import time

import gymnasium
import numpy as np

import racecar_gym.envs.gym_api  # registers gymnasium env IDs
from policy.dqn_policy import DQNPolicy


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained DQN model on racecar_gym (2-agent Austria)"
    )
    parser.add_argument(
        "--model_path",
        default=os.path.join("checkpoints", "dqn", "final_model.zip"),
        help="Path to the trained SB3 DQN .zip file "
             "(default: checkpoints/dqn/final_model.zip)",
    )
    parser.add_argument(
        "--scenario",
        default=os.path.join("scenarios", "eval_austria-single.yml"),
        help="Path to the scenario YAML file (default: scenarios/eval_austria-single.yml)",
    )
    parser.add_argument(
        "--episodes", type=int, default=10,
        help="Number of evaluation episodes (default: 10)",
    )
    parser.add_argument(
        "--render_mode",
        choices=["human", "rgb_array_follow", "rgb_array_birds_eye"],
        default="human",
        help="Rendering mode: human | rgb_array_follow | rgb_array_birds_eye "
             "(default: human)",
    )
    parser.add_argument(
        "--reset_mode",
        choices=["grid", "random"],
        default="grid",
        help="Starting position mode (default: grid)",
    )
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    render_mode = None if args.render_mode == "none" else args.render_mode

    # Load the multi-agent env
    env = gymnasium.make(
        "MultiAgentRaceEnv-v0",
        scenario=args.scenario,
        render_mode=render_mode,
    )
    agent_ids = list(env.observation_space.spaces.keys())

    # Load the trained DQN policy (obs preprocessing and action decoding are encapsulated)
    policy = DQNPolicy(model_path=args.model_path)
    print(f"[eval_dqn] Loaded model from {args.model_path}")
    print(f"[eval_dqn] Evaluating {args.episodes} episode(s) on {args.scenario}\n")

    # ── Episode loop ──────────────────────────────────────────────────────────
    ep_returns = []
    ep_progresses = []
    ep_laps = []

    for ep in range(1, args.episodes + 1):
        obs_dict, _ = env.reset(options={"mode": args.reset_mode})

        done = False
        ep_reward = 0.0
        final_info = {}

        while not done:
            # Apply the shared DQN policy to every agent independently
            multi_action = {aid: policy.act(obs_dict[aid]) for aid in agent_ids}

            obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict = (
                env.step(multi_action)
            )
            sleep = 0.01
            # time.sleep(sleep)

            primary_id = agent_ids[0]
            ep_reward += float(reward_dict[primary_id])
            final_info = info_dict.get(primary_id, {})

            done = any(terminated_dict.values())

            if render_mode == "human":
                env.render()

        # ── Per-episode stats ──────────────────────────────────────────────
        progress = final_info.get("progress", float("nan"))
        lap = final_info.get("lap", float("nan"))
        time_elapsed = final_info.get("time", float("nan"))
        ep_returns.append(ep_reward)
        ep_progresses.append(progress)
        ep_laps.append(lap)

        print(
            f"  Episode {ep:>3d} | "
            f"return={ep_reward:8.2f} | "
            f"lap={lap} | "
            f"progress={progress:.4f} | "
            f"time={time_elapsed:.1f}s"
        )

    # ── Aggregate statistics ───────────────────────────────────────────────────
    print("\n── Summary ─────────────────────────────────────────────────────────")
    print(f"  Episodes        : {args.episodes}")
    print(f"  Mean return     : {np.mean(ep_returns):.2f} ± {np.std(ep_returns):.2f}")
    print(f"  Mean progress   : {np.mean(ep_progresses):.4f}")
    print(f"  Laps completed  : {sum(l > 1 for l in ep_laps)} / {args.episodes}")

    env.close()


if __name__ == "__main__":
    main()

