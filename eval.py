import argparse
from statistics import mean

import racecar_gym.envs.gym_api  # noqa: F401
from racecar_gym.envs.gym_api import MultiAgentRaceEnv

from policy.ppo_policy import PPOPolicy


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a trained PPO policy in racecar_gym.')
    parser.add_argument('--scenario', type=str, default='scenarios/eval_austria.yml')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/ppo_shared.pt')
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--render-mode', type=str, default='human',
                        choices=['human', 'rgb_array_follow', 'rgb_array_birds_eye'])
    parser.add_argument('--reset-mode', type=str, default='grid',
                        choices=['grid', 'random', 'random_bidirectional'])
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--stochastic', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    env = MultiAgentRaceEnv(scenario=args.scenario, render_mode=args.render_mode)
    observations, _ = env.reset(seed=args.seed, options={'mode': args.reset_mode})
    agent_ids = list(env.scenario.agents.keys())
    policy = PPOPolicy(
        checkpoint_path=args.checkpoint,
        device=args.device,
        deterministic=not args.stochastic,
    )

    episode_joint_returns = []
    episode_lengths = []

    for episode in range(1, args.episodes + 1):
        if episode > 1:
            observations, _ = env.reset(options={'mode': args.reset_mode})

        episode_returns = {aid: 0.0 for aid in agent_ids}
        steps = 0

        while True:
            actions = {aid: policy.act(observations[aid]) for aid in agent_ids}
            observations, rewards, dones, _, _ = env.step(actions)
            steps += 1

            for aid in agent_ids:
                episode_returns[aid] += float(rewards[aid])

            if args.render_mode != 'human':
                env.render()

            if any(dones.values()):
                break

        joint_return = sum(episode_returns.values())
        episode_joint_returns.append(joint_return)
        episode_lengths.append(steps)
        per_agent = ' '.join(f'{aid}={episode_returns[aid]:.3f}' for aid in agent_ids)
        print(f'episode={episode} steps={steps} joint_return={joint_return:.3f} {per_agent}')

    env.close()
    print(
        f'mean_joint_return={mean(episode_joint_returns):.3f} '
        f'mean_episode_length={mean(episode_lengths):.1f}'
    )


if __name__ == '__main__':
    main()
