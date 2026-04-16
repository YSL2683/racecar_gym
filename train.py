import argparse
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm

from racecar_gym.envs.gym_api import MultiAgentRaceEnv
from policy.ppo_policy import (
    RacecarPPOModel,
    infer_action_spec,
    infer_observation_keys,
    observations_to_tensor,
    action_vector_to_dict,
    save_checkpoint,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a shared-policy PPO agent for racecar_gym.')
    parser.add_argument('--scenario', type=str, default='scenarios/austria_2agents.yml')
    parser.add_argument('--total-timesteps', type=int, default=int(5e6))
    parser.add_argument('--rollout-steps', type=int, default=512)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--clip-coef', type=float, default=0.2)
    parser.add_argument('--update-epochs', type=int, default=10)
    parser.add_argument('--minibatch-size', type=int, default=256)
    parser.add_argument('--ent-coef', type=float, default=0.01)
    parser.add_argument('--vf-coef', type=float, default=0.5)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    parser.add_argument('--hidden-sizes', type=int, nargs='+', default=[256, 256])
    parser.add_argument('--reset-mode', type=str, default='grid', choices=['grid', 'random', 'random_bidirectional'])
    parser.add_argument('--checkpoint', type=str, default='checkpoints/ppo_shared.pt')
    parser.add_argument('--save-interval', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))

    env = MultiAgentRaceEnv(scenario=args.scenario, render_mode=None)
    env.reset(seed=args.seed, options={'mode': args.reset_mode})
    agent_ids = list(env.scenario.agents.keys())
    if len(agent_ids) != 2:
        raise ValueError(f'Expected a 2-agent scenario, got {len(agent_ids)} agents: {agent_ids}')

    observations, _ = env.reset(seed=args.seed, options={'mode': args.reset_mode})
    observation_keys = infer_observation_keys(observations[agent_ids[0]])
    action_keys, action_sizes = infer_action_spec(env.action_space.spaces[agent_ids[0]])
    observation_dim = int(sum(np.asarray(observations[agent_ids[0]][key]).size for key in observation_keys))
    action_dim = int(sum(action_sizes))

    model = RacecarPPOModel(observation_dim, action_dim, hidden_sizes=args.hidden_sizes).to(device)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    num_agents = len(agent_ids)
    batch_size = args.rollout_steps * num_agents
    if args.minibatch_size > batch_size:
        raise ValueError(f'--minibatch-size must be <= rollout_steps * num_agents ({batch_size}).')

    observations_tensor = observations_to_tensor([observations[aid] for aid in agent_ids], observation_keys, device)
    next_done = torch.zeros(num_agents, dtype=torch.float32, device=device)
    total_updates = max(args.total_timesteps // batch_size, 1)

    recent_returns = deque(maxlen=20)
    episode_returns = {aid: 0.0 for aid in agent_ids}
    completed_episodes = 0
    global_step = 0

    checkpoint_path = Path(args.checkpoint)
    run_dir = checkpoint_path.parent / datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / checkpoint_path.name

    for update in tqdm(range(1, total_updates + 1), desc='Training', unit='update', dynamic_ncols=True):
        rollout_observations = torch.zeros((args.rollout_steps, num_agents, observation_dim), dtype=torch.float32, device=device)
        rollout_raw_actions = torch.zeros((args.rollout_steps, num_agents, action_dim), dtype=torch.float32, device=device)
        rollout_log_probs = torch.zeros((args.rollout_steps, num_agents), dtype=torch.float32, device=device)
        rollout_rewards = torch.zeros((args.rollout_steps, num_agents), dtype=torch.float32, device=device)
        rollout_dones = torch.zeros((args.rollout_steps, num_agents), dtype=torch.float32, device=device)
        rollout_values = torch.zeros((args.rollout_steps, num_agents), dtype=torch.float32, device=device)

        model.train()
        for step in range(args.rollout_steps):
            rollout_observations[step] = observations_tensor

            with torch.no_grad():
                actions, log_probs, _, values, raw_actions = model.get_action_and_value(observations_tensor)

            rollout_raw_actions[step] = raw_actions
            rollout_log_probs[step] = log_probs
            rollout_values[step] = values

            action_dict = {
                aid: action_vector_to_dict(actions[idx].cpu().numpy(), action_keys, action_sizes)
                for idx, aid in enumerate(agent_ids)
            }

            next_observations, rewards, dones, _, _ = env.step(action_dict)
            episode_done = any(dones.values())
            reward_vector = torch.as_tensor(
                [rewards[aid] for aid in agent_ids],
                dtype=torch.float32,
                device=device,
            )
            done_vector = torch.full((num_agents,), float(episode_done), dtype=torch.float32, device=device)

            rollout_rewards[step] = reward_vector
            rollout_dones[step] = done_vector

            global_step += num_agents

            for idx, aid in enumerate(agent_ids):
                episode_returns[aid] += float(reward_vector[idx].item())

            if episode_done:
                recent_returns.append(sum(episode_returns.values()) / num_agents)
                completed_episodes += 1
                next_observations, _ = env.reset(options={'mode': args.reset_mode})
                next_done = torch.zeros(num_agents, dtype=torch.float32, device=device)
                episode_returns = {aid: 0.0 for aid in agent_ids}
            else:
                next_done = done_vector

            observations_tensor = observations_to_tensor(
                [next_observations[aid] for aid in agent_ids],
                observation_keys,
                device,
            )

        with torch.no_grad():
            next_value = model.get_value(observations_tensor)
            advantages = torch.zeros_like(rollout_rewards)
            last_advantage = torch.zeros(num_agents, dtype=torch.float32, device=device)

            for step in reversed(range(args.rollout_steps)):
                if step == args.rollout_steps - 1:
                    next_non_terminal = 1.0 - next_done
                    next_values = next_value
                else:
                    next_non_terminal = 1.0 - rollout_dones[step]
                    next_values = rollout_values[step + 1]

                delta = rollout_rewards[step] + args.gamma * next_values * next_non_terminal - rollout_values[step]
                last_advantage = delta + args.gamma * args.gae_lambda * next_non_terminal * last_advantage
                advantages[step] = last_advantage

            returns = advantages + rollout_values

        b_observations = rollout_observations.reshape(-1, observation_dim)
        b_raw_actions = rollout_raw_actions.reshape(-1, action_dim)
        b_log_probs = rollout_log_probs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = rollout_values.reshape(-1)

        model.train()
        indices = np.arange(batch_size)
        for _ in range(args.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, batch_size, args.minibatch_size):
                batch_indices = indices[start:start + args.minibatch_size]
                mb_observations = b_observations[batch_indices]
                mb_raw_actions = b_raw_actions[batch_indices]
                mb_old_log_probs = b_log_probs[batch_indices]
                mb_advantages = b_advantages[batch_indices]
                mb_returns = b_returns[batch_indices]
                mb_old_values = b_values[batch_indices]

                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std(unbiased=False) + 1e-8)

                _, new_log_probs, entropy, new_values, _ = model.get_action_and_value(
                    mb_observations,
                    raw_action=mb_raw_actions,
                )
                log_ratio = new_log_probs - mb_old_log_probs
                ratio = log_ratio.exp()

                pg_loss_1 = -mb_advantages * ratio
                pg_loss_2 = -mb_advantages * torch.clamp(ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef)
                policy_loss = torch.max(pg_loss_1, pg_loss_2).mean()

                value_delta = new_values - mb_old_values
                clipped_values = mb_old_values + value_delta.clamp(-args.clip_coef, args.clip_coef)
                value_loss_unclipped = (new_values - mb_returns).pow(2)
                value_loss_clipped = (clipped_values - mb_returns).pow(2)
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

                entropy_loss = entropy.mean()
                loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

        mean_recent_return = float(np.mean(recent_returns)) if recent_returns else 0.0
        tqdm.write(
            f'update={update}/{total_updates} '
            f'step={global_step} '
            f'eps={completed_episodes} '
            f'return={mean_recent_return:.3f}'
        )

        if update % args.save_interval == 0 or update == total_updates:
            save_checkpoint(
                checkpoint_path,
                model,
                observation_keys=observation_keys,
                action_keys=action_keys,
                action_sizes=action_sizes,
            )

    env.close()
    tqdm.write(f'Saved checkpoint to {checkpoint_path}')


if __name__ == '__main__':
    main()
