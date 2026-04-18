"""DQN policy for racecar multi-agent environment.

Usage (client):
    DQN_CHECKPOINT=checkpoints/dqn/checkpoint_000100 \\
        python -m server_client.client --agent-id A --policy policy/dqn_policy.py
"""
import os

import numpy as np
import gymnasium
from gymnasium.spaces import Box, Discrete
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env

from racecar_gym.envs.gym_api import MultiAgentRaceEnv
from policy.base_policy import Policy

# ── Action / observation constants ────────────────────────────────────────────

MOTOR_VALUES    = [-1.0, 0.0, 1.0]
STEERING_VALUES = list(np.linspace(-1.0, 1.0, 7))
ACTIONS = np.array([[m ,s] for m in MOTOR_VALUES for s in STEERING_VALUES], dtype=np.float32)
N_ACTIONS       = len(MOTOR_VALUES) * len(STEERING_VALUES)   # 21 motor/steering combos
OBS_DIM         = 1080 + 6 + 6 + 6                          # 1098 lidar + pose + velocity + acceleration

# ── Helpers ───────────────────────────────────────────────────────────────────

def _flatten(obs: dict) -> np.ndarray:
    """Flatten sensor dict to a 1-D float32 vector (1098 dims)."""
    return np.concatenate([
        obs['lidar'].flatten(),
        obs['pose'].flatten(),
        obs['velocity'].flatten(),
        obs['acceleration'].flatten(),
    ]).astype(np.float32)


def _action_mapping(action_id: int) -> dict:
    """Map a discrete action index to {motor, steering} arrays."""
    if not (0 <= action_id < N_ACTIONS):
        raise IndexError(f'Invalid action_id {action_id}, must be in [0, {N_ACTIONS})')
    motor, steering = ACTIONS[action_id]
    return {'motor': motor, 'steering': steering}


# ── RLlib environment adapter ─────────────────────────────────────────────────

class RacecarDQNEnv(MultiAgentEnv):
    """Wraps MultiAgentRaceEnv for RLlib DQN.

    Observations are flattened to a 1098-dim float32 vector.
    Actions are encoded as integers in [0, 20].
    """

    def __init__(self, config=None):
        super().__init__()
        config      = config or {}
        scenario    = config.get('scenario',    'scenarios/train_austria.yml')
        render_mode = config.get('render_mode', 'rgb_array_follow')

        self._env    = MultiAgentRaceEnv(scenario=scenario, render_mode=render_mode)
        self._agents = sorted(self._env.scenario.agents.keys())
        self._agent_ids = set(self._agents)

        _obs = Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)
        _act = Discrete(N_ACTIONS)
        # Dict spaces keyed by agent ID — required by RLlib MultiAgentEnv
        self.observation_space = gymnasium.spaces.Dict({aid: _obs for aid in self._agents})
        self.action_space      = gymnasium.spaces.Dict({aid: _act for aid in self._agents})

    def reset(self, *, seed=None, options=None):
        obs, info = self._env.reset(seed=seed, options=options)
        return {aid: _flatten(obs[aid]) for aid in self._agents}, info

    def step(self, action_dict):
        env_actions = {aid: _action_mapping(action_dict[aid]) for aid in self._agents}
        obs, rewards, dones, _, infos = self._env.step(env_actions)
        flat_obs    = {aid: _flatten(obs[aid]) for aid in self._agents}
        terminateds = {aid: bool(dones[aid]) for aid in self._agents}
        terminateds['__all__'] = any(terminateds.values())
        truncateds  = {aid: False for aid in self._agents}
        truncateds['__all__'] = False
        return flat_obs, rewards, terminateds, truncateds, infos


# ── Trained policy (client-compatible) ───────────────────────────────────────

class DQNPolicy(Policy):
    """Loads a saved RLlib DQN checkpoint and acts on raw sensor observations."""

    def __init__(self, checkpoint_path: str):
        import ray
        from ray.rllib.algorithms.algorithm import Algorithm

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        # Register env so the checkpoint can rebuild algorithm config
        register_env('racecar_dqn', lambda cfg: RacecarDQNEnv(cfg))
        self._algo = Algorithm.from_checkpoint(checkpoint_path)

    def act(self, obs: dict) -> dict:
        flat      = _flatten(obs)
        action_id = int(
            self._algo.compute_single_action(flat, policy_id='shared', explore=False)
        )
        action = _action_mapping(action_id)
        return action


def make_policy() -> Policy:
    """Entry point called by server_client/client.py.

    Set the DQN_CHECKPOINT environment variable to the checkpoint directory,
    e.g. checkpoints/dqn/checkpoint_000100
    """
    checkpoint = os.environ.get('DQN_CHECKPOINT', 'checkpoints/dqn')
    return DQNPolicy(checkpoint)
