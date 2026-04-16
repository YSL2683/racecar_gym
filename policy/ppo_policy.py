import os
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from policy.base_policy import Policy

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0
EPS = 1e-6
DEFAULT_OBSERVATION_KEYS = ('lidar', 'pose', 'velocity', 'acceleration', 'time')
CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / 'checkpoints' / '20260415_220701' / 'ppo_shared.pt'

def infer_observation_keys(observation: Dict[str, np.ndarray]) -> Tuple[str, ...]:
    preferred = [key for key in DEFAULT_OBSERVATION_KEYS if key in observation]
    remaining = [key for key in observation.keys() if key not in preferred]
    return tuple(preferred + remaining)


def infer_action_spec(action_space) -> Tuple[Tuple[str, ...], Tuple[int, ...]]:
    action_keys = tuple(sorted(action_space.spaces.keys()))
    action_sizes = tuple(int(np.prod(action_space.spaces[key].shape)) for key in action_keys)
    return action_keys, action_sizes


def flatten_observation(
    observation: Dict[str, np.ndarray],
    observation_keys: Sequence[str],
) -> np.ndarray:
    chunks = []
    for key in observation_keys:
        if key not in observation:
            raise KeyError(f"Observation missing required key '{key}'.")
        value = np.asarray(observation[key])
        if value.dtype == np.uint8:
            value = value.astype(np.float32) / 255.0
        else:
            value = value.astype(np.float32)
        chunks.append(value.reshape(-1))
    if not chunks:
        raise ValueError('At least one observation key is required.')
    return np.concatenate(chunks, axis=0).astype(np.float32)


def action_vector_to_dict(
    action: np.ndarray,
    action_keys: Sequence[str],
    action_sizes: Sequence[int],
) -> Dict[str, np.ndarray]:
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    expected_dim = int(sum(action_sizes))
    if action.shape[0] != expected_dim:
        raise ValueError(f'Expected action dimension {expected_dim}, got {action.shape[0]}.')

    result = {}
    offset = 0
    for key, size in zip(action_keys, action_sizes):
        result[key] = np.clip(action[offset:offset + size], -1.0, 1.0).astype(np.float32)
        offset += size
    return result


def observations_to_tensor(
    observations: Iterable[Dict[str, np.ndarray]],
    observation_keys: Sequence[str],
    device: torch.device,
) -> torch.Tensor:
    stacked = np.stack([flatten_observation(obs, observation_keys) for obs in observations], axis=0)
    return torch.as_tensor(stacked, dtype=torch.float32, device=device)


def build_mlp(input_dim: int, hidden_sizes: Sequence[int], output_dim: int) -> nn.Sequential:
    layers = []
    prev_dim = input_dim
    for hidden_dim in hidden_sizes:
        layers.extend([nn.Linear(prev_dim, hidden_dim), nn.Tanh()])
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


class RacecarPPOModel(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_sizes: Sequence[int] = (256, 256)):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_sizes = tuple(hidden_sizes)
        self.actor = build_mlp(observation_dim, hidden_sizes, action_dim)
        self.critic = build_mlp(observation_dim, hidden_sizes, 1)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def _distribution(self, observations: torch.Tensor) -> Normal:
        mean = self.actor(observations)
        log_std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).expand_as(mean)
        return Normal(mean, log_std.exp())

    @staticmethod
    def _squash_log_prob(distribution: Normal, raw_action: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        correction = torch.log(1.0 - action.pow(2) + EPS)
        return (distribution.log_prob(raw_action) - correction).sum(dim=-1)

    def get_value(self, observations: torch.Tensor) -> torch.Tensor:
        return self.critic(observations).squeeze(-1)

    def get_action_and_value(
        self,
        observations: torch.Tensor,
        raw_action: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self._distribution(observations)
        if raw_action is None:
            raw_action = distribution.mean if deterministic else distribution.rsample()
        action = torch.tanh(raw_action)
        log_prob = self._squash_log_prob(distribution, raw_action, action)
        entropy = distribution.entropy().sum(dim=-1)
        value = self.get_value(observations)
        return action, log_prob, entropy, value, raw_action


def save_checkpoint(
    path: str | os.PathLike,
    model: RacecarPPOModel,
    observation_keys: Sequence[str],
    action_keys: Sequence[str],
    action_sizes: Sequence[int],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'state_dict': model.state_dict(),
        'observation_keys': list(observation_keys),
        'action_keys': list(action_keys),
        'action_sizes': list(action_sizes),
        'hidden_sizes': list(model.hidden_sizes),
        'observation_dim': model.observation_dim,
        'action_dim': model.action_dim,
    }
    torch.save(payload, path)


def load_checkpoint(
    path: str | os.PathLike,
    device: torch.device,
) -> Tuple[RacecarPPOModel, Dict[str, object]]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = RacecarPPOModel(
        observation_dim=int(checkpoint['observation_dim']),
        action_dim=int(checkpoint['action_dim']),
        hidden_sizes=tuple(int(v) for v in checkpoint['hidden_sizes']),
    )
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    metadata = {
        'observation_keys': tuple(checkpoint['observation_keys']),
        'action_keys': tuple(checkpoint['action_keys']),
        'action_sizes': tuple(int(v) for v in checkpoint['action_sizes']),
    }
    return model, metadata


class PPOPolicy(Policy):
    def __init__(
        self,
        checkpoint_path: str | os.PathLike | None = None,
        device: str | None = None,
        deterministic: bool = True,
    ):
        self._checkpoint_path = Path(checkpoint_path or CHECKPOINT_PATH)
        if not self._checkpoint_path.exists():
            raise FileNotFoundError(
                f'Checkpoint not found: {self._checkpoint_path}. Train first or set '
                'RACECAR_GYM_POLICY_CHECKPOINT.'
            )
        self._device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self._model, metadata = load_checkpoint(self._checkpoint_path, self._device)
        self._observation_keys = metadata['observation_keys']
        self._action_keys = metadata['action_keys']
        self._action_sizes = metadata['action_sizes']
        self._deterministic = deterministic

    def act(self, observation: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        observation_tensor = observations_to_tensor([observation], self._observation_keys, self._device)
        with torch.no_grad():
            action, _, _, _, _ = self._model.get_action_and_value(
                observation_tensor,
                deterministic=self._deterministic,
            )
        return action_vector_to_dict(
            action.squeeze(0).cpu().numpy(),
            self._action_keys,
            self._action_sizes,
        )


def make_policy(checkpoint_path: str | None = None) -> Policy:
    path = checkpoint_path or CHECKPOINT_PATH
    return PPOPolicy(checkpoint_path=path)
