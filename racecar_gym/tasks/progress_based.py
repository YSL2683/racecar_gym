from .task import Task
import numpy as np


class MaximizeProgressTask(Task):
    def __init__(self, laps: int, time_limit: float, terminate_on_collision: bool,
                 delta_progress: float = 0.0, collision_reward: float = 0.0,
                 frame_reward: float = 0.0, progress_reward: float = 100.0, n_min_rays_termination=1080):
        self._time_limit = time_limit
        self._laps = laps
        self._terminate_on_collision = terminate_on_collision
        self._n_min_rays_termination = n_min_rays_termination
        self._last_stored_progress = None
        # reward params
        self._delta_progress = delta_progress
        self._progress_reward = progress_reward
        self._collision_reward = collision_reward
        self._frame_reward = frame_reward

    def reward(self, agent_id, state, action) -> float:
        agent_state = state[agent_id]
        progress = agent_state['lap'] + agent_state['progress']
        if self._last_stored_progress is None:
            self._last_stored_progress = progress
        delta = abs(progress - self._last_stored_progress)
        if delta > .5:  # the agent is crossing the starting line in the wrong direction
            delta = (1 - progress) + self._last_stored_progress
        reward = self._frame_reward
        if self._check_collision(agent_state):
            reward += self._collision_reward
        reward += delta * self._progress_reward
        self._last_stored_progress = progress
        return reward

    def done(self, agent_id, state) -> bool:
        agent_state = state[agent_id]
        if self._terminate_on_collision and self._check_collision(agent_state):
            return True
        return agent_state['lap'] > self._laps or self._time_limit < agent_state['time']

    def _check_collision(self, agent_state):
        safe_margin = 0.25
        collision = agent_state['wall_collision'] or len(agent_state['opponent_collisions']) > 0
        if 'observations' in agent_state and 'lidar' in agent_state['observations']:
            n_min_rays = sum(np.where(agent_state['observations']['lidar'] <= safe_margin, 1, 0))
            return n_min_rays > self._n_min_rays_termination or collision
        return collision

    def reset(self):
        self._last_stored_progress = None


class MaximizeProgressMaskObstacleTask(MaximizeProgressTask):
    def __init__(self, laps: int, time_limit: float, terminate_on_collision: bool, delta_progress=0.0,
                 collision_reward=0, frame_reward=0, progress_reward=100):
        super().__init__(laps, time_limit, terminate_on_collision, delta_progress, collision_reward, frame_reward,
                         progress_reward)

    def reward(self, agent_id, state, action) -> float:
        progress_reward = super().reward(agent_id, state, action)
        distance_to_obstacle = state[agent_id]['obstacle']
        if distance_to_obstacle < .3:  # max distance = 1, meaning perfectly centered in the widest point of the track
            return 0.0
        else:
            return progress_reward


class MaximizeProgressRegularizeAction(MaximizeProgressTask):
    def __init__(self, laps: int, time_limit: float, terminate_on_collision: bool, delta_progress=0.0,
                 collision_reward=0, frame_reward=0, progress_reward=100, action_reg=0.25):
        super().__init__(laps, time_limit, terminate_on_collision, delta_progress, collision_reward, frame_reward,
                         progress_reward)
        self._action_reg = action_reg
        self._last_action = None

    def reset(self):
        super(MaximizeProgressRegularizeAction, self).reset()
        self._last_action = None

    def reward(self, agent_id, state, action) -> float:
        """ Progress-based with action regularization: penalize sharp change in control"""
        reward = super().reward(agent_id, state, action)
        action = np.array(list(action.values()))
        if self._last_action is not None:
            reward -= self._action_reg * np.linalg.norm(action - self._last_action)
        self._last_action = action
        return reward


class RankDiscountedMaximizeProgressTask(MaximizeProgressTask):
    def __init__(self, laps: int, time_limit: float, terminate_on_collision: bool, delta_progress=0.001,
                 collision_reward=-100, frame_reward=-0.1, progress_reward=1):
        super().__init__(laps, time_limit, terminate_on_collision, delta_progress, collision_reward, frame_reward,
                         progress_reward)

    def reward(self, agent_id, state, action) -> float:
        rank = state[agent_id]['rank']
        reward = super().reward(agent_id, state, action)
        reward = reward / float(rank)
        return reward


class DQNProgressTask(Task):
    """Reward task for DQN discrete control — inherits directly from Task.

    Reward components (all weights configurable via YAML params):

      1. progress_reward * abs(delta)
            Same abs-delta calculation as MaximizeProgressTask,
            but applied only when agent is NOT going the wrong way.

      2. motor_weight * ||v_linear||
            Reward proportional to the linear speed magnitude (body-frame [:3]).

      3. action_change_penalty * Σ|Δaction|
            Penalises sudden changes in motor/steering between consecutive steps.

      4. collision_penalty * (1 - lap + progress)
            Collision penalty scaled so that collisions deeper into the lap
            (high progress) are penalised less, and near lap completion more.

    Episode termination: time limit exceeded, laps completed, or
    (optionally) on collision.
    """

    def __init__(
        self,
        laps: int,
        time_limit: float,
        terminate_on_collision: bool,
        progress_reward: float = 1000.0,
        frame_reward: float = -0.01,
        motor_weight: float = 0.1,
        action_change_penalty: float = -0.01,
        collision_penalty: float = 0.0,
        checkpoint_reward: float = 100.0,
    ):
        self._laps = laps
        self._time_limit = time_limit
        self._terminate_on_collision = terminate_on_collision
        self._progress_reward = progress_reward
        self._frame_reward = frame_reward
        self._motor_weight = motor_weight
        self._action_change_penalty = action_change_penalty
        self._collision_penalty = collision_penalty
        self._checkpoint_reward = checkpoint_reward

        self._last_progress = None   # lap + progress at previous step
        self._last_action = None     # np.array([motor, steering])
        self._last_checkpoint = {}   # per-agent last checkpoint index

    # ------------------------------------------------------------------
    def reset(self):
        self._last_progress = None
        self._last_action = None
        self._last_checkpoint = {}

    # ------------------------------------------------------------------
    def reward(self, agent_id, state, action) -> float:
        agent_state = state[agent_id]
        reward = 0.0
        reward += self._frame_reward

        # ── 1. Progress delta (MaximizeProgressTask style, wrong_way guard) ──
        current_progress = agent_state['lap'] + agent_state['progress']
        if self._last_progress is None:
            self._last_progress = current_progress

        if not agent_state['wrong_way']:
            delta = current_progress - self._last_progress
            delta = max(delta, 0.0)  # only reward forward progress, not backward
            if delta > 0.5:  # wrap-around at start/finish line
                # delta = (1 - current_progress) + self._last_progress
                delta = 0.0
            reward += self._progress_reward * delta

        self._last_progress = current_progress

        # ── 2. motor reward ─────────────────────────────────────────
        if action["motor"].flat[0] > 0:  # reward forward throttle, penalize reverse
            reward += self._motor_weight * action["motor"].flat[0]

        # ── 3. Action-change penalty ───────────────────────────────────
        action = np.array(list(action.values()))
        if self._last_action is not None:
            reward -= self._action_change_penalty * np.linalg.norm(action - self._last_action)
        self._last_action = action

        # ── 4. Collision penalty  ───────
        if self._check_collision(agent_state):
            reward += self._collision_penalty

        #── 5. Checkpoint reward  ───────────────────────────────────
        if agent_state['checkpoint'] != self._last_checkpoint.get(agent_id, None):
            reward += self._checkpoint_reward
            self._last_checkpoint[agent_id] = agent_state['checkpoint']

        return reward

    # ------------------------------------------------------------------
    def done(self, agent_id, state) -> bool:
        agent_state = state[agent_id]
        if self._terminate_on_collision and self._check_collision(agent_state):
            return True
        return agent_state['lap'] > self._laps or agent_state['time'] > self._time_limit

    # ------------------------------------------------------------------
    def _check_collision(self, agent_state) -> bool:
        collision = agent_state['wall_collision'] or len(agent_state['opponent_collisions']) > 0
        if 'observations' in agent_state and 'lidar' in agent_state['observations']:
            safe_margin = 0.25
            n_close = int(np.sum(agent_state['observations']['lidar'] <= safe_margin))
            return n_close > 0 or collision
        return collision