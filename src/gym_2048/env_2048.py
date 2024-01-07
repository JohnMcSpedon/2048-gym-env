from dataclasses import dataclass
from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium.core import ActType, ObsType

from . import logic

INFO_KEY_GRID_SEED = "grid_seed"
INFO_KEY_GAME_OVER_REASON = "game_over_reason"
INFO_KEY_DELTA_SCORE = "delta_score"
INFO_KEY_TOTAL_SCORE = "total_score"

NO_VALID_MOVES = "no_valid_moves"
ILLEGAL_MOVE_CHOSEN = "illegal_move_chosen"


@dataclass
class RewardConfig:
    lambda_step_reward: float = 1.0
    lambda_score_reward: float = 1.0
    illegal_move_penalty: float = -10
    game_over_penalty: float = -5


class Env2048(gym.Env):
    r"""
    Attributes:
        action_space: The Space object corresponding to valid actions
        observation_space: The Space object corresponding to valid observations
        reward_range: A tuple corresponding to the min and max possible rewards
    """
    metadata = {"render.modes": ["ansi"]}  # TODO: implement rgb_array, human
    reward_range = (-float("inf"), float("inf"))  # TODO: narrow this range?

    action_space = gym.spaces.Discrete(4)  # up, down, left, right
    observation_space = gym.spaces.Space(shape=[4, 4], dtype=np.uint8)

    def __init__(self, reward_config: RewardConfig, render_mode: str):
        super(Env2048, self).__init__()
        self.reward_cfg = reward_config
        if render_mode != "ansi":
            raise NotImplementedError(f"Haven't implemented render mode {render_mode}")
        self.render_mode = render_mode
        self.reset()

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Run one timestep of the environment’s dynamics using the agent actions.

        When the end of an episode is reached (terminated or truncated), it is necessary to call reset() to reset this
        environment’s state for the next episode.

        Args:
            action (ActType) – an action provided by the agent to update the environment state.
        Returns:
            observation (ObsType): An element of the environment’s observation_space as the next observation due to the
                agent actions.  For 2048, this is the tiles
            reward (float) : amount of reward returned after previous action
            terminated (bool): whether the agent reaches the terminal state (e.g. game is finished).
            truncated (bool): whether the truncation condition outside the scope of the MDP is satisfied
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        valid_moves = [mv.value for mv in self._grid.get_valid_moves()]
        info = {}

        truncated = False
        if len(valid_moves) == 0:
            reward = self.reward_cfg.game_over_penalty
            terminated = True  # TODO: should this have been done in a previous session?
            info[INFO_KEY_GAME_OVER_REASON] = NO_VALID_MOVES
        elif action not in valid_moves:
            # NOTE: choosing to end the game when an illegal move is taken
            reward = self.reward_cfg.illegal_move_penalty
            terminated = False
            info[INFO_KEY_GAME_OVER_REASON] = ILLEGAL_MOVE_CHOSEN
        else:
            prev_score = self._grid.score
            self._grid.apply_move(logic.Move(action))
            delta_score = self._grid.score - prev_score
            reward = self.reward_cfg.lambda_step_reward + self.reward_cfg.lambda_score_reward * delta_score
            info[INFO_KEY_DELTA_SCORE] = delta_score
            terminated = False

        info[INFO_KEY_TOTAL_SCORE] = self._grid.score
        return self._grid.tiles, reward, terminated, truncated, info

    def reset(self, seed: int | None = None, options: dict[str, Any] = None) -> tuple[ObsType, dict[str, Any]]:
        """Resets the state of the environment and returns an initial observation.
        Returns:
            observation (object): the initial observation.
        """
        super().reset(seed=seed)
        self._grid = logic.Grid()
        # add 2 initial tiles
        for _ in range(2):
            self._grid.add_tile(log_val_tile=1)

        info = {INFO_KEY_GRID_SEED: int(self._grid_seed)}
        return self._grid.tiles, info

    def render(self):
        """Renders the environment.
        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:
        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).
            metadata = {'render.modes': ['human', 'rgb_array']}
            def render(self, mode='human'):
                if mode == 'rgb_array':
                    return np.array(...) # return RGB frame suitable for video
                elif mode == 'human':
                    ... # pop up a window and render
                else:
                    super(MyEnv, self).render(mode=mode) # just raise an exception
        """
        if self.render_mode == "ansi":
            print(self._grid.tiles)
            print(f"score: {self._grid.score}")
        else:
            super(Env2048, self).render()
