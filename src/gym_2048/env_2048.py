from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.core import ObsType
from gymnasium.utils import seeding

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
    metadata = {"render.modes": ["ansi"]}  # TODO: implement this
    reward_range = (-float("inf"), float("inf"))  # TODO: narrow this range?

    action_space = gym.spaces.Discrete(4)  # up, down, left, right
    observation_space = gym.spaces.Space(shape=[4, 4], dtype=np.uint8)

    def __init__(self, reward_config: RewardConfig):
        super(Env2048, self).__init__()
        self.reward_cfg = reward_config
        self.reset()

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (object): an action provided by the agent
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        valid_moves = [mv.value for mv in self._grid.get_valid_moves()]
        info = {INFO_KEY_GRID_SEED: int(self._grid_seed)}

        if len(valid_moves) == 0:
            reward = self.reward_cfg.game_over_penalty
            done = True  # TODO: should this have been done in a previus session?
            info[INFO_KEY_GAME_OVER_REASON] = NO_VALID_MOVES
        elif action not in valid_moves:
            # NOTE: choosing to end the game when an illegal move is taken
            reward = self.reward_cfg.illegal_move_penalty
            done = True
            info[INFO_KEY_GAME_OVER_REASON] = ILLEGAL_MOVE_CHOSEN
        else:
            prev_score = self._grid.score
            self._grid.apply_move(logic.Move(action))
            delta_score = self._grid.score - prev_score
            reward = self.reward_cfg.lambda_step_reward + self.reward_cfg.lambda_score_reward * delta_score
            info[INFO_KEY_DELTA_SCORE] = delta_score
            done = False

        info[INFO_KEY_TOTAL_SCORE] = self._grid.score
        return self._grid.tiles, reward, done, info

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
        # TODO: ret val

    def render(self, mode="ansi"):
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
        Note:
            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.
        Args:
            mode (str): the mode to render with
        Example:
        class MyEnv(Env):
            metadata = {'render.modes': ['human', 'rgb_array']}
            def render(self, mode='human'):
                if mode == 'rgb_array':
                    return np.array(...) # return RGB frame suitable for video
                elif mode == 'human':
                    ... # pop up a window and render
                else:
                    super(MyEnv, self).render(mode=mode) # just raise an exception
        """
        if mode == "ansi":
            print(self._grid.tiles)
            print(f"score: {self._grid.score}")
        else:
            super(Env2048, self).render(mode=mode)
