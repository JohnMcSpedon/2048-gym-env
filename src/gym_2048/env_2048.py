import logging
from dataclasses import dataclass
from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium.core import ActType, ObsType
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import logic

INFO_KEY_GRID_SEED = "grid_seed"
INFO_KEY_GAME_OVER_REASON = "game_over_reason"
INFO_KEY_DELTA_SCORE = "delta_score"
INFO_KEY_TOTAL_SCORE = "total_score"
INFO_KEY_MAX_TILE = "max_tile"

NO_VALID_MOVES = "no_valid_moves"
ILLEGAL_MOVE_CHOSEN = "illegal_move_chosen"


def hex_to_rgb(hex_color):
    """Convert hex color string to RGB tuple."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


BACKGROUND_COLOR_GAME = hex_to_rgb("#92877d")
BACKGROUND_COLOR_CELL_EMPTY = hex_to_rgb("#9e948a")
BACKGROUND_COLOR_DICT = {
    2: hex_to_rgb("#eee4da"),
    4: hex_to_rgb("#ede0c8"),
    8: hex_to_rgb("#f2b179"),
    16: hex_to_rgb("#f59563"),
    32: hex_to_rgb("#f67c5f"),
    64: hex_to_rgb("#f65e3b"),
    128: hex_to_rgb("#edcf72"),
    256: hex_to_rgb("#edcc61"),
    512: hex_to_rgb("#edc850"),
    1024: hex_to_rgb("#edc53f"),
    2048: hex_to_rgb("#edc22e"),
}
CELL_COLOR_DICT = {
    2: hex_to_rgb("#776e65"),
    4: hex_to_rgb("#776e65"),
    8: hex_to_rgb("#f9f6f2"),
    16: hex_to_rgb("#f9f6f2"),
    32: hex_to_rgb("#f9f6f2"),
    64: hex_to_rgb("#f9f6f2"),
    128: hex_to_rgb("#f9f6f2"),
    256: hex_to_rgb("#f9f6f2"),
    512: hex_to_rgb("#f9f6f2"),
    1024: hex_to_rgb("#f9f6f2"),
    2048: hex_to_rgb("#f9f6f2"),
}
FONT_SIZE = 40
try:
    FONT = ImageFont.truetype("Verdana.ttf", FONT_SIZE)  # second value is font size
except IOError:
    print("loading verdana failed")
    FONT = ImageFont.load_default()


@dataclass
class RewardConfig:
    lambda_step_reward: float = 0.0
    lambda_score_reward: float = 1.0
    illegal_move_penalty: float = 0.0
    game_over_penalty: float = 0.0
    # Bonus rewards for achieving high tiles
    tile_bonus_512: float = 50.0
    tile_bonus_1024: float = 200.0
    tile_bonus_2048: float = 1000.0


class Env2048(gym.Env):
    r"""
    Attributes:
        action_space: The Space object corresponding to valid actions
        observation_space: The Space object corresponding to valid observations
        reward_range: A tuple corresponding to the min and max possible rewards
    """
    metadata = {"render.modes": ["ansi", "human", "rgb_array"]}
    reward_range = (-float("inf"), float("inf"))  # TODO: narrow this range?

    action_space = gym.spaces.Discrete(4)  # up, down, left, right
    observation_space = gym.spaces.Space(shape=[4, 4], dtype=np.uint8)

    def __init__(self, reward_config: RewardConfig, render_mode: str):
        super(Env2048, self).__init__()
        self.reward_cfg = reward_config
        self.render_mode = render_mode
        self.achieved_tiles = set()  # Track tiles achieved this episode
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
            logging.warning("No valid moves before action taken. This should've been caught previous step.")
            reward = self.reward_cfg.game_over_penalty
            terminated = True
            info[INFO_KEY_GAME_OVER_REASON] = NO_VALID_MOVES
        elif action not in valid_moves:
            # Illegal move - no state change, small negative reward
            reward = self.reward_cfg.illegal_move_penalty
            terminated = False
            info[INFO_KEY_GAME_OVER_REASON] = ILLEGAL_MOVE_CHOSEN
            info[INFO_KEY_DELTA_SCORE] = 0
        else:
            prev_score = self._grid.score
            self._grid.apply_move(logic.Move(action))
            delta_score = self._grid.score - prev_score
            reward = self.reward_cfg.lambda_step_reward + self.reward_cfg.lambda_score_reward * delta_score
            info[INFO_KEY_DELTA_SCORE] = delta_score

            # Check for new high tiles and give bonuses
            max_tile = np.max(self._grid.tiles)
            if max_tile >= 9 and max_tile not in self.achieved_tiles:  # 9 = log2(512)
                self.achieved_tiles.add(max_tile)
                if max_tile == 9:  # 512
                    reward += self.reward_cfg.tile_bonus_512
                elif max_tile == 10:  # 1024
                    reward += self.reward_cfg.tile_bonus_1024
                elif max_tile == 11:  # 2048
                    reward += self.reward_cfg.tile_bonus_2048

            terminated = False

        valid_moves = [mv.value for mv in self._grid.get_valid_moves()]
        if len(valid_moves) == 0:
            reward += self.reward_cfg.game_over_penalty
            terminated = True
            info[INFO_KEY_GAME_OVER_REASON] = NO_VALID_MOVES

        info[INFO_KEY_TOTAL_SCORE] = self._grid.score
        info[INFO_KEY_MAX_TILE] = int(2 ** np.max(self._grid.tiles)) if np.max(self._grid.tiles) > 0 else 0
        return self._grid.tiles, reward, terminated, truncated, info

    def reset(self, seed: int | None = None, options: dict[str, Any] = None) -> tuple[ObsType, dict[str, Any]]:
        """Resets the state of the environment and returns an initial observation.
        Returns:
            observation (object): the initial observation.
        """
        super().reset(seed=seed)
        self._grid = logic.Grid()
        self.achieved_tiles = set()  # Reset achieved tiles
        # add 2 initial tiles
        for _ in range(2):
            self._grid.add_tile(log_val_tile=1)

        info = {INFO_KEY_GRID_SEED: seed}
        return self._grid.tiles, info

    def render(self):
        if self.render_mode == "ansi":
            print(self._grid.tiles)
            print(f"score: {self._grid.score}")
        elif self.render_mode in ["human", "rgb_array"]:
            tile_size = 100
            grid_size = tile_size * self._grid.tiles.shape[0]
            image = Image.new("RGB", (grid_size, grid_size), BACKGROUND_COLOR_GAME)
            draw = ImageDraw.Draw(image)

            for i in range(self._grid.tiles.shape[0]):
                for j in range(self._grid.tiles.shape[1]):
                    value = self._grid.tiles[i][j]
                    x = j * tile_size
                    y = i * tile_size
                    rect = [x + 5, y + 5, x + tile_size - 5, y + tile_size - 5]

                    # Get tile background color
                    if value == 0:
                        color = BACKGROUND_COLOR_CELL_EMPTY
                    else:
                        number = 2 ** value
                        color = BACKGROUND_COLOR_DICT.get(number, "#3c3a32")

                    # Draw tile
                    draw.rectangle(rect, fill=color, outline=BACKGROUND_COLOR_GAME)

                    # Draw number
                    if value != 0:
                        number = 2 ** value
                        text = str(number)
                        text_color = CELL_COLOR_DICT[number]
                        # Adjust font size based on number length
                        text_length = draw.textlength(text, font=FONT)
                        text_x = x + (tile_size - text_length) / 2
                        text_y = y + (tile_size - FONT_SIZE) / 2
                        draw.text((text_x, text_y), text, fill=text_color)

            if self.render_mode == "human":
                image.show()
            else:
                return np.array(image)
        else:
            super(Env2048, self).render()
