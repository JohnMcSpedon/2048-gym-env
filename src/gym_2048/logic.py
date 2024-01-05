import enum
import logging
from typing import List, Tuple

import numpy as np

GRID_HEIGHT = 4
GRID_WIDTH = 4
TILE_SIZE_PROB = 0.9  # 90% 2's, 10% 4's


class Move(enum.Enum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3


NUM_ACTIONS = len(Move)


def _compress_tiles(tiles: np.ndarray) -> np.ndarray:
    """Push all tiles left. (leaving all empty tiles on the right). """
    # TODO: vectorize over rows
    for row in tiles:
        nonempty = row[row != 0]
        row[: len(nonempty)] = nonempty
        row[len(nonempty) :] = 0
    return tiles


_TWO_EXP = {n: int(2 ** n) for n in range(20)}
_TWO_EXP[0] = 0  # NOTE: 2^0 = 1, but we instead use 0 to denote an empty cell


def two_exp(exponent: int) -> int:
    return _TWO_EXP[exponent]


def _merge_tiles(tiles: np.ndarray) -> Tuple[np.ndarray, int]:
    """Merge tiles left. Pairs of adjacent tiles with equal values are combined together."""
    delta_score = 0
    for row in tiles:
        for col_idx in range(len(row) - 1):
            if row[col_idx] != 0 and row[col_idx] == row[col_idx + 1]:
                row[col_idx] += 1
                delta_score += two_exp(row[col_idx])
                row[col_idx + 1] = 0
    return tiles, delta_score


def _get_valid_moves(tiles: np.ndarray) -> List[Move]:
    valid_moves = []
    for move in Move:
        moved_tiles, _ = _apply_move(tiles, move)
        if np.not_equal(tiles, moved_tiles).any():
            valid_moves.append(move)
    return valid_moves


def _apply_move(tiles: np.ndarray, move: Move) -> Tuple[np.ndarray, int]:
    tiles = tiles.copy()  # TODO: what is the performance hit of this lazy implementation?
    if move == Move.LEFT:
        pass
    elif move == Move.RIGHT:
        tiles = np.flip(tiles, axis=1)
    elif move == Move.UP:
        tiles = tiles.T
    elif move == Move.DOWN:
        tiles = np.flip(tiles, axis=0).T

    tiles = _compress_tiles(tiles)
    tiles, delta_score = _merge_tiles(tiles)
    tiles = _compress_tiles(tiles)

    if move == Move.LEFT:
        pass
    elif move == Move.RIGHT:
        tiles = np.flip(tiles, axis=1)
    elif move == Move.UP:
        tiles = tiles.T
    elif move == Move.DOWN:
        tiles = np.flip(tiles.T, axis=0)
    return tiles, delta_score


class Grid:
    """Stores / mutates grid of tiles.

    Tiles are represented as an M x N integer array.
    Empty tiles are represented with 0.  Otherwise a tiles value is stored as log base 2.
    """

    def __init__(self, random_seed=None):
        # TODO: accept tiles as an init argument.  factor above methods into this class?
        self.tiles = np.zeros([GRID_HEIGHT, GRID_WIDTH], dtype=np.uint8)
        self.score = 0

        if random_seed is not None:
            logging.debug(f"Setting random seed to {random_seed}")
            np.random.seed(random_seed)

    def _replay_moves(self, moves):
        # TODO: add ability to replay game up to certain state?
        raise NotImplementedError()

    def add_tile(self, log_val_tile=None):
        open_indices = np.argwhere(self.tiles == 0)
        num_open = len(open_indices)
        if num_open > 0:
            sample_idx = np.random.choice(num_open)
            if log_val_tile is None:
                log_val_tile = 1 if np.random.random() <= TILE_SIZE_PROB else 2
            self.tiles[tuple(open_indices[sample_idx])] = log_val_tile  # convert to tuple for basic indexing
            return True
        else:
            logging.debug("Game over")
            return False

    def get_valid_moves(self):
        return _get_valid_moves(self.tiles)

    def apply_move(self, move):
        valid_moves = self.get_valid_moves()
        assert move in valid_moves, "Invalid move; {} is not in {}".format(move, valid_moves)
        self.tiles, delta_score = _apply_move(self.tiles, move)
        self.score += delta_score
        return self.add_tile()
