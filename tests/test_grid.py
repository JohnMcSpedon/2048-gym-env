"""Tests for Grid seeding and valid-move caching."""
import numpy as np

from gym_2048 import logic
from gym_2048.env_2048 import Env2048, RewardConfig
from gym_2048.logic import Grid, Move


def _play_seeded_episode(seed: int, num_steps: int = 30):
    """Play a fixed policy (first valid move) and record boards."""
    env = Env2048(reward_config=RewardConfig())
    obs, info = env.reset(seed=seed)
    boards = [obs.copy()]
    for _ in range(num_steps):
        valid_moves = info["valid_moves"]
        if not valid_moves:
            break
        obs, _, terminated, _, info = env.step(valid_moves[0])
        boards.append(obs.copy())
        if terminated:
            break
    return boards


def test_grid_seeded_spawns_are_deterministic():
    grid_a = Grid(rng=np.random.default_rng(42))
    grid_b = Grid(rng=np.random.default_rng(42))
    for _ in range(10):
        grid_a.add_tile()
        grid_b.add_tile()
    assert np.array_equal(grid_a.tiles, grid_b.tiles)


def test_env_reset_seed_controls_spawns():
    boards_a = _play_seeded_episode(seed=123)
    boards_b = _play_seeded_episode(seed=123)
    boards_c = _play_seeded_episode(seed=456)
    assert len(boards_a) == len(boards_b)
    for a, b in zip(boards_a, boards_b):
        assert np.array_equal(a, b)
    # Different seed should diverge (boards or episode length)
    diverged = len(boards_a) != len(boards_c) or any(
        not np.array_equal(a, c) for a, c in zip(boards_a, boards_c)
    )
    assert diverged


def test_valid_moves_cache_invalidated_on_mutation():
    grid = Grid(rng=np.random.default_rng(0))
    grid.tiles = np.array(
        [
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    assert set(grid.get_valid_moves()) == {Move.RIGHT, Move.DOWN}
    # Cached result is returned for an unchanged board
    assert grid.get_valid_moves() is grid.get_valid_moves()

    grid.apply_move(Move.RIGHT)
    # After the move (and random spawn) the cache must be recomputed, matching
    # a fresh computation on the current board.
    assert grid.get_valid_moves() == logic._get_valid_moves(grid.tiles)

    grid.add_tile(log_val_tile=1)
    assert grid.get_valid_moves() == logic._get_valid_moves(grid.tiles)
