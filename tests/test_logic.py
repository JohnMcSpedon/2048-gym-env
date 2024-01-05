import sys
print(sys.path)

import numpy as np

from gym_2048 import logic
from gym_2048.logic import Move


def test_compress_tiles():
    for input, expected in [
        ([0, 0, 0], [0, 0, 0]),
        ([1, 0, 0], [1, 0, 0]),
        ([0, 1, 0], [1, 0, 0]),
        ([0, 0, 1], [1, 0, 0]),
        ([0, 1, 1], [1, 1, 0]),
        ([1, 1, 0], [1, 1, 0]),
        ([1, 0, 1], [1, 1, 0]),
        ([0, 1, 1], [1, 1, 0]),
        ([1, 1, 1], [1, 1, 1]),
    ]:
        output = logic._compress_tiles(np.array([input]))
        assert np.array_equal(output, np.array([expected]))


def test_merge_tiles():
    for input, expected_tiles, expected_delta_score in [
        ([0, 0, 0], [0, 0, 0], 0),
        ([1, 1, 0], [2, 0, 0], 4),
        ([1, 0, 1], [1, 0, 1], 0),
        ([1, 1, 1], [2, 0, 1], 4),
        ([1, 1, 1, 1], [2, 0, 2, 0], 8),
        ([1, 2, 1, 2], [1, 2, 1, 2], 0),
        ([1, 2, 2, 1], [1, 3, 0, 1], 8),

    ]:
        tiles, delta_score = logic._merge_tiles(np.array([input]))
        assert np.array_equal(tiles, np.array([expected_tiles]))
        assert delta_score == expected_delta_score


def test_apply_move():
    input = np.array([
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
    ])
    expected_score_delta = 32

    expected_left = np.array([
        [2, 2, 0, 0],
        [2, 2, 0, 0],
        [2, 2, 0, 0],
        [2, 2, 0, 0],
    ])
    tiles, score_delta = logic._apply_move(input.copy(), Move.LEFT)
    assert np.array_equal(tiles, expected_left)
    assert score_delta == expected_score_delta

    expected_right = np.array([
        [0, 0, 2, 2],
        [0, 0, 2, 2],
        [0, 0, 2, 2],
        [0, 0, 2, 2],
    ])
    tiles, score_delta = logic._apply_move(input.copy(), Move.RIGHT)
    assert np.array_equal(tiles, expected_right)
    assert score_delta == expected_score_delta

    expected_up = np.array([
        [2, 2, 2, 2],
        [2, 2, 2, 2],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ])
    tiles, score_delta = logic._apply_move(input.copy(), Move.UP)
    assert np.array_equal(tiles, expected_up)
    assert score_delta == expected_score_delta

    expected_down = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [2, 2, 2, 2],
        [2, 2, 2, 2],
    ])
    tiles, score_delta = logic._apply_move(input.copy(), Move.DOWN)
    assert np.array_equal(tiles, expected_down)
    assert score_delta == expected_score_delta


def test_get_valid_moves():
    tiles = np.array([
        [0, 0, 0, 0],
        [0, 2, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ])
    expected_moves = {
        Move.LEFT,
        Move.RIGHT,
        Move.DOWN,
        Move.UP,
    }
    assert set(logic._get_valid_moves(tiles)) == expected_moves

    tiles = np.array([
        [2, 2, 2, 2],
        [2, 2, 2, 2],
        [2, 2, 2, 2],
        [2, 2, 2, 2],
    ])
    expected_moves = {
        Move.LEFT,
        Move.RIGHT,
        Move.DOWN,
        Move.UP,
    }
    assert set(logic._get_valid_moves(tiles)) == expected_moves

    tiles = np.array([
        [2, 2, 2, 2],
        [4, 4, 4, 4],
        [6, 6, 6, 6],
        [8, 8, 8, 8],
    ])
    expected_moves = {
        Move.LEFT,
        Move.RIGHT,
    }
    assert set(logic._get_valid_moves(tiles)) == expected_moves

    tiles = np.array([
        [2, 4, 6, 8],
        [2, 4, 6, 8],
        [2, 4, 6, 8],
        [2, 4, 6, 8],
    ])
    expected_moves = {
        Move.DOWN,
        Move.UP,
    }
    assert set(logic._get_valid_moves(tiles)) == expected_moves

    tiles = np.array([
        [2, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ])
    expected_moves = {
        Move.DOWN,
        Move.RIGHT,
    }
    assert set(logic._get_valid_moves(tiles)) == expected_moves

    tiles = np.array([
        [2, 0, 0, 0],
        [4, 0, 0, 0],
        [6, 0, 0, 0],
        [8, 0, 0, 0],
    ])
    expected_moves = {
        Move.RIGHT,
    }
    assert set(logic._get_valid_moves(tiles)) == expected_moves

    tiles = np.array([
        [2, 0, 0, 0],
        [4, 0, 0, 0],
        [4, 0, 0, 0],
        [8, 0, 0, 0],
    ])
    expected_moves = {
        Move.UP,
        Move.DOWN,
        Move.RIGHT,
    }
    assert set(logic._get_valid_moves(tiles)) == expected_moves

    tiles = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [2, 4, 6, 8],
        [1, 3, 5, 7],
    ])
    expected_moves = {
        Move.UP,
    }
    assert set(logic._get_valid_moves(tiles)) == expected_moves