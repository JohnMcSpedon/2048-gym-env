import random

from . import logic

KEY_TO_MOVE = {
    "w": logic.Move.UP,
    "s": logic.Move.DOWN,
    "a": logic.Move.LEFT,
    "d": logic.Move.RIGHT,
}


class KeyboardAgent:
    def get_move(self, grid: logic.Grid):
        while True:
            display_grid(grid)
            key = input("which move?")
            key = key.strip()
            move = KEY_TO_MOVE.get(key, None)
            valid_moves = grid.get_valid_moves()
            if move in valid_moves:
                return move
            else:
                print(f"{move} is not a valid move ({valid_moves})")


def display_grid(grid: logic.Grid):
    print(grid.tiles)
    print(f"score: {grid.score}")


class DownLeftAgent:
    def get_move(self, grid: logic.Grid):
        display_grid(grid)
        valid_moves = grid.get_valid_moves()
        if logic.Move.DOWN in valid_moves:
            return logic.Move.DOWN
        elif logic.Move.LEFT in valid_moves:
            return logic.Move.LEFT
        else:
            return random.choice(valid_moves)


class CmdLineGame:
    def __init__(self, agent):
        self.grid = logic.Grid()
        self.agent = agent

    def main_loop(self):
        # add 2 initial tiles
        for _ in range(2):
            self.grid.add_tile(log_val_tile=1)

        while True:
            valid_moves = self.grid.get_valid_moves()
            if len(valid_moves) == 0:
                break
            # get move from agent
            move = self.agent.get_move(self.grid)
            assert move in valid_moves, f"{move} is not a valid move ({valid_moves})"
            self.grid.apply_move(move)

        print("Game Over")
        print(f"Final score: {self.grid.score}")


if __name__ == "__main__":
    # game = CmdLineGame(KeyboardAgent())
    game = CmdLineGame(DownLeftAgent())
    game.main_loop()
