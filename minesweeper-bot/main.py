import keyboard
import numpy as np
import random
import time

from bot import Cell, MinesweeperBot


def main(restart=False):
    running = True

    bot = MinesweeperBot()

    print("Starting...")

    state = bot.build_state()

    if state is None:
        print("Window not found, exiting...")
        return

    bot.focus_window(state)

    while not keyboard.is_pressed("down"):
        grid = bot.grid_from_state(state)

        if np.any(grid == Cell.MINE.value):
            if restart:
                print("Game over, restarting...")
                bot.click("restart")
                state = bot.build_state()
                continue
            else:
                print("Game over...")
                break

        # print(grid)

        move = find_next_move(grid)

        if move is None:
            if restart:
                print("Game won, restarting...")
                bot.click("restart")
                state = bot.build_state()
                continue
            else:
                print("Game won...")
                break

        print(move)

        x, y, button = move
        rows, cols = grid.shape
        print(x, y, grid[y, x])
        state = bot.click_cell(state, x, y, button)
        # time.sleep(.1)

    print("Stopping...")


def neighbors_of_type(grid, x, y, cond):
    rows, cols = grid.shape

    match (x, y):
        case (0, 0):
            return [
                (x + nx, y + ny)
                for ny, nx in np.argwhere(cond(grid[y : y + 2, x : x + 2]))
            ]
        case (0, y) if y == rows - 1:
            return [
                (x + nx, y + ny - 1)
                for ny, nx in np.argwhere(cond(grid[y - 1 : y + 1, x : x + 2]))
            ]
        case (x, 0) if x == cols - 1:
            return [
                (x + nx - 1, y + ny)
                for ny, nx in np.argwhere(cond(grid[y : y + 2, x - 1 : x + 1]))
            ]
        case (x, y) if x == cols - 1 and y == rows - 1:
            return [
                (x + nx - 1, y + ny - 1)
                for ny, nx in np.argwhere(cond(grid[y - 1 : y + 1, x - 1 : x + 1]))
            ]
        case (0, y):
            return [
                (x + nx, y + ny - 1)
                for ny, nx in np.argwhere(cond(grid[y - 1 : y + 2, x : x + 2]))
            ]
        case (x, 0):
            return [
                (x + nx - 1, y + ny)
                for ny, nx in np.argwhere(cond(grid[y : y + 2, x - 1 : x + 2]))
            ]
        case (x, y) if x == cols - 1:
            return [
                (x + nx - 1, y + ny - 1)
                for ny, nx in np.argwhere(cond(grid[y - 1 : y + 2, x - 1 : x + 1]))
            ]
        case (x, y) if y == rows - 1:
            return [
                (x + nx - 1, y + ny - 1)
                for ny, nx in np.argwhere(cond(grid[y - 1 : y + 1, x - 1 : x + 2]))
            ]
        case _:
            return [
                (x + nx - 1, y + ny - 1)
                for ny, nx in np.argwhere(cond(grid[y - 1 : y + 2, x - 1 : x + 2]))
            ]


def find_next_move(grid):
    rows, cols = grid.shape

    if np.all(grid == Cell.HIDDEN.value):
        return (random.randint(0, cols - 1), random.randint(0, rows - 1), "left")

    for y in range(rows):
        for x in range(cols):
            match grid[y, x]:
                case value if value > 0:
                    flag_neighbors = neighbors_of_type(
                        grid, x, y, lambda g: g == Cell.FLAG.value
                    )
                    hidden_neighbors = neighbors_of_type(
                        grid, x, y, lambda g: g == Cell.HIDDEN.value
                    )

                    if len(flag_neighbors) == value:
                        for nx, ny in hidden_neighbors:
                            return (nx, ny, "left")

                    if (len(flag_neighbors) + len(hidden_neighbors)) == value:
                        for nx, ny in hidden_neighbors:
                            return (nx, ny, "right")

                    if (
                        value - len(flag_neighbors) + 1 == 2
                        and len(hidden_neighbors) == 2
                    ):
                        gt_zero_neighbors = neighbors_of_type(
                            grid, x, y, lambda g: g > 0
                        )

                        for nx, ny in gt_zero_neighbors:
                            if nx == x and ny == y:
                                continue

                            n_flag_neighbors = neighbors_of_type(
                                grid, nx, ny, lambda g: g == Cell.FLAG.value
                            )
                            n_hidden_neighbors = neighbors_of_type(
                                grid, nx, ny, lambda g: g == Cell.HIDDEN.value
                            )

                            if len(n_hidden_neighbors) != 3:
                                continue

                            if grid[ny, nx] - len(n_flag_neighbors) + 1 == len(
                                n_hidden_neighbors
                            ):
                                uncommon_neighbors = [
                                    pos
                                    for pos in n_hidden_neighbors
                                    if not pos in hidden_neighbors
                                ]
                                if (
                                    len(n_hidden_neighbors) - len(uncommon_neighbors)
                                    == 2
                                ):
                                    for nnx, nny in uncommon_neighbors:
                                        return (nnx, nny, "right")
                            elif grid[ny, nx] - len(n_flag_neighbors) - 1 == 0:
                                uncommon_neighbors = [
                                    pos
                                    for pos in n_hidden_neighbors
                                    if not pos in hidden_neighbors
                                ]
                                if (
                                    len(n_hidden_neighbors) - len(uncommon_neighbors)
                                    == 2
                                ):
                                    for nnx, nny in uncommon_neighbors:
                                        return (nnx, nny, "left")
                case _:
                    pass

    hidden_cells = [(x, y) for y, x in np.argwhere(grid == Cell.HIDDEN.value)]

    if len(hidden_cells) > 0:
        rx, ry = hidden_cells[random.randint(0, len(hidden_cells) - 1)]
        return (rx, ry, "left")

    return None


if __name__ == "__main__":
    main()
