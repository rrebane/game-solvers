import heapq
import itertools
import keyboard
import numpy as np
import random
import time

from bot import Cell, NonogramBot


def main(restart=False):
    running = True

    bot = NonogramBot()

    print("Starting...")

    state = bot.build_state()

    if state is None:
        print("Nonogram not found, exiting...")
        return

    v_numbers = bot.vertical_numbers_from_state(state)
    h_numbers = bot.horizontal_numbers_from_state(state)

    print(v_numbers)
    print(h_numbers)

    bot.focus_window(state)

    grid = bot.grid_from_state(state)
    move = None

    move_gen = find_next_move(grid.copy(), v_numbers, h_numbers)

    print("Grid size:", (grid.shape[1], grid.shape[0]))
    print(grid)

    while not keyboard.is_pressed("up"):
        try:
            move = next(move_gen)
        except StopIteration:
            print("Game won...")
            break

        print("Move:", move)

        match move:
            case (start_x, start_y, end_x, end_y, button):
                state = bot.drag_cell(state, start_x, start_y, end_x, end_y, button)
            case (x, y, button):
                state = bot.click_cell(state, x, y, button)
            case _:
                assert False

        # print(bot.grid_from_state(state))

        # time.sleep(.1)

    print("Stopping...")


def permutations(row_or_col, numbers, offset=0, current=None, common=None):
    if len(numbers) <= 0:
        return None

    if current is None:
        current = np.zeros(row_or_col.shape[0], dtype=np.uint8)

    mask = np.ones(row_or_col.shape[0], dtype=bool)

    n_total = np.sum(numbers) + len(numbers) - 1

    if n_total + offset > row_or_col.shape[0]:
        return None

    index_range = row_or_col.shape[0] - n_total - offset + 1

    indices = [
        idx
        for idx in itertools.chain.from_iterable(
            itertools.zip_longest(
                range(0, index_range if index_range % 2 == 0 else index_range + 1, 2),
                range(
                    index_range - 1 if index_range % 2 == 0 else index_range - 2, -1, -2
                ),
            )
        )
        if not idx is None
    ]

    for i in indices:
        if np.all(~mask):
            break

        if np.any(
            row_or_col[offset + i : offset + i + numbers[0]] == Cell.MARKED.value
        ):
            continue

        if np.any(row_or_col[offset : offset + i] == Cell.FILLED.value):
            continue

        if offset + i > 0 and row_or_col[offset + i - 1] == Cell.FILLED.value:
            continue

        if (
            offset + i + numbers[0] < row_or_col.shape[0]
            and row_or_col[offset + i + numbers[0]] == Cell.FILLED.value
        ):
            continue

        if len(numbers[1:]) > 0:
            current[offset : offset + i] = Cell.MARKED.value
            current[offset + i : offset + i + numbers[0]] = Cell.FILLED.value
            current[offset + i + numbers[0]] = Cell.MARKED.value

            c_perm = permutations(
                row_or_col, numbers[1:], offset + i + numbers[0] + 1, current, common
            )

            if not c_perm is None:
                common, c_mask = c_perm
                mask = np.logical_and(c_mask, mask)

            current[offset : offset + i + numbers[0] + 1] = Cell.EMPTY.value
        else:
            if offset + i + numbers[0] < row_or_col.shape[0] and np.any(
                row_or_col[offset + i + numbers[0] :] == Cell.FILLED.value
            ):
                continue

            current[offset : offset + i] = Cell.MARKED.value
            current[offset + i : offset + i + numbers[0]] = Cell.FILLED.value

            if current.shape[0] > offset + i + numbers[0]:
                current[offset + i + numbers[0] :] = Cell.MARKED.value

            if common is None:
                common = current.copy()

            mask = np.logical_and(common == current, mask)

            current[offset:] = Cell.EMPTY.value

    return common, mask


def stable_cells(shape, perms):
    if len(perms) <= 0:
        return np.zeros(shape, dtype=np.uint8), np.zeros(shape, dtype=bool)

    cells = np.full((len(perms), shape), Cell.MARKED.value, dtype=np.uint8)

    for i, groups in enumerate(perms):
        for (offset, size) in groups:
            cells[i, offset : offset + size] = Cell.FILLED.value

    return cells[0, :], np.all(cells == cells[0, :], axis=0)


def v_moves(x, rows, mask, button):
    moves = []
    prev_val = False
    start_y = 0

    for y in range(rows):
        if mask[y] != prev_val:
            if mask[y]:
                start_y = y
            else:
                if start_y + 1 == y:
                    yield (x, start_y, button)
                else:
                    yield (x, start_y, x, y - 1, button)

        prev_val = mask[y]

    if mask[rows - 1]:
        if start_y + 1 == rows:
            yield (x, start_y, button)
        else:
            yield (x, start_y, x, rows - 1, button)


def h_moves(y, cols, mask, button):
    prev_val = False
    start_x = 0

    for x in range(cols):
        if mask[x] != prev_val:
            if mask[x]:
                start_x = x
            else:
                if start_x + 1 == x:
                    yield (start_x, y, button)
                else:
                    yield (start_x, y, x - 1, y, button)

        prev_val = mask[x]

    if mask[cols - 1]:
        if start_x + 1 == cols:
            yield (start_x, y, button)
        else:
            yield (start_x, y, cols - 1, y, button)


def v_update(grid, v_numbers, v_numbers_mask, v_completed, x):
    if v_completed[x]:
        return

    rows, _ = grid.shape

    common, common_mask = permutations(grid[:, x], v_numbers[x][v_numbers_mask[x]])
    mask = np.logical_and(grid[:, x] == Cell.EMPTY.value, common_mask)
    filled_mask = np.logical_and(common == Cell.FILLED.value, mask)
    ##  marked_mask = np.logical_and(common == Cell.MARKED.value, mask)

    grid[mask, x] = common[mask]

    for move in v_moves(x, rows, filled_mask, "left"):
        yield move


def h_update(grid, h_numbers, h_numbers_mask, h_completed, y):
    if h_completed[y]:
        return

    _, cols = grid.shape

    common, common_mask = permutations(grid[y, :], h_numbers[y][h_numbers_mask[y]])
    mask = np.logical_and(grid[y, :] == Cell.EMPTY.value, common_mask)
    filled_mask = np.logical_and(common == Cell.FILLED.value, mask)
    ##  marked_mask = np.logical_and(common == Cell.MARKED.value, mask)

    grid[y, mask] = common[mask]

    for move in h_moves(y, cols, filled_mask, "left"):
        yield move


def find_next_move(grid, v_numbers, h_numbers):
    rows, cols = grid.shape

    v_numbers_mask = v_numbers > 0
    h_numbers_mask = h_numbers > 0

    v_completed = np.zeros(cols, dtype=bool)
    h_completed = np.zeros(rows, dtype=bool)

    v_number_max = np.max(v_numbers, axis=1)
    h_number_max = np.max(h_numbers, axis=1)

    v_order = np.argsort(v_number_max)[::-1]
    h_order = np.argsort(h_number_max)[::-1]

    # https://www.nonograms.org/methods

    first_pass = True

    while True:
        moved = False

        # Check for completed rows and columns
        for x in range(cols):
            if np.sum(grid[:, x] == Cell.FILLED.value) == np.sum(v_numbers[x]):
                mask = grid[:, x] == Cell.EMPTY.value
                grid[mask, x] = Cell.MARKED.value
                v_completed[x] = True

        for y in range(rows):
            if np.sum(grid[y, :] == Cell.FILLED.value) == np.sum(h_numbers[y]):
                mask = grid[y, :] == Cell.EMPTY.value
                grid[y, mask] = Cell.MARKED.value
                h_completed[y] = True

        v_idx = 0
        h_idx = 0

        index_queue = []
        index_set = set()

        while v_idx < cols or h_idx < rows:
            if not first_pass and index_queue:
                match heapq.heappop(index_queue):
                    case (_, "v", x):
                        index_set.remove(("v", x))

                        print("Checking column #", x + 1)

                        for move in v_update(
                            grid, v_numbers, v_numbers_mask, v_completed, x
                        ):
                            moved = True

                            match move:
                                case (_, start_y, _, end_y, "left"):
                                    for y in range(start_y, end_y + 1):
                                        if not ("h", y) in index_set:
                                            heapq.heappush(
                                                index_queue,
                                                (
                                                    grid.shape[0] - h_number_max[y],
                                                    "h",
                                                    y,
                                                ),
                                            )
                                            index_set.add(("h", y))
                                    pass
                                case (_, y, "left"):
                                    if not ("h", y) in index_set:
                                        heapq.heappush(
                                            index_queue,
                                            (grid.shape[0] - h_number_max[y], "h", y),
                                        )
                                        index_set.add(("h", y))
                                    pass

                            yield move
                    case (_, "h", y):
                        index_set.remove(("h", y))

                        print("Checking row #", y + 1)

                        for move in h_update(
                            grid, h_numbers, h_numbers_mask, h_completed, y
                        ):
                            moved = True

                            match move:
                                case (start_x, _, end_x, _, "left"):
                                    for x in range(start_x, end_x + 1):
                                        if not ("v", x) in index_set:
                                            heapq.heappush(
                                                index_queue,
                                                (
                                                    grid.shape[1] - v_number_max[x],
                                                    "v",
                                                    x,
                                                ),
                                            )
                                            index_set.add(("v", x))
                                    pass
                                case (x, _, "left"):
                                    if not ("v", x) in index_set:
                                        heapq.heappush(
                                            index_queue,
                                            (grid.shape[1] - v_number_max[x], "v", x),
                                        )
                                        index_set.add(("v", x))
                                    pass

                            yield move
                    case _:
                        assert False
            else:
                v_max = v_number_max[v_order[v_idx]] if v_idx < cols else 0
                h_max = h_number_max[h_order[h_idx]] if h_idx < rows else 0

                if first_pass and max(v_max, h_max) < 10:
                    break

                if v_max > h_max:
                    x = v_order[v_idx]
                    print("Checking column #", x + 1)

                    for move in v_update(
                        grid, v_numbers, v_numbers_mask, v_completed, x
                    ):
                        moved = True

                        match move:
                            case (_, start_y, _, end_y, "left"):
                                for y in range(start_y, end_y + 1):
                                    if not ("h", y) in index_set:
                                        heapq.heappush(
                                            index_queue,
                                            (
                                                grid.shape[0]
                                                - np.sum(
                                                    grid[y, :] == Cell.FILLED.value
                                                ),
                                                "h",
                                                y,
                                            ),
                                        )
                                        index_set.add(("h", y))
                                pass
                            case (_, y, "left"):
                                if not ("h", y) in index_set:
                                    heapq.heappush(
                                        index_queue,
                                        (
                                            grid.shape[0]
                                            - np.sum(grid[y, :] == Cell.FILLED.value),
                                            "h",
                                            y,
                                        ),
                                    )
                                    index_set.add(("h", y))
                                pass

                        yield move

                    v_idx += 1
                else:
                    y = h_order[h_idx]
                    print("Checking row #", y + 1)

                    for move in h_update(
                        grid, h_numbers, h_numbers_mask, h_completed, y
                    ):
                        moved = True

                        match move:
                            case (start_x, _, end_x, _, "left"):
                                for x in range(start_x, end_x + 1):
                                    if not ("v", x) in index_set:
                                        heapq.heappush(
                                            index_queue,
                                            (
                                                grid.shape[1]
                                                - np.sum(
                                                    grid[:, x] == Cell.FILLED.value
                                                ),
                                                "v",
                                                x,
                                            ),
                                        )
                                        index_set.add(("v", x))
                                pass
                            case (x, _, "left"):
                                if not ("v", x) in index_set:
                                    heapq.heappush(
                                        index_queue,
                                        (
                                            grid.shape[1]
                                            - np.sum(grid[:, x] == Cell.FILLED.value),
                                            "v",
                                            x,
                                        ),
                                    )
                                    index_set.add(("v", x))
                                pass

                        yield move

                    h_idx += 1

        first_pass = False

        if not moved:
            break

    return None


if __name__ == "__main__":
    main()
