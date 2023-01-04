import cv2
import numpy as np
import math
import mss
import os
import pyautogui

from enum import Enum


class Cell(Enum):
    EMPTY = 0
    FILLED = 1
    MARKED = 2


class NonogramBot:
    def __init__(self):
        self.puzzle_top_left_offset = (9, 10)
        self.puzzle_bottom_right_offset = (-1, -1)
        self.grid_top_left_offset = (24, 23)
        self.cell_size = (19, 18)
        self.grid_cell_images = [
            "filled",
            "marked",
        ]
        self.bold_border_size = 1

    def build_state(self, p_bounds=None):
        if p_bounds is None:
            p_bounds = find_puzzle_bounds(
                self.puzzle_top_left_offset, self.puzzle_bottom_right_offset
            )

        if p_bounds is None:
            return None

        g_bounds = find_grid_bounds(p_bounds, self.grid_top_left_offset)

        vertical_numbers_bounds = (
            (g_bounds[0][0], p_bounds[0][1]),
            (g_bounds[1][0], g_bounds[0][1]),
        )

        horizontal_numbers_bounds = (
            (p_bounds[0][0], g_bounds[0][1]),
            (g_bounds[0][0], p_bounds[1][1]),
        )

        vertical_numbers = parse_number_cells(
            vertical_numbers_bounds, 1, self.cell_size, self.bold_border_size
        )
        horizontal_numbers = parse_number_cells(
            horizontal_numbers_bounds, 0, self.cell_size, self.bold_border_size
        )
        grid = parse_grid_cells(
            g_bounds, self.grid_cell_images, self.cell_size, self.bold_border_size
        )

        assert vertical_numbers.shape[0] == grid.shape[1]
        assert horizontal_numbers.shape[0] == grid.shape[0]

        return (
            grid,
            vertical_numbers,
            horizontal_numbers,
            p_bounds,
            g_bounds,
        )

    def click_cell(self, state, cell_x, cell_y, button="left"):
        grid, vertical_numbers, horizontal_numbers, p_bounds, g_bounds = state

        (g_min_x, g_min_y), _ = g_bounds

        x = (
            g_min_x
            + cell_x * self.cell_size[0]
            + self.cell_size[0] // 2
            + (cell_x // 5) * self.bold_border_size
        )
        y = (
            g_min_y
            + cell_y * self.cell_size[1]
            + self.cell_size[1] // 2
            + (cell_y // 5) * self.bold_border_size
        )

        pyautogui.click(x, y, button=button)

        match button:
            case "left":
                if grid[cell_y, cell_x] == Cell.FILLED.value:
                    grid[cell_y, cell_x] = Cell.EMPTY.value
                else:
                    grid[cell_y, cell_x] = Cell.FILLED.value
            case "right":
                if grid[cell_y, cell_x] == Cell.MARKED.value:
                    grid[cell_y, cell_x] = Cell.EMPTY.value
                else:
                    grid[cell_y, cell_x] = Cell.MARKED.value
            case _:
                assert False

        return (grid, vertical_numbers, horizontal_numbers, p_bounds, g_bounds)

    def drag_cell(
        self, state, start_cell_x, start_cell_y, end_cell_x, end_cell_y, button="left"
    ):
        grid, vertical_numbers, horizontal_numbers, p_bounds, g_bounds = state

        (g_min_x, g_min_y), _ = g_bounds

        assert start_cell_x <= end_cell_x and start_cell_y <= end_cell_y
        assert start_cell_x == end_cell_x or start_cell_y == end_cell_y

        start_x = (
            g_min_x
            + start_cell_x * self.cell_size[0]
            + self.cell_size[0] // 2
            + (start_cell_x // 5) * self.bold_border_size
        )
        start_y = (
            g_min_y
            + start_cell_y * self.cell_size[1]
            + self.cell_size[1] // 2
            + (start_cell_y // 5) * self.bold_border_size
        )

        end_x = (
            g_min_x
            + end_cell_x * self.cell_size[0]
            + self.cell_size[0] // 2
            + (end_cell_x // 5) * self.bold_border_size
        )
        end_y = (
            g_min_y
            + end_cell_y * self.cell_size[1]
            + self.cell_size[1] // 2
            + (end_cell_y // 5) * self.bold_border_size
        )

        duration = (end_cell_x - start_cell_x + end_cell_y - start_cell_y) // 5 * 0.01

        pyautogui.moveTo(start_x, start_y)
        pyautogui.dragTo(end_x, end_y, duration, button=button)

        match button:
            case "left":
                mask = (
                    grid[start_y : end_y + 1, start_x : end_x + 1] == Cell.FILLED.value
                )
                grid[start_y : end_y + 1, start_x : end_x + 1][mask] = Cell.EMPTY.value
                grid[start_y : end_y + 1, start_x : end_x + 1][
                    ~mask
                ] = Cell.FILLED.value
            case "right":
                mask = (
                    grid[start_y : end_y + 1, start_x : end_x + 1] == Cell.MARKED.value
                )
                grid[start_y : end_y + 1, start_x : end_x + 1][mask] = Cell.EMPTY.value
                grid[start_y : end_y + 1, start_x : end_x + 1][
                    ~mask
                ] = Cell.MARKED.value
            case _:
                assert False

        return (grid, vertical_numbers, horizontal_numbers, p_bounds, g_bounds)

    def focus_window(self, state, threshold=0.95):
        mon = None
        scr = None

        with mss.mss() as sct:
            mon = sct.monitors[0]
            scr = np.array(sct.grab(mon), dtype=np.uint8)[:, :, :3]

        header_img = cv2.imread(".\\img\\header.jpg", cv2.IMREAD_UNCHANGED)
        header_res = cv2.matchTemplate(scr, header_img, cv2.TM_CCOEFF_NORMED)
        _, header_val, _, header_loc = cv2.minMaxLoc(header_res)

        if header_val < threshold:
            return None

        pyautogui.click(header_loc)

    def grid_from_state(self, state):
        return state[0]

    def vertical_numbers_from_state(self, state):
        return state[1]

    def horizontal_numbers_from_state(self, state):
        return state[2]


def find_puzzle_bounds(
    top_left_offset=(0, 0), bottom_right_offset=(0, 0), threshold=0.95
):
    mon = None
    scr = None

    with mss.mss() as sct:
        mon = sct.monitors[0]
        scr = np.array(sct.grab(mon), dtype=np.uint8)[:, :, :3]

    header_img = cv2.imread(".\\img\\header.jpg", cv2.IMREAD_UNCHANGED)
    header_res = cv2.matchTemplate(scr, header_img, cv2.TM_CCOEFF_NORMED)
    _, header_val, _, header_loc = cv2.minMaxLoc(header_res)

    if header_val < threshold:
        return None

    top_left_img = cv2.imread(".\\img\\top_left.jpg", cv2.IMREAD_UNCHANGED)
    top_left_res = cv2.matchTemplate(scr, top_left_img, cv2.TM_CCOEFF_NORMED)
    _, top_left_val, _, top_left_loc = cv2.minMaxLoc(top_left_res)

    if top_left_val < threshold:
        return None

    bottom_right_img = cv2.imread(f".\\img\\bottom_right.jpg", cv2.IMREAD_UNCHANGED)
    bottom_right_res = cv2.matchTemplate(scr, bottom_right_img, cv2.TM_CCOEFF_NORMED)
    _, bottom_right_val, _, bottom_right_loc = cv2.minMaxLoc(bottom_right_res)

    if bottom_right_val < threshold:
        return None

    bottom_right_img_h, bottom_right_img_w = bottom_right_img.shape[:2]

    return (
        (
            top_left_loc[0] + mon["left"] + top_left_offset[0],
            top_left_loc[1] + mon["top"] + top_left_offset[1],
        ),
        (
            bottom_right_loc[0]
            + mon["left"]
            + bottom_right_img_w
            + bottom_right_offset[0],
            bottom_right_loc[1]
            + mon["top"]
            + bottom_right_img_h
            + bottom_right_offset[1],
        ),
    )


def find_grid_bounds(p_bounds, top_left_offset=(0, 0), threshold=0.95):
    mon = {
        "left": p_bounds[0][0],
        "top": p_bounds[0][1],
        "width": p_bounds[1][0] - p_bounds[0][0],
        "height": p_bounds[1][1] - p_bounds[0][1],
    }

    scr = None

    with mss.mss() as sct:
        scr = np.array(sct.grab(mon))[:, :, :3]

    top_left_img = cv2.imread(".\\img\\grid_top_left.jpg", cv2.IMREAD_UNCHANGED)
    top_left_res = cv2.matchTemplate(scr, top_left_img, cv2.TM_CCOEFF_NORMED)
    _, top_left_val, _, top_left_loc = cv2.minMaxLoc(top_left_res)

    if top_left_val < threshold:
        return None

    return (
        (
            top_left_loc[0] + mon["left"] + top_left_offset[0],
            top_left_loc[1] + mon["top"] + top_left_offset[1],
        ),
        (
            p_bounds[1][0],
            p_bounds[1][1],
        ),
    )


def parse_number_cells(bounds, axis, cell_size, border_size, threshold=0.92):
    b_min, b_max = bounds

    mon = {
        "left": b_min[0],
        "top": b_min[1],
        "width": b_max[0] - b_min[0],
        "height": b_max[1] - b_min[1],
    }

    scr = None

    with mss.mss() as sct:
        scr = np.array(sct.grab(mon), dtype=np.uint8)[:, :, :3]

    rows, cols = grid_size(bounds, cell_size, border_size)
    numbers = (
        np.full((rows, cols), -1, dtype=np.int8)
        if axis == 0
        else np.full((cols, rows), -1, dtype=np.int8)
    )

    for i in range(60):
        if not os.path.exists(f".\\img\\{i}.jpg"):
            continue

        num_img = cv2.imread(f".\\img\\{i}.jpg", cv2.IMREAD_UNCHANGED)
        num_res = cv2.matchTemplate(scr, num_img, cv2.TM_CCOEFF_NORMED)

        for y, x in np.argwhere(num_res >= threshold):
            cell_pri = y // cell_size[1] if axis == 0 else x // cell_size[0]
            cell_sec = x // cell_size[0] if axis == 0 else y // cell_size[1]
            numbers[cell_pri, cell_sec] = i

    if not np.all(numbers >= 0):
        print(numbers)
        assert False

    return numbers.astype(np.uint8)


def image_to_cell_type(name):
    match name:
        case "empty":
            return Cell.EMPTY
        case "filled":
            return Cell.FILLED
        case "marked":
            return Cell.MARKED
        case _:
            assert False

    return None


def grid_size(bounds, cell_size, border_size):
    b_min, b_max = bounds

    rows = (b_max[1] - b_min[1]) // (5 * cell_size[1] + border_size) * 5 + (
        (b_max[1] - b_min[1]) % (5 * cell_size[1] + border_size)
    ) // cell_size[1]
    cols = (b_max[0] - b_min[0]) // (5 * cell_size[0] + border_size) * 5 + (
        (b_max[0] - b_min[0]) % (5 * cell_size[0] + border_size)
    ) // cell_size[0]

    return (rows, cols)


def parse_grid_cells(
    bounds, images, cell_size, border_size, threshold=0.95, debug=False
):
    b_min, b_max = bounds

    mon = {
        "left": b_min[0],
        "top": b_min[1],
        "width": b_max[0] - b_min[0],
        "height": b_max[1] - b_min[1],
    }

    scr = None

    with mss.mss() as sct:
        scr = np.array(sct.grab(mon), dtype=np.uint8)[:, :, :3]

    rows, cols = grid_size(bounds, cell_size, border_size)
    grid = np.zeros((rows, cols), dtype=np.uint8)

    for name in images:
        cell_img = cv2.imread(f".\\img\\{name}.jpg", cv2.IMREAD_UNCHANGED)
        cell_res = cv2.matchTemplate(scr, cell_img, cv2.TM_CCOEFF_NORMED)

        h, w = cell_img.shape[:2]
        max_val = 1

        while max_val > threshold:
            _, max_val, _, max_loc = cv2.minMaxLoc(cell_res)

            if max_val > threshold:
                min_x = np.clip(max_loc[0] - w // 2, 0, cell_res.shape[1] - 1)
                min_y = np.clip(max_loc[1] - h // 2, 0, cell_res.shape[0] - 1)
                max_x = np.clip(max_loc[0] + w // 2, 0, cell_res.shape[1] - 1)
                max_y = np.clip(max_loc[1] + h // 2, 0, cell_res.shape[0] - 1)

                cell_res[min_y : max_y + 1, min_x : max_x + 1] = 0

                cell_y = max_loc[1] // cell_size[1]
                cell_x = max_loc[0] // cell_size[0]

                if debug:
                    print(name, cell_x, cell_y)

                grid[cell_y, cell_x] = image_to_cell_type(name).value

    return grid


if __name__ == "__main__":
    bot = NonogramBot()
    p_bounds = find_puzzle_bounds((9, 10), (-5, -4))
    print("p_bounds", p_bounds)

    if p_bounds is None:
        print("Puzzle not found")
    else:
        state = bot.build_state(p_bounds)
        print(state)
        grid = bot.grid_from_state(state)
        print(grid.shape)

        for y in range(grid.shape[0]):
            bot.drag_cell(state, 0, y, grid.shape[1] - 1, y, "left")
