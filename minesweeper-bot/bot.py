import cv2
import numpy as np
import math
import mss
import pyautogui

from enum import Enum


class Cell(Enum):
    MINE = -3
    FLAG = -2
    HIDDEN = -1
    EMPTY = 0
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8


class MinesweeperBot:
    def __init__(self):
        self.tracked_locations = {
            "restart": [
                ("face_happy", 0.95),
                ("face_anticipation", 0.95),
                ("face_dead", 0.95),
            ],
            "hidden": [
                ("hidden_cell", 0.95),
            ],
            "flag": [
                ("flag", 0.95),
            ],
            "empty": [
                ("empty_cell_center", 0.8),
                ("empty_cell_n", 0.8),
                ("empty_cell_s", 0.8),
                ("empty_cell_e", 0.8),
                ("empty_cell_w", 0.8),
                ("empty_cell_ne", 0.8),
                ("empty_cell_nw", 0.8),
                ("empty_cell_se", 0.8),
                ("empty_cell_sw", 0.8),
            ],
            "mine": [
                ("mine_hit", 0.9),
                ("mine", 0.95),
            ],
            "1": [
                ("1_cell", 0.9),
            ],
            "2": [
                ("2_cell", 0.9),
            ],
            "3": [
                ("3_cell", 0.9),
            ],
            "4": [
                ("4_cell", 0.9),
            ],
            "5": [
                ("5_cell", 0.9),
            ],
            "6": [
                ("6_cell", 0.9),
            ],
            "7": [
                ("7_cell", 0.9),
            ],
            "8": [
                ("8_cell", 0.9),
            ],
        }

        self.cell_size = 16
        self.cell_locations = [
            "hidden",
            "flag",
            "empty",
            "mine",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
        ]

    def update_locations(self, w_bounds):
        (w_min_x, w_min_y), (w_max_x, w_max_y) = w_bounds

        dimensions = {
            "left": w_min_x,
            "top": w_min_y,
            "width": w_max_x - w_min_x,
            "height": w_max_y - w_min_y,
        }

        screen = None
        with mss.mss() as sct:
            screen = np.array(sct.grab(dimensions))[:, :, :3]

        known_locations = {}

        for name, images in self.tracked_locations.items():
            for image, threshold in images:
                locations = find_locations(screen, image, dimensions, threshold)
                if name in known_locations:
                    known_locations[name].extend(locations)
                else:
                    known_locations[name] = locations

        return known_locations

    def build_state(self, w_bounds=None):
        if w_bounds is None:
            w_bounds = find_window_bounds()

        if w_bounds is None:
            return None

        locations = self.update_locations(w_bounds)

        (g_min_x, g_min_y), (g_max_x, g_max_y) = grid_location_bounds(w_bounds)

        rows = math.ceil((g_max_y - g_min_y) / self.cell_size)
        cols = math.ceil((g_max_x - g_min_x) / self.cell_size)

        grid = np.full((rows, cols), Cell.HIDDEN.value, dtype=np.int8)

        for name in self.cell_locations:
            for (x, y), (x_size, y_size) in locations[name]:
                cell_x = ((x + x_size // 2) - g_min_x) // self.cell_size
                cell_y = ((y + y_size // 2) - g_min_y) // self.cell_size
                grid[cell_y, cell_x] = location_to_cell(name).value

        return (grid, w_bounds)

    def find_cell_type(self, screen, dimensions):
        for name in self.cell_locations:
            for image, threshold in self.tracked_locations[name]:
                locations = find_locations(screen, image, dimensions, threshold)
                if len(locations) > 0:
                    return (name, locations[0])

        return None

    def click_cell(self, state, cell_x, cell_y, button="left"):
        grid, w_bounds = state

        if grid[cell_y, cell_x] != Cell.HIDDEN.value:
            return (grid, w_bounds)

        (g_min_x, g_min_y), _ = grid_location_bounds(w_bounds)

        x = g_min_x + cell_x * self.cell_size + self.cell_size // 2
        y = g_min_y + cell_y * self.cell_size + self.cell_size // 2
        pyautogui.click(x, y, button=button)

        dimensions = {
            "left": x - self.cell_size,
            "top": y - self.cell_size,
            "width": 2 * self.cell_size,
            "height": 2 * self.cell_size,
        }

        screen = None
        with mss.mss() as sct:
            screen = np.array(sct.grab(dimensions))[:, :, :3]

        match self.find_cell_type(screen, dimensions):
            case (name, _):
                cell = location_to_cell(name).value

                if cell == Cell.EMPTY.value:
                    grid, w_bounds = self.build_state(w_bounds)
                else:
                    grid[cell_y, cell_x] = cell
            case _:
                assert False

        return (grid, w_bounds)

    def focus_window(self, state):
        _, w_bounds = state
        w_min, _ = w_bounds
        pyautogui.click(w_min)

    def click(self, name, button="left"):
        if name in self.tracked_locations:
            (w_min_x, w_min_y), (w_max_x, w_max_y) = find_window_bounds()

            dimensions = {
                "left": w_min_x,
                "top": w_min_y,
                "width": w_max_x - w_min_x,
                "height": w_max_y - w_min_y,
            }

            screen = None
            with mss.mss() as sct:
                screen = np.array(sct.grab(dimensions))[:, :, :3]

            for image, threshold in self.tracked_locations[name]:
                locations = find_locations(screen, image, dimensions, threshold)
                for (x, y), (x_size, y_size) in locations:
                    pyautogui.click(x + x_size // 2, y + y_size // 2, button=button)

    def grid_from_state(self, state):
        return state[0]


def grid_location_bounds(w_bounds):
    (w_min_x, w_min_y), (w_max_x, w_max_y) = w_bounds

    g_min_x = w_min_x + 12
    g_min_y = w_min_y + 100
    g_max_x = w_max_x - 13
    g_max_y = w_max_y - 13

    return ((g_min_x, g_min_y), (g_max_x, g_max_y))


def find_window_bounds(threshold=0.95):
    dimensions = {
        "left": 0,
        "top": 0,
        "width": 2560,
        "height": 1440,
    }

    scr = None

    with mss.mss() as sct:
        scr = np.array(sct.grab(dimensions))[:, :, :3]

    top_left_img = cv2.imread(".\\img\\top_left.jpg", cv2.IMREAD_UNCHANGED)
    top_left_res = cv2.matchTemplate(scr, top_left_img, cv2.TM_CCOEFF_NORMED)
    _, top_left_val, _, top_left_loc = cv2.minMaxLoc(top_left_res)

    if top_left_val < threshold:
        return None

    bottom_right_img = None
    bottom_right_val = 0

    for i in range(4):
        bottom_right_img = cv2.imread(
            f".\\img\\bottom_right_{i}.jpg", cv2.IMREAD_UNCHANGED
        )
        bottom_right_res = cv2.matchTemplate(
            scr, bottom_right_img, cv2.TM_CCOEFF_NORMED
        )
        _, bottom_right_val, _, bottom_right_loc = cv2.minMaxLoc(bottom_right_res)

        if bottom_right_val >= threshold:
            break

    if bottom_right_val < threshold:
        return None

    bottom_right_img_h, bottom_right_img_w = bottom_right_img.shape[:2]

    return (
        (top_left_loc[0], top_left_loc[1]),
        (
            bottom_right_loc[0] + bottom_right_img_w,
            bottom_right_loc[1] + bottom_right_img_h,
        ),
    )


def find_locations(screen, needle, dimensions, threshold=0.95, debug=False):
    locations = []

    needle_img = cv2.imread(f".\img\{needle}.jpg", cv2.IMREAD_UNCHANGED)
    result = cv2.matchTemplate(screen, needle_img, cv2.TM_CCOEFF_NORMED)

    h, w = needle_img.shape[:2]
    max_val = 1

    while max_val > threshold:
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if debug:
            print(needle, max_val, max_loc)

        if max_val > threshold:
            min_x = np.clip(max_loc[0] - w // 2, 0, result.shape[1] - 1)
            min_y = np.clip(max_loc[1] - h // 2, 0, result.shape[0] - 1)
            max_x = np.clip(max_loc[0] + w // 2, 0, result.shape[1] - 1)
            max_y = np.clip(max_loc[1] + h // 2, 0, result.shape[0] - 1)
            result[min_y : max_y + 1, min_x : max_x + 1] = -1
            loc = (max_loc[0] + dimensions["left"], max_loc[1] + dimensions["top"])
            locations.append((loc, (w, h)))

    return locations


def location_to_cell(name):
    match name:
        case "hidden":
            return Cell.HIDDEN
        case "empty":
            return Cell.EMPTY
        case "flag":
            return Cell.FLAG
        case "mine":
            return Cell.MINE
        case "1":
            return Cell.ONE
        case "2":
            return Cell.TWO
        case "3":
            return Cell.THREE
        case "4":
            return Cell.FOUR
        case "5":
            return Cell.FIVE
        case "6":
            return Cell.SIX
        case "7":
            return Cell.SEVEN
        case "8":
            return Cell.EIGHT
        case _:
            assert False

    return None


if __name__ == "__main__":
    bot = MinesweeperBot()
    w_bounds = find_window_bounds()
    if w_bounds is None:
        print("Window not found")
    else:
        locations = bot.update_locations(w_bounds)
        for name in bot.tracked_locations:
            if name in locations:
                print(name, len(locations[name]))
            else:
                print(name, None)
    # state = bot.build_state()
    # print(state[0])
    # state = bot.click_cell(state, 0, 0)
    # print(state[0])
    # bot.click("restart")
