# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Render the ten synthetic scenes of ``tests/examples/llm_correctness_vqa.json``.

Each scene is a 256 x 192 RGB image of a few flat shapes, so a question about
it has one unambiguous answer: the color of a shape, how many there are, where
something sits, whether a line rises. The drawing is deterministic; running the
script again rewrites identical files into ``tests/examples/vqa_scene_images/``.

    pip install ".[media]"
    PYTHONPATH=. python3 tests/examples/make_vqa_scene_images.py
"""

from pathlib import Path

from PIL import Image, ImageDraw

OUTPUT_DIR = Path(__file__).resolve().parent / "vqa_scene_images"
WIDTH, HEIGHT = 256, 192

WHITE = (255, 255, 255)
RED = (220, 30, 30)
BLUE = (40, 90, 220)
GREEN = (40, 160, 70)
YELLOW = (240, 200, 30)
ORANGE = (245, 140, 30)
PURPLE = (140, 60, 180)
CYAN = (40, 200, 220)
BLACK = (0, 0, 0)
SKY = (135, 206, 235)
NIGHT = (10, 15, 50)
SLATE = (30, 30, 40)


def _canvas(color):
    image = Image.new("RGB", (WIDTH, HEIGHT), color)
    return image, ImageDraw.Draw(image)


def scene_01_red_circle():
    image, draw = _canvas(WHITE)
    draw.ellipse((78, 46, 178, 146), fill=RED)
    return image


def scene_02_three_blue_squares():
    image, draw = _canvas(WHITE)
    for column in range(3):
        left = 28 + column * 76
        draw.rectangle((left, 66, left + 56, 122), fill=BLUE)
    return image


def scene_03_triangle_above_rectangle():
    image, draw = _canvas(WHITE)
    draw.polygon([(128, 20), (78, 90), (178, 90)], fill=YELLOW)
    draw.rectangle((68, 110, 188, 170), fill=GREEN)
    return image


def scene_04_sun_top_right():
    image, draw = _canvas(SKY)
    draw.rectangle((0, 140, WIDTH, HEIGHT), fill=GREEN)
    draw.ellipse((196, 18, 240, 62), fill=YELLOW)
    return image


def scene_05_black_cross_on_orange():
    image, draw = _canvas(ORANGE)
    draw.rectangle((116, 36, 140, 156), fill=BLACK)
    draw.rectangle((68, 84, 188, 108), fill=BLACK)
    return image


def scene_06_large_left_small_right_circles():
    image, draw = _canvas(WHITE)
    draw.ellipse((20, 36, 140, 156), fill=PURPLE)
    draw.ellipse((176, 76, 216, 116), fill=CYAN)
    return image


def scene_07_three_red_one_green_square():
    image, draw = _canvas(WHITE)
    corners = [(56, 36), (144, 36), (56, 108), (144, 108)]
    for index, (left, top) in enumerate(corners):
        color = GREEN if index == 3 else RED
        draw.rectangle((left, top, left + 56, top + 48), fill=color)
    return image


def scene_08_rising_line():
    image, draw = _canvas(SLATE)
    draw.line((20, 170, 236, 22), fill=WHITE, width=6)
    return image


def scene_09_night_sky():
    image, draw = _canvas(NIGHT)
    stars = [
        (30, 30),
        (70, 60),
        (120, 25),
        (160, 70),
        (90, 120),
        (230, 110),
        (40, 150),
        (180, 150),
    ]
    for x, y in stars:
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=WHITE)
    draw.ellipse((180, 20, 240, 80), fill=(250, 245, 200))
    draw.ellipse((196, 12, 254, 70), fill=NIGHT)  # bite out of the disc: a crescent
    return image


def scene_10_square_outline():
    image, draw = _canvas(WHITE)
    draw.rectangle((68, 36, 188, 156), outline=BLUE, width=8)
    return image


SCENES = [
    scene_01_red_circle,
    scene_02_three_blue_squares,
    scene_03_triangle_above_rectangle,
    scene_04_sun_top_right,
    scene_05_black_cross_on_orange,
    scene_06_large_left_small_right_circles,
    scene_07_three_red_one_green_square,
    scene_08_rising_line,
    scene_09_night_sky,
    scene_10_square_outline,
]


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    for number, scene in enumerate(SCENES, start=1):
        path = OUTPUT_DIR / f"scene_{number:02d}.png"
        scene().save(path, format="PNG", optimize=True)
        print(f"wrote {path.relative_to(OUTPUT_DIR.parent.parent)}")


if __name__ == "__main__":
    main()
