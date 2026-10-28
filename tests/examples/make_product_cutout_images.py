# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Render the images of ``tests/examples/llm_correctness_product_cutout.json``.

The dataset evaluates a product-listing assistant: given a photo of a few items
on a table, it cuts out the requested item onto a plain white background and
describes it in one sentence. Each record has three images, written to
``tests/examples/product_cutout_images/``:

- ``scene_NN.png``: the input photo, three items standing on a table
- ``output_NN.png``: the cutout the assistant returned
- ``reference_NN.png``: the correct cutout

Every item is drawn from flat shapes into a bounding box, so the same drawing
appears in the scene, in the reference cutout and, for a correct answer, in
the output cutout. The assistant's mistakes are deliberate: the wrong item
(record 3), half the item cut off (record 4) and a recolored item (record 7);
record 8 is correct but smaller and off-center. The drawing is deterministic;
running the script again rewrites identical files.

    pip install ".[media]"
    PYTHONPATH=. python3 tests/examples/make_product_cutout_images.py
"""

import math
from pathlib import Path

from PIL import Image, ImageDraw

OUTPUT_DIR = Path(__file__).resolve().parent / "product_cutout_images"

SCENE_SIZE = (384, 256)
CUTOUT_SIZE = (256, 256)
CUTOUT_BOX = (36, 36, 220, 220)
SLOT_CENTERS = {"left": 72, "middle": 192, "right": 312}
SLOT_SIZE = (104, 120)
TABLE_TOP = 176
STANDING_LINE = 214

WHITE = (255, 255, 255)
WALL = (236, 228, 214)
TABLE = (186, 140, 96)
TABLE_EDGE = (150, 108, 70)
SHADOW = (160, 118, 78)

RED = (200, 38, 38)
RED_DARK = (150, 25, 25)
RED_INSIDE = (90, 20, 20)
RED_HIGHLIGHT = (230, 95, 95)
TERRACOTTA = (196, 98, 55)
TERRACOTTA_DARK = (160, 75, 40)
LEAF = (52, 140, 70)
LEAF_DARK = (34, 105, 50)
LAMP_YELLOW = (238, 192, 36)
LAMP_METAL = (70, 70, 78)
BULB = (255, 240, 170)
LENS = (28, 28, 34)
LENS_SHINE = (90, 90, 102)
TEAL = (0, 140, 140)
TEAL_LIGHT = (120, 200, 200)
CAP = (245, 245, 245)
CAP_OUTLINE = (120, 120, 120)
STRAP = (122, 74, 40)
STRAP_HOLE = (80, 45, 20)
SILVER = (170, 170, 178)
DIAL = (250, 250, 250)
HANDS = (20, 20, 20)
HEADPHONE_BLACK = (32, 32, 38)
HEADPHONE_CUSHION = (75, 75, 82)
HEADPHONE_RED = (200, 40, 40)
HEADPHONE_RED_CUSHION = (140, 30, 30)
BOOK_BLUE = (38, 78, 170)
BOOK_SPINE = (25, 52, 120)
PAGES = (245, 240, 225)


class Box:
    """A bounding box with relative coordinates: ``x(0.5)`` is its center column."""

    def __init__(self, x0, y0, x1, y1):
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1
        self.w, self.h = x1 - x0, y1 - y0

    def x(self, u):
        return self.x0 + u * self.w

    def y(self, v):
        return self.y0 + v * self.h

    def rect(self, u0, v0, u1, v1):
        return (self.x(u0), self.y(v0), self.x(u1), self.y(v1))

    def line_width(self, fraction=0.06):
        return max(2, round(fraction * min(self.w, self.h)))


def draw_mug(draw, box, color=RED):
    width = box.line_width(0.09)
    draw.ellipse(box.rect(0.58, 0.3, 1.0, 0.78), outline=color, width=width)
    draw.rounded_rectangle(
        box.rect(0.0, 0.12, 0.74, 1.0), radius=0.08 * box.w, fill=color
    )
    draw.ellipse(box.rect(0.0, 0.04, 0.74, 0.2), fill=RED_DARK)
    draw.ellipse(box.rect(0.06, 0.07, 0.68, 0.17), fill=RED_INSIDE)
    draw.rectangle(box.rect(0.1, 0.3, 0.16, 0.85), fill=RED_HIGHLIGHT)


def _leaf(draw, base, angle_degrees, length, width, fill):
    angle = math.radians(angle_degrees)
    direction = (math.sin(angle), -math.cos(angle))
    normal = (math.cos(angle), math.sin(angle))
    center = (base[0] + direction[0] * length / 2, base[1] + direction[1] * length / 2)
    points = []
    for step in range(32):
        t = 2 * math.pi * step / 32
        along, across = length / 2 * math.cos(t), width / 2 * math.sin(t)
        points.append(
            (
                center[0] + direction[0] * along + normal[0] * across,
                center[1] + direction[1] * along + normal[1] * across,
            )
        )
    draw.polygon(points, fill=fill)


def draw_plant(draw, box):
    base = (box.x(0.5), box.y(0.57))
    leaves = [(-62, 0.42), (62, 0.42), (-32, 0.5), (32, 0.5), (0, 0.56)]
    for index, (angle, length) in enumerate(leaves):
        color = LEAF if index % 2 == 0 else LEAF_DARK
        _leaf(draw, base, angle, length * box.h, 0.2 * box.w, color)
    draw.polygon(
        [
            (box.x(0.18), box.y(0.6)),
            (box.x(0.82), box.y(0.6)),
            (box.x(0.72), box.y(1.0)),
            (box.x(0.28), box.y(1.0)),
        ],
        fill=TERRACOTTA,
    )
    draw.rectangle(box.rect(0.14, 0.55, 0.86, 0.65), fill=TERRACOTTA_DARK)


def draw_lamp(draw, box):
    width = box.line_width(0.07)
    joint = (box.x(0.24), box.y(0.5))
    draw.ellipse(box.rect(0.06, 0.88, 0.62, 1.0), fill=LAMP_METAL)
    draw.line([(box.x(0.34), box.y(0.92)), joint], fill=LAMP_METAL, width=width)
    draw.line([joint, (box.x(0.6), box.y(0.22))], fill=LAMP_METAL, width=width)
    draw.ellipse(
        (joint[0] - width, joint[1] - width, joint[0] + width, joint[1] + width),
        fill=LAMP_METAL,
    )
    draw.polygon(
        [
            (box.x(0.52), box.y(0.16)),
            (box.x(0.66), box.y(0.06)),
            (box.x(1.0), box.y(0.4)),
            (box.x(0.7), box.y(0.56)),
        ],
        fill=LAMP_YELLOW,
    )
    draw.ellipse(box.rect(0.74, 0.4, 0.88, 0.52), fill=BULB)


def draw_sunglasses(draw, box):
    width = box.line_width(0.12)
    draw.line(
        [(box.x(0.4), box.y(0.4)), (box.x(0.6), box.y(0.4))], fill=LENS, width=width
    )
    draw.ellipse(box.rect(0.02, 0.16, 0.44, 0.98), fill=LENS)
    draw.ellipse(box.rect(0.56, 0.16, 0.98, 0.98), fill=LENS)
    draw.ellipse(box.rect(0.1, 0.3, 0.17, 0.52), fill=LENS_SHINE)
    draw.ellipse(box.rect(0.64, 0.3, 0.71, 0.52), fill=LENS_SHINE)


def draw_bottle(draw, box):
    draw.rounded_rectangle(
        box.rect(0.06, 0.3, 0.94, 1.0), radius=0.25 * box.w, fill=TEAL
    )
    draw.polygon(
        [
            (box.x(0.1), box.y(0.38)),
            (box.x(0.3), box.y(0.2)),
            (box.x(0.7), box.y(0.2)),
            (box.x(0.9), box.y(0.38)),
        ],
        fill=TEAL,
    )
    draw.rectangle(box.rect(0.3, 0.11, 0.7, 0.22), fill=TEAL)
    draw.rectangle(box.rect(0.06, 0.56, 0.94, 0.72), fill=TEAL_LIGHT)
    draw.rounded_rectangle(
        box.rect(0.24, 0.0, 0.76, 0.13),
        radius=0.04 * box.w,
        fill=CAP,
        outline=CAP_OUTLINE,
        width=2,
    )


def draw_watch(draw, box):
    draw.rounded_rectangle(
        box.rect(0.0, 0.3, 1.0, 0.7), radius=0.08 * box.h, fill=STRAP
    )
    for u in (0.8, 0.87, 0.94):
        r = 0.03 * box.h
        draw.ellipse(
            (box.x(u) - r, box.y(0.5) - r, box.x(u) + r, box.y(0.5) + r),
            fill=STRAP_HOLE,
        )
    draw.rectangle(
        box.rect(0.03, 0.33, 0.11, 0.67), outline=SILVER, width=box.line_width(0.04)
    )
    cx, cy, r = box.x(0.5), box.y(0.5), 0.5 * box.h
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=SILVER)
    inner = 0.82 * r
    draw.ellipse((cx - inner, cy - inner, cx + inner, cy + inner), fill=DIAL)
    tick = box.line_width(0.04)
    for angle in (0, 90, 180, 270):
        a = math.radians(angle)
        outer_point = (cx + math.sin(a) * inner * 0.92, cy - math.cos(a) * inner * 0.92)
        inner_point = (cx + math.sin(a) * inner * 0.72, cy - math.cos(a) * inner * 0.72)
        draw.line([inner_point, outer_point], fill=HANDS, width=tick)
    for angle, length in ((300, 0.45), (0, 0.7)):
        a = math.radians(angle)
        tip = (cx + math.sin(a) * inner * length, cy - math.cos(a) * inner * length)
        draw.line([(cx, cy), tip], fill=HANDS, width=tick + 1)


def draw_headphones(draw, box, color=HEADPHONE_BLACK, cushion=HEADPHONE_CUSHION):
    draw.arc(
        box.rect(0.12, 0.04, 0.88, 0.96),
        start=180,
        end=360,
        fill=color,
        width=box.line_width(0.1),
    )
    radius = 0.08 * box.w
    draw.rounded_rectangle(box.rect(0.02, 0.46, 0.3, 0.98), radius=radius, fill=color)
    draw.rounded_rectangle(box.rect(0.7, 0.46, 0.98, 0.98), radius=radius, fill=color)
    draw.rounded_rectangle(
        box.rect(0.2, 0.54, 0.3, 0.9), radius=radius / 2, fill=cushion
    )
    draw.rounded_rectangle(
        box.rect(0.7, 0.54, 0.8, 0.9), radius=radius / 2, fill=cushion
    )


def draw_book(draw, box):
    draw.rectangle(box.rect(0.08, 0.04, 1.0, 0.98), fill=PAGES)
    draw.rectangle(box.rect(0.0, 0.0, 0.93, 0.95), fill=BOOK_BLUE)
    draw.rectangle(box.rect(0.0, 0.0, 0.12, 0.95), fill=BOOK_SPINE)


# Each item: its drawing and its width-to-height ratio.
ITEMS = {
    "mug": (draw_mug, 1.15),
    "plant": (draw_plant, 0.8),
    "lamp": (draw_lamp, 0.75),
    "sunglasses": (draw_sunglasses, 2.2),
    "bottle": (draw_bottle, 0.42),
    "watch": (draw_watch, 2.0),
    "headphones": (draw_headphones, 1.0),
    "book": (draw_book, 0.72),
}


def _fit(aspect, max_w, max_h):
    """The largest width and height with ``aspect`` inside ``max_w`` x ``max_h``."""
    if max_w / max_h > aspect:
        return max_h * aspect, max_h
    return max_w, max_w / aspect


def render_scene(left, middle, right):
    """The input photo: three items standing on a table in front of a wall."""
    image = Image.new("RGB", SCENE_SIZE, WALL)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, TABLE_TOP, SCENE_SIZE[0], SCENE_SIZE[1]), fill=TABLE)
    draw.rectangle((0, TABLE_TOP, SCENE_SIZE[0], TABLE_TOP + 4), fill=TABLE_EDGE)
    for position, item in (("left", left), ("middle", middle), ("right", right)):
        drawer, aspect = ITEMS[item]
        w, h = _fit(aspect, *SLOT_SIZE)
        cx = SLOT_CENTERS[position]
        box = Box(cx - w / 2, STANDING_LINE - h, cx + w / 2, STANDING_LINE)
        draw.ellipse(
            (box.x0 - 4, STANDING_LINE - 6, box.x1 + 4, STANDING_LINE + 6), fill=SHADOW
        )
        drawer(draw, box)
    return image


def render_cutout(item, region=CUTOUT_BOX, **style):
    """One item centered in ``region`` on a plain white background."""
    image = Image.new("RGB", CUTOUT_SIZE, WHITE)
    drawer, aspect = ITEMS[item]
    x0, y0, x1, y1 = region
    w, h = _fit(aspect, x1 - x0, y1 - y0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    drawer(
        ImageDraw.Draw(image),
        Box(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2),
        **style,
    )
    return image


def render_half_cut_sunglasses():
    """The sunglasses cutout with everything right of the bridge's middle missing."""
    image = render_cutout("sunglasses")
    ImageDraw.Draw(image).rectangle(
        (132, 0, CUTOUT_SIZE[0], CUTOUT_SIZE[1]), fill=WHITE
    )
    return image


# One entry per record: the scene's items left to right, the item the request
# names, and the assistant's output cutout.
RECORDS = [
    (("book", "plant", "mug"), "mug", lambda: render_cutout("mug")),
    (("mug", "plant", "bottle"), "plant", lambda: render_cutout("plant")),
    (("lamp", "book", "sunglasses"), "lamp", lambda: render_cutout("book")),
    (("watch", "sunglasses", "mug"), "sunglasses", render_half_cut_sunglasses),
    (("plant", "headphones", "bottle"), "bottle", lambda: render_cutout("bottle")),
    (("sunglasses", "watch", "book"), "watch", lambda: render_cutout("watch")),
    (
        ("mug", "lamp", "headphones"),
        "headphones",
        lambda: render_cutout(
            "headphones", color=HEADPHONE_RED, cushion=HEADPHONE_RED_CUSHION
        ),
    ),
    (
        ("book", "plant", "watch"),
        "book",
        lambda: render_cutout("book", region=(132, 24, 232, 124)),
    ),
    (("lamp", "bottle", "headphones"), "lamp", lambda: render_cutout("lamp")),
    (("sunglasses", "plant", "headphones"), "plant", lambda: render_cutout("plant")),
]


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    for number, (scene_items, target, output) in enumerate(RECORDS, start=1):
        images = {
            "scene": render_scene(*scene_items),
            "output": output(),
            "reference": render_cutout(target),
        }
        for kind, image in images.items():
            image.save(
                OUTPUT_DIR / f"{kind}_{number:02d}.png", format="PNG", optimize=True
            )
    print(f"wrote {3 * len(RECORDS)} images to {OUTPUT_DIR.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
