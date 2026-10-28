# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Render the scanned knowledge-base pages of ``tests/examples/rag_support_kb.json``.

Records 1, 5 and 7 retrieve a page instead of a text passage. Each page is a
640 x 360 image of a few lines of policy text, drawn with Pillow's built-in
font so that no font file is needed. The rendering is deterministic; running
the script again rewrites identical files into ``tests/examples/rag_page_images/``.
Two pages are PNG files and one is a TIFF, the format document scanners
produce, which the judge receives as PNG.

    pip install ".[media]"
    PYTHONPATH=. python3 tests/examples/make_rag_page_images.py
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUTPUT_DIR = Path(__file__).resolve().parent / "rag_page_images"
WIDTH, HEIGHT = 640, 360
PAPER = (250, 248, 240)
INK = (25, 25, 30)
RULE = (120, 120, 130)

PAGES = {
    "returns_policy.png": (
        "RETURNS POLICY",
        [
            "Items can be returned within 30 days of delivery",
            "for a full refund.",
            "Items must be unused and in their original packaging.",
            "Refunds are issued to the original payment method",
            "within 5 business days of receiving the item.",
        ],
    ),
    "warranty_terms.tif": (
        "WARRANTY TERMS",
        [
            "Kitchen appliances: 2 years against manufacturing",
            "defects.",
            "Headphones and earbuds: 1 year limited warranty.",
            "Claims are handled through the support portal.",
        ],
    ),
    "shipping_destinations.png": (
        "SHIPPING DESTINATIONS",
        [
            "EU countries: 2 to 4 business days.",
            "Norway and Switzerland: 4 to 6 business days.",
            "United Kingdom: 3 to 5 business days.",
            "Orders over 60 euro ship free within the EU.",
        ],
    ),
}


def render(title, lines):
    page = Image.new("RGB", (WIDTH, HEIGHT), PAPER)
    draw = ImageDraw.Draw(page)
    heading = ImageFont.load_default(size=30)
    body = ImageFont.load_default(size=24)
    draw.text((40, 32), title, fill=INK, font=heading)
    draw.line((40, 76, WIDTH - 40, 76), fill=RULE, width=2)
    for index, line in enumerate(lines):
        draw.text((40, 100 + index * 40), line, fill=INK, font=body)
    return page


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    for name, (title, lines) in PAGES.items():
        path = OUTPUT_DIR / name
        page = render(title, lines)
        if path.suffix == ".tif":
            # Scanners save single-page bilevel TIFFs; Group 4 is their compression.
            page.convert("1").save(path, format="TIFF", compression="group4")
        else:
            page.save(path, format="PNG", optimize=True)
        print(f"wrote {path.relative_to(OUTPUT_DIR.parent.parent)}")


if __name__ == "__main__":
    main()
