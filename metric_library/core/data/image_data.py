# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import functools
import io

from .modality import Modality
from .data_type import DataType
from . import factory
import numpy as np
from PIL import Image, UnidentifiedImageError

# The largest image block the Bedrock Converse API accepts (API reference,
# Message): at most 8000 pixels per side and 3.75 MB, that is 3,932,160 bytes,
# whose base64 form fills the 5,242,880 bytes Bedrock checks.
MAX_JUDGE_IMAGE_SIDE = 8000
MAX_JUDGE_IMAGE_BYTES = 3_932_160

# The most pixels an image may have, those of the largest image the judge
# takes. The error metrics decode every pixel to float64, 8 bytes a channel.
MAX_IMAGE_PIXEL_COUNT = MAX_JUDGE_IMAGE_SIDE * MAX_JUDGE_IMAGE_SIDE

# The Pillow formats of the extensions DataHandler loads as images (.jpg .jpeg
# .png .bmp .tif .tiff .webp .gif). Image.open tries only these, so a file in
# another format, such as PSD, EPS or JPEG 2000, never reaches its decoder.
IMAGE_FORMATS = ("JPEG", "PNG", "BMP", "TIFF", "WEBP", "GIF")


class ImageData(DataType):
    def __init__(self, path: str):
        """Initialize with path to image"""
        super().__init__(path)

    def preprocess(self, path: str) -> None:
        """Read the file and check its header; the pixels decode on first use.

        The bytes must open as one of ``IMAGE_FORMATS`` (else
        ``UnidentifiedImageError`` names the file) and the image may have at
        most ``MAX_IMAGE_PIXEL_COUNT`` pixels (else ``ValueError``). Opening
        reads only the header.
        """
        with open(path, "rb") as f:
            self.raw_data = f.read()
        try:
            image = Image.open(io.BytesIO(self.raw_data), formats=IMAGE_FORMATS)
        except UnidentifiedImageError:
            raise UnidentifiedImageError(
                f"cannot identify image file {path!r}"
            ) from None
        width, height = image.size
        if width * height > MAX_IMAGE_PIXEL_COUNT:
            raise ValueError(
                f"cannot load the {width}x{height} image {path!r}; an image has "
                f"at most {MAX_IMAGE_PIXEL_COUNT:,} pixels "
                f"({MAX_JUDGE_IMAGE_SIDE}x{MAX_JUDGE_IMAGE_SIDE})"
            )

    @functools.cached_property
    def data(self) -> np.ndarray:
        """The float64 pixels, decoded from ``raw_data`` when first read.

        Only the error metrics read them, through ``get_data``; the judge
        reads ``raw_data``, or a PNG copy of a BMP or TIFF image
        (``_judge_image``), so a run that only judges never builds them.
        """
        image = Image.open(io.BytesIO(self.raw_data), formats=IMAGE_FORMATS)
        return np.array(image).astype(np.float64)

    @property
    def modality(self) -> Modality:
        return Modality.IMAGE

    def get_data_for_LLM_message(self) -> dict:
        """The Converse image block, re-encoded as PNG when Converse cannot read it.

        A PNG, JPEG, GIF or WebP file is sent as it is. Any other image, such as
        BMP or TIFF, is sent as PNG, converted to RGB when PNG cannot store its
        mode (CMYK, 16-bit); ``get_data`` still returns the file's own pixels.
        An image the judge cannot take fails with ``ValueError``: more than
        ``MAX_JUDGE_IMAGE_SIDE`` pixels on a side, a TIFF with several pages
        (one image block holds one page), or a block of more than
        ``MAX_JUDGE_IMAGE_BYTES`` bytes, counted after any conversion to PNG.
        The image is checked and converted once: later calls, such as the
        requests after input validation, reuse the result.
        """
        image_format, data = self._judge_image
        return {"image": {"format": image_format, "source": {"bytes": data}}}

    @functools.cached_property
    def _judge_image(self) -> tuple[str, bytes]:
        """The format and bytes the judge receives, checked against the limits."""
        image = Image.open(io.BytesIO(self.raw_data), formats=IMAGE_FORMATS)
        width, height = image.size
        if max(width, height) > MAX_JUDGE_IMAGE_SIDE:
            raise ValueError(
                f"cannot send a {width}x{height} {self.format.upper()} image to the "
                f"LLM judge; it accepts at most {MAX_JUDGE_IMAGE_SIDE} pixels per side"
            )
        if self.format in ("png", "jpeg", "gif", "webp"):
            image_format, data, sent_as = self.format, self.raw_data, ""
        else:
            pages = getattr(image, "n_frames", 1)
            if pages > 1:
                raise ValueError(
                    f"cannot send a {pages}-page {self.format.upper()} image to the "
                    "LLM judge; split it into one image per page"
                )
            if image.mode not in ("1", "L", "LA", "P", "RGB", "RGBA"):
                image = image.convert("RGB")
            png = io.BytesIO()
            image.save(png, format="PNG")
            image_format, data, sent_as = "png", png.getvalue(), " as PNG"
        if len(data) > MAX_JUDGE_IMAGE_BYTES:
            raise ValueError(
                f"cannot send a {self.format.upper()} image of {len(data):,} bytes"
                f"{sent_as} to the LLM judge; it accepts at most "
                f"{MAX_JUDGE_IMAGE_BYTES:,} bytes (3.75 MB) per image"
            )
        return image_format, data


factory.register(Modality.IMAGE, ImageData)
