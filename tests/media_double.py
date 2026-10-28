# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""In-memory image data for the tests of the error metrics.

The error metrics accept audio, image and video data, and the engine now
rejects any other modality before scoring. ``ImageData`` needs Pillow and a
file on disk; this double carries a NumPy array under the image modality so
the tests stay in memory and free of the ``media`` extra.
"""

import numpy as np

from metric_library.core.data import DataType, Modality


class ArrayImageData(DataType):
    """An image-modality value backed by an in-memory array."""

    def preprocess(self, path) -> None:
        self.raw_data = path
        self.data = np.asarray(path, dtype=np.float64)

    @property
    def modality(self) -> Modality:
        return Modality.IMAGE


class PngImageData(DataType):
    """An image-modality value backed by PNG bytes, for the judge message path."""

    def preprocess(self, path) -> None:
        self.raw_data = path
        self.format = "png"

    @property
    def modality(self) -> Modality:
        return Modality.IMAGE


def images(values):
    return [ArrayImageData(value) for value in values]
