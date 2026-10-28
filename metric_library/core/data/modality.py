# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

from enum import Enum, auto


class Modality(Enum):
    """Smart enum for supported data modalities."""

    TEXT = auto()
    IMAGE = auto()
    AUDIO = auto()
    VIDEO = auto()

    def __str__(self):
        return self.name.lower()
