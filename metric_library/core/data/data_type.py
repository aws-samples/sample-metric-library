# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

from abc import ABC, abstractmethod
from typing import Union
from .modality import Modality
import numpy as np
import os


class DataType(ABC):
    """Base class for all data types supported by the metrics library."""

    def __init__(self, path: str):
        # TODO see if there is a better way to update the format or make more support for when data isn't passed in as a file path
        # Only a media path is checked on disk. A text value is never looked
        # up: on Windows, checking a value such as \\host\share\a would make
        # the host open a connection to that server.
        if (
            self.modality is not Modality.TEXT
            and isinstance(path, str)
            and os.path.isfile(path)
        ):
            # Bedrock content blocks expect a lowercase format name and call the
            # JPEG format "jpeg", while image files are commonly named ".jpg".
            extension = path.split(".")[-1].lower()
            self.format = "jpeg" if extension == "jpg" else extension
        else:
            self.format = "text"
        self.raw_data: Union[bytes | str]
        self.data: np.ndarray
        self.preprocess(path)

    # TODO Preprocessing design needs revisiting, currently just extending init
    @abstractmethod
    def preprocess(self, path: str) -> None:
        """Apply modality-specific preprocessing and return processed DataType."""
        pass

    def get_data(self) -> np.ndarray:
        """Get the processed data in a format suitable for metric computation."""
        return self.data

    def get_raw_data(self) -> Union[str | bytes]:
        """Get the original unprocessed data. Useful for structured data"""
        return self.raw_data

    @property
    @abstractmethod
    def modality(self) -> Modality:
        """Return the modality of this data type."""
        pass

    def get_data_for_LLM_message(self) -> dict:
        """The Converse block ``{<modality>: {"format": ..., "source": {...}}}``."""
        return {
            f"{self.modality}": {
                "format": self.format,
                "source": {"bytes": self.raw_data},
            }
        }
