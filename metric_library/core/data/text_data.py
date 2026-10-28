# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

from .modality import Modality
from .data_type import DataType
from . import factory
import numpy as np
from typing import Dict


class TextData(DataType):
    def __init__(self, path: str):
        """
        Wrap a value as text.

        Args:
            path: The text value. A string that names a file stays that
                string; read a text file with ``TextData.from_file``.
        """
        super().__init__(path)

    def preprocess(self, path: str) -> None:
        """Keep the value as the raw data and the data. Read no file."""
        self.raw_data = path
        self.data = path

    @classmethod
    def from_file(cls, path: str) -> "TextData":
        """Read a UTF-8 text file: raw_data is its content and data its word array."""
        with open(path, "r", encoding="utf-8") as text_file:
            content = text_file.read()
        text = cls(content)
        text.data = np.asarray(content.split())
        return text

    @property
    def modality(self) -> Modality:
        return Modality.TEXT

    def get_data_for_LLM_message(self) -> Dict[str, str]:
        """Overridden DataType function: the Converse text block is ``{"text": ...}`` with no format or source field"""

        return {"text": str(self.raw_data)}


factory.register(Modality.TEXT, TextData)
