# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Factory for creating DataType instances by modality.

Data type modules register themselves by calling `register()` at import time.
The package imports only `text_data` eagerly. When `create()` finds no entry
for a modality, it imports that modality's module (`_MODULES`) on first use
and retries the lookup, so heavy optional dependencies (opencv, pillow,
soundfile) load only when a path of that modality is actually created.
"""

import importlib
from typing import Dict, Type
from .modality import Modality
from .data_type import DataType

_registry: Dict[Modality, Type[DataType]] = {}

# Module (inside this package) that registers the DataType for each modality.
_MODULES: Dict[Modality, str] = {
    Modality.TEXT: "text_data",
    Modality.IMAGE: "image_data",
    Modality.AUDIO: "audio_data",
    Modality.VIDEO: "video_data",
}


def register(modality: Modality, cls: Type[DataType]) -> None:
    """Register a DataType class for a given modality."""
    _registry[modality] = cls


def create(modality: Modality, path: str) -> DataType:
    """
    Create a DataType instance for the given modality.

    On the first miss for a modality, imports its data module so the module
    registers its class, then retries the lookup.

    Raises:
        ValueError: If the modality's module cannot be imported (missing
            optional dependency) or no data type is registered for the modality.
    """
    cls = _registry.get(modality)
    if cls is None:
        module_name = _MODULES.get(modality)
        if module_name is not None:
            try:
                importlib.import_module(f"{__package__}.{module_name}")
            except ImportError as error:
                raise ValueError(
                    f"No data type available for {modality}: importing "
                    f"{module_name} failed ({error}). Install the media extra "
                    f'from the metric-library source folder: pip install ".[media]".'
                ) from error
            cls = _registry.get(modality)
    if cls is None:
        raise ValueError(
            f"No data type registered for {modality}. "
            f"Install the required package and import the corresponding "
            f"data module (e.g., from metric_library.core.data.image_data import ImageData)."
        )
    return cls(path)
