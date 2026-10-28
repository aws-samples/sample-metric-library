# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

from .modality import Modality
from .data_type import DataType
from . import factory
import numpy as np

import soundfile as sf


class AudioData(DataType):
    def __init__(self, path: str):
        super().__init__(path)

    def preprocess(self, path: str) -> None:
        """
        Apply audio preprocessing from path
        - Converting to mono if stereo
        """

        # Load audio from path
        audio, sample_rate = sf.read(path)

        # Store initial Audio Data
        self.raw_data = audio

        # Convert to mono if stereo and convert to float for processing and store
        if len(audio.shape) > 1 and audio.shape[1] > 1:
            audio = np.mean(audio, axis=1)

        # Convert to float for processing and store
        self.data = audio.astype(np.float32)

    @property
    def modality(self) -> Modality:
        return Modality.AUDIO

    def get_data_for_LLM_message(self) -> None:
        """Overridden DataType function because audio input is not compatible with LLM metrics"""
        raise ValueError("Audio inputs are not currently compatible with LLM metrics")


factory.register(Modality.AUDIO, AudioData)
