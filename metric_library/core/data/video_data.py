# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import numpy as np
import cv2
from .modality import Modality
from .data_type import DataType
from . import factory


class VideoData(DataType):
    def __init__(self, path: str):
        super().__init__(path)

    def preprocess(self, path: str) -> None:
        """Loads Video from path with provided string"""

        # Store raw data as bytes
        with open(path, "rb") as f:
            self.raw_data = f.read()

        # Load video from path
        cap = cv2.VideoCapture(path)

        # Extract frames
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()

        # Convert list of frames to numpy array and store
        self.data = np.array(frames).astype(np.float64)

    @property
    def modality(self) -> Modality:
        return Modality.VIDEO


factory.register(Modality.VIDEO, VideoData)
