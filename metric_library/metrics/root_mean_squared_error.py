# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import math

from ..core.aggregation import SampleOutput
from ..core.data import DataType, Modality
from ..core.registry import MetricRegistry
from .mean_squared_error import MeanSquaredError
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from ..core.bedrock_client import BedrockClient
    from ..core.evaluation import EvaluationSample


@MetricRegistry.register
class RootMeanSquaredError(MeanSquaredError):
    """
    Square root of the per-sample MSE; the final score is the mean of the per-sample values.

    Root Mean Squared Error (RMSE) scores each audio, image or video sample as the
    square root of that sample's Mean Squared Error, in the units of the data.
    Large errors count more than small ones. The final score is the mean of the
    per-sample values.

    With several aligned pairs in one sample, each pair's RMSE is an item score
    and the sample score is their mean.
    """

    # Works with numeric data types
    supported_data_types: List[Modality] = [
        Modality.AUDIO,
        Modality.IMAGE,
        Modality.VIDEO,
    ]

    def compute(
        self,
        sample: "EvaluationSample",
        client: "BedrockClient",
    ) -> "float | SampleOutput":
        """The pair's RMSE; with several aligned pairs, the mean of their RMSEs."""
        return self._score_aligned(sample, self._pair_rmse)

    @classmethod
    def _pair_rmse(cls, prediction: DataType, reference: DataType) -> float:
        return math.sqrt(cls._pair_error(prediction, reference))
