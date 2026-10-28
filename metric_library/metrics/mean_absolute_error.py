# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import numpy as np
from typing import TYPE_CHECKING, List
from ..core.aggregation import SampleOutput
from ..core.base import Metric
from ..core.data import DataType, Modality
from ..core.multiplicity import Multiplicity
from ..core.registry import MetricRegistry

if TYPE_CHECKING:
    from ..core.bedrock_client import BedrockClient
    from ..core.evaluation import EvaluationSample


@MetricRegistry.register
class MeanAbsoluteError(Metric):
    """
    Measures the average magnitude of errors between predictions and actual values.

    Mean Absolute Error (MAE) is calculated as the average of absolute differences between
    predicted and actual values, without considering their direction. It's useful for
    regression tasks and gives equal weight to all errors.

    A sample may carry several prediction items paired by position with as many
    reference items. Each pair is scored on its own, the sample score is the
    mean of the pair errors, and the sample details carry every pair error
    under ``item_scores``.
    """

    # Works with numeric data types
    supported_data_types: List[Modality] = [
        Modality.AUDIO,
        Modality.IMAGE,
        Modality.VIDEO,
    ]

    prediction_multiplicity = Multiplicity.ALIGNED
    reference_multiplicity = Multiplicity.ALIGNED

    def compute(
        self,
        sample: "EvaluationSample",
        client: "BedrockClient",
    ) -> "float | SampleOutput":
        """
        Mean absolute difference between the elements of each pair.

        This is the per-sample value; the aggregator gives the dataset value.
        """
        return self._score_aligned(sample, self._pair_error)

    @staticmethod
    def _pair_error(prediction: DataType, reference: DataType) -> float:
        abs_diff = np.abs(prediction.get_data() - reference.get_data())
        return float(np.mean(abs_diff))
