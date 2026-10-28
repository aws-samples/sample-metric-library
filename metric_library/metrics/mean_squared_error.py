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
class MeanSquaredError(Metric):
    """
    Measures the average squared difference between predictions and actual values.

    Mean Squared Error (MSE) penalizes larger errors more heavily than smaller ones
    by squaring the differences. It's commonly used in regression tasks.

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
        Mean squared difference between the elements of each pair.

        This is the per-sample value; the aggregator gives the dataset value.
        """
        return self._score_aligned(sample, self._pair_error)

    @staticmethod
    def _pair_error(prediction: DataType, reference: DataType) -> float:
        sqr_diff = (prediction.get_data() - reference.get_data()) ** 2
        return float(np.mean(sqr_diff))
