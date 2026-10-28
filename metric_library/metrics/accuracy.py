# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

from typing import TYPE_CHECKING

from ..core.base import Metric
from ..core.data import Modality
from ..core.multiplicity import Multiplicity
from ..core.registry import MetricRegistry
from ..utils.comparison_functions import label_set

if TYPE_CHECKING:
    from ..core.bedrock_client import BedrockClient
    from ..core.evaluation import EvaluationSample


@MetricRegistry.register
class Accuracy(Metric):
    """
    Exact-match rate of label sets.

    Each sample scores 1.0 when the prediction and the reference are the same
    label set, else 0.0. A string is one label; a list is a set of labels;
    None, an empty list and a blank string are the empty set. Labels are
    trimmed and lowercased; punctuation is kept; an integral float such as
    1.0 is the label 1. Several items in a role are the labels of one set,
    exactly like one list-valued item. The final score is the mean. With the
    labels matcher and the prf aggregator, the run gives the precision,
    recall and F1 of the labels instead.
    """

    # Labels are strings or lists of strings, which load as text.
    supported_data_types = [Modality.TEXT]
    # Several items in a role are the labels of one set.
    prediction_multiplicity = Multiplicity.COMPOSITE
    reference_multiplicity = Multiplicity.COMPOSITE

    def compute(
        self,
        sample: "EvaluationSample",
        client: "BedrockClient",
    ) -> float:
        """Score one pair: 1.0 when the two label sets are equal, else 0.0."""
        predicted = label_set(self._raw_value(sample.predictions))
        expected = label_set(self._raw_value(sample.references))
        return 1.0 if predicted == expected else 0.0
