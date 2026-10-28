# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

from typing import TYPE_CHECKING
from ..core.base import Metric
from ..core.data import Modality
from ..core.multiplicity import Multiplicity
from ..core.registry import MetricRegistry
from ..utils.comparison_functions import (
    compare_exact,
    compare_fuzzy,
    compare_hungarian,
    compare_numeric,
    compare_semantic,
    NormalizationType,
    ComparatorType,
)

if TYPE_CHECKING:
    from ..core.bedrock_client import BedrockClient
    from ..core.evaluation import EvaluationSample


class DynamicMatch(Metric):
    """
    Base class for dynamically generated metric combinations.
    Supports any combination of normalization and comparison types.
    """

    # The comparators work on strings, numbers and lists, which all load as
    # text; media inputs are rejected before scoring.
    supported_data_types = [Modality.TEXT]

    def __init__(
        self,
        normalization_type: str = NormalizationType.TEXT.value,
        comparator_type: str = ComparatorType.EXACT.value,
    ):
        """
        Initialize a dynamic metric with specified normalization and comparison types.

        Args:
            normalization_type: Type of normalization ('raw', 'text', 'numeric', 'list', 'item')
            comparator_type: Type of comparison ('exact', 'fuzzy', 'hungarian', 'semantic', 'numeric')
        """
        super().__init__()
        self.normalization_type = normalization_type
        self.comparator_type = comparator_type

        # Validate types
        valid_normalizations = [norm_type.value for norm_type in NormalizationType]
        if normalization_type not in valid_normalizations:
            raise ValueError(f"Unknown normalization type: {normalization_type}")
        valid_comparators = [comp_type.value for comp_type in ComparatorType]
        if comparator_type not in valid_comparators:
            raise ValueError(f"Unknown comparator type: {comparator_type}")

        # Set up the comparator function
        self._setup_comparator()

    def _setup_comparator(self):
        """Setup the appropriate comparator function based on type."""
        if self.comparator_type == ComparatorType.EXACT.value:
            self.comparator = lambda x, y: compare_exact(x, y, self.normalization_type)
        elif self.comparator_type == ComparatorType.FUZZY.value:
            self.comparator = lambda x, y: compare_fuzzy(x, y, self.normalization_type)
        elif self.comparator_type == ComparatorType.HUNGARIAN.value:
            self.comparator = lambda x, y: compare_hungarian(
                x, y, ComparatorType.EXACT.value, self.normalization_type
            )
        elif self.comparator_type == ComparatorType.SEMANTIC.value:
            # Semantic comparison uses shared bedrock client from MetricRegistry
            bedrock_client = MetricRegistry.get_shared_bedrock_client()
            self.comparator = lambda x, y: compare_semantic(x, y, bedrock_client)
        elif self.comparator_type == ComparatorType.NUMERIC.value:
            self.comparator = compare_numeric

    # Several items in a role form one list value for the comparator.
    prediction_multiplicity = Multiplicity.COMPOSITE
    reference_multiplicity = Multiplicity.COMPOSITE

    def compute(
        self,
        sample: "EvaluationSample",
        client: "BedrockClient",
    ) -> float:
        """
        Compare one prediction-reference pair as a whole value.

        The comparator owns all normalization and list handling (text
        stripping, numeric parsing, list parsing, Hungarian assignment) and
        the null cases, so the raw value must reach it intact rather than
        being split into characters or list elements. Several items in a role
        reach it as one list of raw values. One comparator call per sample.
        """
        return self.comparator(
            self._raw_value(sample.predictions), self._raw_value(sample.references)
        )


# ============================================================================
# INDIVIDUAL METRIC CLASSES WITH DECORATORS
# ============================================================================
# Extended Combinations - Text Normalization
@MetricRegistry.register
class TextExactMatch(DynamicMatch):
    """Text normalization with exact comparison"""

    def __init__(self):
        super().__init__(
            normalization_type=NormalizationType.TEXT.value,
            comparator_type=ComparatorType.EXACT.value,
        )


@MetricRegistry.register
class TextFuzzyMatch(DynamicMatch):
    """Text normalization with fuzzy comparison"""

    def __init__(self):
        super().__init__(
            normalization_type=NormalizationType.TEXT.value,
            comparator_type=ComparatorType.FUZZY.value,
        )


@MetricRegistry.register
class TextHungarianMatch(DynamicMatch):
    """Text normalization with hungarian comparison"""

    def __init__(self):
        super().__init__(
            normalization_type=NormalizationType.TEXT.value,
            comparator_type=ComparatorType.HUNGARIAN.value,
        )


# Extended Combinations - Numeric Normalization
@MetricRegistry.register
class NumericExactMatch(DynamicMatch):
    """Numeric normalization with exact comparison"""

    def __init__(self):
        super().__init__(
            normalization_type=NormalizationType.NUMERIC.value,
            comparator_type=ComparatorType.EXACT.value,
        )


@MetricRegistry.register
class NumericFuzzyMatch(DynamicMatch):
    """Numeric normalization with fuzzy comparison"""

    def __init__(self):
        super().__init__(
            normalization_type=NormalizationType.NUMERIC.value,
            comparator_type=ComparatorType.FUZZY.value,
        )


# Extended Combinations - List Normalization
@MetricRegistry.register
class ListExactMatch(DynamicMatch):
    """List normalization with exact comparison"""

    def __init__(self):
        super().__init__(
            normalization_type=NormalizationType.LIST.value,
            comparator_type=ComparatorType.EXACT.value,
        )


@MetricRegistry.register
class ListFuzzyMatch(DynamicMatch):
    """List normalization with fuzzy comparison"""

    def __init__(self):
        super().__init__(
            normalization_type=NormalizationType.LIST.value,
            comparator_type=ComparatorType.FUZZY.value,
        )


@MetricRegistry.register
class ListHungarianMatch(DynamicMatch):
    """List normalization with hungarian comparison"""

    def __init__(self):
        super().__init__(
            normalization_type=NormalizationType.LIST.value,
            comparator_type=ComparatorType.HUNGARIAN.value,
        )


# Extended Combinations - Item Normalization
@MetricRegistry.register
class ItemExactMatch(DynamicMatch):
    """Item normalization with exact comparison"""

    def __init__(self):
        super().__init__(
            normalization_type=NormalizationType.ITEM.value,
            comparator_type=ComparatorType.EXACT.value,
        )


@MetricRegistry.register
class ItemFuzzyMatch(DynamicMatch):
    """Item normalization with fuzzy comparison"""

    def __init__(self):
        super().__init__(
            normalization_type=NormalizationType.ITEM.value,
            comparator_type=ComparatorType.FUZZY.value,
        )


@MetricRegistry.register
class ItemHungarianMatch(DynamicMatch):
    """Item normalization with hungarian comparison"""

    def __init__(self):
        super().__init__(
            normalization_type=NormalizationType.ITEM.value,
            comparator_type=ComparatorType.HUNGARIAN.value,
        )


@MetricRegistry.register
class ItemSemanticMatch(DynamicMatch):
    """Item normalization with semantic comparison"""

    def __init__(self):
        super().__init__(
            normalization_type=NormalizationType.ITEM.value,
            comparator_type=ComparatorType.SEMANTIC.value,
        )


@MetricRegistry.register
class RawExactMatch(DynamicMatch):
    """Raw / None normalization with exact comparison"""

    def __init__(self):
        super().__init__(
            normalization_type=NormalizationType.RAW.value,
            comparator_type=ComparatorType.EXACT.value,
        )


@MetricRegistry.register
class RawFuzzyMatch(DynamicMatch):
    """Raw normalization with fuzzy comparison"""

    def __init__(self):
        super().__init__(
            normalization_type=NormalizationType.RAW.value,
            comparator_type=ComparatorType.FUZZY.value,
        )


@MetricRegistry.register
class RawHungarianMatch(DynamicMatch):
    """Raw normalization with hungarian comparison"""

    def __init__(self):
        super().__init__(
            normalization_type=NormalizationType.RAW.value,
            comparator_type=ComparatorType.HUNGARIAN.value,
        )


@MetricRegistry.register
class RawSemanticMatch(DynamicMatch):
    """Raw normalization with semantic comparison"""

    def __init__(self):
        super().__init__(
            normalization_type=NormalizationType.RAW.value,
            comparator_type=ComparatorType.SEMANTIC.value,
        )
