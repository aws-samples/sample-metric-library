# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Base classes for all metrics in the library.

This module defines the core interfaces and base classes that all metrics inherit from,
providing a standardized API.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, List, Optional
import inspect

from .aggregation import AggregationMethod, SampleOutput
from .data import DataType, Modality
from .input_policy import InputPolicy
from .multiplicity import Multiplicity

if TYPE_CHECKING:
    from .bedrock_client import BedrockClient
    from .evaluation import EvaluationSample


class Metric(ABC):
    """
    Base class for all metrics.

    This class provides the foundation for all metrics in the library,
    including the per-sample scoring interface.
    """

    # Default supported metric modalities, can be overwritten by other classes
    supported_data_types: list[Modality] = [
        Modality.TEXT,
        Modality.AUDIO,
        Modality.IMAGE,
        Modality.VIDEO,
    ]

    # Default input contracts, can be overwritten by other classes
    reference_policy: InputPolicy = InputPolicy.REQUIRED
    prompt_policy: InputPolicy = InputPolicy.OMIT
    context_policy: InputPolicy = InputPolicy.OMIT

    # How many items each sample may carry per role, and how they relate. The
    # default is one value per role; a metric that reads several items declares
    # how (see ``Multiplicity``). Input validation enforces the counts.
    prediction_multiplicity: Multiplicity = Multiplicity.SINGLE
    reference_multiplicity: Multiplicity = Multiplicity.SINGLE

    # Default aggregator for the engine; None means numeric_mean
    default_aggregation_method: AggregationMethod | None = None

    def __init__(self, name: str = None):
        """
        Initialize the metric.

        Args:
            name: Optional name for the metric (defaults to class name)
        """
        self.name = name or self.__class__.__name__

    @abstractmethod
    def compute(
        self, sample: "EvaluationSample", client: "BedrockClient"
    ) -> "float | SampleOutput":
        """
        Score one sample. Return a finite real, or a ``SampleOutput`` that
        carries the score and a ``details`` mapping that the sample result
        keeps. With an item matcher, the sample holds one predicted item and
        one reference item, and the score must be from 0.0 to 1.0.
        Deterministic metrics ignore ``client``; LLM metrics use it for the
        judge call.

        Args:
            sample: One aligned prediction-reference pair with its optional
                context and prompt
            client: Bedrock client shared across the samples of one evaluation

        Returns:
            Finite real score for the sample, or a ``SampleOutput``
        """
        pass

    def validate_evaluation_inputs(
        self,
        predictions: List[DataType],
        references: List[DataType],
        context: Optional[List[List[DataType]]],
        prompts: Optional[List[DataType]],
    ) -> None:
        """
        Fail before compute when inputs are missing, unaligned or of a modality
        the metric does not declare.

        Each auxiliary input is checked against the metric's declared policy:
        OMIT skips the role, REQUIRED fails when the role is missing, and a
        supplied REQUIRED or OPTIONAL role must have one item per prediction.
        Every item of a role the metric reads must have a modality listed in
        ``supported_data_types``. Item content is never inspected.

        Args:
            predictions: Predicted values, one per sample
            references: Ground truth values, one per prediction
            context: Optional context data - List of context sources, each containing
                one DataType object per prediction
            prompts: Optional prompt data, one per prediction

        Raises:
            ValueError: If predictions are empty, a declared policy is not an
                ``InputPolicy`` value, a REQUIRED input is missing, a supplied
                input does not align with predictions, or an item has a
                modality outside ``supported_data_types``
        """
        if not predictions:
            raise ValueError("predictions must contain at least one item")

        prediction_count = len(predictions)
        reference_policy = self._validate_policy("reference", self.reference_policy)
        prompt_policy = self._validate_policy("prompt", self.prompt_policy)
        context_policy = self._validate_policy("context", self.context_policy)

        if reference_policy != InputPolicy.OMIT:
            if not references:
                if reference_policy == InputPolicy.REQUIRED:
                    raise ValueError(f"references are required for {self.name}")
            elif len(references) != prediction_count:
                raise ValueError(
                    f"reference count ({len(references)}) must match prediction "
                    f"count ({prediction_count}) for {self.name}"
                )

        if prompt_policy != InputPolicy.OMIT:
            if not prompts:
                if prompt_policy == InputPolicy.REQUIRED:
                    raise ValueError(f"prompts are required for {self.name}")
            elif len(prompts) != prediction_count:
                raise ValueError(
                    f"prompts count ({len(prompts)}) must match prediction "
                    f"count ({prediction_count}) for {self.name}"
                )

        if context_policy != InputPolicy.OMIT:
            if not context:
                if context_policy == InputPolicy.REQUIRED:
                    raise ValueError(f"context is required for {self.name}")
            else:
                for source_index, context_source in enumerate(context):
                    if len(context_source) != prediction_count:
                        raise ValueError(
                            f"context source {source_index} count "
                            f"({len(context_source)}) must match prediction "
                            f"count ({prediction_count}) for {self.name}"
                        )

        self._validate_modalities("prediction", predictions)
        if reference_policy != InputPolicy.OMIT:
            self._validate_modalities("reference", references)
        if prompt_policy != InputPolicy.OMIT:
            self._validate_modalities("prompt", prompts)
        if context_policy != InputPolicy.OMIT and context:
            for source_index, context_source in enumerate(context):
                self._validate_modalities(
                    f"context source {source_index} item", context_source
                )

    def _validate_modalities(self, role: str, items: Optional[List[object]]) -> None:
        """Fail when an item of a role the metric reads has an undeclared modality.

        An entry of a role is one ``DataType`` or a list of them when the
        sample carries several items; every part is checked.
        """
        for index, entry in enumerate(items or ()):
            parts = entry if isinstance(entry, (list, tuple)) else (entry,)
            for item in parts:
                if item is None or item.modality in self.supported_data_types:
                    continue
                accepted = ", ".join(str(m) for m in self.supported_data_types)
                raise ValueError(
                    f"{self.name} does not accept {item.modality} data "
                    f"({role} {index}); supported: {accepted}"
                )

    def _validate_item_counts(
        self,
        predictions: list,
        references: list | None,
    ) -> None:
        """Enforce each role's declared ``Multiplicity`` on every sample's items.

        A sample's prediction or reference may be one ``DataType`` or a list
        of them. ``SINGLE`` requires exactly one, ``ALIGNED`` requires the two
        roles to have the same number, and every role needs at least one item.

        The multiplicity is the metric's reading of several items in one
        ``compute`` call, so the engine calls this with the ``whole`` matcher
        only, after ``validate_evaluation_inputs``. With ``labels`` or ``list``
        the matcher replaces it: the matcher reads the items, and ``compute``
        gets one item per role.
        """
        reference_policy = self._validate_policy("reference", self.reference_policy)
        prediction_rule = self._validate_multiplicity(
            "prediction", self.prediction_multiplicity
        )
        reference_rule = self._validate_multiplicity(
            "reference", self.reference_multiplicity
        )
        aligned = Multiplicity.ALIGNED in (prediction_rule, reference_rule)
        if aligned and reference_policy == InputPolicy.OMIT:
            raise ValueError(
                f"{self.name} declares aligned prediction and reference items "
                "but omits references"
            )
        check_references = reference_policy != InputPolicy.OMIT and bool(references)

        for index, prediction in enumerate(predictions):
            prediction_count = self._item_count(prediction)
            self._check_item_count(
                "prediction", prediction_rule, prediction_count, index
            )
            if not check_references:
                continue
            reference_count = self._item_count(references[index])
            self._check_item_count("reference", reference_rule, reference_count, index)
            if aligned and prediction_count != reference_count:
                raise ValueError(
                    f"prediction item count ({prediction_count}) must match "
                    f"reference item count ({reference_count}) for sample "
                    f"{index} of {self.name}"
                )

    def _check_item_count(
        self, role: str, rule: Multiplicity, count: int, index: int
    ) -> None:
        if count == 0:
            raise ValueError(f"sample {index} has no {role} items for {self.name}")
        if rule is Multiplicity.SINGLE and count != 1:
            raise ValueError(
                f"{self.name} accepts exactly one {role} item per sample; "
                f"sample {index} has {count}"
            )

    @staticmethod
    def _item_count(value: object) -> int:
        return len(value) if isinstance(value, (list, tuple)) else 1

    def _validate_multiplicity(self, role: str, value: object) -> Multiplicity:
        """Coerce a declared multiplicity or fail with a clear message."""
        try:
            return Multiplicity(value)
        except ValueError as error:
            expected = [member.value for member in Multiplicity]
            raise ValueError(
                f"Invalid {role} multiplicity {value!r} for {self.name}; "
                f"expected one of {expected}"
            ) from error

    @staticmethod
    def _raw_value(items: "tuple[DataType, ...]"):
        """One item's raw value, or the list of raw values when there are several.

        A single item hands its raw value through untouched, so a metric that
        gives list-valued raw data its own meaning (a label set, a list to
        match) sees exactly what it saw when the role was one ``DataType``.
        """
        if len(items) == 1:
            return items[0].get_raw_data()
        return [item.get_raw_data() for item in items]

    def _score_aligned(
        self,
        sample: "EvaluationSample",
        score_pair: "Callable[[DataType, DataType], float]",
        reduce: "Callable[[list[float]], float] | None" = None,
    ) -> "float | SampleOutput":
        """Score ``ALIGNED`` items pairwise and reduce them to one sample score.

        Item ``i`` of the prediction is scored against item ``i`` of the
        reference, inside one ``compute`` call of the whole path. One pair
        returns its bare score, unchanged from scoring one value per role.
        Several pairs return a ``SampleOutput`` whose score is ``reduce`` of the
        item scores (their mean by default) and whose details carry every item
        score under ``item_scores``. The labels and list matchers pair items by
        score in the engine instead, so an ``ALIGNED`` metric accepts only the
        whole matcher.
        """
        scores = [
            score_pair(prediction, reference)
            for prediction, reference in zip(sample.predictions, sample.references)
        ]
        if len(scores) == 1:
            return scores[0]
        score = reduce(scores) if reduce is not None else sum(scores) / len(scores)
        return SampleOutput(score=score, details={"item_scores": scores})

    def _validate_policy(self, input_name: str, policy: object) -> InputPolicy:
        """Coerce a declared policy to ``InputPolicy`` or fail with a clear message."""
        try:
            return InputPolicy(policy)
        except ValueError as error:
            expected = [member.value for member in InputPolicy]
            raise ValueError(
                f"Invalid {input_name} policy {policy!r} for {self.name}; "
                f"expected one of {expected}"
            ) from error

    @classmethod
    def get_info(cls) -> str:
        return (
            f"* {cls.__name__}: {inspect.getdoc(cls)}\n"
            f"  - Supported Data Types: {', '.join(str(dt) for dt in cls.supported_data_types)}\n\n"
        )
