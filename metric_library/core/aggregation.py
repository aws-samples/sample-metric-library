# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Explicit numeric sample aggregation contracts."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from numbers import Real
from typing import Any, Literal, Sequence, TypeAlias

from munkres import Munkres, make_cost_matrix

AggregationMethod: TypeAlias = Literal[
    "numeric_mean",
    "weighted_mean",
    "median",
    "quantile",
    "prf",
]
_METHODS = frozenset(
    {
        "numeric_mean",
        "weighted_mean",
        "median",
        "quantile",
        "prf",
    }
)


class SampleStatus(str, Enum):
    """Terminal status for one sample evaluation."""

    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class EvaluationStatus(str, Enum):
    """Status for one metric's detailed evaluation.

    Evaluation stops at the first sample that still fails after retries, so an
    aggregate is either complete or failed; there is no partial aggregate.
    """

    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class SampleOutput:
    """The return value of ``compute`` that carries a score and a details mapping."""

    score: float
    details: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "score", _validate_score(self.score))
        object.__setattr__(self, "details", _validate_details(self.details))


@dataclass(frozen=True)
class Matches:
    """The candidate pair scores and the item counts of one class in one sample.

    ``scores`` has one row for each predicted item, with one score for each
    reference item. The ``prf`` aggregator pairs the items at each threshold.
    """

    scores: tuple[tuple[float, ...], ...]
    n_pred: int
    n_ref: int

    def __post_init__(self) -> None:
        for name in ("n_pred", "n_ref"):
            count = getattr(self, name)
            if isinstance(count, bool) or not isinstance(count, int):
                raise TypeError(f"{name} must be an integer")
            if count < 0:
                raise ValueError(f"{name} must be non-negative")
        rows = tuple(tuple(row) for row in self.scores)
        if len(rows) != self.n_pred or any(len(row) != self.n_ref for row in rows):
            raise ValueError("scores must have n_pred rows of n_ref scores")
        object.__setattr__(
            self,
            "scores",
            tuple(tuple(_validate_pair_score(score) for score in row) for row in rows),
        )

    def pair_count(self, threshold: float) -> int:
        """Return the most one-to-one pairs with a score of ``threshold`` or more.

        Pairs below the threshold do not count, so they cannot take the place
        of a pair at or above it.
        """
        eligible = [
            [1 if score >= threshold else 0 for score in row] for row in self.scores
        ]
        if not any(map(any, eligible)):
            return 0
        pairs = Munkres().compute(make_cost_matrix(eligible, lambda x: 1 - x))
        return sum(eligible[row][column] for row, column in pairs)


@dataclass(frozen=True)
class SampleResult:
    """One numeric sample outcome consumed by an aggregator.

    Skipped samples carry no payload: a sample is skipped only when evaluation
    stops after an earlier failure, which the status already says. ``matches``
    holds one ``Matches`` for each class when an item matcher scored the sample.
    """

    sample_id: str
    ordinal: int
    status: SampleStatus
    score: float | None = None
    error: str | None = None
    details: dict | None = None
    matches: dict | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id.strip():
            raise ValueError("sample_id must be a non-empty string")
        if isinstance(self.ordinal, bool) or not isinstance(self.ordinal, int):
            raise TypeError("ordinal must be an integer")
        if self.ordinal < 0:
            raise ValueError("ordinal must be non-negative")
        if not isinstance(self.status, SampleStatus):
            raise TypeError("status must be a SampleStatus")

        if self.status is SampleStatus.COMPLETED:
            normalized_score = _validate_score(self.score)
            if self.error is not None:
                raise ValueError("completed samples cannot contain an error")
            object.__setattr__(self, "score", normalized_score)
            if self.details is not None:
                object.__setattr__(self, "details", _validate_details(self.details))
            if self.matches is not None:
                object.__setattr__(self, "matches", _validate_matches(self.matches))
            return

        if self.score is not None:
            raise ValueError("non-completed samples cannot contain a score")
        if self.details is not None:
            raise ValueError("non-completed samples cannot contain details")
        if self.matches is not None:
            raise ValueError("non-completed samples cannot contain matches")

        if self.status is SampleStatus.FAILED:
            if not isinstance(self.error, str) or not self.error.strip():
                raise ValueError("failed samples require a non-empty error")
            return

        if self.error is not None:
            raise ValueError("skipped samples cannot contain an error")


@dataclass(frozen=True)
class AggregationSpec:
    """Closed configuration for the supported numeric aggregators."""

    method: AggregationMethod = "numeric_mean"
    weights: tuple[float, ...] | None = None
    quantile: float | None = None
    thresholds: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.method not in _METHODS:
            raise ValueError(f"unknown aggregation method: {self.method!r}")

        normalized_weights = None
        if self.weights is not None:
            if not isinstance(self.weights, tuple):
                raise TypeError("weights must be a tuple")
            normalized_weights = tuple(
                _validate_weight(weight) for weight in self.weights
            )
            object.__setattr__(self, "weights", normalized_weights)

        if self.quantile is not None:
            object.__setattr__(
                self,
                "quantile",
                _validate_unit_parameter(self.quantile, "quantile"),
            )

        if self.thresholds is not None:
            if not isinstance(self.thresholds, tuple):
                raise TypeError("thresholds must be a tuple")
            normalized_thresholds = tuple(
                _validate_threshold(threshold) for threshold in self.thresholds
            )
            if not normalized_thresholds:
                raise ValueError("thresholds must not be empty")
            if len(set(normalized_thresholds)) != len(normalized_thresholds):
                raise ValueError("thresholds must be unique")
            object.__setattr__(self, "thresholds", normalized_thresholds)

        if self.method == "weighted_mean":
            if normalized_weights is None:
                raise ValueError("weighted_mean requires weights")
            if self.quantile is not None or self.thresholds is not None:
                raise ValueError("weighted_mean accepts only weights")
        elif self.method == "quantile":
            if self.quantile is None:
                raise ValueError("quantile aggregation requires quantile")
            if self.weights is not None or self.thresholds is not None:
                raise ValueError("quantile aggregation accepts only quantile")
        elif self.method == "prf":
            if self.weights is not None or self.quantile is not None:
                raise ValueError("prf accepts only thresholds")
            if self.thresholds is None:
                # Without thresholds, a pair with a score of 0.5 or more counts.
                object.__setattr__(self, "thresholds", (0.5,))
        elif (
            self.weights is not None
            or self.quantile is not None
            or self.thresholds is not None
        ):
            raise ValueError(f"{self.method} does not accept aggregation parameters")

    def validate_for(self, total_count: int) -> None:
        """Validate parameters that depend on the complete sample count."""
        if isinstance(total_count, bool) or not isinstance(total_count, int):
            raise TypeError("total_count must be an integer")
        if total_count <= 0:
            raise ValueError("at least one sample is required")
        if self.method != "weighted_mean":
            return
        if len(self.weights) != total_count:
            raise ValueError("weights must align one-for-one with all samples")
        if not math.isfinite(sum(self.weights)):
            raise ValueError("weight sum must be finite")


@dataclass(frozen=True)
class AggregationResult:
    """Numeric aggregate with explicit accounting and coverage."""

    method: AggregationMethod
    status: EvaluationStatus
    score: float | None
    total_count: int
    succeeded_count: int
    failed_count: int
    skipped_count: int
    denominator: int | float
    coverage: float
    spec: AggregationSpec = field(default_factory=AggregationSpec)
    values: dict | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.spec, AggregationSpec):
            raise TypeError("spec must be an AggregationSpec")
        if self.method != self.spec.method:
            raise ValueError("method must match spec.method")
        if not isinstance(self.status, EvaluationStatus):
            raise TypeError("status must be an EvaluationStatus")

        counts = (
            self.total_count,
            self.succeeded_count,
            self.failed_count,
            self.skipped_count,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) for value in counts
        ):
            raise TypeError("aggregation counts must be integers")
        if self.total_count <= 0:
            raise ValueError("total_count must be positive")
        if any(value < 0 for value in counts[1:]):
            raise ValueError("aggregation counts cannot be negative")
        if self.total_count != (
            self.succeeded_count + self.failed_count + self.skipped_count
        ):
            raise ValueError("sample counts must add up to total_count")
        self.spec.validate_for(self.total_count)
        self._validate_denominator()

        expected_coverage = self.succeeded_count / self.total_count
        if isinstance(self.coverage, bool) or not isinstance(self.coverage, Real):
            raise TypeError("coverage must be a real number")
        normalized_coverage = float(self.coverage)
        if not math.isfinite(normalized_coverage):
            raise ValueError("coverage must be finite")
        if normalized_coverage != expected_coverage:
            raise ValueError("coverage must equal succeeded_count / total_count")
        object.__setattr__(self, "coverage", normalized_coverage)

        if self.method == "prf" and self.status is EvaluationStatus.COMPLETED:
            if self.values is None:
                raise ValueError("completed prf aggregation requires values")
            if not isinstance(self.values, dict):
                raise TypeError("values must be a dict")
            if self.values.get("f1") != self.score:
                raise ValueError("values['f1'] must equal score")
        elif self.values is not None:
            raise ValueError("values are allowed only for a completed prf aggregation")

        if self.status is EvaluationStatus.COMPLETED:
            object.__setattr__(self, "score", _validate_score(self.score))
            if self.succeeded_count != self.total_count:
                raise ValueError(
                    "completed aggregation requires every sample to succeed"
                )
            return

        if self.score is not None:
            raise ValueError("failed aggregation cannot contain a score")
        if self.failed_count != 1:
            raise ValueError("failed aggregation requires exactly one failed sample")

    def _validate_denominator(self) -> None:
        if self.method == "weighted_mean":
            if isinstance(self.denominator, bool) or not isinstance(
                self.denominator, Real
            ):
                raise TypeError("weighted denominator must be a real number")
            normalized = float(self.denominator)
            if not math.isfinite(normalized):
                raise ValueError("weighted denominator must be finite")
            if self.succeeded_count == 0 and normalized != 0.0:
                raise ValueError("weighted denominator must be zero without successes")
            if self.succeeded_count > 0 and normalized <= 0.0:
                raise ValueError("weighted denominator must be positive with successes")
            object.__setattr__(self, "denominator", normalized)
            return

        if isinstance(self.denominator, bool) or not isinstance(self.denominator, int):
            raise TypeError("denominator must be an integer")
        if self.denominator != self.succeeded_count:
            raise ValueError("denominator must equal succeeded_count")


def numeric_mean(results: Sequence[SampleResult]) -> AggregationResult:
    """Return the ordered arithmetic mean of successful sample scores."""
    return _aggregate_numeric(results, AggregationSpec())


def weighted_mean(
    results: Sequence[SampleResult],
    weights: Sequence[float],
) -> AggregationResult:
    """Return an ordinal-aligned positive-weight mean."""
    return _aggregate_numeric(
        results,
        AggregationSpec(method="weighted_mean", weights=tuple(weights)),
    )


def median(results: Sequence[SampleResult]) -> AggregationResult:
    """Return the median of successful sample scores."""
    return _aggregate_numeric(results, AggregationSpec(method="median"))


def quantile(results: Sequence[SampleResult], q: float) -> AggregationResult:
    """Return an inclusive type-7 linear quantile of successful scores."""
    return _aggregate_numeric(results, AggregationSpec(method="quantile", quantile=q))


def prf(
    results: Sequence[SampleResult],
    thresholds: Sequence[float] | None = None,
) -> AggregationResult:
    """Return the class means of precision, recall and F1 from the match records.

    A pair counts as a true positive when its score is at least the threshold.
    The first threshold gives the score; without thresholds it is 0.5.
    """
    if thresholds is not None:
        thresholds = tuple(thresholds)
    return _aggregate_prf(results, AggregationSpec(method="prf", thresholds=thresholds))


def aggregate(
    results: Sequence[SampleResult],
    spec: AggregationSpec,
) -> AggregationResult:
    if spec.method == "prf":
        return _aggregate_prf(results, spec)
    return _aggregate_numeric(results, spec)


def _aggregate_numeric(
    results: Sequence[SampleResult],
    spec: AggregationSpec,
) -> AggregationResult:
    samples = tuple(results)
    if not samples:
        raise ValueError("at least one sample result is required")
    _validate_sample_sequence(samples)
    validate_fail_fast_sequence(samples)

    total_count = len(samples)
    spec.validate_for(total_count)
    completed = tuple(
        sample for sample in samples if sample.status is SampleStatus.COMPLETED
    )
    failed_count = sum(sample.status is SampleStatus.FAILED for sample in samples)
    skipped_count = sum(sample.status is SampleStatus.SKIPPED for sample in samples)
    succeeded_count = len(completed)
    coverage = succeeded_count / total_count

    status = (
        EvaluationStatus.COMPLETED
        if succeeded_count == total_count
        else EvaluationStatus.FAILED
    )
    candidate_score, denominator = _reduce_successful(completed, spec)
    score = candidate_score if status is EvaluationStatus.COMPLETED else None

    return AggregationResult(
        method=spec.method,
        status=status,
        score=score,
        total_count=total_count,
        succeeded_count=succeeded_count,
        failed_count=failed_count,
        skipped_count=skipped_count,
        denominator=denominator,
        coverage=coverage,
        spec=spec,
    )


def _reduce_successful(
    completed: Sequence[SampleResult],
    spec: AggregationSpec,
) -> tuple[float | None, int | float]:
    samples = tuple(completed)
    if not samples:
        return None, 0.0 if spec.method == "weighted_mean" else 0

    scores = tuple(sample.score for sample in samples)
    if spec.method == "numeric_mean":
        return sum(scores) / len(scores), len(scores)
    if spec.method == "weighted_mean":
        weighted = tuple(
            (sample.score, spec.weights[sample.ordinal]) for sample in samples
        )
        denominator = sum(weight for _, weight in weighted)
        numerator = sum(score * weight for score, weight in weighted)
        if not math.isfinite(denominator) or not math.isfinite(numerator):
            raise ValueError("weighted aggregation must remain finite")
        return numerator / denominator, denominator
    if spec.method == "median":
        ordered = sorted(scores)
        midpoint = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[midpoint], len(ordered)
        return (ordered[midpoint - 1] + ordered[midpoint]) / 2, len(ordered)

    # quantile: inclusive type-7 linear interpolation
    ordered = sorted(scores)
    position = (len(ordered) - 1) * spec.quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    value = ordered[lower] + fraction * (ordered[upper] - ordered[lower])
    return value, len(ordered)


def _aggregate_prf(
    results: Sequence[SampleResult],
    spec: AggregationSpec,
) -> AggregationResult:
    samples = tuple(results)
    if not samples:
        raise ValueError("at least one sample result is required")
    _validate_sample_sequence(samples)
    validate_fail_fast_sequence(samples)

    total_count = len(samples)
    spec.validate_for(total_count)
    completed = tuple(
        sample for sample in samples if sample.status is SampleStatus.COMPLETED
    )
    failed_count = sum(sample.status is SampleStatus.FAILED for sample in samples)
    skipped_count = sum(sample.status is SampleStatus.SKIPPED for sample in samples)
    succeeded_count = len(completed)
    coverage = succeeded_count / total_count

    status = (
        EvaluationStatus.COMPLETED
        if succeeded_count == total_count
        else EvaluationStatus.FAILED
    )
    values = (
        _prf_values(completed, spec) if status is EvaluationStatus.COMPLETED else None
    )

    return AggregationResult(
        method=spec.method,
        status=status,
        score=None if values is None else values["f1"],
        total_count=total_count,
        succeeded_count=succeeded_count,
        failed_count=failed_count,
        skipped_count=skipped_count,
        denominator=succeeded_count,
        coverage=coverage,
        spec=spec,
        values=values,
    )


def _prf_values(completed: Sequence[SampleResult], spec: AggregationSpec) -> dict:
    per_sample = []
    for sample in completed:
        if sample.matches is None:
            raise ValueError(f"{spec.method} requires per-sample matches")
        per_sample.append(sample.matches)

    universe = sorted({item_class for matches in per_sample for item_class in matches})
    if not universe:
        raise ValueError("no items in any prediction or reference")

    by_threshold = [
        _prf_at(per_sample, universe, threshold) for threshold in spec.thresholds
    ]
    values = dict(by_threshold[0])
    if len(by_threshold) > 1:
        values["by_threshold"] = by_threshold
    return values


def _prf_at(per_sample: list[dict], universe: list[str], threshold: float) -> dict:
    """Count the pairs of every class at one threshold and take the class means."""
    per_class = {}
    for item_class in universe:
        records = [
            matches[item_class] for matches in per_sample if item_class in matches
        ]
        tp = sum(record.pair_count(threshold) for record in records)
        fp = sum(record.n_pred for record in records) - tp
        fn = sum(record.n_ref for record in records) - tp
        precision, recall, f1 = _prf_trio(tp, fp, fn)
        per_class[item_class] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": len(per_sample) - len(records),
            "support": tp + fn,
        }

    return {
        "threshold": threshold,
        "precision": sum(c["precision"] for c in per_class.values()) / len(universe),
        "recall": sum(c["recall"] for c in per_class.values()) / len(universe),
        "f1": sum(c["f1"] for c in per_class.values()) / len(universe),
        "per_class": per_class,
    }


def _prf_trio(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Apply the zero rule: a ratio with a zero denominator is 0.0."""
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _validate_score(score: object) -> float:
    """Accept any finite real score; each metric owns its own range."""
    if isinstance(score, bool) or not isinstance(score, Real):
        raise TypeError("score must be a real number")
    normalized = float(score)
    if not math.isfinite(normalized):
        raise ValueError("score must be finite")
    return normalized


def _validate_details(details: object) -> dict:
    """Accept a mapping with string keys and return it as a dict copy."""
    if not isinstance(details, Mapping):
        raise TypeError("details must be a mapping")
    if any(not isinstance(key, str) for key in details):
        raise TypeError("details keys must be strings")
    return dict(details)


def _validate_matches(matches: object) -> dict:
    """Accept a mapping of class names to ``Matches`` and return a dict copy."""
    if not isinstance(matches, Mapping):
        raise TypeError("matches must be a mapping")
    if any(
        not isinstance(item_class, str) or not isinstance(record, Matches)
        for item_class, record in matches.items()
    ):
        raise TypeError("matches must map class names to Matches")
    return dict(matches)


def _validate_pair_score(score: object) -> float:
    """Accept a finite real pair score from 0.0 to 1.0."""
    normalized = _validate_score(score)
    if not 0.0 <= normalized <= 1.0:
        raise ValueError(f"pair score must be between 0.0 and 1.0, got {score!r}")
    return normalized


def _validate_threshold(threshold: object) -> float:
    if isinstance(threshold, bool) or not isinstance(threshold, Real):
        raise TypeError("thresholds must be real numbers")
    normalized = float(threshold)
    if not 0.0 < normalized <= 1.0:
        raise ValueError("thresholds must be greater than 0.0 and at most 1.0")
    return normalized


def _validate_weight(weight: object) -> float:
    if isinstance(weight, bool) or not isinstance(weight, Real):
        raise TypeError("weights must be real numbers")
    normalized = float(weight)
    if not math.isfinite(normalized):
        raise ValueError("weights must be finite")
    if normalized <= 0.0:
        raise ValueError("weights must be strictly positive")
    return normalized


def _validate_unit_parameter(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    if not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0")
    return normalized


def _validate_sample_sequence(results: tuple[SampleResult, ...]) -> None:
    expected_ordinals = tuple(range(len(results)))
    actual_ordinals = tuple(result.ordinal for result in results)
    if actual_ordinals != expected_ordinals:
        raise ValueError("sample results must be in contiguous ordinal order")
    sample_ids = tuple(result.sample_id for result in results)
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("sample result IDs must be unique")


def validate_fail_fast_sequence(results: tuple[SampleResult, ...]) -> None:
    failure_seen = False
    for result in results:
        if result.status is SampleStatus.COMPLETED:
            if failure_seen:
                raise ValueError(
                    "evaluation cannot contain completed results after failure"
                )
            continue
        if result.status is SampleStatus.FAILED:
            if failure_seen:
                raise ValueError("evaluation can contain only one failed result")
            failure_seen = True
            continue
        if not failure_seen:
            raise ValueError("skipped samples require a prior failure")


__all__ = [
    "aggregate",
    "validate_fail_fast_sequence",
    "AggregationMethod",
    "AggregationResult",
    "AggregationSpec",
    "EvaluationStatus",
    "Matches",
    "SampleOutput",
    "SampleResult",
    "SampleStatus",
    "median",
    "numeric_mean",
    "prf",
    "quantile",
    "weighted_mean",
]
