# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Engine functions: score every sample of one metric with retries and
aggregate the per-sample results. The registry calls ``evaluate_metric``.
The engine builds the Bedrock client when a metric first reads it.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from numbers import Real
from typing import Callable

from botocore.exceptions import (
    ClientError,
    ConnectionClosedError,
    ConnectTimeoutError,
    EndpointConnectionError,
    ReadTimeoutError,
)

from .aggregation import (
    AggregationMethod,
    AggregationResult,
    AggregationSpec,
    EvaluationStatus,
    Matches,
    SampleOutput,
    SampleResult,
    SampleStatus,
    aggregate as aggregate_samples,
    validate_fail_fast_sequence,
)
from .base import Metric
from .bedrock_client import MAX_POOL_CONNECTIONS, BedrockClient
from .input_policy import InputPolicy
from .data import DataType, Modality
from .data.text_data import TextData
from .llm_metric import LLMBatchMetric, LLMMetric
from .matching import Matcher, WholeValue
from .multiplicity import Multiplicity

logger = logging.getLogger(__name__)


# Upper bounds keep ``multiplier ** (max_attempts - 1)`` far below float
# overflow (5 ** 9), so the backoff formula needs no special cases.
MAX_ATTEMPTS_LIMIT = 10
MULTIPLIER_LIMIT = 5.0

# How many samples of a thread-safe metric are scored at the same time, by
# default and at most. The limit is the connection pool of the client the
# workers share, so that every worker has a pooled connection.
DEFAULT_MAX_WORKERS = 4
MAX_WORKERS_LIMIT = MAX_POOL_CONNECTIONS

# One ``compute_matrix`` call scores at most this many predicted items and
# this many reference items.
MATRIX_BLOCK_SIZE = 10


def _normalize_timing_value(field_name: str, value: Real) -> float:
    """Return one retry timing value as a finite float."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a real number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite")
    return normalized


def _as_items(value: object) -> tuple:
    """Normalize one role's value: ``None`` to ``()``, one item to a one-tuple."""
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return (value,)


@dataclass(frozen=True, init=False)
class EvaluationSample:
    """One aligned sample passed to ``Metric.compute``.

    Every role is stored as a tuple of ``DataType`` items: ``predictions``,
    ``references``, ``prompts`` and ``contexts``. A role may be supplied as
    ``None``, one item or a sequence of items and is normalized on
    construction, so one item and a one-tuple are the same sample; an omitted
    role is the empty tuple. How many items a role may carry is the metric's
    declaration (``Multiplicity``), enforced by its input validation.

    ``prediction``, ``reference`` and ``prompt`` read the one item of a role
    (``None`` when the role is empty), which is all a metric that scores one
    value per role ever needs. A metric that reads several items uses the
    plural tuples.
    """

    sample_id: str
    ordinal: int
    predictions: tuple[DataType, ...]
    references: tuple[DataType, ...]
    prompts: tuple[DataType, ...]
    contexts: tuple[DataType, ...]

    def __init__(
        self,
        sample_id: str,
        ordinal: int,
        prediction: object = None,
        reference: object = None,
        prompt: object = None,
        contexts: tuple[DataType, ...] = (),
    ) -> None:
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError("sample_id must be a non-empty string")
        if isinstance(ordinal, bool) or not isinstance(ordinal, int):
            raise TypeError("ordinal must be an integer")
        if ordinal < 0:
            raise ValueError("ordinal must be non-negative")
        if not isinstance(contexts, tuple):
            raise TypeError("contexts must be a tuple")
        object.__setattr__(self, "sample_id", sample_id)
        object.__setattr__(self, "ordinal", ordinal)
        object.__setattr__(self, "predictions", _as_items(prediction))
        object.__setattr__(self, "references", _as_items(reference))
        object.__setattr__(self, "prompts", _as_items(prompt))
        object.__setattr__(self, "contexts", contexts)

    @property
    def prediction(self) -> DataType | None:
        """The one prediction item; see ``predictions`` when there are several."""
        return self._single("prediction", self.predictions)

    @property
    def reference(self) -> DataType | None:
        """The one reference item, or ``None`` when the role is empty."""
        return self._single("reference", self.references)

    @property
    def prompt(self) -> DataType | None:
        """The one prompt item, or ``None`` when the role is empty."""
        return self._single("prompt", self.prompts)

    @staticmethod
    def _single(role: str, items: tuple[DataType, ...]) -> DataType | None:
        if len(items) > 1:
            raise ValueError(
                f"the sample carries {len(items)} {role} items; read sample.{role}s"
            )
        return items[0] if items else None


@dataclass(frozen=True)
class EvaluationConfiguration:
    """Caller overrides for one metric evaluation.

    ``max_attempts`` is how many times one sample may be scored before it is
    recorded as failed. A sample that still fails after its last attempt stops
    the evaluation: earlier successes are kept, later samples are skipped, and
    the aggregate status is failed.

    Retries wait ``min(max_delay, initial_delay * multiplier ** (n - 1))``
    seconds after the ``n``-th failed attempt.

    When ``aggregation_method`` is set, construction validates it together
    with ``weights``, ``quantile`` and ``thresholds`` through
    ``AggregationSpec``. ``resolve`` validates the weight count against the
    sample count, and validates parameters given without a method against the
    metric's default aggregator.

    ``matcher`` gives the items of each value to the metric. ``WholeValue``,
    the default, scores each sample as it is. ``LabelSet`` and ``ListItems``
    score each candidate pair of items and require the ``prf`` aggregator,
    whose ``thresholds`` set the lowest pair score of a true positive.

    ``max_workers`` is how many samples of a metric whose ``thread_safe`` is
    True are scored at the same time, each on its own thread; any other
    metric, or ``max_workers=1``, is scored one sample at a time on the
    calling thread. Samples are handed out in input order, and the results
    keep the input order. Once a sample fails, no sample starts and no later
    sample makes another attempt; the failure reported is the first in input
    order, and every later sample is skipped, also one that was already
    scored.
    """

    max_attempts: int = 3
    aggregation_method: AggregationMethod | None = None
    weights: tuple[float, ...] | None = None
    quantile: float | None = None
    initial_delay: float = 1.0
    multiplier: float = 2.0
    max_delay: float = 30.0
    thresholds: tuple[float, ...] | None = None
    matcher: Matcher = WholeValue()
    # Last, so that a configuration built positionally keeps its meaning.
    max_workers: int = DEFAULT_MAX_WORKERS

    def __post_init__(self) -> None:
        if isinstance(self.max_attempts, bool) or not isinstance(
            self.max_attempts, int
        ):
            raise TypeError("max_attempts must be an integer")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.max_attempts > MAX_ATTEMPTS_LIMIT:
            raise ValueError(f"max_attempts must be at most {MAX_ATTEMPTS_LIMIT}")

        if isinstance(self.max_workers, bool) or not isinstance(self.max_workers, int):
            raise TypeError("max_workers must be an integer")
        if self.max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        if self.max_workers > MAX_WORKERS_LIMIT:
            raise ValueError(f"max_workers must be at most {MAX_WORKERS_LIMIT}")

        initial_delay = _normalize_timing_value("initial_delay", self.initial_delay)
        multiplier = _normalize_timing_value("multiplier", self.multiplier)
        max_delay = _normalize_timing_value("max_delay", self.max_delay)

        if initial_delay < 0:
            raise ValueError("initial_delay must be non-negative")
        if multiplier < 1:
            raise ValueError("multiplier must be at least 1")
        if multiplier > MULTIPLIER_LIMIT:
            raise ValueError(f"multiplier must be at most {MULTIPLIER_LIMIT:g}")
        if max_delay < 0:
            raise ValueError("max_delay must be non-negative")
        if max_delay < initial_delay:
            raise ValueError("max_delay must be greater than or equal to initial_delay")

        object.__setattr__(self, "initial_delay", initial_delay)
        object.__setattr__(self, "multiplier", multiplier)
        object.__setattr__(self, "max_delay", max_delay)

        if not isinstance(self.matcher, Matcher):
            raise TypeError("matcher must be a Matcher")

        if self.aggregation_method is not None:
            AggregationSpec(
                method=self.aggregation_method,
                weights=self.weights,
                quantile=self.quantile,
                thresholds=self.thresholds,
            )

    def resolve(self, metric: Metric, total_count: int) -> AggregationSpec:
        """Resolve caller overrides and metric metadata into a validated spec."""
        method = self.aggregation_method
        if method is None:
            method = getattr(metric, "default_aggregation_method", None)
        if method is None:
            method = "numeric_mean"

        aggregation = AggregationSpec(
            method=method,
            weights=self.weights,
            quantile=self.quantile,
            thresholds=self.thresholds,
        )
        aggregation.validate_for(total_count)
        return aggregation


@dataclass(frozen=True)
class MetricEvaluationResult:
    """Ordered sample outcomes and their explicit aggregate."""

    metric_name: str
    samples: tuple[SampleResult, ...]
    aggregate: AggregationResult

    @property
    def status(self) -> EvaluationStatus:
        return self.aggregate.status

    @property
    def score(self) -> float | None:
        return self.aggregate.score

    def __post_init__(self) -> None:
        if not isinstance(self.metric_name, str) or not self.metric_name.strip():
            raise ValueError("metric_name must be a non-empty string")
        if not isinstance(self.samples, tuple):
            raise TypeError("samples must be a tuple")
        if len(self.samples) != self.aggregate.total_count:
            raise ValueError("sample count must match aggregate total_count")
        expected_ordinals = tuple(range(len(self.samples)))
        actual_ordinals = tuple(sample.ordinal for sample in self.samples)
        if actual_ordinals != expected_ordinals:
            raise ValueError("samples must be in contiguous ordinal order")
        sample_ids = tuple(sample.sample_id for sample in self.samples)
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("sample IDs must be unique")

        succeeded_count = sum(
            sample.status is SampleStatus.COMPLETED for sample in self.samples
        )
        failed_count = sum(
            sample.status is SampleStatus.FAILED for sample in self.samples
        )
        skipped_count = sum(
            sample.status is SampleStatus.SKIPPED for sample in self.samples
        )
        if (
            succeeded_count != self.aggregate.succeeded_count
            or failed_count != self.aggregate.failed_count
            or skipped_count != self.aggregate.skipped_count
        ):
            raise ValueError("sample statuses must match aggregate counts")
        validate_fail_fast_sequence(self.samples)
        expected_coverage = succeeded_count / len(self.samples)
        if self.aggregate.coverage != expected_coverage:
            raise ValueError("aggregate coverage must match successful samples")


class _LazyClient:
    """Build the client when a metric first reads it.

    The factory runs at the first attribute read, and the engine keeps that
    client for the rest of the evaluation, so it builds at most one client. A
    lock makes workers that first read it at the same time wait for that one
    client. A metric that never reads the client never builds it. A factory
    error surfaces at that first read, so it fails the sample that made it. A
    lasting error, such as a missing profile, is raised again at every later
    read, so the factory runs once; a transient one is retried with its sample.
    """

    def __init__(self, factory: Callable[[], BedrockClient]):
        self._factory = factory
        self._client: BedrockClient | None = None
        self._error: Exception | None = None
        self._lock = threading.Lock()

    def __getattr__(self, name: str):
        with self._lock:
            if self._error is not None:
                raise self._error
            if self._client is None:
                try:
                    self._client = self._factory()
                except Exception as error:
                    if not _classify_failure(error)[0]:
                        self._error = error
                    raise
            client = self._client
        return getattr(client, name)


class _SampleSkipped(Exception):
    """A sample stopped because an earlier sample failed: it is SKIPPED."""


class _Cutoff:
    """The ordinal of the earliest failed sample, which every worker reads.

    Only the scheduler writes it. ``check`` raises ``_SampleSkipped`` for a
    later sample, so a sample whose result would be SKIPPED makes no further
    Bedrock call; after ``stop_all`` it raises for every sample.
    """

    def __init__(self) -> None:
        self.ordinal: int | None = None

    def record(self, ordinal: int) -> None:
        if self.ordinal is None or ordinal < self.ordinal:
            self.ordinal = ordinal

    def stop_all(self) -> None:
        self.ordinal = -1

    def check(self, sample: EvaluationSample) -> None:
        if self.ordinal is not None and sample.ordinal > self.ordinal:
            raise _SampleSkipped


def evaluate_metric(
    metric: Metric,
    predictions: list[DataType],
    references: list[DataType],
    context: list[list[DataType]] | None = None,
    prompts: list[DataType] | None = None,
    *,
    configuration: EvaluationConfiguration | None = None,
    client_factory: Callable[[], BedrockClient] | None = None,
    sleeper: Callable[[float], None] | None = None,
) -> MetricEvaluationResult:
    """Score every sample of one metric with retries and aggregate the results.

    The client comes from ``client_factory`` (``BedrockClient`` by default)
    when the metric first reads it. Up to ``configuration.max_workers`` samples
    of a thread-safe metric are scored at the same time; the results keep the
    input order.
    """
    if sleeper is None:
        sleeper = time.sleep
    elif not callable(sleeper):
        raise TypeError("sleeper must be callable")

    metric_name = metric.name
    if not isinstance(metric_name, str) or not metric_name.strip():
        raise ValueError("metric.name must be a non-empty string")
    if not isinstance(metric.thread_safe, bool):
        raise TypeError(f"thread_safe must be True or False for {metric_name}")

    configuration = configuration or EvaluationConfiguration()
    matcher = configuration.matcher
    if isinstance(metric, LLMMetric) and not isinstance(matcher, WholeValue):
        # An empty value has no items on the item path: the LLM metric checks
        # its prompts and context, and _score_items checks each item.
        metric.validate_evaluation_inputs(
            predictions, references, context, prompts, whole_values=False
        )
    else:
        metric.validate_evaluation_inputs(predictions, references, context, prompts)
    if isinstance(matcher, WholeValue):
        # Multiplicity is for the whole path, and the matcher replaces it on
        # the item path: there the matcher reads the items of each role and
        # compute gets one item per role, so the item counts do not apply.
        metric._validate_item_counts(predictions, references)
    prediction_count = len(predictions)

    aggregation = configuration.resolve(metric, prediction_count)
    _check_matcher(metric, matcher, aggregation.method, predictions, references)

    def _omitted(role: str) -> bool:
        return getattr(metric, role, None) == InputPolicy.OMIT

    effective_references = [] if _omitted("reference_policy") else references
    effective_prompts = None if _omitted("prompt_policy") else prompts
    effective_context = None if _omitted("context_policy") else context
    samples = _build_samples(
        predictions,
        effective_references,
        effective_context,
        effective_prompts,
        prediction_count,
    )
    client = _LazyClient(client_factory or BedrockClient)
    cutoff = _Cutoff()

    def score(sample: EvaluationSample) -> SampleResult:
        if isinstance(matcher, WholeValue):
            return _score_with_retries(
                metric, sample, client, configuration, sleeper, cutoff
            )
        return _score_items(
            metric,
            sample,
            matcher,
            aggregation.thresholds[0],
            client,
            configuration,
            sleeper,
            cutoff,
        )

    workers = min(configuration.max_workers, len(samples))
    # Only a metric that declares itself thread-safe gets several workers.
    if workers > 1 and metric.thread_safe:
        outcomes = _score_concurrently(samples, score, workers, cutoff)
    else:
        outcomes = _score_sequentially(samples, score)

    sample_results = _fail_fast_results(samples, outcomes)
    aggregate = aggregate_samples(sample_results, aggregation)
    return MetricEvaluationResult(
        metric_name=metric_name,
        samples=sample_results,
        aggregate=aggregate,
    )


def _score_sequentially(
    samples: tuple[EvaluationSample, ...],
    score: Callable[[EvaluationSample], SampleResult],
) -> dict[int, SampleResult | Exception]:
    """Score the samples one after another on the calling thread.

    Returns the result or the error of each scored sample by ordinal; the
    first error stops the scoring.
    """
    outcomes: dict[int, SampleResult | Exception] = {}
    for sample in samples:
        try:
            outcomes[sample.ordinal] = score(sample)
        except Exception as error:
            outcomes[sample.ordinal] = error
            break
    return outcomes


def _score_concurrently(
    samples: tuple[EvaluationSample, ...],
    score: Callable[[EvaluationSample], SampleResult],
    workers: int,
    cutoff: _Cutoff,
) -> dict[int, SampleResult | Exception]:
    """Score up to ``workers`` samples at the same time.

    Returns the result or the error of each scored sample by ordinal. Samples
    are submitted in input order, the next one as soon as a worker is free. A
    failure sets ``cutoff``: no sample is submitted after it, and a later
    sample stops before its next attempt, though a call in flight runs to its
    end, as a thread cannot be stopped. Every sample before the first failure
    in input order was submitted earlier and has finished, so
    ``_fail_fast_results`` gives the results of scoring one sample after
    another.
    """
    outcomes: dict[int, SampleResult | Exception] = {}
    running: dict[Future, EvaluationSample] = {}
    upcoming = 0  # the ordinal of the next sample to submit
    executor = ThreadPoolExecutor(max_workers=workers)
    try:
        while True:
            while (
                cutoff.ordinal is None
                and upcoming < len(samples)
                and len(running) < workers
            ):
                sample = samples[upcoming]
                running[executor.submit(score, sample)] = sample
                upcoming += 1
            if not running:
                break
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in done:
                sample = running.pop(future)
                try:
                    outcomes[sample.ordinal] = future.result()
                except _SampleSkipped:
                    pass  # an earlier sample failed: this one is SKIPPED
                except Exception as error:
                    # Only an Exception fails the sample. A BaseException,
                    # such as KeyboardInterrupt, propagates, as with one worker.
                    outcomes[sample.ordinal] = error
                    cutoff.record(sample.ordinal)
    except BaseException:
        # Return at once: start nothing more, stop every running sample before
        # its next attempt, and let the calls in flight end on their own.
        cutoff.stop_all()
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    executor.shutdown()
    return outcomes


def _fail_fast_results(
    samples: tuple[EvaluationSample, ...],
    outcomes: dict[int, SampleResult | Exception],
) -> tuple[SampleResult, ...]:
    """Keep the results up to the first error in input order, then stop.

    That sample is FAILED and every later sample is SKIPPED, even one that
    completed on another worker, so the aggregate is explicitly failed.
    """
    results: list[SampleResult] = []
    for index, sample in enumerate(samples):
        outcome = outcomes[sample.ordinal]
        if isinstance(outcome, SampleResult):
            results.append(outcome)
            continue
        results.append(
            SampleResult(
                sample_id=sample.sample_id,
                ordinal=sample.ordinal,
                status=SampleStatus.FAILED,
                error=_safe_error(outcome),
            )
        )
        results.extend(
            SampleResult(
                sample_id=remaining.sample_id,
                ordinal=remaining.ordinal,
                status=SampleStatus.SKIPPED,
            )
            for remaining in samples[index + 1 :]
        )
        break
    return tuple(results)


def _check_matcher(
    metric: Metric,
    matcher: Matcher,
    method: str,
    predictions: list,
    references: list | None,
) -> None:
    """Fail before the first sample when the matcher does not fit the metric or the data."""
    if isinstance(matcher, WholeValue):
        if method == "prf":
            raise ValueError("prf requires the labels or list matcher")
        return
    if method != "prf":
        raise ValueError(
            "the labels and list matchers require the prf aggregation method"
        )
    if isinstance(metric, LLMMetric) and not isinstance(metric, LLMBatchMetric):
        raise ValueError(f"{metric.name} accepts only the whole matcher")
    if getattr(metric, "reference_policy", None) == InputPolicy.OMIT:
        raise ValueError("prf requires a metric that reads references")
    # The matcher replaces the multiplicity on the item path: it pairs the
    # items of each class by score and counts every unpaired item. A metric
    # whose items pair by position or are alternatives cannot keep that
    # reading there.
    rules = {
        "prediction": metric._validate_multiplicity(
            "prediction", metric.prediction_multiplicity
        ),
        "reference": metric._validate_multiplicity(
            "reference", metric.reference_multiplicity
        ),
    }
    if Multiplicity.ALIGNED in rules.values():
        raise ValueError(
            f"{metric.name} accepts only the whole matcher: it pairs prediction "
            "and reference items by position, and the labels and list matchers "
            "pair them by score"
        )
    for role, rule in rules.items():
        if rule is Multiplicity.ANY_ACCEPTABLE:
            raise ValueError(
                f"{metric.name} accepts only the whole matcher: it reads several "
                f"{role} items as alternatives, and the labels and list matchers "
                "read them as separate items"
            )
    # The matchers read text values. An image item would reach the metric,
    # and the judge, as the text of its bytes.
    for role, values in (("prediction", predictions), ("reference", references)):
        for index, value in enumerate(values or ()):
            for item in _as_items(value):
                if item is not None and item.modality is not Modality.TEXT:
                    raise ValueError(
                        "the labels and list matchers read only text, not "
                        f"{item.modality} data ({role} {index}); use the whole matcher"
                    )


def _score_with_retries(
    metric: Metric,
    sample: EvaluationSample,
    client: BedrockClient,
    configuration: EvaluationConfiguration,
    sleeper: Callable[[float], None],
    cutoff: _Cutoff,
) -> SampleResult:
    """Score one whole sample: the metric score is the sample score."""
    output = _call_with_retries(
        lambda: metric.compute(sample, client), sample, configuration, sleeper, cutoff
    )
    if isinstance(output, SampleOutput):
        return SampleResult(
            sample_id=sample.sample_id,
            ordinal=sample.ordinal,
            status=SampleStatus.COMPLETED,
            score=output.score,
            details=output.details,
        )
    return SampleResult(
        sample_id=sample.sample_id,
        ordinal=sample.ordinal,
        status=SampleStatus.COMPLETED,
        score=output,
    )


def _score_items(
    metric: Metric,
    sample: EvaluationSample,
    matcher: Matcher,
    threshold: float,
    client: BedrockClient,
    configuration: EvaluationConfiguration,
    sleeper: Callable[[float], None],
    cutoff: _Cutoff,
) -> SampleResult:
    """Score every candidate pair of items of each class and record the matches.

    An ``LLMBatchMetric`` scores a class with two or more items on one side with
    ``compute_matrix``; every other class gets one ``compute`` call per pair.
    The sample score is the F1 of the sample at ``threshold``:
    ``2 * pairs / (n_pred + n_ref)``, or 1.0 when no side has items.
    """
    predicted = _matcher_items(matcher, "prediction", sample.predictions)
    expected = _matcher_items(matcher, "reference", sample.references)
    if isinstance(metric, LLMMetric):
        # The empty-content check of an LLM metric, applied to each item: a
        # blank item fails the sample instead of reaching the judge.
        for role, entries in (("prediction", predicted), ("reference", expected)):
            for index, (_, item) in enumerate(entries):
                if not metric._has_usable_content(item):
                    raise ValueError(f"{role} item {index} is empty")
    matches = {}
    for item_class in sorted({c for c, _ in predicted} | {c for c, _ in expected}):
        rows = [item for c, item in predicted if c == item_class]
        columns = [item for c, item in expected if c == item_class]
        if isinstance(metric, LLMBatchMetric) and len(rows) * len(columns) > 1:
            scores = _matrix_scores(
                metric, sample, rows, columns, client, configuration, sleeper, cutoff
            )
        else:
            scores = []
            for prediction in rows:
                row = []
                for reference in columns:
                    pair = EvaluationSample(
                        sample_id=sample.sample_id,
                        ordinal=sample.ordinal,
                        prediction=prediction,
                        reference=reference,
                        prompt=sample.prompts,
                        contexts=sample.contexts,
                    )
                    output = _call_with_retries(
                        lambda: metric.compute(pair, client),
                        pair,
                        configuration,
                        sleeper,
                        cutoff,
                    )
                    row.append(
                        output.score if isinstance(output, SampleOutput) else output
                    )
                scores.append(tuple(row))
        matches[item_class] = Matches(
            scores=tuple(scores), n_pred=len(rows), n_ref=len(columns)
        )

    item_count = sum(record.n_pred + record.n_ref for record in matches.values())
    pairs = sum(record.pair_count(threshold) for record in matches.values())
    return SampleResult(
        sample_id=sample.sample_id,
        ordinal=sample.ordinal,
        status=SampleStatus.COMPLETED,
        score=2 * pairs / item_count if item_count else 1.0,
        matches=matches,
    )


def _matcher_items(
    matcher: Matcher, role: str, items: tuple[DataType, ...]
) -> list[tuple[str, DataType]]:
    """Return the matcher items of one role; an item error names the role.

    Several items in a role are one value, read by the rule of
    ``Metric._raw_value``: the matcher gets the list of their raw values, as
    it gets one list-valued item.
    """
    if len(items) > 1:
        value = TextData(Metric._raw_value(items))
    else:
        value = items[0] if items else None
    try:
        return matcher.items(value)
    except ValueError as error:
        raise ValueError(f"{role} {error}") from error


def _matrix_scores(
    metric: LLMBatchMetric,
    sample: EvaluationSample,
    rows: list[DataType],
    columns: list[DataType],
    client: BedrockClient,
    configuration: EvaluationConfiguration,
    sleeper: Callable[[float], None],
    cutoff: _Cutoff,
) -> list[list[float]]:
    """Score one class with one ``compute_matrix`` call for each block of items."""
    scores = [[] for _ in rows]
    for top in range(0, len(rows), MATRIX_BLOCK_SIZE):
        block_rows = rows[top : top + MATRIX_BLOCK_SIZE]
        for left in range(0, len(columns), MATRIX_BLOCK_SIZE):
            block_columns = columns[left : left + MATRIX_BLOCK_SIZE]
            block = _call_with_retries(
                lambda: metric.compute_matrix(
                    sample, block_rows, block_columns, client
                ),
                sample,
                configuration,
                sleeper,
                cutoff,
            )
            if not (
                isinstance(block, list)
                and len(block) == len(block_rows)
                and all(
                    isinstance(row, list) and len(row) == len(block_columns)
                    for row in block
                )
            ):
                raise ValueError(
                    f"judge matrix must have {len(block_rows)} rows and "
                    f"{len(block_columns)} columns"
                )
            for offset, row in enumerate(block):
                scores[top + offset].extend(row)
    return scores


def _call_with_retries(
    call: Callable[[], object],
    sample: EvaluationSample,
    configuration: EvaluationConfiguration,
    sleeper: Callable[[float], None],
    cutoff: _Cutoff,
):
    """Run one metric call for ``sample`` and retry only eligible transient failures.

    Before each attempt and each wait, ``cutoff`` stops a sample that an
    earlier failed sample has made SKIPPED.
    """
    for failed_attempt in range(1, configuration.max_attempts + 1):
        cutoff.check(sample)
        try:
            return call()
        except Exception as error:
            is_transient, category = _classify_failure(error)
            if not is_transient or failed_attempt == configuration.max_attempts:
                raise
            cutoff.check(sample)

            delay = min(
                configuration.max_delay,
                configuration.initial_delay
                * configuration.multiplier ** (failed_attempt - 1),
            )
            # The category, not the message: the message can quote dataset
            # content, and the envelope reports it if the sample fails.
            logger.warning(
                "Sample %s attempt %d/%d failed with %s; retrying in %s seconds",
                sample.sample_id,
                failed_attempt,
                configuration.max_attempts,
                category,
                delay,
            )
            sleeper(delay)

    raise AssertionError("retry loop exited without a result")


def _build_samples(
    predictions: list[DataType],
    references: list[DataType],
    context: list[list[DataType]] | None,
    prompts: list[DataType] | None,
    prediction_count: int,
) -> tuple[EvaluationSample, ...]:
    context_sources = context or []
    return tuple(
        EvaluationSample(
            sample_id=str(index),
            ordinal=index,
            prediction=predictions[index],
            reference=references[index] if references else None,
            prompt=prompts[index] if prompts else None,
            contexts=tuple(
                item for source in context_sources for item in _as_items(source[index])
            ),
        )
        for index in range(prediction_count)
    )


_TRANSIENT_CLIENT_ERROR_CODES = frozenset(
    {
        "ThrottlingException",
        "ModelTimeoutException",
        "ServiceUnavailableException",
        "InternalServerException",
        "ModelNotReadyException",
    }
)
_TRANSIENT_TRANSPORT_EXCEPTIONS = (
    EndpointConnectionError,
    ConnectTimeoutError,
    ReadTimeoutError,
    ConnectionClosedError,
)


def _safe_type_name(error: BaseException) -> str:
    """Return a non-empty exception type name without invoking the instance."""
    try:
        type_name = type(error).__name__
        if isinstance(type_name, str):
            type_name = type_name.strip()
            if type_name:
                return type_name
    except Exception:
        pass
    return "Exception"


def _safe_client_error_code(error: ClientError) -> str | None:
    """Extract plain dictionary-shaped ClientError metadata conservatively."""
    try:
        response = error.response
        if not isinstance(response, dict):
            return None
        error_metadata = response.get("Error")
        if not isinstance(error_metadata, dict):
            return None
        code = error_metadata.get("Code")
        if type(code) is not str:
            return None
        return code
    except Exception:
        return None


def _classify_failure(error: Exception) -> tuple[bool, str]:
    """Return whether a failure is explicitly transient and its safe category."""
    if isinstance(error, ClientError):
        code = _safe_client_error_code(error)
        if code is None:
            return False, "bedrock:ClientError"
        return code in _TRANSIENT_CLIENT_ERROR_CODES, f"bedrock:{code}"

    if isinstance(error, _TRANSIENT_TRANSPORT_EXCEPTIONS):
        return True, f"transport:{_safe_type_name(error)}"

    return False, f"exception:{_safe_type_name(error)}"


# The longest error message a result or envelope carries. A message can quote
# a dataset value or a request of any size.
MAX_ERROR_LENGTH = 500


def _safe_error(error: Exception) -> str:
    """Render a caught failure without allowing diagnostics to mask it.

    The text is the stripped ``str(error)``, else the type name. A text longer
    than ``MAX_ERROR_LENGTH`` keeps its start and ends with `` [truncated]``,
    in ``MAX_ERROR_LENGTH`` characters.
    """
    try:
        message = str(error)
        if isinstance(message, str):
            message = message.strip()
            if message:
                if len(message) > MAX_ERROR_LENGTH:
                    marker = " [truncated]"
                    message = message[: MAX_ERROR_LENGTH - len(marker)] + marker
                return message
    except Exception:
        pass
    return _safe_type_name(error)


__all__ = [
    "EvaluationConfiguration",
    "EvaluationSample",
    "MetricEvaluationResult",
]
