# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Characterization and regression tests for explicit aggregation."""

import dataclasses
import math
from fractions import Fraction
from unittest.mock import Mock

import numpy as np
import pytest

import metric_library.metrics  # noqa: F401 - registers built-in metrics
from metric_library.core.aggregation import (
    AggregationResult,
    AggregationSpec,
    EvaluationStatus,
    Matches,
    SampleOutput,
    SampleResult,
    SampleStatus,
    median,
    numeric_mean,
    prf,
    quantile,
    weighted_mean,
)
from metric_library.core.base import Metric
from metric_library.core.data.text_data import TextData
from media_double import images
from metric_library.core.evaluation import (
    EvaluationConfiguration,
    MetricEvaluationResult,
)
from metric_library.core.matching import LabelSet
from metric_library.core.registry import MetricRegistry


class TestRmseDefinition:
    """Lock the RMSE aggregate: the mean of the per-sample values."""

    # Two four-value arrays against zeros: one large error in the first pair, a
    # constant error in the second.
    PREDICTIONS = (np.array([0.0, 0.0, 0.0, 4.0]), np.array([3.0, 3.0, 3.0, 3.0]))
    REFERENCES = (np.zeros(4), np.zeros(4))

    @staticmethod
    def _compute(name, predictions, references, **kwargs):
        # Deterministic metrics never touch the client, so a spec-less Mock
        # keeps the engine off Bedrock while still satisfying client_factory.
        return MetricRegistry.compute_metric(
            name,
            images(predictions),
            images(references),
            client_factory=Mock(spec=[]),
            **kwargs,
        )

    def test_rmse_is_the_mean_of_the_per_sample_rmse_values(self):
        result = self._compute(
            "RootMeanSquaredError", self.PREDICTIONS, self.REFERENCES
        )

        assert [sample.score for sample in result.samples] == [2.0, 3.0]
        assert result.aggregate.method == "numeric_mean"
        assert result.score == 2.5
        assert result.aggregate == numeric_mean(result.samples)

    def test_rmse_is_the_square_root_of_mse_per_sample_not_in_aggregate(self):
        rmse = self._compute("RootMeanSquaredError", self.PREDICTIONS, self.REFERENCES)
        mse = self._compute("MeanSquaredError", self.PREDICTIONS, self.REFERENCES)
        mae = self._compute("MeanAbsoluteError", self.PREDICTIONS, self.REFERENCES)

        assert [sample.score for sample in mse.samples] == [4.0, 9.0]
        assert mse.score == 6.5
        for rmse_sample, mse_sample in zip(rmse.samples, mse.samples):
            assert rmse_sample.score == pytest.approx(math.sqrt(mse_sample.score))
        # The mean of the per-sample values is below the pooled RMSE (2.5495).
        assert rmse.score == 2.5
        assert rmse.score < math.sqrt(mse.score)
        assert [sample.score for sample in mae.samples] == [1.0, 3.0]
        assert mae.score == 2.0


class PrfByDefaultMetric(Metric):
    """Fixture: 1.0 for equal text, else 0.0; the default aggregator is prf."""

    default_aggregation_method = "prf"

    def compute(self, sample, client):
        prediction = sample.prediction.get_raw_data()
        return float(prediction == sample.reference.get_raw_data())


class TestMetricDefaultAggregator:
    """A metric's default aggregator yields to an explicit request."""

    def test_prf_default_yields_to_a_caller_override(self, monkeypatch):
        monkeypatch.setitem(
            MetricRegistry.metrics, "prfbydefaultmetric", PrfByDefaultMetric
        )
        predictions = [TextData("a"), TextData("b")]
        references = [TextData("a"), TextData("c")]

        # The prf default needs an item matcher from the caller.
        default = MetricRegistry.compute_metric(
            "PrfByDefaultMetric",
            predictions,
            references,
            configuration=EvaluationConfiguration(matcher=LabelSet()),
            client_factory=Mock(spec=[]),
        )
        override = MetricRegistry.compute_metric(
            "PrfByDefaultMetric",
            predictions,
            references,
            configuration=EvaluationConfiguration(aggregation_method="numeric_mean"),
            client_factory=Mock(spec=[]),
        )

        assert default.aggregate.method == "prf"
        assert default.score == pytest.approx(1 / 3)
        assert override.aggregate.method == "numeric_mean"
        assert override.score == 0.5
        assert override.aggregate.values is None


def _completed(ordinal, score):
    return SampleResult(
        sample_id=str(ordinal),
        ordinal=ordinal,
        status=SampleStatus.COMPLETED,
        score=score,
    )


def _failed(ordinal, error="judge failed"):
    return SampleResult(
        sample_id=str(ordinal),
        ordinal=ordinal,
        status=SampleStatus.FAILED,
        error=error,
    )


def _skipped(ordinal):
    return SampleResult(
        sample_id=str(ordinal),
        ordinal=ordinal,
        status=SampleStatus.SKIPPED,
    )


def _completed_aggregate(score):
    return AggregationResult(
        method="numeric_mean",
        status=EvaluationStatus.COMPLETED,
        score=score,
        total_count=1,
        succeeded_count=1,
        failed_count=0,
        skipped_count=0,
        denominator=1,
        coverage=1.0,
    )


def _prf_aggregate(**overrides):
    """A one-sample completed ``prf`` aggregate; ``values`` is left out."""
    kwargs = dict(
        method="prf",
        status=EvaluationStatus.COMPLETED,
        score=0.5,
        total_count=1,
        succeeded_count=1,
        failed_count=0,
        skipped_count=0,
        denominator=1,
        coverage=1.0,
        spec=AggregationSpec(method="prf"),
    )
    kwargs.update(overrides)
    return AggregationResult(**kwargs)


# The match records of one class with one item: a pair with the full score, a
# predicted item without a reference item, a reference item without a
# predicted item.
TP = Matches(((1.0,),), 1, 1)
FP = Matches(((),), 1, 0)
FN = Matches((), 0, 1)


def _match_sample(ordinal, score, matches):
    """A completed sample carrying the match record of each of its classes."""
    return SampleResult(
        sample_id=str(ordinal),
        ordinal=ordinal,
        status=SampleStatus.COMPLETED,
        score=score,
        matches=matches,
    )


# The small fixture ``A, B, C`` vs ``A, B, D`` as per-sample match records.
SMALL_FIXTURE = (
    _match_sample(0, 1.0, {"a": TP}),
    _match_sample(1, 1.0, {"b": TP}),
    _match_sample(2, 0.0, {"c": FP, "d": FN}),
)

# ``a, a, a`` vs ``a, b, b`` as per-sample match records: the macro precision
# (1/6) is not the macro recall (0.5).
UNEVEN_FIXTURE = (
    _match_sample(0, 1.0, {"a": TP}),
    _match_sample(1, 0.0, {"a": FP, "b": FN}),
    _match_sample(2, 0.0, {"a": FP, "b": FN}),
)


def _assert_counts(result, *, total, succeeded, failed, skipped):
    assert result.total_count == total
    assert result.succeeded_count == succeeded
    assert result.failed_count == failed
    assert result.skipped_count == skipped
    assert result.total_count == succeeded + failed + skipped
    assert result.denominator == succeeded
    assert result.coverage == succeeded / total


class TestSampleResultContracts:
    """Reject impossible sample result states at construction."""

    @pytest.mark.parametrize(
        "kwargs",
        [
            {
                "status": SampleStatus.COMPLETED,
                "score": 0.5,
                "error": "unexpected",
            },
            {"status": SampleStatus.FAILED, "score": 0.5, "error": "failed"},
            {"status": SampleStatus.FAILED, "error": "   "},
            {"status": SampleStatus.SKIPPED, "error": "not an error"},
            {"status": SampleStatus.SKIPPED, "score": 0.5},
        ],
    )
    def test_invalid_state_is_rejected(self, kwargs):
        with pytest.raises(ValueError):
            SampleResult(sample_id="0", ordinal=0, **kwargs)

    @pytest.mark.parametrize(
        ("score", "exception"),
        [
            (True, TypeError),
            ("0.5", TypeError),
            (float("nan"), ValueError),
            (float("inf"), ValueError),
            (float("-inf"), ValueError),
        ],
    )
    def test_invalid_completed_score_is_rejected(self, score, exception):
        with pytest.raises(exception):
            _completed(0, score)

    @pytest.mark.parametrize("score", [-0.01, 1.01, 2.5])
    def test_finite_score_outside_unit_interval_is_accepted(self, score):
        result = SampleResult("0", 0, SampleStatus.COMPLETED, score=score)

        assert result.score == score

    def test_completed_details_are_stored_as_a_dict_copy(self):
        source = {"outcomes": {"a": "TP"}}

        result = SampleResult("0", 0, SampleStatus.COMPLETED, score=1.0, details=source)
        source["extra"] = "mutated afterwards"

        assert result.details == {"outcomes": {"a": "TP"}}
        assert result.details is not source

    def test_details_default_to_none(self):
        result = SampleResult("0", 0, SampleStatus.COMPLETED, score=1.0)

        assert result.details is None

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"status": SampleStatus.FAILED, "error": "failed"},
            {"status": SampleStatus.SKIPPED},
        ],
        ids=["failed", "skipped"],
    )
    def test_non_completed_samples_reject_details(self, kwargs):
        with pytest.raises(
            ValueError, match="non-completed samples cannot contain details"
        ):
            SampleResult("0", 0, details={"outcomes": {}}, **kwargs)

    @pytest.mark.parametrize(
        ("details", "message"),
        [
            (["x"], "details must be a mapping"),
            ({1: "x"}, "details keys must be strings"),
        ],
        ids=["list", "non-string-key"],
    )
    def test_invalid_details_are_rejected(self, details, message):
        with pytest.raises(TypeError, match=message):
            SampleResult("0", 0, SampleStatus.COMPLETED, score=1.0, details=details)

    @pytest.mark.parametrize(
        ("kwargs", "exception", "message"),
        [
            pytest.param(
                {
                    "status": SampleStatus.FAILED,
                    "error": "failed",
                    "matches": {"a": TP},
                },
                ValueError,
                "non-completed samples cannot contain matches",
                id="failed-sample",
            ),
            pytest.param(
                {
                    "status": SampleStatus.COMPLETED,
                    "score": 1.0,
                    "matches": {"a": "TP"},
                },
                TypeError,
                "matches must map class names to Matches",
                id="non-matches-value",
            ),
        ],
    )
    def test_invalid_matches_are_rejected(self, kwargs, exception, message):
        with pytest.raises(exception, match=message):
            SampleResult("0", 0, **kwargs)


class TestSampleOutput:
    """Validate the score and the details that ``compute`` may return."""

    def test_stores_the_score_and_the_details(self):
        output = SampleOutput(0.5, {"outcomes": {}})

        assert output.score == 0.5
        assert output.details == {"outcomes": {}}

    def test_score_is_normalized_to_float(self):
        output = SampleOutput(Fraction(1, 2), {})

        assert output.score == 0.5
        assert type(output.score) is float

    @pytest.mark.parametrize(
        ("score", "details", "exception", "message"),
        [
            ("0.5", {}, TypeError, "score must be a real number"),
            (math.nan, {}, ValueError, "score must be finite"),
            (1.0, "x", TypeError, "details must be a mapping"),
            (1.0, {1: 2}, TypeError, "details keys must be strings"),
        ],
        ids=["string-score", "nan-score", "string-details", "non-string-key"],
    )
    def test_invalid_score_or_details_is_rejected(
        self, score, details, exception, message
    ):
        with pytest.raises(exception, match=message):
            SampleOutput(score, details)


class TestMatches:
    """Validate the match record of one class and count its pairs."""

    def test_rows_are_normalized_to_tuples(self):
        record = Matches([[1.0, 0.5], [0.0, 0.25]], 2, 2)

        assert record.scores == ((1.0, 0.5), (0.0, 0.25))

    @pytest.mark.parametrize(
        ("scores", "n_pred", "n_ref", "exception", "message"),
        [
            pytest.param(
                ((1.0,),),
                2,
                1,
                ValueError,
                "scores must have n_pred rows of n_ref scores",
                id="missing-row",
            ),
            pytest.param(
                ((1.0,),),
                1,
                2,
                ValueError,
                "scores must have n_pred rows of n_ref scores",
                id="short-row",
            ),
            pytest.param(
                (),
                -1,
                0,
                ValueError,
                "n_pred must be non-negative",
                id="negative-n-pred",
            ),
            pytest.param(
                (), 0, -1, ValueError, "n_ref must be non-negative", id="negative-n-ref"
            ),
            pytest.param(
                ((1.0,),),
                True,
                1,
                TypeError,
                "n_pred must be an integer",
                id="bool-n-pred",
            ),
            pytest.param(
                ((1.0,),),
                1,
                True,
                TypeError,
                "n_ref must be an integer",
                id="bool-n-ref",
            ),
            pytest.param(
                ((3.2,),),
                1,
                1,
                ValueError,
                "pair score must be between 0.0 and 1.0, got 3.2",
                id="score-above-one",
            ),
            pytest.param(
                ((-0.1,),),
                1,
                1,
                ValueError,
                "pair score must be between 0.0 and 1.0, got -0.1",
                id="score-below-zero",
            ),
            pytest.param(
                ((math.nan,),), 1, 1, ValueError, "score must be finite", id="nan-score"
            ),
            pytest.param(
                ((math.inf,),), 1, 1, ValueError, "score must be finite", id="inf-score"
            ),
        ],
    )
    def test_invalid_record_is_rejected(
        self, scores, n_pred, n_ref, exception, message
    ):
        with pytest.raises(exception) as exc:
            Matches(scores, n_pred, n_ref)

        assert str(exc.value) == message

    # In each 2 x 2 matrix, the pairs with the highest total score are not the
    # most pairs at the threshold.
    @pytest.mark.parametrize(
        ("scores", "threshold", "expected"),
        [
            pytest.param(((0.99, 0.49), (0.49, 0.0)), 0.4, 2, id="two-pairs-at-0.4"),
            pytest.param(((0.99, 0.49), (0.49, 0.0)), 0.5, 1, id="one-pair-at-0.5"),
            pytest.param(
                ((0.6, 0.45), (0.45, 0.0)), 0.5, 1, id="lower-pairs-do-not-block"
            ),
            pytest.param(((1.0, 0.62), (0.62, 0.38)), 0.5, 2, id="two-pairs-at-0.5"),
            pytest.param(((1.0, 0.62), (0.62, 0.38)), 0.8, 1, id="one-pair-at-0.8"),
            pytest.param(((), ()), 0.5, 0, id="predicted-items-only"),
        ],
    )
    def test_pair_count_is_the_most_one_to_one_pairs_at_the_threshold(
        self, scores, threshold, expected
    ):
        record = Matches(scores, len(scores), len(scores[0]))

        assert record.pair_count(threshold) == expected


class TestNumericMean:
    """Prove explicit denominator, coverage, and stop-on-failure accounting."""

    def test_all_success_uses_exact_ordered_builtin_mean(self):
        scores = [1.0, 1e-16, 1e-16]

        result = numeric_mean(
            tuple(_completed(index, score) for index, score in enumerate(scores))
        )

        assert result.status is EvaluationStatus.COMPLETED
        assert result.score == sum(scores) / len(scores)
        _assert_counts(result, total=3, succeeded=3, failed=0, skipped=0)

    def test_failure_retains_counts_and_earlier_successes_without_score(self):
        results = (_completed(0, 0.2), _failed(1), _skipped(2), _skipped(3))

        result = numeric_mean(results)

        assert result.status is EvaluationStatus.FAILED
        assert result.score is None
        assert results[0].score == 0.2
        _assert_counts(result, total=4, succeeded=1, failed=1, skipped=2)

    def test_first_sample_failure_has_zero_denominator_and_coverage(self):
        result = numeric_mean((_failed(0), _skipped(1)))

        assert result.status is EvaluationStatus.FAILED
        assert result.score is None
        _assert_counts(result, total=2, succeeded=0, failed=1, skipped=1)

    def test_empty_results_are_rejected(self):
        with pytest.raises(ValueError, match="at least one sample result"):
            numeric_mean(())

    def test_skips_require_a_prior_failure(self):
        with pytest.raises(ValueError, match="prior failure"):
            numeric_mean((_skipped(0), _failed(1)))

    def test_only_one_failure_is_allowed(self):
        with pytest.raises(ValueError, match="only one failed result"):
            numeric_mean((_failed(0), _failed(1)))

    def test_completed_results_cannot_follow_a_failure(self):
        with pytest.raises(ValueError, match="completed results after failure"):
            numeric_mean((_completed(0, 0.2), _failed(1), _completed(2, 0.8)))

    def test_results_must_have_contiguous_ordinals_and_unique_ids(self):
        with pytest.raises(ValueError, match="contiguous ordinal"):
            numeric_mean((_completed(1, 0.5),))

        duplicate_id = SampleResult(
            sample_id="same",
            ordinal=0,
            status=SampleStatus.COMPLETED,
            score=0.5,
        )
        second_duplicate = SampleResult(
            sample_id="same",
            ordinal=1,
            status=SampleStatus.COMPLETED,
            score=0.5,
        )
        with pytest.raises(ValueError, match="IDs must be unique"):
            numeric_mean((duplicate_id, second_duplicate))


class TestAggregationResultContracts:
    """Reject aggregate states the public aggregator can never produce."""

    @pytest.mark.parametrize(
        ("succeeded_count", "failed_count", "skipped_count"),
        [
            (2, 0, 1),
            (0, 2, 1),
            (3, 0, 0),
        ],
    )
    def test_failed_status_requires_exactly_one_failed_sample(
        self, succeeded_count, failed_count, skipped_count
    ):
        with pytest.raises(ValueError, match="exactly one failed sample"):
            AggregationResult(
                method="numeric_mean",
                status=EvaluationStatus.FAILED,
                score=None,
                total_count=3,
                succeeded_count=succeeded_count,
                failed_count=failed_count,
                skipped_count=skipped_count,
                denominator=succeeded_count,
                coverage=succeeded_count / 3,
            )

    def test_completed_status_requires_every_sample_to_succeed(self):
        with pytest.raises(ValueError, match="every sample to succeed"):
            AggregationResult(
                method="numeric_mean",
                status=EvaluationStatus.COMPLETED,
                score=0.5,
                total_count=2,
                succeeded_count=1,
                failed_count=1,
                skipped_count=0,
                denominator=1,
                coverage=0.5,
            )

    def test_failed_status_cannot_contain_a_score(self):
        with pytest.raises(ValueError, match="cannot contain a score"):
            AggregationResult(
                method="numeric_mean",
                status=EvaluationStatus.FAILED,
                score=0.5,
                total_count=2,
                succeeded_count=1,
                failed_count=1,
                skipped_count=0,
                denominator=1,
                coverage=0.5,
            )

    @pytest.mark.parametrize(
        ("score", "exception"),
        [
            (True, TypeError),
            ("0.5", TypeError),
            (float("nan"), ValueError),
            (float("inf"), ValueError),
            (float("-inf"), ValueError),
        ],
    )
    def test_invalid_completed_score_is_rejected(self, score, exception):
        with pytest.raises(exception):
            _completed_aggregate(score)

    @pytest.mark.parametrize("score", [-0.01, 1.01, 2.5])
    def test_finite_score_outside_unit_interval_is_accepted(self, score):
        result = _completed_aggregate(score)

        assert result.status is EvaluationStatus.COMPLETED
        assert result.score == score

    def test_coverage_rejects_boolean(self):
        with pytest.raises(TypeError, match="coverage must be a real number"):
            AggregationResult(
                method="numeric_mean",
                status=EvaluationStatus.COMPLETED,
                score=0.5,
                total_count=1,
                succeeded_count=1,
                failed_count=0,
                skipped_count=0,
                denominator=1,
                coverage=True,
            )

    def test_numeric_fields_are_normalized_to_float(self):
        result = AggregationResult(
            method="numeric_mean",
            status=EvaluationStatus.COMPLETED,
            score=np.float32(0.5),
            total_count=1,
            succeeded_count=1,
            failed_count=0,
            skipped_count=0,
            denominator=1,
            coverage=np.float64(1.0),
        )

        assert type(result.score) is float
        assert type(result.coverage) is float

    def test_numeric_aggregation_rejects_values(self):
        with pytest.raises(
            ValueError, match="values are allowed only for a completed prf aggregation"
        ):
            AggregationResult(
                method="numeric_mean",
                status=EvaluationStatus.COMPLETED,
                score=0.5,
                total_count=1,
                succeeded_count=1,
                failed_count=0,
                skipped_count=0,
                denominator=1,
                coverage=1.0,
                values={"f1": 0.5},
            )

    def test_completed_prf_aggregation_requires_values(self):
        with pytest.raises(
            ValueError, match="completed prf aggregation requires values"
        ):
            _prf_aggregate()

    def test_prf_values_must_be_a_dict(self):
        with pytest.raises(TypeError, match="values must be a dict"):
            _prf_aggregate(values=["f1"])

    def test_prf_values_f1_must_equal_the_score(self):
        with pytest.raises(ValueError, match=r"values\['f1'\] must equal score"):
            _prf_aggregate(score=0.5, values={"f1": 0.25})

    def test_failed_prf_aggregation_rejects_values(self):
        with pytest.raises(
            ValueError, match="values are allowed only for a completed prf aggregation"
        ):
            _prf_aggregate(
                status=EvaluationStatus.FAILED,
                score=None,
                succeeded_count=0,
                failed_count=1,
                denominator=0,
                coverage=0.0,
                values={"f1": 0.5},
            )

    def test_completed_prf_aggregation_with_matching_values_constructs(self):
        result = _prf_aggregate(score=0.5, values={"f1": 0.5})

        assert result.status is EvaluationStatus.COMPLETED
        assert result.values == {"f1": 0.5}
        assert result.denominator == result.succeeded_count
        assert type(result.denominator) is int


class TestAggregationSpec:
    """Reject unknown methods and ignored or incompatible parameters."""

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"method": "unknown"},
            {"method": "weighted_mean"},
            {"method": "weighted_mean", "weights": (1.0,), "quantile": 0.5},
            {"method": "median", "weights": (1.0,)},
            {"method": "quantile"},
            {"method": "quantile", "quantile": 0.5, "weights": (1.0,)},
        ],
    )
    def test_invalid_configuration_is_rejected(self, kwargs):
        with pytest.raises((TypeError, ValueError)):
            AggregationSpec(**kwargs)

    @pytest.mark.parametrize(
        "value",
        [True, float("nan"), float("inf"), -0.01, 1.01],
    )
    def test_quantile_parameter_is_strict(self, value):
        with pytest.raises((TypeError, ValueError)):
            AggregationSpec(method="quantile", quantile=value)

    def test_prf_constructs_without_parameters(self):
        spec = AggregationSpec(method="prf")

        assert spec.method == "prf"
        assert spec.weights is None
        assert spec.quantile is None

    @pytest.mark.parametrize(
        ("thresholds", "expected"),
        [
            pytest.param(None, (0.5,), id="default"),
            pytest.param((0.75, 0.5, 0.6), (0.75, 0.5, 0.6), id="given-order"),
        ],
    )
    def test_prf_thresholds_default_to_one_half_and_keep_the_given_order(
        self, thresholds, expected
    ):
        spec = AggregationSpec(method="prf", thresholds=thresholds)

        assert spec.thresholds == expected

    @pytest.mark.parametrize(
        ("kwargs", "exception", "message"),
        [
            pytest.param(
                {"method": "prf", "weights": (1.0,)},
                ValueError,
                "prf accepts only thresholds",
                id="weights-for-prf",
            ),
            pytest.param(
                {"method": "prf", "quantile": 0.5},
                ValueError,
                "prf accepts only thresholds",
                id="quantile-for-prf",
            ),
            *[
                pytest.param(
                    {"method": "prf", "thresholds": (threshold,)},
                    ValueError,
                    "thresholds must be greater than 0.0 and at most 1.0",
                    id=f"threshold-{threshold}",
                )
                for threshold in (0.0, 1.5, math.nan)
            ],
            pytest.param(
                {"method": "prf", "thresholds": (0.5, 0.5)},
                ValueError,
                "thresholds must be unique",
                id="duplicate-thresholds",
            ),
            pytest.param(
                {"method": "prf", "thresholds": ()},
                ValueError,
                "thresholds must not be empty",
                id="empty-thresholds",
            ),
            pytest.param(
                {"method": "prf", "thresholds": [0.5]},
                TypeError,
                "thresholds must be a tuple",
                id="list-thresholds",
            ),
            pytest.param(
                {"method": "median", "thresholds": (0.5,)},
                ValueError,
                "median does not accept aggregation parameters",
                id="thresholds-for-median",
            ),
        ],
    )
    def test_prf_parameters_are_rejected_with_exact_messages(
        self, kwargs, exception, message
    ):
        with pytest.raises(exception) as exc:
            AggregationSpec(**kwargs)

        assert str(exc.value) == message

    def test_method_and_result_method_must_match(self):
        with pytest.raises(ValueError, match="method must match"):
            AggregationResult(
                method="median",
                status=EvaluationStatus.COMPLETED,
                score=0.5,
                total_count=1,
                succeeded_count=1,
                failed_count=0,
                skipped_count=0,
                denominator=1,
                coverage=1.0,
            )


class TestWeightedMean:
    """Verify ordinal-aligned positive weights and weight-mass denominator."""

    def test_all_success_uses_weight_mass_denominator(self):
        result = weighted_mean(
            (_completed(0, 0.2), _completed(1, 0.8)),
            (1.0, 3.0),
        )

        assert result.method == "weighted_mean"
        assert result.score == (0.2 * 1.0 + 0.8 * 3.0) / 4.0
        assert result.denominator == 4.0
        assert type(result.denominator) is float
        assert result.spec.weights == (1.0, 3.0)
        assert result.coverage == 1.0

    def test_failure_reports_successful_weight_mass_without_score(self):
        results = (_completed(0, 0.2), _completed(1, 0.8), _failed(2), _skipped(3))

        result = weighted_mean(results, (1.0, 3.0, 100.0, 100.0))

        assert result.status is EvaluationStatus.FAILED
        assert result.score is None
        assert result.denominator == 4.0
        assert result.coverage == 0.5

    def test_first_sample_failure_has_float_zero_denominator(self):
        result = weighted_mean((_failed(0), _skipped(1)), (1.0, 2.0))

        assert result.status is EvaluationStatus.FAILED
        assert result.denominator == 0.0
        assert type(result.denominator) is float

    def test_weight_count_must_match_all_samples(self):
        with pytest.raises(ValueError, match="one-for-one"):
            weighted_mean((_completed(0, 0.5), _completed(1, 0.5)), (1.0,))

    @pytest.mark.parametrize(
        "weight",
        [True, "1", float("nan"), float("inf"), 0.0, -1.0],
    )
    def test_invalid_weight_is_rejected(self, weight):
        with pytest.raises((TypeError, ValueError)):
            weighted_mean((_completed(0, 0.5),), (weight,))

    def test_overflowing_weight_sum_is_rejected(self):
        with pytest.raises(ValueError, match="weight sum must be finite"):
            weighted_mean(
                (_completed(0, 0.5), _completed(1, 0.5)),
                (1e308, 1e308),
            )


class TestMedianAndQuantile:
    """Verify median and inclusive type-7 quantile semantics."""

    @pytest.mark.parametrize(
        ("scores", "expected"),
        [
            ([0.9, 0.1, 0.5], 0.5),
            ([0.9, 0.1], 0.5),
            ([0.4], 0.4),
        ],
    )
    def test_median(self, scores, expected):
        result = median(
            tuple(_completed(index, score) for index, score in enumerate(scores))
        )

        assert result.method == "median"
        assert result.score == expected
        assert result.denominator == len(scores)

    @pytest.mark.parametrize(
        ("q", "expected"),
        [
            (0.0, 0.0),
            (0.25, 0.3),
            (0.5, 0.6),
            (1.0, 1.0),
        ],
    )
    def test_quantile_uses_inclusive_linear_interpolation(self, q, expected):
        scores = [0.0, 0.4, 0.8, 1.0]

        result = quantile(
            tuple(_completed(index, score) for index, score in enumerate(scores)),
            q,
        )

        assert result.method == "quantile"
        assert result.score == pytest.approx(expected)
        assert result.denominator == 4
        assert result.spec.quantile == q

    def test_quantile_singleton_returns_the_value(self):
        for q in (0.0, 0.25, 0.5, 1.0):
            assert quantile((_completed(0, 0.7),), q).score == 0.7


class TestPrfAggregators:
    """Count per-class pairs over completed samples into precision, recall and F1."""

    def test_prf_averages_the_per_class_table_of_the_small_fixture(self):
        result = prf(SMALL_FIXTURE)
        values = result.values

        assert result.method == "prf"
        assert result.status is EvaluationStatus.COMPLETED
        assert values["threshold"] == 0.5
        assert values["precision"] == 0.5
        assert values["recall"] == 0.5
        assert values["f1"] == 0.5
        assert set(values) == {"threshold", "precision", "recall", "f1", "per_class"}
        assert list(values["per_class"]) == ["a", "b", "c", "d"]
        assert values["per_class"]["a"] == {
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "tp": 1,
            "fp": 0,
            "fn": 0,
            "tn": 2,
            "support": 1,
        }
        assert values["per_class"]["b"] == values["per_class"]["a"]
        assert values["per_class"]["c"] == {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "tp": 0,
            "fp": 1,
            "fn": 0,
            "tn": 2,
            "support": 0,
        }
        assert values["per_class"]["d"] == {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "tp": 0,
            "fp": 0,
            "fn": 1,
            "tn": 2,
            "support": 1,
        }
        assert result.score == values["f1"]
        assert result.denominator == 3
        assert result.coverage == 1.0
        _assert_counts(result, total=3, succeeded=3, failed=0, skipped=0)

    def test_per_class_counts_add_up_to_the_completed_count(self):
        per_class = prf(SMALL_FIXTURE).values["per_class"]

        assert set(per_class) == {"a", "b", "c", "d"}
        for counts in per_class.values():
            assert counts["tp"] + counts["fp"] + counts["fn"] + counts["tn"] == 3
            assert counts["support"] == counts["tp"] + counts["fn"]

    def test_failure_reports_the_numeric_mean_counts_without_values(self):
        samples = (
            _match_sample(0, 1.0, {"a": TP}),
            _failed(1),
            _skipped(2),
        )

        result = prf(samples)

        assert result.status is EvaluationStatus.FAILED
        assert result.score is None
        assert result.values is None
        baseline = numeric_mean(samples)
        _assert_counts(result, total=3, succeeded=1, failed=1, skipped=1)
        assert (result.succeeded_count, result.failed_count, result.skipped_count) == (
            baseline.succeeded_count,
            baseline.failed_count,
            baseline.skipped_count,
        )
        assert result.denominator == baseline.denominator
        assert result.coverage == baseline.coverage

    def test_completed_sample_without_matches_is_rejected(self):
        samples = (_completed(0, 1.0), _completed(1, 0.0))

        with pytest.raises(ValueError) as exc:
            prf(samples)

        assert str(exc.value) == "prf requires per-sample matches"

    def test_empty_matches_everywhere_are_rejected(self):
        samples = (_match_sample(0, 1.0, {}), _match_sample(1, 1.0, {}))

        with pytest.raises(ValueError) as exc:
            prf(samples)

        assert str(exc.value) == "no items in any prediction or reference"

    def test_two_thresholds_add_by_threshold_in_the_given_order(self):
        # One pair with the score 0.6: no pair at 0.8, one pair at 0.5.
        samples = (_match_sample(0, 1.0, {"car": Matches(((0.6,),), 1, 1)}),)

        result = prf(samples, thresholds=(0.8, 0.5))
        by_threshold = result.values["by_threshold"]
        top_level = {
            key: value for key, value in result.values.items() if key != "by_threshold"
        }

        assert [entry["threshold"] for entry in by_threshold] == [0.8, 0.5]
        assert [entry["f1"] for entry in by_threshold] == [0.0, 1.0]
        assert top_level == by_threshold[0]
        assert result.score == 0.0
        assert prf(samples, thresholds=(0.5,)).values == by_threshold[1]

    def test_tn_counts_only_the_samples_without_the_class(self):
        # Two predicted and two reference cars in the first sample, no item in
        # the second. One pair reaches 0.5, so tp + fp + fn (3) is more than
        # the sample count (2), and tn is the one sample without cars.
        cars = Matches(((0.9, 0.0), (0.0, 0.3)), 2, 2)
        samples = (_match_sample(0, 0.5, {"car": cars}), _match_sample(1, 1.0, {}))

        counts = prf(samples).values["per_class"]["car"]

        assert cars.pair_count(0.5) == 1
        assert (counts["tp"], counts["fp"], counts["fn"]) == (1, 1, 1)
        assert counts["tn"] == 1

    def test_metric_evaluation_result_accepts_a_matching_prf_aggregate(self):
        aggregate = prf(UNEVEN_FIXTURE)
        assert aggregate.values["precision"] == pytest.approx(1 / 6)
        assert aggregate.values["recall"] == 0.5
        assert aggregate.values["f1"] == pytest.approx(0.25)

        result = MetricEvaluationResult("m", UNEVEN_FIXTURE, aggregate)

        assert result.status is EvaluationStatus.COMPLETED
        assert result.score == aggregate.values["f1"]
        assert result.aggregate == aggregate

    def test_aggregation_result_rejects_a_score_that_differs_from_f1(self):
        aggregate = prf(SMALL_FIXTURE)

        with pytest.raises(ValueError, match=r"values\['f1'\] must equal score"):
            dataclasses.replace(aggregate, score=0.75)
