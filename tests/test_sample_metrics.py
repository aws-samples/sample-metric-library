# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Deterministic metrics through the single engine: one kind, per-sample scores, aggregators, matchers, pinned values."""

from unittest.mock import Mock, call, patch

import numpy as np
import pytest

import metric_library.metrics  # noqa: F401 - registers built-in metrics
from metric_library import (
    EvaluationConfiguration,
    EvaluationSample,
    EvaluationStatus,
    LabelSet,
    ListItems,
    Matches,
    MetricEvaluationResult,
    SampleStatus,
)
from metric_library.core.aggregation import numeric_mean, prf
from metric_library.core.base import Metric
from metric_library.core.data import Modality
from metric_library.core.data.text_data import TextData
from media_double import images
from metric_library.core.llm_metric import LLMMetric
from metric_library.core.registry import MetricRegistry
from metric_library.metrics.match import DynamicMatch
from metric_library.utils.comparison_functions import NormalizationType


@pytest.fixture(autouse=True)
def _forbid_live_aws(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("live AWS is forbidden in tests")

    monkeypatch.setattr("metric_library.core.bedrock_client.boto3.client", fail)
    monkeypatch.setattr("metric_library.core.bedrock_client.boto3.Session", fail)


@pytest.fixture(autouse=True)
def _shared_embedding_client():
    # compute_metric builds a fresh instance per call, and the two semantic match
    # metrics ask the registry for the shared embedding client while they construct.
    with patch.object(
        MetricRegistry, "get_shared_bedrock_client", return_value=EMBEDDING_CLIENT
    ):
        yield


# One fixed vector per word keeps the semantic scores deterministic however many
# times a test embeds them: identical words score 1.0, "bird" vs "fish" scores 0.5.
_EMBEDDINGS = {
    "cat": [1.0, 0.0],
    "dog": [0.0, 1.0],
    "bird": [1.0, 1.0],
    "fish": [1.0, -1.0],
}


def _embed(texts):
    return [_EMBEDDINGS[text] for text in texts]


EMBEDDING_CLIENT = Mock(spec=["get_embeddings"])
EMBEDDING_CLIENT.get_embeddings.side_effect = _embed


def _texts(values):
    return [TextData(value) for value in values]


TEXT_PREDICTIONS = _texts(["cat", "dog", "bird"])
TEXT_REFERENCES = _texts(["cat", "dog", "fish"])
NUMERIC_PREDICTIONS = _texts([1.5, 3.0, 2.0])
NUMERIC_REFERENCES = _texts([1.0, 3.0, 4.5])
# In-memory image data for the error metrics, which take audio, image and
# video data and reject text before scoring.
ARRAY_PREDICTIONS = images(
    [
        np.array([0.0, 0.0, 0.0, 4.0]),
        np.array([3.0, 3.0, 3.0, 3.0]),
        np.array([1.0, 2.0, 3.0, 4.0]),
    ]
)
ARRAY_REFERENCES = images([np.zeros(4), np.zeros(4), np.zeros(4)])
LIST_PREDICTIONS = _texts([["a", "b"], ["c"], ["d", "e", "f"]])
LIST_REFERENCES = _texts([["b", "a"], ["c"], ["d", "e", "g"]])
SHORT_PROMPTS = _texts(["question-0"])
SHORT_CONTEXT = [_texts(["context-0"])]
LONG_PROMPTS = _texts([f"question-{index}" for index in range(4)])
LONG_CONTEXT = [_texts([f"context-{index}" for index in range(4)])]


def _inputs_for(metric):
    """Pick a small input without a degenerate pair that the metric's family accepts.

    The error metrics, whose declared types exclude text, get in-memory images.
    """
    if isinstance(metric, DynamicMatch):
        if metric.normalization_type == NormalizationType.NUMERIC.value:
            return NUMERIC_PREDICTIONS, NUMERIC_REFERENCES
        if metric.normalization_type == NormalizationType.LIST.value:
            return LIST_PREDICTIONS, LIST_REFERENCES
        return TEXT_PREDICTIONS, TEXT_REFERENCES
    if Modality.TEXT in metric.supported_data_types:
        return TEXT_PREDICTIONS, TEXT_REFERENCES
    return ARRAY_PREDICTIONS, ARRAY_REFERENCES


# Every registered class enters the one-kind check; the deterministic ones (the
# classes that are not LLMMetric subclasses) enter the scoring parametrizations.
REGISTERED_CLASSES = [
    pytest.param(metric_class, id=metric_class.__name__)
    for metric_class in MetricRegistry.metrics.values()
]
with patch.object(
    MetricRegistry, "get_shared_bedrock_client", return_value=EMBEDDING_CLIENT
):
    DETERMINISTIC_METRICS = [
        metric_class()
        for metric_class in MetricRegistry.metrics.values()
        if not issubclass(metric_class, LLMMetric)
    ]
DETERMINISTIC_METRIC_CASES = [
    pytest.param(metric, *_inputs_for(metric), id=metric.name)
    for metric in DETERMINISTIC_METRICS
]

AGGREGATOR_CONFIGURATIONS = [
    pytest.param(
        EvaluationConfiguration(aggregation_method="numeric_mean"), id="numeric_mean"
    ),
    pytest.param(
        EvaluationConfiguration(
            aggregation_method="weighted_mean", weights=(1.0, 2.0, 3.0)
        ),
        id="weighted_mean",
    ),
    pytest.param(EvaluationConfiguration(aggregation_method="median"), id="median"),
    pytest.param(
        EvaluationConfiguration(aggregation_method="quantile", quantile=0.5),
        id="quantile",
    ),
]

REFERENCE_DEFECTS = [
    pytest.param(0, "references are required for {name}", id="missing-references"),
    pytest.param(
        1,
        "reference count (1) must match prediction count (3) for {name}",
        id="misaligned-references",
    ),
]

AUXILIARY_COUNTS = [
    pytest.param(SHORT_CONTEXT, SHORT_PROMPTS, id="fewer-than-predictions"),
    pytest.param(LONG_CONTEXT, LONG_PROMPTS, id="more-than-predictions"),
]

# Classification precision, recall and F1 is Accuracy with the labels matcher and
# prf; the list matcher gives one item for each element of a list.
LABELS_PRF = EvaluationConfiguration(aggregation_method="prf", matcher=LabelSet())
LIST_PRF = EvaluationConfiguration(aggregation_method="prf", matcher=ListItems())

# The two configurations of Accuracy that share the degenerate-pair rule (1.0) and
# the one-sided rule (0.0): the whole value, and the labels matcher with prf.
DEGENERATE_RULE_CONFIGURATIONS = [
    pytest.param(None, id="whole"),
    pytest.param(LABELS_PRF, id="labels"),
]

# Two single-label inputs for the labels matcher and prf. On the second, the macro
# precision is not the macro recall.
PRF_INPUTS = [
    pytest.param(TEXT_PREDICTIONS, TEXT_REFERENCES, id="text-fixture"),
    pytest.param(_texts(["a", "a", "a"]), _texts(["a", "b", "b"]), id="uneven"),
]

# The match record of one label of Accuracy with the labels matcher: the label is
# in both sets (one pair, score 1.0), in the prediction only, or in the reference only.
PAIRED_LABEL = Matches(((1.0,),), 1, 1)
PREDICTION_ONLY_LABEL = Matches(((),), 1, 0)
REFERENCE_ONLY_LABEL = Matches((), 0, 1)

# A valid entity list for ListItems(class_key="type", value_key="text").
ENTITY = [{"type": "PER", "text": "Smith"}]

# The pair scores of PairTableMetric. No single pairing gives the most pairs at
# both 0.5 (the 0.6 pair) and 0.4 (the two 0.45 pairs).
PAIR_SCORES = {
    ("p1", "r1"): 0.6,
    ("p1", "r2"): 0.45,
    ("p2", "r1"): 0.45,
    ("p2", "r2"): 0.0,
}


def _compute_metric(
    name, predictions, references, context=None, prompts=None, configuration=None
):
    """Run the registry entry point with a stub factory; no deterministic run reaches AWS."""
    return MetricRegistry.compute_metric(
        name,
        predictions,
        references,
        context,
        prompts,
        configuration=configuration,
        client_factory=Mock(spec=[]),
    )


class RecordingMetric(Metric):
    """Fixture metric that hands every sample the engine builds to the injected client."""

    def compute(self, sample, client):
        client.record(sample)
        return 1.0


class PairTableMetric(Metric):
    """Fixture metric that scores each item pair from ``PAIR_SCORES``."""

    def compute(self, sample, client):
        return PAIR_SCORES[
            (sample.prediction.get_raw_data(), sample.reference.get_raw_data())
        ]


class OutOfRangeMetric(Metric):
    """Fixture metric with a finite score above 1.0."""

    def compute(self, sample, client):
        return 3.2


class TestOneMetricKind:
    """Every registered class is a ``Metric`` scored through ``compute``."""

    def test_registry_holds_the_built_in_metrics(self):
        # Guards the parametrizations below against passing on an empty or
        # partial discovery. The bounds are the built-in counts; custom or new
        # metrics may add to them.
        assert len(MetricRegistry.metrics) >= 46
        assert len(DETERMINISTIC_METRICS) >= 20
        for name in ("F1Score", "Precision", "Recall", "PrecisionRecallF1"):
            assert not MetricRegistry.is_registered(name)

    @pytest.mark.parametrize("metric_class", REGISTERED_CLASSES)
    def test_registered_class_scores_per_sample(self, metric_class):
        assert issubclass(metric_class, Metric)
        metric = metric_class()
        assert callable(metric.compute)


class TestPerSampleAlignment:
    """Each sample scores its own pair: a single-pair run scores that sample the same.

    The check reads the sample score of the single-pair run, not its aggregate,
    so it holds for a metric whose aggregate is not a mean of the sample scores.
    """

    @pytest.mark.parametrize(
        ("metric", "predictions", "references"), DETERMINISTIC_METRIC_CASES
    )
    def test_samples_are_aligned_and_score_their_own_pair(
        self, metric, predictions, references
    ):
        result = _compute_metric(metric.name, predictions, references)

        assert len(result.samples) == len(predictions)
        for index, sample in enumerate(result.samples):
            assert sample.sample_id == str(index)
            assert sample.ordinal == index
            assert sample.status is SampleStatus.COMPLETED
            assert (
                sample.score
                == _compute_metric(
                    metric.name, [predictions[index]], [references[index]]
                )
                .samples[0]
                .score
            )


class TestRequiredReferences:
    """Missing or misaligned references fail with one message before any client exists."""

    @pytest.mark.parametrize(("kept", "message"), REFERENCE_DEFECTS)
    @pytest.mark.parametrize(
        ("metric", "predictions", "references"), DETERMINISTIC_METRIC_CASES
    )
    def test_defective_references_are_rejected_before_the_client_factory_runs(
        self, metric, predictions, references, kept, message
    ):
        client_factory = Mock(spec=[])

        with pytest.raises(ValueError) as error:
            MetricRegistry.compute_metric(
                metric.name,
                predictions,
                references[:kept],
                client_factory=client_factory,
            )

        assert str(error.value) == message.format(name=metric.name)
        client_factory.assert_not_called()


class TestAggregators:
    """Every aggregator applies to every deterministic metric, and the request wins."""

    @pytest.mark.parametrize("configuration", AGGREGATOR_CONFIGURATIONS)
    @pytest.mark.parametrize(
        ("metric", "predictions", "references"), DETERMINISTIC_METRIC_CASES
    )
    def test_requested_aggregator_completes(
        self, metric, predictions, references, configuration
    ):
        # An explicit request wins over the metric default.
        result = _compute_metric(
            metric.name, predictions, references, configuration=configuration
        )

        assert result.aggregate.status is EvaluationStatus.COMPLETED
        assert result.aggregate.method == configuration.aggregation_method


class TestAuxiliaryInputs:
    """Prompts and context are OMIT roles: any count is dropped, never rejected."""

    @pytest.mark.parametrize(("context", "prompts"), AUXILIARY_COUNTS)
    @pytest.mark.parametrize(
        ("metric", "predictions", "references"), DETERMINISTIC_METRIC_CASES
    )
    def test_any_prompt_and_context_count_changes_nothing(
        self, metric, predictions, references, context, prompts
    ):
        baseline = _compute_metric(metric.name, predictions, references)

        result = _compute_metric(metric.name, predictions, references, context, prompts)

        assert result.status is EvaluationStatus.COMPLETED
        assert result.score == baseline.score
        assert [sample.score for sample in result.samples] == [
            sample.score for sample in baseline.samples
        ]

    @pytest.mark.parametrize(("context", "prompts"), AUXILIARY_COUNTS)
    def test_engine_builds_each_sample_without_prompt_or_context(
        self, monkeypatch, context, prompts
    ):
        monkeypatch.setitem(MetricRegistry.metrics, "recordingmetric", RecordingMetric)
        client = Mock()

        MetricRegistry.compute_metric(
            "RecordingMetric",
            TEXT_PREDICTIONS,
            TEXT_REFERENCES,
            context,
            prompts,
            client_factory=Mock(return_value=client),
        )

        assert client.record.call_args_list == [
            call(EvaluationSample(str(index), index, prediction, reference, None, ()))
            for index, (prediction, reference) in enumerate(
                zip(TEXT_PREDICTIONS, TEXT_REFERENCES)
            )
        ]


class TestMatchFamily:
    """``DynamicMatch`` hands each raw pair to its comparator exactly once."""

    def test_comparator_runs_once_per_pair_with_raw_values(self, monkeypatch):
        comparator = Mock(return_value=1.0)

        class RecordingMatch(DynamicMatch):
            def __init__(self):
                super().__init__()
                self.comparator = comparator

        monkeypatch.setitem(MetricRegistry.metrics, "recordingmatch", RecordingMatch)

        result = _compute_metric("RecordingMatch", TEXT_PREDICTIONS, TEXT_REFERENCES)

        assert result.score == 1.0
        assert comparator.call_args_list == [
            call(prediction.get_raw_data(), reference.get_raw_data())
            for prediction, reference in zip(TEXT_PREDICTIONS, TEXT_REFERENCES)
        ]


class TestClientIsolation:
    """Deterministic metrics never build the client."""

    @pytest.mark.parametrize(
        ("metric", "predictions", "references"), DETERMINISTIC_METRIC_CASES
    )
    def test_deterministic_metrics_never_build_the_client(
        self, metric, predictions, references
    ):
        # The engine builds the client when a metric first reads it, so the
        # factory never runs for a metric that ignores the client.
        client_factory = Mock(return_value=Mock(spec=[]))

        result = MetricRegistry.compute_metric(
            metric.name, predictions, references, client_factory=client_factory
        )

        client_factory.assert_not_called()
        assert result.status is EvaluationStatus.COMPLETED
        assert all(sample.status is SampleStatus.COMPLETED for sample in result.samples)


class TestDegeneratePairs:
    """A degenerate pair scores 1.0 on the single path; a one-sided empty pair scores 0.0."""

    @pytest.mark.parametrize(
        ("configuration", "value"),
        [
            pytest.param(None, "", id="whole-both-empty"),
            pytest.param(LABELS_PRF, "", id="labels-both-empty"),
            pytest.param(LABELS_PRF, [None], id="labels-no-labels"),
            pytest.param(LABELS_PRF, [], id="labels-empty-list"),
            pytest.param(LABELS_PRF, None, id="labels-none"),
        ],
    )
    def test_degenerate_sample_is_completed_with_score_one(self, configuration, value):
        # The prf aggregator raises over a dataset with no labels, so a second,
        # matching pair carries one. Both pairs score 1.0, so the aggregate is 1.0.
        result = _compute_metric(
            "Accuracy",
            [TextData(value), TextData("a")],
            [TextData(value), TextData("a")],
            configuration=configuration,
        )

        assert result.score == 1.0
        assert result.samples[0].status is SampleStatus.COMPLETED
        assert result.samples[0].score == 1.0

    def test_no_labels_anywhere_fails_under_prf(self):
        with pytest.raises(ValueError) as error:
            _compute_metric(
                "Accuracy", [TextData("")], [TextData("")], configuration=LABELS_PRF
            )

        assert str(error.value) == "no items in any prediction or reference"

    def test_degenerate_pair_counts_as_one_in_the_aggregate(self):
        result = _compute_metric("Accuracy", _texts(["", "a"]), _texts(["", "b"]))

        assert result.score == 0.5
        assert [sample.score for sample in result.samples] == [1.0, 0.0]
        assert all(sample.status is SampleStatus.COMPLETED for sample in result.samples)

    def test_degenerate_sample_adds_no_label_and_counts_in_the_denominator(self):
        result = _compute_metric(
            "Accuracy", _texts(["", "a"]), _texts(["", "b"]), configuration=LABELS_PRF
        )

        assert [sample.score for sample in result.samples] == [1.0, 0.0]
        per_class = result.aggregate.values["per_class"]
        assert per_class["a"] == {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "tp": 0,
            "fp": 1,
            "fn": 0,
            "tn": 1,
            "support": 0,
        }
        assert per_class["b"] == {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "tp": 0,
            "fp": 0,
            "fn": 1,
            "tn": 1,
            "support": 1,
        }
        assert result.score == 0.0

    @pytest.mark.parametrize(
        ("prediction", "reference"),
        [
            pytest.param("", "a", id="empty-prediction"),
            pytest.param("a", "", id="empty-reference"),
        ],
    )
    @pytest.mark.parametrize("configuration", DEGENERATE_RULE_CONFIGURATIONS)
    def test_one_sided_empty_pair_scores_zero(
        self, configuration, prediction, reference
    ):
        result = _compute_metric(
            "Accuracy",
            _texts([prediction]),
            _texts([reference]),
            configuration=configuration,
        )

        assert result.samples[0].status is SampleStatus.COMPLETED
        assert result.samples[0].score == 0.0
        assert result.score == 0.0


class TestValueLocks:
    """Pinned dataset values for the headline metrics on the single path."""

    def test_accuracy_scores_each_pair_and_averages(self):
        result = _compute_metric(
            "Accuracy", _texts(["cat", "dog", "bird"]), _texts(["cat", "dog", "fish"])
        )

        assert [sample.score for sample in result.samples] == [1.0, 1.0, 0.0]
        assert result.score == pytest.approx(2 / 3)

    def test_labels_prf_macro_on_the_text_fixture(self):
        result = _compute_metric(
            "Accuracy", TEXT_PREDICTIONS, TEXT_REFERENCES, configuration=LABELS_PRF
        )

        assert [sample.score for sample in result.samples] == [1.0, 1.0, 0.0]
        assert [sample.matches for sample in result.samples] == [
            {"cat": PAIRED_LABEL},
            {"dog": PAIRED_LABEL},
            {"bird": PREDICTION_ONLY_LABEL, "fish": REFERENCE_ONLY_LABEL},
        ]
        assert result.aggregate.method == "prf"
        assert set(result.aggregate.values) == {
            "threshold",
            "precision",
            "recall",
            "f1",
            "per_class",
        }
        assert result.aggregate.values["threshold"] == 0.5
        assert result.score == 0.5
        assert result.aggregate.values["precision"] == 0.5
        assert result.aggregate.values["recall"] == 0.5
        per_class = result.aggregate.values["per_class"]
        assert list(per_class) == ["bird", "cat", "dog", "fish"]
        assert per_class["bird"] == {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "tp": 0,
            "fp": 1,
            "fn": 0,
            "tn": 2,
            "support": 0,
        }
        assert per_class["fish"]["fn"] == 1

    def test_rmse_defaults_to_the_mean_of_the_per_sample_values(self):
        predictions, references = ARRAY_PREDICTIONS[:2], ARRAY_REFERENCES[:2]
        result = _compute_metric("RootMeanSquaredError", predictions, references)

        assert [sample.score for sample in result.samples] == [2.0, 3.0]
        assert result.aggregate.method == "numeric_mean"
        assert result.score == 2.5
        assert result.aggregate == numeric_mean(result.samples)
        mae = _compute_metric("MeanAbsoluteError", predictions, references)
        mse = _compute_metric("MeanSquaredError", predictions, references)
        assert mae.score == 2.0
        assert mse.score == 6.5


class TestLabelsMatcherPrf:
    """Precision, recall and F1 of labels: Accuracy with the labels matcher and prf."""

    @staticmethod
    def _compute(predictions, references):
        return _compute_metric(
            "Accuracy", predictions, references, configuration=LABELS_PRF
        )

    @pytest.mark.parametrize(
        ("prediction", "reference", "score", "matches"),
        [
            pytest.param(
                "positive",
                "negative",
                0.0,
                {"negative": REFERENCE_ONLY_LABEL, "positive": PREDICTION_ONLY_LABEL},
                id="mismatch",
            ),
            pytest.param(
                "negative", "negative", 1.0, {"negative": PAIRED_LABEL}, id="match"
            ),
            # The sample score is the F1 of the sample: 2 * 2 pairs / (2 + 3) items.
            pytest.param(
                ["economics", "politics"],
                ["economics", "history", "politics"],
                0.8,
                {
                    "economics": PAIRED_LABEL,
                    "history": REFERENCE_ONLY_LABEL,
                    "politics": PAIRED_LABEL,
                },
                id="multi-label",
            ),
            pytest.param(
                "Positive ",
                "positive",
                1.0,
                {"positive": PAIRED_LABEL},
                id="trim-lowercase",
            ),
            pytest.param(True, True, 1.0, {"true": PAIRED_LABEL}, id="json-true"),
            pytest.param(1, "1", 1.0, {"1": PAIRED_LABEL}, id="json-one-vs-text-one"),
            pytest.param(
                ["a", "a", "b"],
                ["b", "a"],
                1.0,
                {"a": PAIRED_LABEL, "b": PAIRED_LABEL},
                id="set",
            ),
            pytest.param(
                {"k": 1},
                [{"k": 1}],
                1.0,
                {"[('k', 1)]": PAIRED_LABEL},
                id="dict-label",
            ),
            pytest.param(
                "", "a", 0.0, {"a": REFERENCE_ONLY_LABEL}, id="empty-prediction"
            ),
            pytest.param(
                "a", "", 0.0, {"a": PREDICTION_ONLY_LABEL}, id="empty-reference"
            ),
            pytest.param(
                True,
                1,
                0.0,
                {"1": REFERENCE_ONLY_LABEL, "true": PREDICTION_ONLY_LABEL},
                id="true-vs-one",
            ),
        ],
    )
    def test_each_sample_has_one_match_record_per_label(
        self, prediction, reference, score, matches
    ):
        result = self._compute([TextData(prediction)], [TextData(reference)])

        assert result.samples[0].score == score
        assert result.samples[0].matches == matches

    def test_multi_label_pair_gives_the_class_means(self):
        result = self._compute(
            _texts([["economics", "politics"]]),
            _texts([["economics", "history", "politics"]]),
        )

        values = result.aggregate.values
        assert values["precision"] == pytest.approx(2 / 3)
        assert values["recall"] == pytest.approx(2 / 3)
        assert values["f1"] == pytest.approx(2 / 3)
        assert result.score == pytest.approx(2 / 3)

    def test_multi_label_pair_has_a_zero_match_rate(self):
        # As a whole value, the pair is one label-set mismatch; the labels matcher
        # gives the same pair the sample score 0.8 (the multi-label case above).
        result = _compute_metric(
            "Accuracy",
            _texts([["economics", "politics"]]),
            _texts([["economics", "history", "politics"]]),
            configuration=EvaluationConfiguration(aggregation_method="numeric_mean"),
        )

        assert result.score == 0.0
        assert result.aggregate.values is None

    @pytest.mark.parametrize(("predictions", "references"), PRF_INPUTS)
    def test_prf_values_are_the_means_of_the_per_class_table(
        self, predictions, references
    ):
        result = self._compute(predictions, references)

        assert result.aggregate.method == "prf"
        values = result.aggregate.values
        assert set(values) == {"threshold", "precision", "recall", "f1", "per_class"}
        per_class = values["per_class"]
        for key in ("precision", "recall", "f1"):
            assert values[key] == pytest.approx(
                sum(row[key] for row in per_class.values()) / len(per_class)
            )
        assert result.score == values["f1"]

    @pytest.mark.parametrize(("predictions", "references"), PRF_INPUTS)
    def test_aggregate_is_the_public_aggregator_of_the_samples(
        self, predictions, references
    ):
        result = self._compute(predictions, references)

        assert result.aggregate.method == "prf"
        assert result.aggregate == prf(result.samples)
        assert result.score == result.aggregate.values["f1"]
        rebuilt = MetricEvaluationResult(
            result.metric_name, result.samples, result.aggregate
        )
        assert rebuilt == result

    def test_counts_add_up_per_class(self):
        result = self._compute(_texts(["a", "a", "a"]), _texts(["a", "b", "b"]))

        per_class = result.aggregate.values["per_class"]
        assert list(per_class) == ["a", "b"]
        for row in per_class.values():
            assert row["tp"] + row["fp"] + row["fn"] + row["tn"] == 3
            assert row["support"] == row["tp"] + row["fn"]

    def test_uneven_fixture_separates_precision_from_recall(self):
        result = self._compute(_texts(["a", "a", "a"]), _texts(["a", "b", "b"]))

        assert [sample.matches for sample in result.samples] == [
            {"a": PAIRED_LABEL},
            {"a": PREDICTION_ONLY_LABEL, "b": REFERENCE_ONLY_LABEL},
            {"a": PREDICTION_ONLY_LABEL, "b": REFERENCE_ONLY_LABEL},
        ]
        values = result.aggregate.values
        assert values["precision"] == pytest.approx(1 / 6)
        assert values["recall"] == 0.5
        assert values["f1"] == pytest.approx(0.25)

    @pytest.mark.parametrize(
        ("configuration", "expected"),
        [
            pytest.param(
                EvaluationConfiguration(aggregation_method="median"), 1.0, id="median"
            ),
            pytest.param(
                EvaluationConfiguration(aggregation_method="quantile", quantile=0.0),
                0.0,
                id="quantile",
            ),
            pytest.param(
                EvaluationConfiguration(
                    aggregation_method="weighted_mean", weights=(1.0, 1.0, 2.0)
                ),
                0.5,
                id="weighted_mean",
            ),
        ],
    )
    def test_numeric_aggregators_apply_to_the_match_score(
        self, configuration, expected
    ):
        # Without the labels matcher, Accuracy gives each sample the label-set
        # match score, which the numeric aggregators take.
        result = _compute_metric(
            "Accuracy",
            TEXT_PREDICTIONS,
            TEXT_REFERENCES,
            configuration=configuration,
        )

        assert result.aggregate.method == configuration.aggregation_method
        assert result.score == expected
        assert result.aggregate.values is None


class TestAccuracyLabelSets:
    """Accuracy compares whole label sets with the label rule of the labels matcher."""

    @pytest.mark.parametrize(
        ("prediction", "reference", "score"),
        [
            pytest.param("positive", "negative", 0.0, id="wrong-label"),
            pytest.param("Positive ", "positive", 1.0, id="trim-lowercase"),
            pytest.param("positive.", "positive", 0.0, id="punctuation-kept"),
            pytest.param(1, 1, 1.0, id="integer-match"),
            pytest.param(0, 1, 0.0, id="integer-mismatch"),
            # Models often serialize class ids as floats; 1.0 is the label 1.
            pytest.param(1, 1.0, 1.0, id="integral-float-is-the-integer"),
            pytest.param(-3.0, -3, 1.0, id="negative-integral-float"),
            pytest.param([1, 2.0], [2, 1], 1.0, id="integral-floats-in-a-list"),
            pytest.param(2.5, 2, 0.0, id="fractional-float-stays-distinct"),
            pytest.param("1.0", 1, 0.0, id="numeric-string-is-not-parsed"),
            pytest.param(True, "true", 1.0, id="boolean"),
            pytest.param(True, 1, 0.0, id="boolean-is-not-the-integer"),
            pytest.param(["a", "b"], ["b", "a"], 1.0, id="list-is-a-set"),
            pytest.param(
                "Paris is the capital of France.",
                "The capital of France is Paris.",
                0.0,
                id="free-text",
            ),
            pytest.param(None, None, 1.0, id="null"),
            pytest.param("", "a", 0.0, id="one-side-empty"),
        ],
    )
    def test_accuracy_scores_whole_label_sets(self, prediction, reference, score):
        result = _compute_metric(
            "Accuracy", [TextData(prediction)], [TextData(reference)]
        )

        assert result.samples[0].status is SampleStatus.COMPLETED
        assert result.samples[0].score == score

    @pytest.mark.parametrize(
        ("predictions", "references"),
        [
            pytest.param(TEXT_PREDICTIONS, TEXT_REFERENCES, id="text-fixture"),
            pytest.param(
                _texts([1, 0, 2]), _texts([1.0, 0.0, 3.0]), id="int-vs-float-labels"
            ),
        ],
    )
    def test_labels_matcher_keeps_the_sample_scores_of_single_labels(
        self, predictions, references
    ):
        whole = _compute_metric("Accuracy", predictions, references)
        labels = _compute_metric(
            "Accuracy", predictions, references, configuration=LABELS_PRF
        )

        assert [s.score for s in whole.samples] == [s.score for s in labels.samples]


class TestItemMatchers:
    """Item matchers: the metric scores each candidate pair, and prf pairs the items."""

    def test_retrieval_pools_the_items_in_one_class(self):
        predictions = _texts([["d1", "d2"], ["d3"]])
        references = _texts([["d1"], ["d3", "d4"]])

        result = _compute_metric(
            "TextExactMatch", predictions, references, configuration=LIST_PRF
        )
        labels = _compute_metric(
            "TextExactMatch", predictions, references, configuration=LABELS_PRF
        )

        values = result.aggregate.values
        for key in ("precision", "recall", "f1"):
            assert values[key] == pytest.approx(2 / 3)
        assert list(values["per_class"]) == ["*"]
        row = values["per_class"]["*"]
        counts = {key: row[key] for key in ("tp", "fp", "fn", "tn", "support")}
        assert counts == {"tp": 2, "fp": 1, "fn": 1, "tn": 0, "support": 3}
        assert [sample.score for sample in result.samples] == pytest.approx(
            [2 / 3, 2 / 3]
        )
        # With the labels matcher, each document ID is a class.
        assert labels.aggregate.values["f1"] == 0.5

    def test_entities_pair_within_each_type_at_each_threshold(self):
        configuration = EvaluationConfiguration(
            aggregation_method="prf",
            matcher=ListItems(class_key="type", value_key="text"),
            thresholds=(0.5, 0.8),
        )
        prediction = [
            {"type": "PER", "text": "Jane Smith"},
            {"type": "ORG", "text": "Acme"},
        ]
        reference = [
            {"type": "PER", "text": "Smith"},
            {"type": "ORG", "text": "Acme Corp"},
        ]

        result = _compute_metric(
            "TextFuzzyMatch",
            _texts([prediction]),
            _texts([reference]),
            configuration=configuration,
        )

        values = result.aggregate.values
        assert values["threshold"] == 0.5
        assert values["f1"] == 0.5
        assert {name: row["f1"] for name, row in values["per_class"].items()} == {
            "org": 0.0,
            "per": 1.0,
        }
        assert [(row["threshold"], row["f1"]) for row in values["by_threshold"]] == [
            (0.5, 0.5),
            (0.8, 0.0),
        ]
        sample = result.samples[0]
        assert sample.score == 0.5
        assert sample.matches["per"].scores == ((0.5,),)
        assert sample.matches["org"].scores == ((pytest.approx(4 / 9),),)
        # Another threshold from the stored samples, with no metric call.
        assert prf(result.samples, thresholds=(0.4,)).values["f1"] == 1.0

    @pytest.mark.parametrize(
        ("thresholds", "counts", "score"),
        [
            # The pairs below 0.5 do not block the 0.6 pair.
            pytest.param(None, (1, 1, 1), 0.5, id="default-0.5"),
            pytest.param((0.4,), (2, 0, 0), 1.0, id="0.4"),
        ],
    )
    def test_prf_keeps_the_most_pairs_at_each_threshold(
        self, monkeypatch, thresholds, counts, score
    ):
        monkeypatch.setitem(MetricRegistry.metrics, "pairtablemetric", PairTableMetric)
        configuration = EvaluationConfiguration(
            aggregation_method="prf", matcher=ListItems(), thresholds=thresholds
        )

        result = _compute_metric(
            "PairTableMetric",
            _texts([["p1", "p2"]]),
            _texts([["r1", "r2"]]),
            configuration=configuration,
        )

        row = result.aggregate.values["per_class"]["*"]
        assert (row["tp"], row["fp"], row["fn"]) == counts
        assert result.samples[0].score == score

    def test_classes_follow_the_label_rule(self):
        # The items are the whole elements, and TextExactMatch normalizes their text.
        result = _compute_metric(
            "TextExactMatch",
            _texts([[{"label": "Car"}]]),
            _texts([[{"label": " car "}]]),
            configuration=EvaluationConfiguration(
                aggregation_method="prf", matcher=ListItems(class_key="label")
            ),
        )

        assert result.samples[0].matches == {"car": Matches(((1.0,),), 1, 1)}

    def test_pair_score_above_one_fails_the_sample_only_on_the_item_path(
        self, monkeypatch
    ):
        monkeypatch.setitem(
            MetricRegistry.metrics, "outofrangemetric", OutOfRangeMetric
        )
        predictions, references = _texts([["x"]]), _texts([["x"]])

        items = _compute_metric(
            "OutOfRangeMetric", predictions, references, configuration=LIST_PRF
        )
        whole = _compute_metric("OutOfRangeMetric", predictions, references)

        assert items.samples[0].status is SampleStatus.FAILED
        assert (
            items.samples[0].error == "pair score must be between 0.0 and 1.0, got 3.2"
        )
        assert whole.samples[0].status is SampleStatus.COMPLETED
        assert whole.score == 3.2

    @pytest.mark.parametrize(
        ("prediction", "reference", "error"),
        [
            # A value that is not a list is its only element, item 0.
            pytest.param(
                "Smith",
                ENTITY,
                "prediction item 0 needs a non-empty 'type' field",
                id="not-a-list",
            ),
            pytest.param(
                ENTITY,
                [{"text": "Smith"}],
                "reference item 0 needs a non-empty 'type' field",
                id="no-class-field",
            ),
            pytest.param(
                [{"type": "PER"}],
                ENTITY,
                "prediction item 0 has no 'text' field",
                id="no-value-field",
            ),
        ],
    )
    def test_item_error_fails_the_sample_and_skips_the_rest(
        self, prediction, reference, error
    ):
        result = _compute_metric(
            "TextExactMatch",
            _texts([prediction, ENTITY]),
            _texts([reference, ENTITY]),
            configuration=EvaluationConfiguration(
                aggregation_method="prf",
                matcher=ListItems(class_key="type", value_key="text"),
            ),
        )

        assert [sample.status for sample in result.samples] == [
            SampleStatus.FAILED,
            SampleStatus.SKIPPED,
        ]
        assert result.samples[0].error == error
