# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""The v1 registry and package surface survives the move to the single engine."""

import math
from unittest.mock import Mock, patch

import numpy as np
import pytest

import metric_library
import metric_library.core
import metric_library.core.evaluation
import metric_library.core.registry
import metric_library.metrics  # noqa: F401 - registers built-in metrics
from metric_library.core.aggregation import EvaluationStatus, SampleStatus
from metric_library.core.base import Metric
from metric_library.core.data import Modality
from metric_library.core.data.text_data import TextData
from media_double import images
from metric_library.core.evaluation import MetricEvaluationResult, evaluate_metric
from metric_library.core.input_policy import InputPolicy
from metric_library.core.llm_metric import LLMMetric
from metric_library.core.registry import MetricError, MetricRegistry
from metric_library.metrics.accuracy import Accuracy
from metric_library.metrics.llm_correctness import LLMCorrectness
from metric_library.metrics.match import DynamicMatch
from metric_library.utils.comparison_functions import NormalizationType


@pytest.fixture(autouse=True)
def _forbid_live_aws(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("live AWS is forbidden in tests")

    monkeypatch.setattr("metric_library.core.bedrock_client.boto3.client", fail)
    monkeypatch.setattr("metric_library.core.bedrock_client.boto3.Session", fail)


# Every text embeds to the same vector, so the two semantic match metrics score
# without an embedding provider. compute_metric builds a fresh instance per call,
# and those metrics ask the registry for the shared client while they construct.
EMBEDDING_CLIENT = Mock(spec=["get_embeddings"])
EMBEDDING_CLIENT.get_embeddings.return_value = [[1.0, 0.0]]


@pytest.fixture(autouse=True)
def _shared_embedding_client():
    with patch.object(
        MetricRegistry, "get_shared_bedrock_client", return_value=EMBEDDING_CLIENT
    ):
        yield


def _texts(values):
    return [TextData(value) for value in values]


TEXT_PREDICTIONS = _texts(["cat", "dog", "bird"])
TEXT_REFERENCES = _texts(["cat", "dog", "fish"])
NUMERIC_PREDICTIONS = _texts([1, 3, 2])
NUMERIC_REFERENCES = _texts([0, 0, 4])
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
CONTEXT = [_texts(["context-0", "context-1", "context-2"])]
PROMPTS = _texts(["question-0", "question-1", "question-2"])

# The public contract: the 23 names of the single engine and the matchers.
PUBLIC_EXPORTS = [
    "AggregationMethod",
    "AggregationResult",
    "AggregationSpec",
    "DataHandler",
    "EvaluationConfiguration",
    "EvaluationSample",
    "EvaluationStatus",
    "LabelSet",
    "ListItems",
    "Matcher",
    "Matches",
    "MetricEvaluationResult",
    "MetricRegistry",
    "SampleOutput",
    "SampleResult",
    "SampleStatus",
    "WholeValue",
    "extract_nested_values",
    "median",
    "numeric_mean",
    "prf",
    "quantile",
    "weighted_mean",
]

CORE_EXPORTS = [
    "DataHandler",
    "DataType",
    "InputPolicy",
    "Metric",
    "MetricRegistry",
    "Modality",
    "extract_nested_values",
]


def _inputs_for(metric):
    """Pick the records that a deterministic metric accepts.

    The numeric match metrics take plain numbers, the error metrics take
    in-memory images, and the other metrics take the text records.
    """
    if isinstance(metric, DynamicMatch):
        if metric.normalization_type == NormalizationType.NUMERIC.value:
            return NUMERIC_PREDICTIONS, NUMERIC_REFERENCES
        return TEXT_PREDICTIONS, TEXT_REFERENCES
    if Modality.TEXT in metric.supported_data_types:
        return TEXT_PREDICTIONS, TEXT_REFERENCES
    return ARRAY_PREDICTIONS, ARRAY_REFERENCES


class TestBatchComputeSurface:
    def test_every_registered_metric_yields_result_or_metric_error(self):
        registered = {
            name: metric_class()
            for name, metric_class in MetricRegistry.metrics.items()
        }
        # Guards the sweep against passing on an empty or partial discovery.
        assert len(registered) >= 46

        llm_names = {n for n, m in registered.items() if isinstance(m, LLMMetric)}
        deterministic = {n: m for n, m in registered.items() if n not in llm_names}
        assert llm_names
        assert deterministic

        # Deterministic metrics ignore the client, so a stub factory is enough.
        # One call per input set: text records, plain numbers, in-memory arrays.
        results = {}
        for predictions, references in (
            (TEXT_PREDICTIONS, TEXT_REFERENCES),
            (NUMERIC_PREDICTIONS, NUMERIC_REFERENCES),
            (ARRAY_PREDICTIONS, ARRAY_REFERENCES),
        ):
            names = [
                name
                for name, metric in deterministic.items()
                if _inputs_for(metric)[0] is predictions
            ]
            assert names, predictions
            results.update(
                MetricRegistry.batch_compute(
                    names,
                    predictions,
                    references,
                    CONTEXT,
                    PROMPTS,
                    client_factory=Mock(spec=[]),
                )
            )
        # The LLM judges accept the aligned context and prompts and fail at their
        # first sample, where the engine builds the forbidden Bedrock client
        # through the default factory.
        results.update(
            MetricRegistry.batch_compute(
                sorted(llm_names), TEXT_PREDICTIONS, TEXT_REFERENCES, CONTEXT, PROMPTS
            )
        )

        assert set(results) == set(MetricRegistry.metrics)
        for name, value in results.items():
            assert isinstance(value, (MetricEvaluationResult, MetricError)), name

        for name in deterministic:
            result = results[name]
            assert isinstance(result, MetricEvaluationResult), (name, result)
            assert result.status is EvaluationStatus.COMPLETED, (name, result)
            assert isinstance(result.score, float), (name, result.score)
            assert math.isfinite(result.score), (name, result.score)
            assert result.score == result.aggregate.score, name

        for name in llm_names:
            result = results[name]
            assert isinstance(result, MetricEvaluationResult), (name, result)
            assert result.status is EvaluationStatus.FAILED, name
            assert result.samples[0].error == "live AWS is forbidden in tests", name
            assert all(
                sample.status is SampleStatus.SKIPPED for sample in result.samples[1:]
            ), name


class TestRegistryApi:
    def test_registry_methods_are_callable(self):
        for method in (
            "register",
            "get_metric",
            "is_registered",
            "list_metrics",
            "compute_metric",
            "batch_compute",
        ):
            assert callable(getattr(MetricRegistry, method)), method

    def test_registry_lookups_resolve_built_in_metrics(self):
        assert MetricRegistry.is_registered("Accuracy") is True
        assert MetricRegistry.is_registered("NoSuchMetric") is False
        assert "Accuracy" in MetricRegistry
        assert isinstance(MetricRegistry.get_metric("Accuracy"), Accuracy)
        assert "Accuracy" in {e["name"] for e in MetricRegistry.list_metrics()}
        result = MetricRegistry.compute_metric(
            "Accuracy", TEXT_PREDICTIONS, TEXT_REFERENCES, client_factory=Mock(spec=[])
        )
        assert isinstance(result, MetricEvaluationResult)
        assert result.score == pytest.approx(2 / 3)

    def test_metric_error_keeps_its_v1_shape(self):
        assert callable(MetricError)
        assert MetricError("x") == {"error": "x"}
        assert isinstance(MetricError("x"), dict)

    def test_register_rejects_non_metric_values(self):
        for value in (object(), 42, type("NotAMetric", (), {})):
            with pytest.raises(TypeError, match="register expects a Metric subclass"):
                MetricRegistry.register(value)
        assert "notametric" not in MetricRegistry.metrics

    def test_metric_without_compute_fails_at_instantiation(self):
        class NoCompute(Metric):
            pass

        try:
            # register admits the class; the missing compute surfaces at
            # instantiation, not at registration.
            assert MetricRegistry.register(NoCompute) is NoCompute
            assert MetricRegistry.metrics["nocompute"] is NoCompute
            with pytest.raises(
                TypeError,
                match=r"Can't instantiate abstract class NoCompute .*compute",
            ):
                MetricRegistry.get_metric("NoCompute")

            results = MetricRegistry.batch_compute(
                ["NoCompute"],
                TEXT_PREDICTIONS,
                TEXT_REFERENCES,
                client_factory=Mock(spec=[]),
            )
            assert isinstance(results["NoCompute"], MetricError)
            assert "Can't instantiate abstract class" in results["NoCompute"]["error"]
            assert "compute" in results["NoCompute"]["error"]
        finally:
            MetricRegistry.metrics.pop("nocompute", None)

    def test_failing_client_factory_does_not_affect_a_deterministic_metric(self):
        factory = Mock(side_effect=RuntimeError("no client"))
        result = MetricRegistry.compute_metric(
            "Accuracy", TEXT_PREDICTIONS, TEXT_REFERENCES, client_factory=factory
        )
        assert result.score == pytest.approx(2 / 3)
        factory.assert_not_called()

        results = MetricRegistry.batch_compute(
            ["Accuracy"],
            TEXT_PREDICTIONS,
            TEXT_REFERENCES,
            client_factory=Mock(side_effect=RuntimeError("no client")),
        )
        assert isinstance(results["Accuracy"], MetricEvaluationResult)


def _judge_prompt(metric, predictions, references, prompts=None):
    """Evaluate ``metric`` with a stub judge and return the prompt text it sent."""
    client = Mock(spec=["query"])
    client.query.return_value = {"reasoning": "ok", "score": 1.0}
    result = evaluate_metric(
        metric,
        _texts(predictions),
        _texts(references),
        None,
        _texts(prompts) if prompts else None,
        client_factory=Mock(return_value=client),
    )
    assert result.score == 1.0
    return client.query.call_args.kwargs["prompt"]


class TestOldRubricName:
    """A v1 LLM metric that defines ``get_evaluation_prompt`` keeps its criteria.

    ``rubric`` replaces that name; the old one stays an alias with a warning.
    """

    def test_a_custom_metric_with_the_old_name_sends_its_criteria(self):
        with pytest.warns(
            FutureWarning, match=r"OldNameMetric defines get_evaluation_prompt\(\)"
        ):

            class OldNameMetric(LLMMetric):
                reference_policy = InputPolicy.OMIT
                prompt_policy = InputPolicy.OMIT
                context_policy = InputPolicy.OMIT

                def get_evaluation_prompt(self) -> str:
                    return "Score 1.0 when the response is polite."

        prompt = _judge_prompt(OldNameMetric(), ["Thank you."], [])

        assert prompt == "Score 1.0 when the response is polite."

    def test_a_built_in_subclass_that_overrides_the_old_name_sends_its_criteria(
        self,
    ):
        with pytest.warns(
            FutureWarning, match=r"StricterCorrectness defines get_evaluation_prompt"
        ):

            class StricterCorrectness(LLMCorrectness):
                def get_evaluation_prompt(self) -> str:
                    return "Score 1.0 only for an exact answer."

        prompt = _judge_prompt(
            StricterCorrectness(), ["Paris"], ["Paris"], ["Capital of France?"]
        )

        assert prompt == "Score 1.0 only for an exact answer."

    def test_an_old_override_can_extend_the_built_in_rubric_through_super(self):
        with pytest.warns(FutureWarning, match=r"ExtendedCorrectness defines"):

            class ExtendedCorrectness(LLMCorrectness):
                def get_evaluation_prompt(self) -> str:
                    return super().get_evaluation_prompt() + "\nBe strict."

        prompt = _judge_prompt(
            ExtendedCorrectness(), ["Paris"], ["Paris"], ["Capital of France?"]
        )

        assert prompt == LLMCorrectness().rubric() + "\nBe strict."

    def test_a_custom_metric_with_neither_name_fails_at_instantiation(self):
        class NoRubric(LLMMetric):
            pass

        with pytest.raises(
            TypeError, match=r"Can't instantiate abstract class NoRubric .*rubric"
        ):
            NoRubric()


class TestPackageExports:
    def test_top_level_exports_are_the_public_names(self):
        assert sorted(metric_library.__all__) == PUBLIC_EXPORTS
        for name in PUBLIC_EXPORTS:
            assert getattr(metric_library, name) is not None, name

    def test_core_exports_are_the_public_names(self):
        assert sorted(metric_library.core.__all__) == CORE_EXPORTS
        for name in CORE_EXPORTS:
            assert getattr(metric_library.core, name) is not None, name
        assert metric_library.core.MetricRegistry is metric_library.MetricRegistry

    def test_removed_names_are_gone(self):
        for name in (
            "MetricEvaluator",
            "SampleMetric",
            "DerivedMetric",
            "DerivedMetricEvaluationResult",
        ):
            assert not hasattr(metric_library, name), name
        for name in (
            "MetricEvaluator",
            "SampleMetric",
            "DerivedMetricEvaluationResult",
        ):
            assert not hasattr(metric_library.core.evaluation, name), name
        assert not hasattr(metric_library.core, "DerivedMetric")
        assert not hasattr(metric_library.core.registry, "normalize_metric_result")
