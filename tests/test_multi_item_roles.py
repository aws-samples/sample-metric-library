# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Several items per role: declaration, validation and deterministic scoring.

A sample's prediction or reference may be a list of ``DataType`` items. Each
metric declares with ``Multiplicity`` how many it accepts and how they relate;
with the ``whole`` matcher the engine enforces the counts before any sample is
scored, and with ``labels`` or ``list`` the matcher reads the items instead.
One item per role is unchanged for every metric, which the first class
below checks against every registered non-LLM metric; the remaining classes
cover what several items do.
"""

import json
from unittest.mock import Mock

import pytest

import metric_library.metrics  # noqa: F401 - registers built-in metrics
from metric_library import (
    EvaluationConfiguration,
    EvaluationSample,
    EvaluationStatus,
    LabelSet,
    ListItems,
    Matches,
)
from metric_library.core.base import Metric
from metric_library.core.data.text_data import TextData
from metric_library.core.data_handler import DataHandler
from metric_library.core.evaluation import evaluate_metric
from metric_library.core.input_policy import InputPolicy
from metric_library.core.llm_metric import LLMMetric
from metric_library.core.multiplicity import Multiplicity
from metric_library.core.registry import MetricRegistry
from metric_library.metrics.llm_correctness import LLMCorrectness
from media_double import ArrayImageData, PngImageData


@pytest.fixture(autouse=True)
def _forbid_live_aws(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("live AWS is forbidden in tests")

    monkeypatch.setattr("metric_library.core.bedrock_client.boto3.client", fail)
    monkeypatch.setattr("metric_library.core.bedrock_client.boto3.Session", fail)


@pytest.fixture
def no_embeddings(monkeypatch):
    """Semantic match metrics compare equal strings as 1.0 and others as 0.0."""
    client = Mock(spec=["get_embeddings"])
    client.get_embeddings.side_effect = lambda texts: [
        [1.0, 0.0] if texts[0] == "same" else [0.0, 1.0]
    ]
    monkeypatch.setattr(MetricRegistry, "_shared_bedrock_client", client)


def _text(value):
    return TextData(value)


def _image(value):
    # The error metrics accept media only; a number stands in as an in-memory
    # image, which is what their data types reduce to.
    return ArrayImageData(value)


def _no_judge():
    return Mock(side_effect=AssertionError("judge client must not be constructed"))


class LLMCorrectnessAnyAnswer(LLMCorrectness):
    """The README custom metric for a ground truth of alternative answers."""

    reference_multiplicity = Multiplicity.ANY_ACCEPTABLE


class SameValue(Metric):
    """A custom pair metric with the default multiplicity and every modality."""

    def compute(self, sample, client):
        prediction = sample.prediction.get_raw_data()
        return 1.0 if prediction == sample.reference.get_raw_data() else 0.0


def _snapshot(result):
    return [
        (
            sample.status.value,
            sample.score,
            sample.error,
            sample.details,
            sample.matches,
        )
        for sample in result.samples
    ], (result.aggregate.method, result.status.value, result.score)


NON_LLM_METRICS = sorted(
    cls.__name__
    for cls in MetricRegistry.metrics.values()
    if not issubclass(cls, LLMMetric)
)

# One raw pair per metric family that every metric in it scores successfully.
PAIRS = {
    "text": [("same", "same"), ("cat", "dog")],
    "numeric": [(1.5, 1.0), (3, 3)],
    "lists": [(["a", "b"], ["b", "a"]), (["c"], ["d"])],
}


class TestOneItemPerRoleIsUnchanged:
    """A scalar item and a one-item list are the same input for every metric."""

    @pytest.mark.parametrize("metric_name", NON_LLM_METRICS)
    def test_scalar_and_one_item_list_score_identically(
        self, metric_name, no_embeddings
    ):
        metric = MetricRegistry.get_metric(metric_name)
        if "Error" in metric_name:
            item, pairs = _image, PAIRS["numeric"]
        elif "Numeric" in metric_name:
            item, pairs = _text, PAIRS["numeric"]
        else:
            item, pairs = _text, PAIRS["text"]

        scalar = evaluate_metric(
            metric,
            [item(p) for p, _ in pairs],
            [item(r) for _, r in pairs],
            client_factory=_no_judge(),
        )
        wrapped = evaluate_metric(
            metric,
            [[item(p)] for p, _ in pairs],
            [[item(r)] for _, r in pairs],
            client_factory=_no_judge(),
        )

        assert scalar.status is EvaluationStatus.COMPLETED
        assert _snapshot(wrapped) == _snapshot(scalar)

    @pytest.mark.parametrize("metric_name", NON_LLM_METRICS)
    def test_every_non_llm_metric_declares_how_many_items_it_reads(
        self, metric_name, no_embeddings
    ):
        metric = MetricRegistry.get_metric(metric_name)

        assert isinstance(metric.prediction_multiplicity, Multiplicity)
        assert isinstance(metric.reference_multiplicity, Multiplicity)


class TestEvaluationSampleRoles:
    """Roles are tuples; the singular accessors read the one item."""

    def test_scalar_none_and_sequence_inputs_normalize_to_tuples(self):
        item, other = _text("a"), _text("b")

        sample = EvaluationSample("0", 0, item, None, [item, other], ())

        assert sample.predictions == (item,)
        assert sample.references == ()
        assert sample.prompts == (item, other)
        assert sample.prediction is item
        assert sample.reference is None

    def test_one_item_and_its_one_tuple_are_the_same_sample(self):
        item, reference = _text("a"), _text("b")

        assert EvaluationSample("0", 0, item, reference, None, ()) == EvaluationSample(
            "0", 0, (item,), (reference,), (), ()
        )
        assert hash(EvaluationSample("0", 0, [item, item], reference, None, ()))

    def test_singular_accessor_refuses_to_pick_one_of_several_items(self):
        sample = EvaluationSample(
            "0", 0, [_text("a"), _text("b")], _text("r"), None, ()
        )

        with pytest.raises(
            ValueError, match="2 prediction items; read sample.predictions"
        ):
            sample.prediction


class TestMultiplicityValidation:
    """Declared counts are enforced with the whole matcher, before any sample."""

    def test_single_metric_rejects_several_prediction_items(self):
        class OneValue(Metric):
            def compute(self, sample, client):
                return 1.0

        with pytest.raises(ValueError) as exc_info:
            evaluate_metric(
                OneValue(),
                [_text("a"), [_text("b"), _text("c")]],
                [_text("a"), _text("b")],
                client_factory=_no_judge(),
            )

        assert str(exc_info.value) == (
            "OneValue accepts exactly one prediction item per sample; sample 1 has 2"
        )

    def test_single_metric_rejects_several_reference_items(self):
        class OneValue(Metric):
            def compute(self, sample, client):
                return 1.0

        with pytest.raises(
            ValueError, match="exactly one reference item per sample; sample 0 has 2"
        ):
            evaluate_metric(
                OneValue(),
                [_text("a")],
                [[_text("a"), _text("b")]],
                client_factory=_no_judge(),
            )

    @pytest.mark.parametrize(
        ("matcher", "f1"),
        [
            # One class "*": a-a and d-d pair, b and c stay unpaired.
            pytest.param(ListItems(), 2 / 3, id="list"),
            # Classes a, b, c, d: a and d have F1 1.0, b and c have 0.0.
            pytest.param(LabelSet(), 0.5, id="labels"),
        ],
    )
    def test_item_matchers_read_arrays_for_a_single_metric(self, matcher, f1):
        """The matcher replaces the multiplicity: compute gets one item per role."""
        result = evaluate_metric(
            SameValue(),
            [[_text("a"), _text("b")], [_text("d")]],
            [[_text("a"), _text("c")], [_text("d")]],
            configuration=EvaluationConfiguration(
                aggregation_method="prf", matcher=matcher
            ),
            client_factory=_no_judge(),
        )

        assert result.status is EvaluationStatus.COMPLETED
        assert result.score == pytest.approx(f1)

    @pytest.mark.parametrize(
        ("metric", "predictions", "references", "prompts", "matcher", "message"),
        [
            pytest.param(
                MetricRegistry.get_metric("MeanAbsoluteError"),
                [[_image(1.0), _image(2.0)]],
                [[_image(1.5), _image(2.0)]],
                None,
                ListItems(),
                "MeanAbsoluteError accepts only the whole matcher: it pairs "
                "prediction and reference items by position, and the labels and "
                "list matchers pair them by score",
                id="aligned",
            ),
            pytest.param(
                LLMCorrectnessAnyAnswer(),
                [_text("New York City")],
                [[_text("NYC"), _text("New York City")]],
                [_text("Which city is called the Big Apple?")],
                LabelSet(),
                "LLMCorrectnessAnyAnswer accepts only the whole matcher: it reads "
                "several reference items as alternatives, and the labels and list "
                "matchers read them as separate items",
                id="any-acceptable",
            ),
        ],
    )
    def test_item_matchers_stop_a_reading_they_cannot_replace(
        self, metric, predictions, references, prompts, matcher, message
    ):
        client_factory = _no_judge()

        with pytest.raises(ValueError) as exc_info:
            evaluate_metric(
                metric,
                predictions,
                references,
                None,
                prompts,
                configuration=EvaluationConfiguration(
                    aggregation_method="prf", matcher=matcher
                ),
                client_factory=client_factory,
            )

        assert str(exc_info.value) == message
        client_factory.assert_not_called()

    @pytest.mark.parametrize(
        ("metric", "predictions", "references", "prompts", "matcher", "location"),
        [
            pytest.param(
                MetricRegistry.get_metric("LLMCorrectness"),
                [[_text("A red mug."), PngImageData(b"mug")]],
                [_text("A red mug.")],
                [_text("Describe the product.")],
                ListItems(),
                "prediction 0",
                id="list-image-in-an-array",
            ),
            pytest.param(
                SameValue(),
                [_text("cat")],
                [PngImageData(b"cat")],
                None,
                LabelSet(),
                "reference 0",
                id="labels-one-image",
            ),
        ],
    )
    def test_item_matchers_stop_on_an_item_that_is_not_text(
        self, metric, predictions, references, prompts, matcher, location
    ):
        """The matchers read text, so an image would reach them as its bytes."""
        client_factory = _no_judge()

        with pytest.raises(ValueError) as exc_info:
            evaluate_metric(
                metric,
                predictions,
                references,
                None,
                prompts,
                configuration=EvaluationConfiguration(
                    aggregation_method="prf", matcher=matcher
                ),
                client_factory=client_factory,
            )

        assert str(exc_info.value) == (
            "the labels and list matchers read only text, not image data "
            f"({location}); use the whole matcher"
        )
        client_factory.assert_not_called()

    def test_aligned_metric_rejects_unequal_counts(self):
        with pytest.raises(ValueError) as exc_info:
            evaluate_metric(
                MetricRegistry.get_metric("MeanAbsoluteError"),
                [[_image(1.0), _image(2.0), _image(3.0)]],
                [[_image(1.0), _image(2.0)]],
                client_factory=_no_judge(),
            )

        assert str(exc_info.value) == (
            "prediction item count (3) must match reference item count (2) for "
            "sample 0 of MeanAbsoluteError"
        )

    def test_an_empty_list_is_no_items(self):
        with pytest.raises(
            ValueError, match="sample 0 has no prediction items for Accuracy"
        ):
            evaluate_metric(
                MetricRegistry.get_metric("Accuracy"),
                [[]],
                [_text("a")],
                client_factory=_no_judge(),
            )

    def test_aligned_metric_must_consume_references(self):
        class Misdeclared(Metric):
            reference_policy = InputPolicy.OMIT
            prediction_multiplicity = Multiplicity.ALIGNED

            def compute(self, sample, client):
                return 1.0

        with pytest.raises(
            ValueError,
            match="declares aligned prediction and reference items but omits references",
        ):
            evaluate_metric(Misdeclared(), [_text("a")], [], client_factory=_no_judge())

    def test_unknown_multiplicity_is_rejected_with_the_declared_options(self):
        class Typo(Metric):
            prediction_multiplicity = "several"

            def compute(self, sample, client):
                return 1.0

        with pytest.raises(
            ValueError, match="Invalid prediction multiplicity 'several' for Typo"
        ):
            evaluate_metric(
                Typo(), [_text("a")], [_text("a")], client_factory=_no_judge()
            )

    def test_count_rules_run_after_the_existing_alignment_checks(self):
        with pytest.raises(
            ValueError,
            match="reference count \\(1\\) must match prediction count \\(2\\)",
        ):
            evaluate_metric(
                MetricRegistry.get_metric("Accuracy"),
                [[_text("a"), _text("b")], _text("c")],
                [_text("a")],
                client_factory=_no_judge(),
            )


class TestCompositeLabelSets:
    """Several items in a label-set metric are the labels of one set."""

    @pytest.mark.parametrize(
        ("configuration", "scores"),
        [
            (None, [1.0, 0.0]),
            # The labels matcher reads the items as one set too: in the second
            # sample x pairs and y has no pair, so the sample F1 is 2 / 3.
            (
                EvaluationConfiguration(aggregation_method="prf", matcher=LabelSet()),
                [1.0, 2 / 3],
            ),
        ],
        ids=["whole", "labels"],
    )
    def test_items_and_a_list_valued_item_build_the_same_label_set(
        self, configuration, scores
    ):
        metric = MetricRegistry.get_metric("Accuracy")

        as_items = evaluate_metric(
            metric,
            [[_text("a"), _text("b")], [_text("x")]],
            [[_text("b"), _text("a")], [_text("y"), _text("x")]],
            configuration=configuration,
            client_factory=_no_judge(),
        )
        as_lists = evaluate_metric(
            metric,
            [_text(["a", "b"]), _text(["x"])],
            [_text(["b", "a"]), _text(["y", "x"])],
            configuration=configuration,
            client_factory=_no_judge(),
        )

        assert _snapshot(as_items) == _snapshot(as_lists)
        assert [sample.score for sample in as_items.samples] == pytest.approx(scores)

    def test_match_comparator_receives_the_items_as_one_list(self):
        exact = MetricRegistry.get_metric("ListExactMatch")
        raw = MetricRegistry.get_metric("RawExactMatch")
        predictions = [[_text("a"), _text("b")]]
        references = [[_text("b"), _text("a")]]

        assert (
            evaluate_metric(
                exact, predictions, references, client_factory=_no_judge()
            ).score
            == 1.0
        )
        assert (
            evaluate_metric(
                raw, predictions, references, client_factory=_no_judge()
            ).score
            == 0.0
        )


class TestAlignedNumericErrors:
    """MAE, MSE and RMSE score aligned pairs and report every pair's error."""

    def test_mae_reports_pair_errors_and_their_mean(self):
        result = evaluate_metric(
            MetricRegistry.get_metric("MeanAbsoluteError"),
            [[_image(1.0), _image(4.0)], _image(2.0)],
            [[_image(1.5), _image(2.0)], _image(2.0)],
            client_factory=_no_judge(),
        )

        assert result.status is EvaluationStatus.COMPLETED
        assert result.samples[0].score == 1.25
        assert result.samples[0].details == {"item_scores": [0.5, 2.0]}
        assert result.samples[1].score == 0.0
        assert result.samples[1].details is None
        assert result.score == 0.625

    def test_mse_reports_squared_pair_errors(self):
        result = evaluate_metric(
            MetricRegistry.get_metric("MeanSquaredError"),
            [[_image(1.0), _image(4.0)]],
            [[_image(1.5), _image(2.0)]],
            client_factory=_no_judge(),
        )

        assert result.samples[0].details == {"item_scores": [0.25, 4.0]}
        assert result.samples[0].score == 2.125

    def test_rmse_averages_pairs_within_a_sample_and_samples_overall(self):
        result = evaluate_metric(
            MetricRegistry.get_metric("RootMeanSquaredError"),
            [[_image(1.0), _image(4.0)], _image(2.0)],
            [[_image(1.5), _image(2.0)], _image(5.0)],
            client_factory=_no_judge(),
        )

        assert result.samples[0].details == {"item_scores": [0.5, 2.0]}
        assert result.samples[0].score == (0.5 + 2.0) / 2
        assert result.samples[1].score == 3.0
        assert result.aggregate.method == "numeric_mean"
        assert result.score == (1.25 + 3.0) / 2


LLM_METRICS = sorted(
    cls.__name__
    for cls in MetricRegistry.metrics.values()
    if issubclass(cls, LLMMetric)
)


class TestEveryLLMMetricReadsSeveralItems:
    """Every LLM metric composes a two-part candidate the same way."""

    @pytest.mark.parametrize("metric_name", LLM_METRICS)
    def test_a_two_part_candidate_is_numbered_and_scored_per_part(self, metric_name):
        client = Mock(spec=["query"])
        client.query.return_value = {
            "reasoning": "Both parts were weighed.",
            "score": 0.5,
            "part_scores": [0.25, 0.75],
        }

        result = evaluate_metric(
            MetricRegistry.get_metric(metric_name),
            [[_text("part-a"), _text("part-b")]],
            [_text("reference")],
            [[_text("context")]],
            [_text("question")],
            client_factory=Mock(return_value=client),
        )

        assert result.status is EvaluationStatus.COMPLETED
        assert result.samples[0].details["item_scores"] == [0.25, 0.75]
        request = client.query.call_args.kwargs
        texts = [block.get("text") for block in request["additional_content"]]
        assert '\n<candidate_response part="1" of="2">' in texts
        assert '\n<candidate_response part="2" of="2">' in texts
        part_scores = request["response_schema"]["properties"]["part_scores"]
        assert (part_scores["minItems"], part_scores["maxItems"]) == (2, 2)


class TestBatchMetricsReadSeveralItems:
    """The matrix call of a batch metric reads several items in a role."""

    def test_the_matrix_call_reads_every_prompt_part_and_item(self):
        client = Mock(spec=["query"])
        client.query.return_value = {
            "reasoning": "Paris is in both lists.",
            "scores": [[0.9, 0.1], [0.2, 0.3]],
        }

        result = evaluate_metric(
            MetricRegistry.get_metric("LLMCorrectness"),
            [[_text("Paris"), _text("Lyon")]],
            [_text(["Paris", "Nice"])],
            prompts=[[_text("Name two cities"), _text("in France")]],
            configuration=EvaluationConfiguration(
                aggregation_method="prf", matcher=ListItems()
            ),
            client_factory=Mock(return_value=client),
        )

        # The two prediction items are one list, like the list-valued
        # reference: one 2 x 2 matrix call.
        assert result.samples[0].matches == {
            "*": Matches(((0.9, 0.1), (0.2, 0.3)), 2, 2)
        }
        request = client.query.call_args.kwargs
        texts = [block.get("text") for block in request["additional_content"]]
        assert texts[:6] == [
            '\n<task part="1" of="2">',
            "Name two cities",
            "</task>",
            '\n<task part="2" of="2">',
            "in France",
            "</task>",
        ]


class TestSeveralContextItems:
    """A context entry in several items, such as retrieved passages, is one part each."""

    # LLMFaithfulness requires context; LLMCorrectness accepts it.
    @pytest.mark.parametrize("metric_name", ["LLMFaithfulness", "LLMCorrectness"])
    def test_each_passage_is_a_context_part_and_a_blank_one_is_skipped(
        self, metric_name
    ):
        client = Mock(spec=["query"])
        client.query.return_value = {"reasoning": "Grounded.", "score": 0.5}
        passages = [_text("Meeting room: B-204."), _text(" "), _text("Lunch at noon.")]

        result = evaluate_metric(
            MetricRegistry.get_metric(metric_name),
            [_text("Room B-204.")],
            [_text("Room B-204.")],
            [[passages]],
            [_text("Where is the meeting?")],
            client_factory=Mock(return_value=client),
        )

        assert result.status is EvaluationStatus.COMPLETED
        content = client.query.call_args.kwargs["additional_content"]
        texts = [block["text"] for block in content]
        start = texts.index('\n<context part="1" of="2">')
        assert texts[start : start + 6] == [
            '\n<context part="1" of="2">',
            "Meeting room: B-204.",
            "</context>",
            '\n<context part="2" of="2">',
            "Lunch at noon.",
            "</context>",
        ]


class TestLoaderArrays:
    """A JSON array of scalars under the extracted key is one item per element."""

    def _records(self, tmp_path, *values):
        path = tmp_path / "records.json"
        path.write_text(json.dumps([{"answer": value} for value in values]))
        return str(path)

    def test_scalars_stay_one_item_and_arrays_become_one_item_per_element(
        self, tmp_path
    ):
        path = self._records(tmp_path, "first", ["second", 3], 4.5)

        loaded = DataHandler.parse_json_file(path, "answer")

        assert [type(value).__name__ for value in loaded] == [
            "TextData",
            "list",
            "TextData",
        ]
        assert loaded[0].get_raw_data() == "first"
        assert [part.get_raw_data() for part in loaded[1]] == ["second", 3]
        assert all(isinstance(part, TextData) for part in loaded[1])
        assert loaded[2].get_raw_data() == 4.5

    @pytest.mark.parametrize(
        "value", [[], [["a"], ["b"]], [{"k": "v"}]], ids=["empty", "arrays", "objects"]
    )
    def test_an_empty_or_nested_array_stays_one_item_with_the_array(
        self, tmp_path, value
    ):
        path = self._records(tmp_path, value)

        (loaded,) = DataHandler.parse_json_file(path, "answer")

        assert isinstance(loaded, TextData)
        assert loaded.get_raw_data() == value

    def test_an_empty_label_set_and_line_items_score_as_one_value(self, tmp_path):
        records = tmp_path / "records.json"
        line_items = [{"sku": "A1", "qty": 2}, {"sku": "B2", "qty": 1}]
        records.write_text(
            json.dumps(
                [
                    {
                        "labels": [],
                        "gold_labels": ["refund"],
                        "items": line_items,
                        "gold_items": line_items[::-1],
                    }
                ]
            )
        )

        def score(metric_name, key, configuration=None):
            return evaluate_metric(
                MetricRegistry.get_metric(metric_name),
                DataHandler.parse_json_file(str(records), key),
                DataHandler.parse_json_file(str(records), f"gold_{key}"),
                configuration=configuration,
                client_factory=_no_judge(),
            )

        # No predicted label: the reference label is missed, nothing is predicted.
        labels = score(
            "Accuracy",
            "labels",
            EvaluationConfiguration(aggregation_method="prf", matcher=LabelSet()),
        )
        assert labels.samples[0].matches == {"refund": Matches((), 0, 1)}
        assert score("ListHungarianMatch", "items").score == 1.0

    def test_a_two_part_candidate_from_records_reaches_the_judge_as_parts(
        self, tmp_path
    ):
        records = tmp_path / "records.json"
        records.write_text(
            json.dumps(
                [
                    {
                        "q": "Sum 2 and 3, then double it.",
                        "pred": ["5", "10"],
                        "gold": "10",
                    }
                ]
            )
        )
        client = Mock(spec=["query"])
        client.query.return_value = {
            "reasoning": "Both steps are right.",
            "score": 1.0,
            "part_scores": [1.0, 1.0],
        }

        result = MetricRegistry.compute_metric(
            "LLMCorrectness",
            DataHandler.parse_json_file(str(records), "pred"),
            DataHandler.parse_json_file(str(records), "gold"),
            None,
            DataHandler.parse_json_file(str(records), "q"),
            client_factory=Mock(return_value=client),
        )

        content = client.query.call_args.kwargs["additional_content"]
        texts = [block["text"] for block in content]
        assert (
            "\nThe candidate response consists of 2 parts that together form one response."
            in texts
        )
        assert '\n<candidate_response part="1" of="2">' in texts
        assert '\n<candidate_response part="2" of="2">' in texts
        assert texts.index("5") < texts.index("10")
        schema = client.query.call_args.kwargs["response_schema"]
        assert schema["properties"]["part_scores"]["maxItems"] == 2
        assert result.score == 1.0
        assert result.samples[0].details == {
            "reasoning": "Both steps are right.",
            "item_scores": [1.0, 1.0],
        }
