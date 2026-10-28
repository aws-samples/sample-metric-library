# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Deterministic regressions for ML-01 fail-closed correctness behavior."""

import json
import re
from fractions import Fraction
from unittest.mock import Mock

import numpy as np
import pytest

import metric_library.metrics  # noqa: F401 - registers built-in metrics
from metric_library.core.aggregation import EvaluationStatus, SampleStatus
from metric_library.core.base import Metric
from metric_library.core.data import Modality, factory
from metric_library.core.data_handler import DataHandler
from media_double import PngImageData
from metric_library.core.input_policy import InputPolicy
from metric_library.core.llm_metric import LLMMetric
from metric_library.core.registry import MetricRegistry


@pytest.fixture(autouse=True)
def _forbid_live_aws(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("live AWS is forbidden in tests")

    monkeypatch.setattr("metric_library.core.bedrock_client.boto3.client", fail)
    monkeypatch.setattr("metric_library.core.bedrock_client.boto3.Session", fail)


# Every registered LLMMetric subclass. Discovered from the registry so a new
# metric is covered automatically instead of needing a hand-maintained table.
LLM_METRIC_CLASSES = sorted(
    (cls for cls in MetricRegistry.metrics.values() if issubclass(cls, LLMMetric)),
    key=lambda cls: cls.__name__,
)

# For each policy attribute: the compute_metric input it governs, the opening
# delimiter LLMMetric emits when it forwards that input to the judge, and the
# error raised when a REQUIRED input is missing. This is the contract each
# policy value promises.
INPUT_CONTRACTS = {
    "reference_policy": (
        "references",
        "\n<ground_truth_response>",
        "references are required",
    ),
    "prompt_policy": ("prompts", "\n<task>", "prompts are required"),
    "context_policy": ("context", "\n<context>", "context is required"),
}

SENTINELS = {
    "references": "reference sentinel",
    "prompts": "prompt sentinel",
    "context": "context sentinel",
}


def _text(value):
    return factory.create(Modality.TEXT, value)


class UnsendableImage(PngImageData):
    """An image whose judge block cannot be built, like one above the limits."""

    def get_data_for_LLM_message(self) -> dict:
        raise ValueError("cannot send this image to the LLM judge")


def _class_name(cls):
    return cls.__name__


def _all_inputs(**overrides):
    """One aligned prediction plus every auxiliary input, unless overridden."""
    inputs = {
        "references": [_text(SENTINELS["references"])],
        "prompts": [_text(SENTINELS["prompts"])],
        "context": [[_text(SENTINELS["context"])]],
    }
    inputs.update(overrides)
    return inputs


def _judge_client(*responses):
    """A stub judge whose successive ``query`` calls return ``responses`` in order."""
    client = Mock()
    client.query.side_effect = list(responses)
    return client


def _message_text(judge_call):
    return [item.get("text", "") for item in judge_call.kwargs["additional_content"]]


def _compute_with_mock_judge(metric_class, inputs):
    """Run the metric through the engine with a stubbed judge; return message text."""
    client = _judge_client({"reasoning": "ok", "score": 0.5})
    result = MetricRegistry.compute_metric(
        metric_class.__name__,
        [_text("candidate")],
        inputs["references"],
        inputs["context"],
        inputs["prompts"],
        client_factory=Mock(return_value=client),
    )

    assert result.score == 0.5
    return _message_text(client.query.call_args)


class ScoreFromClient(Metric):
    """Fixture metric whose per-sample score is whatever the injected client carries."""

    def compute(self, sample, client):
        return client.value


def _evaluate_returned_score(returned, monkeypatch):
    """Run ``ScoreFromClient`` through the engine with ``returned`` as its sample score."""
    monkeypatch.setitem(MetricRegistry.metrics, "scorefromclient", ScoreFromClient)

    return MetricRegistry.compute_metric(
        "ScoreFromClient",
        [_text("a")],
        [_text("a")],
        client_factory=Mock(return_value=Mock(value=returned)),
    )


class TestLLMInputPolicies:
    """Verify every built-in LLM metric declares, and honors, its input contract."""

    def test_discovery_finds_builtin_llm_metrics(self):
        # Guard against discovery silently returning nothing (e.g. import order),
        # which would make the parametrized tests below vanish instead of fail.
        names = {cls.__name__ for cls in LLM_METRIC_CLASSES}
        assert {"LLMJudge", "LLMReadability"} <= names

    @pytest.mark.parametrize("metric_class", LLM_METRIC_CLASSES, ids=_class_name)
    def test_policies_are_declared_explicitly_as_enum_members(self, metric_class):
        for attribute in INPUT_CONTRACTS:
            # __dict__ (not getattr) so inheriting the LLMMetric default is a failure:
            # each metric must state its own contract for every input.
            assert attribute in metric_class.__dict__, (
                f"{metric_class.__name__} inherits {attribute} instead of declaring it"
            )
            assert isinstance(metric_class.__dict__[attribute], InputPolicy)

    @pytest.mark.parametrize("metric_class", LLM_METRIC_CLASSES, ids=_class_name)
    def test_engine_contract_is_inherited_not_overridden(self, metric_class):
        # Per-sample scoring and input validation live in LLMMetric; a metric that
        # overrides them can bypass the fail-closed checks and aggregation rules.
        for attribute in ("validate_evaluation_inputs", "compute"):
            assert attribute not in metric_class.__dict__, (
                f"{metric_class.__name__} overrides {attribute}"
            )
        assert issubclass(metric_class, Metric)

    @pytest.mark.parametrize("metric_class", LLM_METRIC_CLASSES, ids=_class_name)
    def test_supplied_inputs_are_forwarded_unless_declared_omit(self, metric_class):
        message_text = _compute_with_mock_judge(metric_class, _all_inputs())

        assert "candidate" in message_text
        for attribute, (input_name, delimiter, _) in INPUT_CONTRACTS.items():
            policy = getattr(metric_class, attribute)
            sentinel = SENTINELS[input_name]
            if policy == InputPolicy.OMIT:
                assert sentinel not in message_text, f"{input_name} sent despite OMIT"
                assert delimiter not in message_text
            else:
                assert sentinel in message_text, (
                    f"{input_name} dropped despite {policy}"
                )
                assert delimiter in message_text

    def test_image_context_items_reach_the_judge(self):
        """A context image is sent as an image block inside its context delimiters.

        Before, only context items with non-blank text were forwarded, so an
        image context item was dropped without a signal even though LLMMetric
        declares the image modality. The blank item is skipped, so the two
        items that remain are numbered.
        """
        png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
        client = _judge_client({"reasoning": "ok", "score": 0.5})
        result = MetricRegistry.compute_metric(
            "LLMJudge",
            [_text("candidate")],
            [_text("truth")],
            [[PngImageData(png)], [_text("   ")], [_text("doc text")]],
            [_text("question")],
            client_factory=Mock(return_value=client),
        )
        assert result.score == 0.5
        content = client.query.call_args.kwargs["additional_content"]
        texts = [item.get("text", "") for item in content]
        openers = [i for i, text in enumerate(texts) if text.startswith("\n<context")]
        assert [texts[i] for i in openers] == [
            '\n<context part="1" of="2" modality="image">',
            '\n<context part="2" of="2">',
        ], texts
        assert content[openers[0] + 1] == {
            "image": {"format": "png", "source": {"bytes": png}}
        }
        assert content[openers[1] + 1] == {"text": "doc text"}
        assert "   " not in texts

    def test_optional_roles_follow_the_same_content_rule_as_context(self):
        """A blank optional prompt is skipped; an image reference is an image block.

        LLMJudge has an optional prompt and a required reference. Before, a
        blank optional prompt was sent as an empty text block while a blank
        context item was skipped.
        """
        png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
        client = _judge_client({"reasoning": "ok", "score": 0.5})
        result = MetricRegistry.compute_metric(
            "LLMJudge",
            [_text("candidate")],
            [PngImageData(png)],
            None,
            [_text("   ")],
            client_factory=Mock(return_value=client),
        )
        assert result.score == 0.5
        content = client.query.call_args.kwargs["additional_content"]
        texts = [item.get("text") for item in content]
        assert "\n<task>" not in texts
        assert "   " not in texts
        reference_opener = texts.index('\n<ground_truth_response modality="image">')
        assert content[reference_opener + 1] == {
            "image": {"format": "png", "source": {"bytes": png}}
        }

    @pytest.mark.parametrize("attribute", list(INPUT_CONTRACTS))
    @pytest.mark.parametrize("metric_class", LLM_METRIC_CLASSES, ids=_class_name)
    def test_missing_input_fails_closed_only_when_declared_required(
        self, metric_class, attribute
    ):
        input_name, delimiter, required_message = INPUT_CONTRACTS[attribute]
        inputs = _all_inputs(**{input_name: []})

        if getattr(metric_class, attribute) == InputPolicy.REQUIRED:
            client_factory = Mock()
            with pytest.raises(ValueError, match=required_message):
                MetricRegistry.compute_metric(
                    metric_class.__name__,
                    [_text("candidate")],
                    inputs["references"],
                    inputs["context"],
                    inputs["prompts"],
                    client_factory=client_factory,
                )
            # Validation runs before the engine builds a client.
            client_factory.assert_not_called()
        else:
            message_text = _compute_with_mock_judge(metric_class, inputs)
            assert SENTINELS[input_name] not in message_text
            assert delimiter not in message_text

    def test_reference_free_metric_runs_with_empty_reference_list(self):
        client = _judge_client({"reasoning": "No reference is needed.", "score": 0.5})

        results = MetricRegistry.batch_compute(
            ["LLMReadability"],
            [_text("candidate")],
            [],
            client_factory=Mock(return_value=client),
        )

        assert list(results) == ["LLMReadability"]
        assert results["LLMReadability"].status is EvaluationStatus.COMPLETED
        assert results["LLMReadability"].score == 0.5


class TestJudgeReasoningIsKept:
    """The judge's reasoning rides with every LLM sample result."""

    def test_reasoning_is_recorded_per_sample(self):
        client = _judge_client(
            {"reasoning": "Short, plain sentences.", "score": 0.75},
            {"reasoning": "Long, nested clauses.", "score": 0.25},
        )

        result = MetricRegistry.compute_metric(
            "LLMReadability",
            [_text("candidate-0"), _text("candidate-1")],
            [],
            client_factory=Mock(return_value=client),
        )

        assert [sample.score for sample in result.samples] == [0.75, 0.25]
        assert [sample.details for sample in result.samples] == [
            {"reasoning": "Short, plain sentences."},
            {"reasoning": "Long, nested clauses."},
        ]
        assert result.score == 0.5


class TestStrictCardinalityAndExtraction:
    """Verify positional inputs cannot be silently dropped or multiplied."""

    def test_reference_count_must_match_predictions(self):
        client_factory = Mock()

        with pytest.raises(
            ValueError, match=r"reference count \(1\).*prediction.*\(2\)"
        ):
            MetricRegistry.compute_metric(
                "LLMJudge",
                [_text("one"), _text("two")],
                [_text("reference")],
                client_factory=client_factory,
            )

        client_factory.assert_not_called()

    def test_optional_prompt_count_must_match_when_supplied(self):
        client_factory = Mock()

        with pytest.raises(
            ValueError, match=r"prompts count \(1\).*prediction count \(2\)"
        ):
            MetricRegistry.compute_metric(
                "LLMReadability",
                [_text("one"), _text("two")],
                [],
                [],
                [_text("only one prompt")],
                client_factory=client_factory,
            )

        client_factory.assert_not_called()

    def test_context_source_count_must_match_predictions(self):
        client_factory = Mock()

        with pytest.raises(
            ValueError, match=r"context source 0 count \(1\).*prediction count \(2\)"
        ):
            MetricRegistry.compute_metric(
                "LLMSelfKnowledge",
                [_text("one"), _text("two")],
                [],
                [[_text("only one context")]],
                client_factory=client_factory,
            )

        client_factory.assert_not_called()

    def test_non_llm_metric_cardinality_fails_in_registry(self):
        client_factory = Mock()

        result = MetricRegistry.batch_compute(
            ["Accuracy"],
            [_text("one"), _text("two")],
            [_text("one")],
            client_factory=client_factory,
        )

        assert "error" in result["Accuracy"]
        assert "must match prediction count" in result["Accuracy"]["error"]
        client_factory.assert_not_called()

    def test_batch_create_rejects_path_key_count_mismatch(self):
        with pytest.raises(
            ValueError, match=r"data_items count \(2\).*extract_keys count \(1\)"
        ):
            DataHandler.batch_create(["one.json", "two.json"], extract_keys=["pred"])

    def test_parse_json_rejects_a_key_missing_from_one_record(self, tmp_path):
        data_path = tmp_path / "missing.json"
        data_path.write_text(
            json.dumps([{"pred": "one"}, {"other": "two"}]),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match=r"Missing key 'pred' in record 1"):
            DataHandler.parse_json_file(str(data_path), "pred")

    def test_parse_json_uses_the_top_level_key_over_nested_occurrences(self, tmp_path):
        # The record's own key wins; a same-named nested key is not a duplicate.
        data_path = tmp_path / "top_level.json"
        data_path.write_text(
            json.dumps([{"pred": "one", "nested": {"pred": "other"}}]),
            encoding="utf-8",
        )
        items = DataHandler.parse_json_file(str(data_path), "pred")
        assert [item.get_raw_data() for item in items] == ["one"]

    def test_parse_json_rejects_duplicate_nested_keys_in_one_record(self, tmp_path):
        # Without a top-level key, two nested occurrences are ambiguous.
        data_path = tmp_path / "duplicate.json"
        data_path.write_text(
            json.dumps([{"a": {"pred": "one"}, "b": {"pred": "duplicate"}}]),
            encoding="utf-8",
        )
        with pytest.raises(
            ValueError, match=r"Duplicate key 'pred' in record 0.*2 occurrences"
        ):
            DataHandler.parse_json_file(str(data_path), "pred")

    def test_parse_json_rejects_empty_record_set(self, tmp_path):
        data_path = tmp_path / "empty.json"
        data_path.write_text("[]", encoding="utf-8")

        with pytest.raises(ValueError, match="No records found"):
            DataHandler.parse_json_file(str(data_path), "pred")


class TestJudgeResponseValidation:
    """Verify malformed or misleading judge responses never become scores."""

    @pytest.mark.parametrize(
        ("response", "message"),
        [
            ([], "JSON object"),
            ({"score": 0.5}, "reasoning"),
            ({"reasoning": "   ", "score": 0.5}, "reasoning"),
            ({"reasoning": 123, "score": 0.5}, "reasoning"),
            ({"reasoning": "valid"}, "missing score"),
            ({"reasoning": "valid", "score": "0.5"}, "JSON number"),
            ({"reasoning": "valid", "score": True}, "JSON number"),
            ({"reasoning": "valid", "score": float("nan")}, "finite"),
            ({"reasoning": "valid", "score": float("inf")}, "finite"),
            ({"reasoning": "valid", "score": -0.01}, "between 0.0 and 1.0"),
            ({"reasoning": "valid", "score": 1.01}, "between 0.0 and 1.0"),
        ],
    )
    def test_invalid_response_fails_the_sample(self, response, message):
        client = _judge_client(response)

        result = MetricRegistry.compute_metric(
            "LLMJudge",
            [_text("candidate")],
            [_text("reference")],
            client_factory=Mock(return_value=client),
        )

        assert result.status is EvaluationStatus.FAILED
        assert result.score is None
        assert result.samples[0].status is SampleStatus.FAILED
        assert re.search(message, result.samples[0].error)
        # A rejected response is terminal: the judge is not asked again.
        client.query.assert_called_once()

    @pytest.mark.parametrize("score", [0, 0.25, 1])
    def test_finite_in_range_numeric_score_is_accepted(self, score):
        client = _judge_client({"reasoning": "Valid reasoning.", "score": score})

        result = MetricRegistry.compute_metric(
            "LLMJudge",
            [_text("candidate")],
            [_text("reference")],
            client_factory=Mock(return_value=client),
        )

        assert result.status is EvaluationStatus.COMPLETED
        assert result.score == float(score)
        assert type(result.score) is float


class TestReviewedFailClosedBoundaries:
    """Cover fail-open boundaries found by the independent semantic review."""

    @pytest.mark.parametrize("context_value", [None, "", "  \n\t"])
    def test_required_blank_context_fails_before_client(self, context_value):
        client_factory = Mock()

        with pytest.raises(ValueError, match="context contains no usable content"):
            MetricRegistry.compute_metric(
                "LLMSelfKnowledge",
                [_text("candidate")],
                [],
                [[_text(context_value)]],
                client_factory=client_factory,
            )

        client_factory.assert_not_called()

    def test_required_blank_prompt_fails_before_client(self):
        client_factory = Mock()

        with pytest.raises(ValueError, match="prompts contain empty content"):
            MetricRegistry.compute_metric(
                "LLMCorrectness",
                [_text("candidate")],
                [_text("reference")],
                [],
                [_text("  ")],
                client_factory=client_factory,
            )

        client_factory.assert_not_called()

    @pytest.mark.parametrize(
        ("role", "location"),
        [
            ("predictions", "prediction 1"),
            ("references", "reference 1"),
            ("prompts", "prompt 1"),
            ("context", "context source 0 item 1"),
        ],
    )
    def test_an_item_the_judge_cannot_take_fails_before_client(self, role, location):
        """The second sample's item fails the metric before the first judge call."""
        inputs = {
            "predictions": [_text("candidate-0"), _text("candidate-1")],
            "references": [_text("reference-0"), _text("reference-1")],
            "prompts": [_text("question-0"), _text("question-1")],
            "context": [[_text("context-0"), _text("context-1")]],
        }
        if role == "context":
            inputs["context"][0][1] = UnsendableImage(b"image")
        else:
            inputs[role][1] = UnsendableImage(b"image")
        client_factory = Mock()

        with pytest.raises(
            ValueError, match=f"^{location}: cannot send this image to the LLM judge$"
        ):
            MetricRegistry.compute_metric(
                "LLMCorrectness",
                inputs["predictions"],
                inputs["references"],
                inputs["context"],
                inputs["prompts"],
                client_factory=client_factory,
            )

        client_factory.assert_not_called()

    def test_a_part_the_judge_cannot_take_fails_before_client(self):
        """Each part of an entry in several parts is built on its own."""
        client_factory = Mock()

        with pytest.raises(
            ValueError, match="^prediction 0: cannot send this image to the LLM judge$"
        ):
            MetricRegistry.compute_metric(
                "LLMCorrectness",
                [[_text("candidate part 1"), UnsendableImage(b"image")]],
                [_text("reference")],
                None,
                [_text("question")],
                client_factory=client_factory,
            )

        client_factory.assert_not_called()

    def test_an_item_in_an_omitted_role_is_not_built(self):
        """LLMFollowingInstructions omits its reference, so it never reaches the judge."""
        client = _judge_client({"reasoning": "ok", "score": 1.0})

        result = MetricRegistry.compute_metric(
            "LLMFollowingInstructions",
            [_text("Hi.")],
            [UnsendableImage(b"image")],
            None,
            [_text("Say hi.")],
            client_factory=Mock(return_value=client),
        )

        assert result.score == 1.0

    def test_keyed_record_naming_a_file_manifest_stays_that_string(self, tmp_path):
        first_path = tmp_path / "first.txt"
        second_path = tmp_path / "second.txt"
        first_path.write_text("first", encoding="utf-8")
        second_path.write_text("second", encoding="utf-8")
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(
            json.dumps([str(first_path), str(second_path)]),
            encoding="utf-8",
        )
        data_path = tmp_path / "records.json"
        data_path.write_text(
            json.dumps([{"pred": str(manifest_path)}]),
            encoding="utf-8",
        )

        items = DataHandler.parse_json_file(str(data_path), "pred")

        assert len(items) == 1
        assert items[0].modality is Modality.TEXT
        assert items[0].get_raw_data() == str(manifest_path)

    @pytest.mark.parametrize(
        ("prompts", "context", "message"),
        [
            ([object()], [], r"prompt count \(1\).*prediction count \(2\)"),
            ([], [[object()]], r"context source 0 count \(1\).*prediction count \(2\)"),
        ],
    )
    def test_runner_auxiliary_cardinality_is_metric_independent(
        self, prompts, context, message
    ):
        from metric_library.run_metrics import _validate_auxiliary_cardinality

        with pytest.raises(ValueError, match=message):
            _validate_auxiliary_cardinality([object(), object()], prompts, context)

    @pytest.mark.parametrize(
        ("returned", "message"),
        [
            pytest.param(None, "score must be a real number", id="none"),
            pytest.param("0.5", "score must be a real number", id="string"),
            pytest.param(True, "score must be a real number", id="bool"),
            pytest.param(float("nan"), "score must be finite", id="nan"),
            pytest.param(float("inf"), "score must be finite", id="inf"),
        ],
    )
    def test_invalid_sample_score_fails_the_metric(
        self, returned, message, monkeypatch
    ):
        result = _evaluate_returned_score(returned, monkeypatch)

        assert result.status is EvaluationStatus.FAILED
        assert result.score is None
        assert result.samples[0].status is SampleStatus.FAILED
        assert result.samples[0].error == message
        assert not any(
            sample.status is SampleStatus.COMPLETED for sample in result.samples
        )

    def test_structured_result_with_error_field_is_rejected(self, monkeypatch):
        result = _evaluate_returned_score(
            {"error": "estimated error rate", "score": 0.8}, monkeypatch
        )

        assert result.status is EvaluationStatus.FAILED
        assert result.score is None
        assert result.samples[0].status is SampleStatus.FAILED
        assert result.samples[0].error == "score must be a real number"
        assert not any(
            sample.status is SampleStatus.COMPLETED for sample in result.samples
        )


class TestStrictJsonMetricResults:
    """Ensure every successful score is a plain float that serializes as strict JSON."""

    @pytest.mark.parametrize(
        "returned",
        [
            pytest.param(Fraction(1, 2), id="fraction"),
            pytest.param(np.longdouble("0.5"), id="longdouble"),
        ],
    )
    def test_exotic_real_scores_become_plain_floats(self, returned, monkeypatch):
        result = _evaluate_returned_score(returned, monkeypatch)

        assert result.status is EvaluationStatus.COMPLETED
        assert result.samples[0].status is SampleStatus.COMPLETED
        assert result.samples[0].score == 0.5
        assert type(result.samples[0].score) is float
        assert result.score == 0.5
        assert type(result.score) is float
        assert json.loads(json.dumps(result.score, allow_nan=False)) == 0.5

    def test_overflowing_integer_score_fails_the_sample(self, monkeypatch):
        result = _evaluate_returned_score(10**1000, monkeypatch)

        assert result.status is EvaluationStatus.FAILED
        assert result.score is None
        assert result.samples[0].status is SampleStatus.FAILED
        # float(10**1000) raises OverflowError("int too large to convert to float").
        assert "too large" in result.samples[0].error


class TestPerExampleLLMJudging:
    """Preserve main's isolated judge call and mean-aggregation behavior."""

    def test_each_sample_gets_an_isolated_call_and_scores_are_averaged(self):
        predictions = [_text("candidate one"), _text("candidate two")]
        references = [_text("reference one"), _text("reference two")]
        prompts = [_text("question one"), _text("question two")]
        context = [[_text("context one"), _text("context two")]]
        client = _judge_client(
            {"reasoning": "first sample", "score": 0.2},
            {"reasoning": "second sample", "score": 0.8},
        )
        client_factory = Mock(return_value=client)

        result = MetricRegistry.compute_metric(
            "LLMJudge",
            predictions,
            references,
            context,
            prompts,
            client_factory=client_factory,
        )

        assert result.status is EvaluationStatus.COMPLETED
        assert [sample.score for sample in result.samples] == [0.2, 0.8]
        assert result.score == pytest.approx(0.5)
        client_factory.assert_called_once_with()
        assert client.query.call_count == len(predictions)

        first_text = _message_text(client.query.call_args_list[0])
        second_text = _message_text(client.query.call_args_list[1])

        for value in ("candidate one", "reference one", "question one", "context one"):
            assert value in first_text
            assert value not in second_text
        for value in ("candidate two", "reference two", "question two", "context two"):
            assert value in second_text
            assert value not in first_text

    def test_reference_free_metric_still_judges_each_prediction(self):
        answers = ["first answer", "second answer"]
        predictions = [_text(answer) for answer in answers]
        client = _judge_client(
            {"reasoning": "first sample", "score": 0.4},
            {"reasoning": "second sample", "score": 0.6},
        )
        client_factory = Mock(return_value=client)

        result = MetricRegistry.compute_metric(
            "LLMReadability",
            predictions,
            [],
            client_factory=client_factory,
        )

        assert result.status is EvaluationStatus.COMPLETED
        assert result.score == pytest.approx(0.5)
        client_factory.assert_called_once_with()
        assert client.query.call_count == len(predictions)
        for index, judge_call in enumerate(client.query.call_args_list):
            content_text = _message_text(judge_call)
            assert answers[index] in content_text
            assert answers[1 - index] not in content_text
            assert "\n<ground_truth_response>" not in content_text


class FailWithLongMessage(Metric):
    """Fixture metric whose every sample fails with a 1,000-character message."""

    def compute(self, sample, client):
        raise ValueError("x" * 1000)


class TestErrorTextIsBounded:
    """A result carries at most 500 characters of an error message."""

    def test_a_failed_sample_keeps_the_start_of_a_long_message(self, monkeypatch):
        monkeypatch.setitem(
            MetricRegistry.metrics, "failwithlongmessage", FailWithLongMessage
        )
        result = MetricRegistry.compute_metric(
            "FailWithLongMessage",
            [_text("a")],
            [_text("a")],
            client_factory=Mock(spec=[]),
        )
        assert result.samples[0].error == "x" * 488 + " [truncated]"

    def test_a_metric_that_fails_before_scoring_keeps_the_start(self):
        name = "m" * 600
        (error,) = MetricRegistry.batch_compute(
            [name], [_text("a")], [_text("a")]
        ).values()
        assert len(error["error"]) == 500
        assert "m" * 400 in error["error"]
        assert error["error"].endswith(" [truncated]")
