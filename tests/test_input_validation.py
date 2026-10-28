# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Input validation messages on the library path and the CLI path.

``MetricRegistry.compute_metric`` validates through
``Metric.validate_evaluation_inputs`` before it builds a client, and the
``run_metrics`` runner reaches the metric through ``batch_compute``. One defect
therefore has one message on both paths, and neither path touches AWS on the
way to it.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

import metric_library.metrics  # noqa: F401 - registers built-in metrics
from metric_library.core.base import Metric
from metric_library.core.data import Modality
from metric_library.core.data.text_data import TextData
from media_double import images
from metric_library.core.registry import MetricRegistry

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _forbid_live_aws(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("live AWS is forbidden in tests")

    monkeypatch.setattr("metric_library.core.bedrock_client.boto3.client", fail)
    monkeypatch.setattr("metric_library.core.bedrock_client.boto3.Session", fail)


PREDICTIONS = [TextData("answer-0"), TextData("answer-1")]
REFERENCES = [TextData("answer-0"), TextData("truth-1")]
PROMPTS = [TextData("question-0"), TextData("question-1")]
CONTEXT = [[TextData("context-0"), TextData("context-1")]]

# Single-defect inputs as (predictions, references, context, prompts, message).
# Prompts and context are aligned wherever they are not the defect, so that a
# REQUIRED or OPTIONAL policy never adds a second defect.
EMPTY_PREDICTIONS = pytest.param(
    [],
    [],
    None,
    None,
    "predictions must contain at least one item",
    id="empty-predictions",
)
MISSING_REFERENCES = pytest.param(
    PREDICTIONS,
    [],
    CONTEXT,
    PROMPTS,
    "references are required for {name}",
    id="missing-references",
)
MISALIGNED_REFERENCES = pytest.param(
    PREDICTIONS,
    REFERENCES[:1],
    CONTEXT,
    PROMPTS,
    "reference count (1) must match prediction count (2) for {name}",
    id="misaligned-references",
)
MISSING_PROMPTS = pytest.param(
    PREDICTIONS,
    REFERENCES,
    CONTEXT,
    [],
    "prompts are required for {name}",
    id="missing-prompts",
)
MISALIGNED_PROMPTS = pytest.param(
    PREDICTIONS,
    REFERENCES,
    CONTEXT,
    PROMPTS[:1],
    "prompts count (1) must match prediction count (2) for {name}",
    id="misaligned-prompts",
)
MISALIGNED_CONTEXT_SOURCE = pytest.param(
    PREDICTIONS,
    REFERENCES,
    [CONTEXT[0][:1]],
    PROMPTS,
    "context source 0 count (1) must match prediction count (2) for {name}",
    id="misaligned-context-source",
)

# Defects every metric reports through the inherited Metric validator.
SHARED_DEFECTS = [EMPTY_PREDICTIONS, MISSING_REFERENCES, MISALIGNED_REFERENCES]

# LLMCorrectness declares prompts REQUIRED and context OPTIONAL.
LLM_CORRECTNESS_DEFECTS = SHARED_DEFECTS + [
    MISSING_PROMPTS,
    MISALIGNED_PROMPTS,
    MISALIGNED_CONTEXT_SOURCE,
]

# The defects the runner hands on to the metric. The loader rejects an empty
# record set, and the runner's own prompt and context cardinality check fires
# before the metric with a different message, so those three never reach it.
CLI_REACHABLE_DEFECTS = [MISSING_REFERENCES, MISALIGNED_REFERENCES, MISSING_PROMPTS]


def _run_cli_on(tmp_path, predictions, references, context, prompts):
    """Run ``run_metrics`` on LLMCorrectness over JSON files encoding the inputs.

    The execution output holds one record per prediction with the prompt and
    context values under their own keys. References go to a separate file so a
    shorter reference list can be encoded. An empty role is left off the command
    line, which the runner turns into the same empty list the in-process call
    passes. Validation runs before the runner builds a client, so the
    subprocess never needs AWS access; it inherits the environment as-is.
    """
    execution_path = tmp_path / "execution.json"
    output_path = tmp_path / "result.json"
    records = [{"pred": prediction.get_raw_data()} for prediction in predictions]
    args = [
        "--execution-output-path",
        str(execution_path),
        "--metrics",
        "LLMCorrectness",
        "--pred-key",
        "pred",
        "--output-path",
        str(output_path),
    ]

    if references:
        reference_path = tmp_path / "references.json"
        reference_path.write_text(
            json.dumps([{"gold": reference.get_raw_data()} for reference in references])
        )
        args += ["--ground-truth-path", str(reference_path)]
        args += ["--ground-truth-key", "gold"]
    if prompts:
        for record, prompt in zip(records, prompts, strict=True):
            record["prompt"] = prompt.get_raw_data()
        args += ["--prompt-key", "prompt"]
    if context:
        context_keys = []
        for source_index, context_source in enumerate(context):
            key = f"context{source_index}"
            for record, item in zip(records, context_source, strict=True):
                record[key] = item.get_raw_data()
            context_keys.append(key)
        args += ["--context", ",".join(context_keys)]
    execution_path.write_text(json.dumps(records))

    # No shell, and every argument comes from the test.
    return subprocess.run(  # nosec B603 # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit
        [sys.executable, "-m", "metric_library.run_metrics", *args],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
    )


class TestSharedCardinalityMessages:
    """A deterministic metric and an LLM metric report one wording per defect."""

    @pytest.mark.parametrize("metric_name", ["Accuracy", "LLMCorrectness"])
    @pytest.mark.parametrize(
        ("predictions", "references", "context", "prompts", "message"),
        SHARED_DEFECTS,
    )
    def test_compute_metric_reports_the_metric_named_message(
        self, metric_name, predictions, references, context, prompts, message
    ):
        expected = message.format(name=metric_name)
        factory = Mock(spec=[])

        with pytest.raises(ValueError, match=re.escape(expected)):
            MetricRegistry.compute_metric(
                metric_name,
                predictions,
                references,
                context,
                prompts,
                client_factory=factory,
            )

        factory.assert_not_called()


class TestLLMCorrectnessPaths:
    """``compute_metric`` raises, and the CLI reports, one message per defect."""

    @pytest.mark.parametrize(
        ("predictions", "references", "context", "prompts", "message"),
        LLM_CORRECTNESS_DEFECTS,
    )
    def test_compute_metric_rejects_every_defect_before_any_client(
        self, predictions, references, context, prompts, message
    ):
        expected = message.format(name="LLMCorrectness")
        factory = Mock(spec=[])

        with pytest.raises(ValueError, match=re.escape(expected)):
            MetricRegistry.compute_metric(
                "LLMCorrectness",
                predictions,
                references,
                context,
                prompts,
                client_factory=factory,
            )

        factory.assert_not_called()

    @pytest.mark.parametrize(
        ("predictions", "references", "context", "prompts", "message"),
        CLI_REACHABLE_DEFECTS,
    )
    def test_compute_metric_and_cli_report_the_same_message(
        self, tmp_path, predictions, references, context, prompts, message
    ):
        expected = message.format(name="LLMCorrectness")
        factory = Mock(spec=[])

        with pytest.raises(ValueError) as excinfo:
            MetricRegistry.compute_metric(
                "LLMCorrectness",
                predictions,
                references,
                context,
                prompts,
                client_factory=factory,
            )
        raised = str(excinfo.value)
        assert raised == expected
        factory.assert_not_called()

        result = _run_cli_on(tmp_path, predictions, references, context, prompts)

        assert result.returncode == 1, result.stderr
        output = json.loads(result.stdout)
        assert output["status"] == "failed"
        assert output["data"]["results"]["LLMCorrectness"] == {"error": raised}
        assert output["errors"] == [f"LLMCorrectness: {raised}"]

    def test_alignment_defect_is_reported_before_content_defect(self):
        predictions = [TextData("answer-0"), TextData("")]
        references = REFERENCES[:1]
        expected = (
            "reference count (1) must match prediction count (2) for LLMCorrectness"
        )
        factory = Mock(spec=[])

        with pytest.raises(ValueError, match=re.escape(expected)):
            MetricRegistry.compute_metric(
                "LLMCorrectness",
                predictions,
                references,
                CONTEXT,
                PROMPTS,
                client_factory=factory,
            )

        factory.assert_not_called()


class TestPolicyCoercion:
    """A declared policy outside InputPolicy fails with the role and metric named."""

    def test_invalid_policy_message_names_role_and_metric(self, monkeypatch):
        class SometimesPrompted(Metric):
            prompt_policy = "sometimes"

            def compute(self, sample, client):
                return 1.0

        monkeypatch.setitem(
            MetricRegistry.metrics, "sometimesprompted", SometimesPrompted
        )
        expected = (
            "Invalid prompt policy 'sometimes' for SometimesPrompted; "
            "expected one of ['required', 'optional', 'omit']"
        )
        factory = Mock(spec=[])

        with pytest.raises(ValueError, match=re.escape(expected)):
            MetricRegistry.compute_metric(
                "SometimesPrompted",
                PREDICTIONS,
                REFERENCES,
                client_factory=factory,
            )

        factory.assert_not_called()


class TestModalityEnforcement:
    """Every item of a role the metric reads must be a declared modality."""

    IMAGES = images([np.array([0.0, 4.0]), np.array([3.0, 3.0])])
    ZEROS = images([np.zeros(2), np.zeros(2)])

    @pytest.mark.parametrize(
        "name", ["RootMeanSquaredError", "MeanSquaredError", "MeanAbsoluteError"]
    )
    def test_error_metric_rejects_plain_numbers_before_any_client(self, name):
        # Plain numbers load as text; the error metrics accept audio, image and
        # video only. Before, the mean of |p - r| came back labelled RMSE.
        expected = (
            f"{name} does not accept text data (prediction 0); "
            "supported: audio, image, video"
        )
        factory = Mock(spec=[])
        with pytest.raises(ValueError, match=re.escape(expected)):
            MetricRegistry.compute_metric(
                name,
                [TextData(2), TextData(-2)],
                [TextData(0), TextData(0)],
                client_factory=factory,
            )
        factory.assert_not_called()

    def test_the_message_names_the_first_offending_role_and_index(self):
        expected = (
            "MeanSquaredError does not accept text data (reference 1); "
            "supported: audio, image, video"
        )
        with pytest.raises(ValueError, match=re.escape(expected)):
            MetricRegistry.compute_metric(
                "MeanSquaredError",
                self.IMAGES,
                [self.ZEROS[0], TextData(0)],
                client_factory=Mock(spec=[]),
            )

    @pytest.mark.parametrize(
        "name", ["Accuracy", "TextExactMatch", "NumericExactMatch"]
    )
    def test_label_and_match_metrics_reject_images(self, name):
        # These metrics compare strings, numbers and lists, which load as text.
        expected = f"{name} does not accept image data (prediction 0); supported: text"
        with pytest.raises(ValueError, match=re.escape(expected)):
            MetricRegistry.compute_metric(
                name, self.IMAGES, self.ZEROS, client_factory=Mock(spec=[])
            )

    def test_omitted_roles_are_not_checked(self):
        # The error metrics drop prompts and context, so text there is fine.
        result = MetricRegistry.compute_metric(
            "MeanSquaredError",
            self.IMAGES,
            self.ZEROS,
            [[TextData("c0"), TextData("c1")]],
            [TextData("q0"), TextData("q1")],
            client_factory=Mock(spec=[]),
        )
        assert result.score == pytest.approx((8.0 + 9.0) / 2)

    def test_a_custom_metric_restricted_to_text_rejects_images(self):
        class TextOnly(Metric):
            supported_data_types = [Modality.TEXT]

            def compute(self, sample, client):
                return 1.0

        try:
            MetricRegistry.register(TextOnly)
            expected = (
                "TextOnly does not accept image data (prediction 0); supported: text"
            )
            with pytest.raises(ValueError, match=re.escape(expected)):
                MetricRegistry.compute_metric(
                    "TextOnly", self.IMAGES, self.ZEROS, client_factory=Mock(spec=[])
                )
            assert (
                MetricRegistry.compute_metric(
                    "TextOnly", PREDICTIONS, REFERENCES, client_factory=Mock(spec=[])
                ).score
                == 1.0
            )
        finally:
            MetricRegistry.metrics.pop("textonly", None)


class TestOmittedRoles:
    """OMIT drops prompts and context of any count instead of rejecting them."""

    @pytest.mark.parametrize(
        ("context", "prompts"),
        [
            pytest.param([CONTEXT[0][:1]], None, id="short-context"),
            pytest.param(None, PROMPTS[:1], id="short-prompts"),
            pytest.param(
                [CONTEXT[0][:1]],
                PROMPTS + [TextData("question-2")],
                id="short-context-and-long-prompts",
            ),
        ],
    )
    def test_accuracy_ignores_misaligned_prompts_and_context(self, context, prompts):
        result = MetricRegistry.compute_metric(
            "Accuracy",
            PREDICTIONS,
            REFERENCES,
            context,
            prompts,
            client_factory=Mock(spec=[]),
        )

        assert isinstance(result.score, float)
        assert (
            result.score
            == MetricRegistry.compute_metric(
                "Accuracy", PREDICTIONS, REFERENCES, client_factory=Mock(spec=[])
            ).score
        )
