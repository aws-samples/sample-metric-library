# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Tests for run_metrics and custom_metrics.

Verifies that run_metrics works end-to-end, loads custom metrics only from
--custom-metrics-path, and that the no-op custom_metrics.py is valid Python.
"""

import ast
import base64
import importlib.util
import json
import logging
import os
import re
import shutil
import stat
import struct
import subprocess
import sys
import textwrap
import threading
import zlib
from datetime import datetime, timedelta, timezone
from pathlib import Path, PureWindowsPath
from unittest.mock import Mock

import pytest

import metric_library.metrics  # noqa: F401 - registers built-in metrics
from metric_library.core.aggregation import (
    Matches,
    SampleResult,
    SampleStatus,
    median,
    numeric_mean,
    prf,
    quantile,
    weighted_mean,
)
from metric_library.core.data import Modality
from metric_library.core.data.text_data import TextData
from metric_library.core.data_handler import DataHandler
from metric_library.core.registry import MetricRegistry

REPO_ROOT = Path(__file__).resolve().parent.parent

ENVELOPE_KEYS = {"ok", "status", "data", "errors", "meta"}
# The printed summary leaves out meta and adds the file that holds the full
# envelope.
STDOUT_KEYS = {"ok", "status", "data", "errors", "output_path"}


def _summary_of(envelope, output_path):
    """Return the envelope the runner prints by default for a file envelope."""
    data = envelope["data"]
    if data is not None:
        data = {key: value for key, value in data.items() if key != "samples"}
    summary = {key: value for key, value in envelope.items() if key != "meta"}
    return {**summary, "data": data, "output_path": str(output_path)}


def _full_output(result, output_path, stream="stdout"):
    """Return the file envelope after checking the printed one is its summary.

    The runner prints the envelope without ``data.samples`` and with
    ``output_path``; the file at ``output_path`` holds the full envelope.
    """
    printed = json.loads(getattr(result, stream))
    full = json.loads(Path(output_path).read_text())
    assert printed == _summary_of(full, output_path)
    return full


SAMPLE_ENTRY_KEYS = {
    "sample_id",
    "status",
    "score",
    "error",
    "matches",
    "item_scores",
    "reasoning",
}

# The labels matcher on the ``A/A, B/B, C/D`` records of ``sample_data``: one
# match record per lowercased label, and the per-class table that the prf
# aggregator reports for it. C and D are in different classes, so they cannot
# pair: class c has one predicted item and no reference item, class d the
# reverse.
SMALL_FIXTURE_MATCHES = [
    {"a": {"scores": [[1.0]], "n_pred": 1, "n_ref": 1}},
    {"b": {"scores": [[1.0]], "n_pred": 1, "n_ref": 1}},
    {
        "c": {"scores": [[]], "n_pred": 1, "n_ref": 0},
        "d": {"scores": [], "n_pred": 0, "n_ref": 1},
    },
]
SMALL_FIXTURE_PER_CLASS = {
    "a": {
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "tp": 1,
        "fp": 0,
        "fn": 0,
        "tn": 2,
        "support": 1,
    },
    "b": {
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "tp": 1,
        "fp": 0,
        "fn": 0,
        "tn": 2,
        "support": 1,
    },
    "c": {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "tp": 0,
        "fp": 1,
        "fn": 0,
        "tn": 2,
        "support": 0,
    },
    "d": {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "tp": 0,
        "fp": 0,
        "fn": 1,
        "tn": 2,
        "support": 1,
    },
}
SMALL_FIXTURE_PRF = {
    "threshold": 0.5,
    "precision": 0.5,
    "recall": 0.5,
    "f1": 0.5,
    "per_class": SMALL_FIXTURE_PER_CLASS,
}


@pytest.fixture()
def sample_data(tmp_path):
    """Create a minimal JSON prediction file for testing."""
    data = [
        {"pred": "A", "gold": "A"},
        {"pred": "B", "gold": "B"},
        {"pred": "C", "gold": "D"},
    ]
    path = tmp_path / "data.json"
    path.write_text(json.dumps(data))
    return path


def _run_metrics(args, extra_pythonpath=None, cwd=None, env=None, stdout=None):
    """Helper to invoke run_metrics as a subprocess.

    Runs in the repository root unless ``cwd`` is given, and puts
    ``extra_pythonpath`` before the repository on PYTHONPATH. ``env`` adds to
    or overrides the inherited environment. ``stdout``, an open file, takes the
    runner's standard output in place of the captured one.
    """
    pythonpath = str(REPO_ROOT)
    if extra_pythonpath:
        pythonpath = f"{extra_pythonpath}:{pythonpath}"
    env = {**os.environ, "PYTHONPATH": pythonpath, **(env or {})}
    # No shell, and every argument comes from the test.
    return subprocess.run(  # nosec B603 # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit
        [sys.executable, "-m", "metric_library.run_metrics", *args],
        stdout=stdout or subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=cwd or str(REPO_ROOT),
        env=env,
    )


def _sample_results(entries):
    """Rebuild the engine's ``SampleResult`` objects from ``data.samples[name]``.

    The public aggregators consume ``SampleResult`` sequences, so this lets a
    test feed the per-sample entries of an envelope back into the aggregator
    the CLI claims to have applied. An entry's ``matches`` mapping becomes the
    sample's ``Matches`` records, which the ``prf`` aggregator reads.
    """
    return [
        SampleResult(
            sample_id=entry["sample_id"],
            ordinal=ordinal,
            status=SampleStatus(entry["status"]),
            score=entry["score"],
            error=entry["error"],
            matches=(
                {
                    item_class: Matches(**record)
                    for item_class, record in entry["matches"].items()
                }
                if entry["matches"] is not None
                else None
            ),
        )
        for ordinal, entry in enumerate(entries)
    ]


def _run_evaluation(
    data_path,
    output_path,
    metrics,
    *extra_args,
    pred_key="pred",
    ground_truth_key="gold",
):
    """Run the runner on ``data_path`` for ``metrics`` with extra options."""
    return _run_metrics(
        [
            "--execution-output-path",
            str(data_path),
            "--metrics",
            metrics,
            "--pred-key",
            pred_key,
            "--ground-truth-key",
            ground_truth_key,
            *extra_args,
            "--output-path",
            str(output_path),
        ]
    )


def _write_identified_records(path, identifiers):
    """Write the ``A/A, B/B, C/D`` records of ``sample_data``, each with an ``id``."""
    first, second, third = identifiers
    path.write_text(
        json.dumps(
            [
                {"id": first, "pred": "A", "gold": "A"},
                {"id": second, "pred": "B", "gold": "B"},
                {"id": third, "pred": "C", "gold": "D"},
            ]
        )
    )
    return path


class TestEndToEnd:
    """End-to-end tests that run the full pipeline."""

    def test_basic_evaluation(self, sample_data, tmp_path):
        """Run Accuracy on sample data and verify the score."""
        output_path = tmp_path / "result.json"
        result = _run_metrics(
            [
                "--execution-output-path",
                str(sample_data),
                "--metrics",
                "Accuracy",
                "--pred-key",
                "pred",
                "--ground-truth-key",
                "gold",
                "--output-path",
                str(output_path),
            ]
        )
        assert result.returncode == 0, result.stderr
        output = _full_output(result, output_path)
        assert output["ok"] is True
        assert output["data"]["results"]["Accuracy"] == pytest.approx(2 / 3)

    def test_multiple_metrics(self, sample_data, tmp_path):
        """Run multiple metrics in a single invocation."""
        output_path = tmp_path / "result.json"
        result = _run_metrics(
            [
                "--execution-output-path",
                str(sample_data),
                "--metrics",
                "Accuracy,TextExactMatch",
                "--pred-key",
                "pred",
                "--ground-truth-key",
                "gold",
                "--output-path",
                str(output_path),
            ]
        )
        assert result.returncode == 0, result.stderr
        output = _full_output(result, output_path)
        assert output["ok"] is True
        assert len(output["data"]["results"]) == 2
        for name, value in output["data"]["results"].items():
            assert isinstance(value, float), f"{name} is not a float"

    def test_classification_metrics_keep_v1_envelope(self, sample_data, tmp_path):
        """Float-valued metrics keep the v1 envelope and values through the engine."""
        output_path = tmp_path / "result.json"
        result = _run_metrics(
            [
                "--execution-output-path",
                str(sample_data),
                "--metrics",
                "Accuracy,TextExactMatch",
                "--pred-key",
                "pred",
                "--ground-truth-key",
                "gold",
                "--output-path",
                str(output_path),
            ]
        )
        assert result.returncode == 0, result.stderr
        output = _full_output(result, output_path)
        assert set(output) == ENVELOPE_KEYS
        assert output["ok"] is True
        assert output["status"] == "completed"
        assert output["errors"] == []
        results = output["data"]["results"]
        assert set(results) == {"Accuracy", "TextExactMatch"}
        assert all(isinstance(value, float) for value in results.values())
        assert results["Accuracy"] == pytest.approx(2 / 3)
        # A/A and B/B match exactly and score 1.0 each; C/D scores 0.0
        assert results["TextExactMatch"] == pytest.approx(2 / 3)

    def test_per_sample_results_are_reported_next_to_the_aggregate(
        self, sample_data, tmp_path
    ):
        """``data.samples[name]`` lists one entry per input pair, in input order.

        Without ``--sample-id-key`` the records are numbered from 1, so the
        ``sample_id`` of the first record is ``"1"``.
        """
        output_path = tmp_path / "result.json"
        result = _run_metrics(
            [
                "--execution-output-path",
                str(sample_data),
                "--metrics",
                "Accuracy,TextExactMatch",
                "--pred-key",
                "pred",
                "--ground-truth-key",
                "gold",
                "--output-path",
                str(output_path),
            ]
        )
        assert result.returncode == 0, result.stderr
        output = _full_output(result, output_path)
        assert set(output["data"]) == {"results", "samples"}
        assert set(output["data"]["samples"]) == set(output["data"]["results"])
        for entries in output["data"]["samples"].values():
            assert all(set(entry) == SAMPLE_ENTRY_KEYS for entry in entries)
            # The default identifier is the 1-based record position.
            assert [entry["sample_id"] for entry in entries] == ["1", "2", "3"]
        accuracy_samples = output["data"]["samples"]["Accuracy"]
        assert accuracy_samples == [
            {
                "sample_id": "1",
                "status": "completed",
                "score": 1.0,
                "error": None,
                "matches": None,
                "item_scores": None,
                "reasoning": None,
            },
            {
                "sample_id": "2",
                "status": "completed",
                "score": 1.0,
                "error": None,
                "matches": None,
                "item_scores": None,
                "reasoning": None,
            },
            {
                "sample_id": "3",
                "status": "completed",
                "score": 0.0,
                "error": None,
                "matches": None,
                "item_scores": None,
                "reasoning": None,
            },
        ]
        mean_score = sum(entry["score"] for entry in accuracy_samples) / len(
            accuracy_samples
        )
        assert mean_score == pytest.approx(output["data"]["results"]["Accuracy"])
        assert json.loads(output_path.read_text()) == output

    def test_error_metrics_report_the_mean_on_the_image_example(self, tmp_path):
        """The error metrics average their per-sample values on the image example.

        The generated images carry the constant offsets 10, 0 and 20, so each
        per-sample RMSE and MAE is that offset and each per-sample MSE its square.
        """
        # A subprocess keeps PIL and the image data type out of this process.
        if importlib.util.find_spec("PIL") is None:
            pytest.skip("Pillow is not installed")
        output_path = tmp_path / "result.json"
        result = _run_evaluation(
            REPO_ROOT / "tests" / "examples" / "rmse_image_renders.json",
            output_path,
            "RootMeanSquaredError,MeanSquaredError,MeanAbsoluteError",
            "--sample-id-key",
            "id",
            pred_key="generated_image",
            ground_truth_key="reference_image",
        )
        assert result.returncode == 0, result.stderr
        output = _full_output(result, output_path)
        assert set(output) == ENVELOPE_KEYS
        assert output["ok"] is True
        assert output["status"] == "completed"
        assert output["errors"] == []
        results = output["data"]["results"]
        samples = output["data"]["samples"]
        expected_samples = {
            "RootMeanSquaredError": [10.0, 0.0, 20.0],
            "MeanSquaredError": [100.0, 0.0, 400.0],
            "MeanAbsoluteError": [10.0, 0.0, 20.0],
        }
        for name, scores in expected_samples.items():
            assert [entry["sample_id"] for entry in samples[name]] == ["1", "2", "3"]
            assert [entry["score"] for entry in samples[name]] == scores, name
            aggregate = numeric_mean(_sample_results(samples[name]))
            assert results[name] == aggregate.score, name
        assert results["RootMeanSquaredError"] == 10.0
        assert results["MeanSquaredError"] == pytest.approx(500 / 3)
        assert results["MeanAbsoluteError"] == 10.0
        assert json.loads(output_path.read_text()) == output

    def test_error_metrics_reject_plain_numbers(self, tmp_path):
        """Plain numbers load as text, which the error metrics do not accept.

        Each metric fails before scoring and names the modality rule. Before,
        the run completed and reported the mean of |p - r| as the RMSE.
        """
        records_path = tmp_path / "numbers.json"
        records_path.write_text(
            json.dumps([{"pred": p, "gold": 0} for p in (2, 0, -2, 4)])
        )
        output_path = tmp_path / "result.json"
        metrics = ["RootMeanSquaredError", "MeanSquaredError", "MeanAbsoluteError"]
        result = _run_evaluation(records_path, output_path, ",".join(metrics))
        assert result.returncode == 1, result.stderr
        output = _full_output(result, output_path)
        assert output["ok"] is False
        assert output["status"] == "failed"
        messages = {
            name: f"{name} does not accept text data (prediction 0); "
            "supported: audio, image, video"
            for name in metrics
        }
        assert output["data"]["results"] == {
            name: {"error": message} for name, message in messages.items()
        }
        assert output["data"]["samples"] == {}
        assert output["errors"] == [
            f"{name}: {message}" for name, message in messages.items()
        ]
        assert json.loads(output_path.read_text()) == output

    def test_results_written_to_file(self, sample_data, tmp_path):
        """Verify results are written to the output file, not just stdout."""
        output_path = tmp_path / "subdir" / "result.json"
        result = _run_metrics(
            [
                "--execution-output-path",
                str(sample_data),
                "--metrics",
                "Accuracy",
                "--pred-key",
                "pred",
                "--ground-truth-key",
                "gold",
                "--output-path",
                str(output_path),
            ]
        )
        assert result.returncode == 0, result.stderr
        assert output_path.exists()
        file_output = json.loads(output_path.read_text())
        assert file_output["ok"] is True

    def test_output_path_no_directory(self, sample_data, tmp_path):
        """Output path with no directory component should not crash."""
        result = _run_metrics(
            [
                "--execution-output-path",
                str(sample_data),
                "--metrics",
                "Accuracy",
                "--pred-key",
                "pred",
                "--ground-truth-key",
                "gold",
                "--output-path",
                "result.json",
            ],
            cwd=str(tmp_path),
        )
        assert result.returncode == 0, result.stderr
        assert (tmp_path / "result.json").exists()

    def test_unknown_metric_returns_partial_failure(self, sample_data, tmp_path):
        """A mixed successful/error result must be authoritative and nonzero."""
        output_path = tmp_path / "result.json"
        result = _run_metrics(
            [
                "--execution-output-path",
                str(sample_data),
                "--metrics",
                "Accuracy,NoSuchMetric",
                "--pred-key",
                "pred",
                "--ground-truth-key",
                "gold",
                "--output-path",
                str(output_path),
            ]
        )
        assert result.returncode == 2, result.stderr
        output = _full_output(result, output_path)
        assert output["ok"] is False
        assert output["status"] == "partial"
        assert isinstance(output["data"]["results"]["Accuracy"], float)
        assert "error" in output["data"]["results"]["NoSuchMetric"]
        assert "Accuracy" in output["data"]["samples"]
        assert "NoSuchMetric" not in output["data"]["samples"]
        assert output["errors"]
        assert json.loads(output_path.read_text()) == output

    def test_all_metrics_failing_returns_failed_status(self, sample_data, tmp_path):
        """No successful metric must return failed with exit code 1."""
        output_path = tmp_path / "result.json"
        result = _run_metrics(
            [
                "--execution-output-path",
                str(sample_data),
                "--metrics",
                "NoSuchMetric",
                "--pred-key",
                "pred",
                "--ground-truth-key",
                "gold",
                "--output-path",
                str(output_path),
            ]
        )

        assert result.returncode == 1, result.stderr
        output = _full_output(result, output_path)
        assert output["ok"] is False
        assert output["status"] == "failed"
        assert "error" in output["data"]["results"]["NoSuchMetric"]
        assert json.loads(output_path.read_text()) == output

    def test_validation_defect_reports_the_compute_metric_message(
        self, sample_data, tmp_path
    ):
        """The CLI reports a validation defect with the exact in-process message.

        A separate ground-truth file with two records against three predictions
        passes the runner's own cardinality checks (which cover prompts and
        context only) and reaches the metric's input validation.
        """
        reference_path = tmp_path / "references.json"
        reference_path.write_text(json.dumps([{"gold": "A"}, {"gold": "B"}]))
        output_path = tmp_path / "result.json"

        # Build the in-process inputs exactly as the runner does: one value per
        # JSON record under the extraction key.
        predictions = DataHandler.parse_json_file(str(sample_data), "pred")
        references = DataHandler.parse_json_file(str(reference_path), "gold")
        client_factory = Mock(spec=[])
        with pytest.raises(ValueError) as excinfo:
            MetricRegistry.compute_metric(
                "Accuracy", predictions, references, client_factory=client_factory
            )
        message = str(excinfo.value)
        assert message == (
            "reference count (2) must match prediction count (3) for Accuracy"
        )
        client_factory.assert_not_called()

        result = _run_metrics(
            [
                "--execution-output-path",
                str(sample_data),
                "--metrics",
                "Accuracy",
                "--pred-key",
                "pred",
                "--ground-truth-path",
                str(reference_path),
                "--ground-truth-key",
                "gold",
                "--output-path",
                str(output_path),
            ]
        )

        assert result.returncode == 1, result.stderr
        output = _full_output(result, output_path)
        assert set(output) == ENVELOPE_KEYS
        assert output["ok"] is False
        assert output["status"] == "failed"
        assert output["data"]["results"]["Accuracy"] == {"error": message}
        assert output["data"]["samples"] == {}
        assert output["errors"] == [f"Accuracy: {message}"]
        assert json.loads(output_path.read_text()) == output

    def test_missing_key_fails_and_persists_envelope(self, sample_data, tmp_path):
        """A missing extraction key must not become an empty successful dataset."""
        output_path = tmp_path / "result.json"
        result = _run_metrics(
            [
                "--execution-output-path",
                str(sample_data),
                "--metrics",
                "Accuracy",
                "--pred-key",
                "missing",
                "--ground-truth-key",
                "gold",
                "--output-path",
                str(output_path),
            ]
        )

        assert result.returncode == 1
        output = _full_output(result, output_path, "stderr")
        assert output["ok"] is False
        assert output["status"] == "failed"
        assert output["data"] is None
        assert "Missing key 'missing'" in output["errors"][0]
        assert json.loads(output_path.read_text()) == output

    def test_separate_prompt_file_uses_prompt_key_for_every_record(
        self, sample_data, tmp_path
    ):
        """A keyed prompt file must align all prompts, not wrap it as sample zero."""
        custom_dir = tmp_path / "custom"
        custom_dir.mkdir()
        (custom_dir / "custom_metrics.py").write_text(
            textwrap.dedent("""\
            from metric_library.core.base import Metric
            from metric_library.core.input_policy import InputPolicy
            from metric_library.core.registry import MetricRegistry

            @MetricRegistry.register
            class PromptCoverage(Metric):
                prompt_policy = InputPolicy.REQUIRED

                def compute(self, sample, client):
                    # The engine enforces the prompt count; every sample must carry one.
                    return 1.0 if sample.prompt is not None else 0.0
            """)
        )
        prompt_path = tmp_path / "prompts.json"
        prompt_path.write_text(
            json.dumps(
                [
                    {"prompt": "question one"},
                    {"prompt": "question two"},
                    {"prompt": "question three"},
                ]
            )
        )
        output_path = tmp_path / "result.json"

        result = _run_metrics(
            [
                "--execution-output-path",
                str(sample_data),
                "--metrics",
                "PromptCoverage",
                "--pred-key",
                "pred",
                "--ground-truth-key",
                "gold",
                "--prompt-path",
                str(prompt_path),
                "--prompt-key",
                "prompt",
                "--output-path",
                str(output_path),
                "--custom-metrics-path",
                str(custom_dir / "custom_metrics.py"),
            ],
            cwd=str(tmp_path),
        )

        assert result.returncode == 0, result.stderr
        output = _full_output(result, output_path)
        assert output["ok"] is True
        assert output["status"] == "completed"
        assert output["data"]["results"]["PromptCoverage"] == 1.0

    def test_keyless_prompt_manifest_loads_one_file_per_prediction(
        self, sample_data, tmp_path
    ):
        """Relative manifest entries are resolved from the manifest directory."""
        custom_dir = tmp_path / "custom"
        custom_dir.mkdir()
        (custom_dir / "custom_metrics.py").write_text(
            textwrap.dedent(
                """\
            from metric_library.core.base import Metric
            from metric_library.core.input_policy import InputPolicy
            from metric_library.core.registry import MetricRegistry

            @MetricRegistry.register
            class PromptManifest(Metric):
                prompt_policy = InputPolicy.REQUIRED

                def compute(self, sample, client):
                    expected = ["question one", "question two", "question three"]
                    return 1.0 if sample.prompt.get_raw_data() == expected[sample.ordinal] else 0.0
            """
            )
        )
        manifest_dir = tmp_path / "manifest"
        manifest_dir.mkdir()
        prompt_files = []
        for index, prompt in enumerate(
            ["question one", "question two", "question three"]
        ):
            prompt_file = manifest_dir / f"prompt-{index}.txt"
            prompt_file.write_text(prompt)
            prompt_files.append(prompt_file.name)

        prompt_path = manifest_dir / "prompts.json"
        prompt_path.write_text(json.dumps(prompt_files))
        output_path = tmp_path / "result.json"
        runner_dir = tmp_path / "runner"
        runner_dir.mkdir()

        result = _run_metrics(
            [
                "--execution-output-path",
                str(sample_data),
                "--metrics",
                "PromptManifest",
                "--pred-key",
                "pred",
                "--ground-truth-key",
                "gold",
                "--prompt-path",
                str(prompt_path),
                "--output-path",
                str(output_path),
                "--custom-metrics-path",
                str(custom_dir / "custom_metrics.py"),
            ],
            cwd=str(runner_dir),
        )

        assert result.returncode == 0, result.stderr
        output = _full_output(result, output_path)
        assert output["status"] == "completed"
        assert output["data"]["results"]["PromptManifest"] == 1.0

    @pytest.mark.parametrize(
        ("prompt_document", "prompt_key"),
        [
            (
                [
                    {"prompt": "question one"},
                    {"prompt": "question two"},
                    {"prompt": "question three"},
                ],
                None,
            ),
            ({"prompt": "question one"}, None),
            ([{"prompt": "question one"}], "null"),
            ([{"prompt": "question one"}], "None"),
        ],
    )
    def test_keyless_json_prompt_records_require_prompt_key(
        self, sample_data, tmp_path, prompt_document, prompt_key
    ):
        """JSON prompt records cannot silently become one whole-file prompt."""
        prompt_path = tmp_path / "prompts.json"
        prompt_path.write_text(json.dumps(prompt_document))
        output_path = tmp_path / "result.json"
        prompt_args = ["--prompt-path", str(prompt_path)]
        if prompt_key is not None:
            prompt_args.extend(["--prompt-key", prompt_key])

        result = _run_metrics(
            [
                "--execution-output-path",
                str(sample_data),
                "--metrics",
                "Accuracy",
                "--pred-key",
                "pred",
                "--ground-truth-key",
                "gold",
                *prompt_args,
                "--output-path",
                str(output_path),
            ]
        )

        assert result.returncode == 1
        output = _full_output(result, output_path, "stderr")
        assert output["status"] == "failed"
        assert output["data"] is None
        assert "A JSON --prompt-path requires --prompt-key" in output["errors"][0]
        assert json.loads(output_path.read_text()) == output

    @pytest.mark.parametrize("manifest", [[], [123]])
    def test_keyless_prompt_manifest_requires_file_path_array(
        self, sample_data, tmp_path, manifest
    ):
        """Only a non-empty JSON array of file paths is a keyless manifest."""
        prompt_path = tmp_path / "prompt-manifest.json"
        prompt_path.write_text(json.dumps(manifest))
        output_path = tmp_path / "result.json"

        result = _run_metrics(
            [
                "--execution-output-path",
                str(sample_data),
                "--metrics",
                "Accuracy",
                "--pred-key",
                "pred",
                "--ground-truth-key",
                "gold",
                "--prompt-path",
                str(prompt_path),
                "--output-path",
                str(output_path),
            ]
        )

        assert result.returncode == 1
        output = _full_output(result, output_path, "stderr")
        assert output["status"] == "failed"
        assert "A JSON --prompt-path requires --prompt-key" in output["errors"][0]
        assert json.loads(output_path.read_text()) == output

    def test_keyless_prompt_manifest_rejects_missing_file(self, sample_data, tmp_path):
        """A manifest must not silently discard missing prompt files."""
        prompt_path = tmp_path / "prompt-manifest.json"
        prompt_path.write_text(json.dumps([str(tmp_path / "missing.txt")]))
        output_path = tmp_path / "result.json"

        result = _run_metrics(
            [
                "--execution-output-path",
                str(sample_data),
                "--metrics",
                "Accuracy",
                "--pred-key",
                "pred",
                "--ground-truth-key",
                "gold",
                "--prompt-path",
                str(prompt_path),
                "--output-path",
                str(output_path),
            ]
        )

        assert result.returncode == 1
        output = _full_output(result, output_path, "stderr")
        assert output["status"] == "failed"
        assert (
            "Prompt manifest entry 0 must be an existing regular file"
            in output["errors"][0]
        )
        assert json.loads(output_path.read_text()) == output

    def test_keyless_ground_truth_list_resolves_from_its_own_folder(self, tmp_path):
        """The entries of a JSON array of paths resolve like record values."""
        gold_dir = tmp_path / "gold"
        gold_dir.mkdir()
        (gold_dir / "first.txt").write_text("Paris", encoding="utf-8")
        (gold_dir / "second.txt").write_text("Rome", encoding="utf-8")
        gold_path = gold_dir / "references.json"
        gold_path.write_text(json.dumps(["first.txt", "second.txt"]))
        data_path = tmp_path / "data.json"
        data_path.write_text(json.dumps([{"pred": "Paris"}, {"pred": "Berlin"}]))
        output_path = tmp_path / "result.json"

        # The working directory holds no first.txt or second.txt.
        result = _run_metrics(
            [
                "--execution-output-path",
                str(data_path),
                "--metrics",
                "TextExactMatch",
                "--pred-key",
                "pred",
                "--ground-truth-path",
                str(gold_path),
                "--output-path",
                str(output_path),
            ],
            cwd=str(tmp_path),
        )

        assert result.returncode == 0, result.stderr
        output = _full_output(result, output_path)
        entries = output["data"]["samples"]["TextExactMatch"]
        assert [entry["score"] for entry in entries] == [1.0, 0.0]

    def test_keyless_ground_truth_list_reports_a_file_that_does_not_load(
        self, tmp_path
    ):
        """The load error stops the run; the list file does not become the reference."""
        # A subprocess keeps PIL and the image data type out of this process.
        if importlib.util.find_spec("PIL") is None:
            pytest.skip("Pillow is not installed")
        source = REPO_ROOT / "tests" / "examples" / "rmse_images" / "reference_0.png"
        shutil.copy(source, tmp_path / "generated.png")
        (tmp_path / "broken.png").write_bytes(b"not an image")
        gold_path = tmp_path / "references.json"
        gold_path.write_text(json.dumps(["broken.png"]))
        data_path = tmp_path / "data.json"
        data_path.write_text(json.dumps([{"pred": "generated.png"}]))
        output_path = tmp_path / "result.json"

        result = _run_metrics(
            [
                "--execution-output-path",
                str(data_path),
                "--metrics",
                "MeanAbsoluteError",
                "--pred-key",
                "pred",
                "--ground-truth-path",
                str(gold_path),
                "--output-path",
                str(output_path),
            ]
        )

        assert result.returncode == 1
        output = _full_output(result, output_path, "stderr")
        assert output["data"] is None
        assert output["errors"] == [
            f"cannot identify image file {str(tmp_path / 'broken.png')!r}"
        ]

    def test_misaligned_separate_prompts_fail_before_custom_metric(
        self, sample_data, tmp_path
    ):
        """Prompt alignment is a runner contract, not delegated to metric code."""
        custom_dir = tmp_path / "custom"
        custom_dir.mkdir()
        (custom_dir / "custom_metrics.py").write_text(
            textwrap.dedent("""\
            from metric_library.core.base import Metric
            from metric_library.core.registry import MetricRegistry

            @MetricRegistry.register
            class PromptBlind(Metric):
                def compute(self, sample, client):
                    return 1.0
            """)
        )
        prompt_path = tmp_path / "prompts.json"
        prompt_path.write_text(json.dumps([{"prompt": "one"}, {"prompt": "two"}]))
        output_path = tmp_path / "result.json"

        result = _run_metrics(
            [
                "--execution-output-path",
                str(sample_data),
                "--metrics",
                "PromptBlind",
                "--pred-key",
                "pred",
                "--ground-truth-key",
                "gold",
                "--prompt-path",
                str(prompt_path),
                "--prompt-key",
                "prompt",
                "--output-path",
                str(output_path),
                "--custom-metrics-path",
                str(custom_dir / "custom_metrics.py"),
            ],
            cwd=str(tmp_path),
        )

        assert result.returncode == 1
        output = _full_output(result, output_path, "stderr")
        assert output["status"] == "failed"
        assert output["data"] is None
        assert "prompt count (2) must match prediction count (3)" in output["errors"]
        assert json.loads(output_path.read_text()) == output

    def test_invalid_per_sample_scores_fail_closed(self, sample_data, tmp_path):
        """Any per-sample score that is not a finite real fails the metric.

        The engine rejects the value at the first sample, the runner maps the
        FAILED result to ``{"error": "sample 1: ..."}`` under the 1-based record
        position, and a dict is no longer valid data even when it carries an
        ``error`` key.
        """
        custom_dir = tmp_path / "custom"
        custom_dir.mkdir()
        (custom_dir / "custom_metrics.py").write_text(
            textwrap.dedent("""\
            from metric_library.core.base import Metric
            from metric_library.core.registry import MetricRegistry

            @MetricRegistry.register
            class ReturnsNaN(Metric):
                def compute(self, sample, client):
                    return float("nan")

            @MetricRegistry.register
            class ReturnsNone(Metric):
                def compute(self, sample, client):
                    return None

            @MetricRegistry.register
            class StructuredErrorResult(Metric):
                def compute(self, sample, client):
                    return {"error": "estimated error rate", "score": 0.8}
            """)
        )
        common_args = [
            "--execution-output-path",
            str(sample_data),
            "--pred-key",
            "pred",
            "--ground-truth-key",
            "gold",
        ]

        partial_path = tmp_path / "partial.json"
        partial_result = _run_metrics(
            [
                *common_args,
                "--metrics",
                "Accuracy,ReturnsNaN",
                "--output-path",
                str(partial_path),
                "--custom-metrics-path",
                str(custom_dir / "custom_metrics.py"),
            ],
            cwd=str(tmp_path),
        )
        assert partial_result.returncode == 2, partial_result.stderr
        partial_output = _full_output(partial_result, partial_path)
        assert partial_output["ok"] is False
        assert partial_output["status"] == "partial"
        assert partial_output["data"]["results"]["Accuracy"] == pytest.approx(2 / 3)
        assert partial_output["data"]["results"]["ReturnsNaN"] == {
            "error": "sample 1: score must be finite"
        }
        assert partial_output["errors"] == [
            "ReturnsNaN: sample 1: score must be finite"
        ]
        assert '"ReturnsNaN": NaN' not in partial_result.stdout
        # The failed metric keeps its full list: the failed sample, then the skipped.
        assert partial_output["data"]["samples"]["ReturnsNaN"] == [
            {
                "sample_id": "1",
                "status": "failed",
                "score": None,
                "error": "score must be finite",
                "matches": None,
                "item_scores": None,
                "reasoning": None,
            },
            {
                "sample_id": "2",
                "status": "skipped",
                "score": None,
                "error": None,
                "matches": None,
                "item_scores": None,
                "reasoning": None,
            },
            {
                "sample_id": "3",
                "status": "skipped",
                "score": None,
                "error": None,
                "matches": None,
                "item_scores": None,
                "reasoning": None,
            },
        ]
        accuracy_samples = partial_output["data"]["samples"]["Accuracy"]
        assert len(accuracy_samples) == 3
        assert all(entry["status"] == "completed" for entry in accuracy_samples)

        failed_path = tmp_path / "failed.json"
        failed_result = _run_metrics(
            [
                *common_args,
                "--metrics",
                "ReturnsNone",
                "--output-path",
                str(failed_path),
                "--custom-metrics-path",
                str(custom_dir / "custom_metrics.py"),
            ],
            cwd=str(tmp_path),
        )
        assert failed_result.returncode == 1, failed_result.stderr
        failed_output = _full_output(failed_result, failed_path)
        assert failed_output["ok"] is False
        assert failed_output["status"] == "failed"
        assert failed_output["data"]["results"]["ReturnsNone"] == {
            "error": "sample 1: score must be a real number"
        }
        assert failed_output["errors"] == [
            "ReturnsNone: sample 1: score must be a real number"
        ]

        structured_path = tmp_path / "structured.json"
        structured_result = _run_metrics(
            [
                *common_args,
                "--metrics",
                "StructuredErrorResult",
                "--output-path",
                str(structured_path),
                "--custom-metrics-path",
                str(custom_dir / "custom_metrics.py"),
            ],
            cwd=str(tmp_path),
        )
        assert structured_result.returncode == 1, structured_result.stderr
        structured_output = _full_output(structured_result, structured_path)
        assert structured_output["ok"] is False
        assert structured_output["status"] == "failed"
        assert structured_output["data"]["results"]["StructuredErrorResult"] == {
            "error": "sample 1: score must be a real number"
        }
        assert structured_output["errors"] == [
            "StructuredErrorResult: sample 1: score must be a real number"
        ]
        assert json.loads(structured_path.read_text()) == structured_output

    @pytest.mark.parametrize(
        ("identifiers", "expected_ids"),
        [
            pytest.param(["r-a", "r-b", "r-c"], ["r-a", "r-b", "r-c"], id="text"),
            pytest.param([7, 8, 9], ["7", "8", "9"], id="numeric"),
        ],
    )
    def test_sample_id_key_replaces_the_positional_identifier(
        self, tmp_path, identifiers, expected_ids
    ):
        """``--sample-id-key`` names every sample after its record identifier.

        The identifier is the ``str()`` of the record's value under the key, so
        a numeric identifier is reported as its decimal text. The aggregates do
        not depend on the option.
        """
        data_path = _write_identified_records(tmp_path / "records.json", identifiers)
        output_path = tmp_path / "result.json"
        result = _run_metrics(
            [
                "--execution-output-path",
                str(data_path),
                "--metrics",
                "Accuracy,TextExactMatch",
                "--pred-key",
                "pred",
                "--ground-truth-key",
                "gold",
                "--sample-id-key",
                "id",
                "--output-path",
                str(output_path),
            ]
        )
        assert result.returncode == 0, result.stderr
        output = _full_output(result, output_path)
        assert output["status"] == "completed"
        assert output["data"]["results"] == pytest.approx(
            {"Accuracy": 2 / 3, "TextExactMatch": 2 / 3}
        )
        samples = output["data"]["samples"]
        assert set(samples) == {"Accuracy", "TextExactMatch"}
        for name, entries in samples.items():
            assert [entry["sample_id"] for entry in entries] == expected_ids, name
            assert all(entry["status"] == "completed" for entry in entries), name
            assert [entry["score"] for entry in entries] == [1.0, 1.0, 0.0], name
        assert json.loads(output_path.read_text()) == output

    def test_sample_id_key_names_the_failed_sample(self, tmp_path):
        """A failed sample is reported under its record identifier, not its position."""
        custom_dir = tmp_path / "custom"
        custom_dir.mkdir()
        (custom_dir / "custom_metrics.py").write_text(
            textwrap.dedent("""\
            from metric_library.core.base import Metric
            from metric_library.core.registry import MetricRegistry

            @MetricRegistry.register
            class ReturnsNaN(Metric):
                def compute(self, sample, client):
                    return float("nan")
            """)
        )
        data_path = _write_identified_records(
            tmp_path / "records.json", ["r-a", "r-b", "r-c"]
        )
        output_path = tmp_path / "result.json"
        result = _run_metrics(
            [
                "--execution-output-path",
                str(data_path),
                "--metrics",
                "ReturnsNaN",
                "--pred-key",
                "pred",
                "--ground-truth-key",
                "gold",
                "--sample-id-key",
                "id",
                "--output-path",
                str(output_path),
                "--custom-metrics-path",
                str(custom_dir / "custom_metrics.py"),
            ],
            cwd=str(tmp_path),
        )
        assert result.returncode == 1, result.stderr
        output = _full_output(result, output_path)
        assert output["ok"] is False
        assert output["status"] == "failed"
        assert output["data"]["results"] == {
            "ReturnsNaN": {"error": "sample r-a: score must be finite"}
        }
        assert output["errors"] == ["ReturnsNaN: sample r-a: score must be finite"]
        # The failed sample keeps its identifier; the skipped ones keep theirs.
        assert output["data"]["samples"]["ReturnsNaN"] == [
            {
                "sample_id": "r-a",
                "status": "failed",
                "score": None,
                "error": "score must be finite",
                "matches": None,
                "item_scores": None,
                "reasoning": None,
            },
            {
                "sample_id": "r-b",
                "status": "skipped",
                "score": None,
                "error": None,
                "matches": None,
                "item_scores": None,
                "reasoning": None,
            },
            {
                "sample_id": "r-c",
                "status": "skipped",
                "score": None,
                "error": None,
                "matches": None,
                "item_scores": None,
                "reasoning": None,
            },
        ]
        assert json.loads(output_path.read_text()) == output

    def test_a_top_level_key_wins_over_nested_occurrences(self, tmp_path):
        """A same-named key inside a nested object is not a duplicate.

        RAG records often carry an ``id`` in their context documents. One
        lookup rule applies to every key: the record's own key is used when
        present, else the key must occur exactly once at any depth.
        """
        data_path = tmp_path / "records.json"
        data_path.write_text(
            json.dumps(
                [
                    {
                        "id": 1,
                        "w": 2,
                        "pred": "A",
                        "gold": "A",
                        "context": [{"id": "doc-7", "w": 9, "text": "x"}],
                    },
                    {
                        "id": 2,
                        "w": 1,
                        "pred": "B",
                        "gold": "C",
                        "context": [{"id": "doc-8", "w": 9, "text": "y"}],
                    },
                ]
            )
        )
        output_path = tmp_path / "result.json"
        result = _run_evaluation(
            data_path,
            output_path,
            "Accuracy",
            "--sample-id-key",
            "id",
            "--aggregation-methods",
            "weighted_mean",
            "--weight-key",
            "w",
        )
        assert result.returncode == 0, result.stderr
        output = _full_output(result, output_path)
        assert output["status"] == "completed"
        # Weights 2 and 1 on scores 1 and 0: (2 + 0) / 3.
        assert output["data"]["results"]["Accuracy"] == pytest.approx(2 / 3)
        assert [e["sample_id"] for e in output["data"]["samples"]["Accuracy"]] == [
            "1",
            "2",
        ]

    @pytest.mark.parametrize(
        ("records", "message"),
        [
            pytest.param(
                [
                    {"id": "r-a", "pred": "A", "gold": "A"},
                    {"id": "r-a", "pred": "B", "gold": "B"},
                    {"id": "r-c", "pred": "C", "gold": "D"},
                ],
                "sample ids under key 'id' must be unique",
                id="duplicate-ids",
            ),
            pytest.param(
                [
                    {"id": "r-a", "pred": "A", "gold": "A"},
                    {"id": " ", "pred": "B", "gold": "B"},
                    {"id": "r-c", "pred": "C", "gold": "D"},
                ],
                "sample id in record 1 under key 'id' must be non-empty",
                id="blank-id",
            ),
            pytest.param(
                [
                    {"id": "r-a", "pred": "A", "gold": "A"},
                    {"pred": "B", "gold": "B"},
                    {"id": "r-c", "pred": "C", "gold": "D"},
                ],
                "Missing key 'id' in record 1 of {path}",
                id="missing-key",
            ),
            pytest.param(
                [
                    {"id": "r-a", "pred": "A", "gold": "A"},
                    {"pred": "B", "gold": "B", "docs": [{"id": "x"}, {"id": "y"}]},
                    {"id": "r-c", "pred": "C", "gold": "D"},
                ],
                "Duplicate key 'id' in record 1 of {path}: found 2 occurrences",
                id="two-nested-ids-no-top-level",
            ),
        ],
    )
    def test_sample_id_key_rejects_invalid_identifiers(
        self, tmp_path, records, message
    ):
        """A duplicate, blank or missing identifier is the pre-execution error envelope.

        The record position in the messages is the loader's 0-based index; the
        message for a missing key is the loader's own.
        """
        data_path = tmp_path / "records.json"
        data_path.write_text(json.dumps(records))
        output_path = tmp_path / "result.json"
        result = _run_metrics(
            [
                "--execution-output-path",
                str(data_path),
                "--metrics",
                "Accuracy",
                "--pred-key",
                "pred",
                "--ground-truth-key",
                "gold",
                "--sample-id-key",
                "id",
                "--output-path",
                str(output_path),
            ]
        )

        assert result.returncode == 1
        output = _full_output(result, output_path, "stderr")
        assert set(output) == ENVELOPE_KEYS
        assert output["ok"] is False
        assert output["status"] == "failed"
        assert output["data"] is None
        assert output["errors"] == [message.format(path=data_path)]
        assert json.loads(output_path.read_text()) == output

    @pytest.mark.parametrize(
        ("metrics", "extra_args", "message"),
        [
            pytest.param(
                "Accuracy,Accuracy",
                ["--aggregation-methods", "median,numeric_mean"],
                "--metrics names one metric more than once: 'Accuracy' and 'Accuracy'",
                id="same-spelling",
            ),
            pytest.param(
                "Accuracy,accuracy",
                [],
                "--metrics names one metric more than once: 'Accuracy' and 'accuracy'",
                id="other-case",
            ),
            pytest.param(
                "TextExactMatch,text_exact_match",
                [],
                "--metrics names one metric more than once: "
                "'TextExactMatch' and 'text_exact_match'",
                id="other-separator",
            ),
        ],
    )
    def test_repeated_metric_name_fails_before_metric_execution(
        self, sample_data, tmp_path, metrics, extra_args, message
    ):
        """A repeated metric is a pre-execution error, so no result is lost."""
        output_path = tmp_path / "result.json"
        result = _run_evaluation(sample_data, output_path, metrics, *extra_args)

        assert result.returncode == 1
        output = _full_output(result, output_path, "stderr")
        assert set(output) == ENVELOPE_KEYS
        assert output["ok"] is False
        assert output["status"] == "failed"
        assert output["data"] is None
        assert output["errors"] == [message]
        assert json.loads(output_path.read_text()) == output

    def test_unknown_aws_profile_fails_only_the_metric_that_uses_the_client(
        self, tmp_path, monkeypatch
    ):
        """A bad profile fails the judge at its first sample; Accuracy still scores."""
        monkeypatch.setenv("AWS_PROFILE", "no-such-profile")
        monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "missing-config"))
        monkeypatch.setenv(
            "AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "missing-credentials")
        )
        records_path = tmp_path / "records.json"
        records_path.write_text(
            json.dumps(
                [
                    {"pred": "a", "gold": "a", "q": "x"},
                    {"pred": "b", "gold": "c", "q": "y"},
                ]
            )
        )
        output_path = tmp_path / "result.json"

        result = _run_evaluation(
            records_path,
            output_path,
            "Accuracy,LLMCorrectness",
            "--prompt-key",
            "q",
        )

        assert result.returncode == 2, result.stderr
        output = _full_output(result, output_path)
        assert output["status"] == "partial"
        assert output["data"]["results"]["Accuracy"] == 0.5
        assert output["data"]["results"]["LLMCorrectness"] == {
            "error": "sample 1: The config profile (no-such-profile) could not be found"
        }
        entries = output["data"]["samples"]["LLMCorrectness"]
        assert [entry["status"] for entry in entries] == ["failed", "skipped"]


class TestAggregationMethodsOption:
    """``--aggregation-methods`` and ``--quantile`` select the aggregator per metric."""

    @pytest.mark.parametrize(
        ("aggregation_args", "expected", "aggregator"),
        [
            pytest.param(
                ["--aggregation-methods", "median"],
                1.0,
                median,
                id="median",
            ),
            pytest.param(
                ["--aggregation-methods", "quantile", "--quantile", "0.0"],
                0.0,
                lambda samples: quantile(samples, 0.0),
                id="quantile",
            ),
            pytest.param(
                ["--aggregation-methods", "numeric_mean"],
                pytest.approx(2 / 3),
                numeric_mean,
                id="numeric_mean",
            ),
        ],
    )
    def test_single_entry_applies_to_every_selected_metric(
        self, sample_data, tmp_path, aggregation_args, expected, aggregator
    ):
        """One entry for the batch: each aggregate is that aggregator of its samples.

        Accuracy and TextExactMatch both score ``1.0, 1.0, 0.0`` on the fixture,
        so the median is ``1.0`` and the 0.0 quantile is ``0.0`` where the default
        mean is ``2/3``.
        """
        output_path = tmp_path / "result.json"
        result = _run_metrics(
            [
                "--execution-output-path",
                str(sample_data),
                "--metrics",
                "Accuracy,TextExactMatch",
                "--pred-key",
                "pred",
                "--ground-truth-key",
                "gold",
                *aggregation_args,
                "--output-path",
                str(output_path),
            ]
        )
        assert result.returncode == 0, result.stderr
        output = _full_output(result, output_path)
        assert output["status"] == "completed"
        results = output["data"]["results"]
        samples = output["data"]["samples"]
        assert set(results) == {"Accuracy", "TextExactMatch"}
        for name in results:
            assert results[name] == expected, name
            # The per-sample entries do not depend on the aggregation method.
            assert [entry["score"] for entry in samples[name]] == [1.0, 1.0, 0.0]
            aggregate = aggregator(_sample_results(samples[name]))
            assert results[name] == pytest.approx(aggregate.score), name
        assert json.loads(output_path.read_text()) == output

    @pytest.mark.parametrize(
        ("aggregation_args", "expected"),
        [
            pytest.param(
                ["--aggregation-methods", "median,quantile", "--quantile", "0.0"],
                {
                    "Accuracy": (1.0, median),
                    "TextExactMatch": (0.0, lambda samples: quantile(samples, 0.0)),
                },
                id="median-and-quantile",
            ),
            pytest.param(
                ["--aggregation-methods", ",median"],
                {
                    "Accuracy": (pytest.approx(2 / 3), numeric_mean),
                    "TextExactMatch": (1.0, median),
                },
                id="default-and-median",
            ),
            pytest.param(
                ["--aggregation-methods", ""],
                {
                    "Accuracy": (pytest.approx(2 / 3), numeric_mean),
                    "TextExactMatch": (pytest.approx(2 / 3), numeric_mean),
                },
                id="empty-string-keeps-defaults",
            ),
        ],
    )
    def test_aggregation_methods_apply_per_metric(
        self, sample_data, tmp_path, aggregation_args, expected
    ):
        """Entry ``i`` aggregates metric ``i``; an empty entry keeps the default mean.

        ``expected`` maps each metric to its aggregate and to the public
        aggregator the runner should have applied to that metric's samples.
        """
        output_path = tmp_path / "result.json"
        result = _run_metrics(
            [
                "--execution-output-path",
                str(sample_data),
                "--metrics",
                "Accuracy,TextExactMatch",
                "--pred-key",
                "pred",
                "--ground-truth-key",
                "gold",
                *aggregation_args,
                "--output-path",
                str(output_path),
            ]
        )
        assert result.returncode == 0, result.stderr
        output = _full_output(result, output_path)
        assert output["status"] == "completed"
        results = output["data"]["results"]
        samples = output["data"]["samples"]
        assert set(results) == set(expected)
        for name, (score, aggregator) in expected.items():
            assert results[name] == score, name
            # The per-sample entries do not depend on the aggregation method.
            assert [entry["score"] for entry in samples[name]] == [1.0, 1.0, 0.0]
            aggregate = aggregator(_sample_results(samples[name]))
            assert results[name] == pytest.approx(aggregate.score), name
        assert json.loads(output_path.read_text()) == output

    def test_results_keep_the_metrics_order(self, sample_data, tmp_path):
        """Metrics on different methods merge back into the ``--metrics`` order."""
        output_path = tmp_path / "result.json"
        result = _run_metrics(
            [
                "--execution-output-path",
                str(sample_data),
                "--metrics",
                "Accuracy,TextExactMatch,TextFuzzyMatch",
                "--pred-key",
                "pred",
                "--ground-truth-key",
                "gold",
                "--aggregation-methods",
                ",median,",
                "--output-path",
                str(output_path),
            ]
        )
        assert result.returncode == 0, result.stderr
        output = _full_output(result, output_path)
        assert output["status"] == "completed"
        results = output["data"]["results"]
        # The runner evaluates the two default metrics in one batch and the
        # median metric in another; the envelope must still follow --metrics.
        assert list(results) == ["Accuracy", "TextExactMatch", "TextFuzzyMatch"]
        assert list(output["data"]["samples"]) == [
            "Accuracy",
            "TextExactMatch",
            "TextFuzzyMatch",
        ]
        assert results["TextExactMatch"] == 1.0
        assert results["Accuracy"] == pytest.approx(2 / 3)
        assert results["TextFuzzyMatch"] == pytest.approx(2 / 3)
        assert json.loads(output_path.read_text()) == output

    @pytest.mark.parametrize(
        ("metrics", "aggregation_args", "message"),
        [
            pytest.param(
                "Accuracy",
                ["--aggregation-methods", "quantile"],
                "quantile aggregation requires quantile",
                id="quantile-without-quantile",
            ),
            pytest.param(
                "Accuracy",
                ["--aggregation-methods", "median", "--quantile", "0.5"],
                "--quantile requires --aggregation-methods quantile",
                id="median-with-quantile",
            ),
            pytest.param(
                "Accuracy",
                ["--aggregation-methods", "quantile", "--quantile", "1.5"],
                "quantile must be between 0.0 and 1.0",
                id="quantile-out-of-range",
            ),
            pytest.param(
                "Accuracy",
                ["--quantile", "0.5"],
                "--quantile requires --aggregation-methods quantile",
                id="quantile-without-method",
            ),
            pytest.param(
                "Accuracy,TextExactMatch",
                ["--aggregation-methods", "median,median,median"],
                "aggregation method count (3) must match metric count (2)",
                id="count-mismatch",
            ),
            pytest.param(
                "Accuracy",
                ["--aggregation-methods", "weighted_mean"],
                "weighted_mean requires --weight-key",
                id="weighted-mean-without-weight-key",
            ),
            pytest.param(
                "Accuracy",
                ["--aggregation-methods", "median", "--weight-key", "w"],
                "--weight-key requires --aggregation-methods weighted_mean",
                id="median-with-weight-key",
            ),
            pytest.param(
                "Accuracy",
                ["--weight-key", "w"],
                "--weight-key requires --aggregation-methods weighted_mean",
                id="weight-key-without-method",
            ),
            pytest.param(
                "Accuracy",
                ["--aggregation-methods", "unknown_mean"],
                "unknown aggregation method 'unknown_mean'; choose from "
                "numeric_mean, weighted_mean, median, quantile, prf",
                id="unknown-method",
            ),
        ],
    )
    def test_invalid_aggregation_options_fail_before_metric_execution(
        self, sample_data, tmp_path, metrics, aggregation_args, message
    ):
        """An invalid methods/quantile combination is the pre-execution error envelope."""
        output_path = tmp_path / "result.json"
        result = _run_metrics(
            [
                "--execution-output-path",
                str(sample_data),
                "--metrics",
                metrics,
                "--pred-key",
                "pred",
                "--ground-truth-key",
                "gold",
                *aggregation_args,
                "--output-path",
                str(output_path),
            ]
        )

        assert result.returncode == 1
        output = _full_output(result, output_path, "stderr")
        assert set(output) == ENVELOPE_KEYS
        assert output["ok"] is False
        assert output["status"] == "failed"
        assert output["data"] is None
        assert output["errors"] == [message]
        assert json.loads(output_path.read_text()) == output


class TestWeightKeyOption:
    """``--weight-key`` carries one ``weighted_mean`` weight per record."""

    @staticmethod
    def _weighted_data(tmp_path, weights):
        records = [
            {"pred": "A", "gold": "A"},
            {"pred": "B", "gold": "B"},
            {"pred": "C", "gold": "D"},
        ]
        for record, weight in zip(records, weights):
            if weight is not ...:
                record["w"] = weight
        path = tmp_path / "weighted.json"
        path.write_text(json.dumps(records))
        return path

    def _run(self, data_path, output_path, metrics, *aggregation_args):
        return _run_metrics(
            [
                "--execution-output-path",
                str(data_path),
                "--metrics",
                metrics,
                "--pred-key",
                "pred",
                "--ground-truth-key",
                "gold",
                *aggregation_args,
                "--output-path",
                str(output_path),
            ]
        )

    def test_a_long_invalid_weight_is_cut_in_the_error(self, tmp_path):
        """The pre-execution error keeps 500 characters of a message."""
        data_path = self._weighted_data(tmp_path, ["x" * 1000, 1, 1])
        output_path = tmp_path / "result.json"

        result = self._run(
            data_path,
            output_path,
            "Accuracy",
            "--aggregation-methods",
            "weighted_mean",
            "--weight-key",
            "w",
        )

        assert result.returncode == 1
        (message,) = json.loads(result.stderr)["errors"]
        assert message.startswith(
            "weight in record 0 under key 'w' must be a finite positive number, "
            "got 'xxx"
        )
        assert len(message) == 500
        assert message.endswith(" [truncated]")

    def test_weights_from_the_records_give_the_weighted_mean(self, tmp_path):
        """Scores ``1, 1, 0`` under weights ``1, 1, 2`` aggregate to ``0.5``.

        The per-sample entries do not change, and the aggregate equals the
        public ``weighted_mean`` of those entries under the same weights.
        """
        data_path = self._weighted_data(tmp_path, [1, 1, 2])
        output_path = tmp_path / "result.json"
        result = self._run(
            data_path,
            output_path,
            "Accuracy,TextExactMatch",
            "--aggregation-methods",
            "weighted_mean",
            "--weight-key",
            "w",
        )
        assert result.returncode == 0, result.stderr
        output = _full_output(result, output_path)
        assert output["status"] == "completed"
        results = output["data"]["results"]
        samples = output["data"]["samples"]
        for name in ("Accuracy", "TextExactMatch"):
            assert results[name] == pytest.approx(0.5), name
            assert [entry["score"] for entry in samples[name]] == [1.0, 1.0, 0.0]
            aggregate = weighted_mean(_sample_results(samples[name]), [1.0, 1.0, 2.0])
            assert results[name] == pytest.approx(aggregate.score), name
        assert json.loads(output_path.read_text()) == output

    def test_weights_apply_only_to_the_weighted_mean_entries(self, tmp_path):
        """``weighted_mean,`` weights the first metric and keeps the second on its mean."""
        data_path = self._weighted_data(tmp_path, [1, 1, 2])
        output_path = tmp_path / "result.json"
        result = self._run(
            data_path,
            output_path,
            "Accuracy,TextExactMatch",
            "--aggregation-methods",
            "weighted_mean,",
            "--weight-key",
            "w",
        )
        assert result.returncode == 0, result.stderr
        results = json.loads(result.stdout)["data"]["results"]
        assert results["Accuracy"] == pytest.approx(0.5)
        assert results["TextExactMatch"] == pytest.approx(2 / 3)

    def test_fractional_weights_are_accepted(self, tmp_path):
        """Weights ``0.5, 0.5, 1.0`` give the same aggregate as ``1, 1, 2``."""
        data_path = self._weighted_data(tmp_path, [0.5, 0.5, 1.0])
        output_path = tmp_path / "result.json"
        result = self._run(
            data_path,
            output_path,
            "Accuracy",
            "--aggregation-methods",
            "weighted_mean",
            "--weight-key",
            "w",
        )
        assert result.returncode == 0, result.stderr
        results = json.loads(result.stdout)["data"]["results"]
        assert results["Accuracy"] == pytest.approx(0.5)

    def test_faq_routing_example_reports_the_traffic_weighted_accuracy(self, tmp_path):
        """``tests/examples/weighted_mean_faq_routing.json`` gives 0.7 plain and 0.6 weighted.

        Records 3, 4 and 5 are misrouted and carry 4000 of the 10000 monthly
        queries, the numbers documented in ``tests/examples/README.md``.
        """
        data_path = REPO_ROOT / "tests" / "examples" / "weighted_mean_faq_routing.json"
        output_path = tmp_path / "result.json"
        common = (
            "--pred-key",
            "predicted_article",
            "--ground-truth-key",
            "true_article",
            "--sample-id-key",
            "id",
            "--output-path",
            str(output_path),
        )
        plain = _run_metrics(
            [
                "--execution-output-path",
                str(data_path),
                "--metrics",
                "Accuracy",
                *common,
            ]
        )
        assert plain.returncode == 0, plain.stderr
        assert json.loads(plain.stdout)["data"]["results"]["Accuracy"] == pytest.approx(
            0.7
        )

        weighted = _run_metrics(
            [
                "--execution-output-path",
                str(data_path),
                "--metrics",
                "Accuracy,TextExactMatch",
                "--aggregation-methods",
                "weighted_mean,",
                "--weight-key",
                "monthly_queries",
                *common,
            ]
        )
        assert weighted.returncode == 0, weighted.stderr
        output = _full_output(weighted, output_path)
        assert output["data"]["results"]["Accuracy"] == pytest.approx(0.6)
        # The empty entry keeps the plain mean of the second metric.
        assert output["data"]["results"]["TextExactMatch"] == pytest.approx(0.7)
        misrouted = [
            entry["sample_id"]
            for entry in output["data"]["samples"]["Accuracy"]
            if entry["score"] == 0.0
        ]
        assert misrouted == ["3", "4", "5"]

    @pytest.mark.parametrize(
        ("weights", "message"),
        [
            pytest.param(
                [1, "1", 2],
                "weight in record 1 under key 'w' must be a finite positive number, "
                "got '1'",
                id="string",
            ),
            pytest.param(
                [1, 0, 2],
                "weight in record 1 under key 'w' must be a finite positive number, "
                "got 0",
                id="zero",
            ),
            pytest.param(
                [1, -1, 2],
                "weight in record 1 under key 'w' must be a finite positive number, "
                "got -1",
                id="negative",
            ),
            pytest.param(
                [1, True, 2],
                "weight in record 1 under key 'w' must be a finite positive number, "
                "got True",
                id="boolean",
            ),
            pytest.param(
                [1, None, 2],
                "weight in record 1 under key 'w' must be a finite positive number, "
                "got None",
                id="null",
            ),
            pytest.param(
                [1, ..., 2],
                "Missing key 'w' in record 1 of {path}",
                id="missing-key",
            ),
        ],
    )
    def test_an_invalid_weight_fails_before_metric_execution(
        self, tmp_path, weights, message
    ):
        """A non-numeric, non-positive or missing weight is the pre-execution error envelope."""
        data_path = self._weighted_data(tmp_path, weights)
        output_path = tmp_path / "result.json"
        result = self._run(
            data_path,
            output_path,
            "Accuracy",
            "--aggregation-methods",
            "weighted_mean",
            "--weight-key",
            "w",
        )
        assert result.returncode == 1
        output = _full_output(result, output_path, "stderr")
        assert output["ok"] is False
        assert output["status"] == "failed"
        assert output["data"] is None
        assert output["errors"] == [message.format(path=data_path)]
        assert json.loads(output_path.read_text()) == output


class TestPrfCli:
    """The prf aggregator with the labels matcher on the ``A/A, B/B, C/D`` fixture.

    ``data.results[name]`` is the values object of the prf aggregator; every
    entry carries ``matches``, which is ``null`` with the whole matcher.
    """

    def test_labels_matcher_reports_the_prf_object_and_the_matches(
        self, sample_data, tmp_path
    ):
        """``prf,`` and ``labels,`` give the prf object to the first metric only.

        TextExactMatch keeps its float and its entries carry ``matches: null``;
        the Accuracy entries carry one match record per label. The object
        equals the public aggregator applied to the entries.
        """
        output_path = tmp_path / "result.json"
        result = _run_evaluation(
            sample_data,
            output_path,
            "Accuracy,TextExactMatch",
            "--aggregation-methods",
            "prf,",
            "--matcher",
            "labels,",
        )
        assert result.returncode == 0, result.stderr
        output = _full_output(result, output_path)
        assert output["ok"] is True
        assert output["status"] == "completed"
        assert output["errors"] == []
        results = output["data"]["results"]
        samples = output["data"]["samples"]
        assert results["TextExactMatch"] == pytest.approx(2 / 3)
        assert results["Accuracy"] == SMALL_FIXTURE_PRF

        for entry in samples["TextExactMatch"]:
            assert set(entry) == SAMPLE_ENTRY_KEYS
            assert entry["matches"] is None
        prf_entries = samples["Accuracy"]
        assert all(set(entry) == SAMPLE_ENTRY_KEYS for entry in prf_entries)
        assert [entry["sample_id"] for entry in prf_entries] == ["1", "2", "3"]
        assert [entry["status"] for entry in prf_entries] == ["completed"] * 3
        assert [entry["score"] for entry in prf_entries] == [1.0, 1.0, 0.0]
        assert [entry["matches"] for entry in prf_entries] == SMALL_FIXTURE_MATCHES
        aggregate = prf(_sample_results(prf_entries))
        assert aggregate.values == results["Accuracy"]
        assert json.loads(output_path.read_text()) == output

    def test_the_object_result_keeps_the_metrics_order(self, sample_data, tmp_path):
        """Float and object results merge back into the ``--metrics`` order."""
        output_path = tmp_path / "result.json"
        result = _run_evaluation(
            sample_data,
            output_path,
            "Accuracy,TextExactMatch,TextFuzzyMatch",
            "--aggregation-methods",
            ",prf,",
            "--matcher",
            ",labels,",
        )
        assert result.returncode == 0, result.stderr
        output = _full_output(result, output_path)
        assert output["status"] == "completed"
        results = output["data"]["results"]
        # The runner evaluates the two whole-matcher metrics in one batch and
        # the prf metric in another; the envelope must still follow --metrics.
        expected_order = ["Accuracy", "TextExactMatch", "TextFuzzyMatch"]
        assert list(results) == expected_order
        assert list(output["data"]["samples"]) == expected_order
        assert results["Accuracy"] == pytest.approx(2 / 3)
        assert results["TextExactMatch"] == SMALL_FIXTURE_PRF
        assert results["TextFuzzyMatch"] == pytest.approx(2 / 3)
        assert json.loads(output_path.read_text()) == output

    @pytest.mark.parametrize(
        "name", ["prf_binary", "prf_macro", "prf_micro", "quadratic_mean"]
    )
    def test_removed_aggregator_name_fails_before_metric_execution(
        self, sample_data, tmp_path, name
    ):
        """A removed aggregator name is an unknown aggregation method.

        The runner writes the error envelope before any metric runs.
        """
        output_path = tmp_path / "result.json"
        result = _run_evaluation(
            sample_data,
            output_path,
            "Accuracy",
            "--aggregation-methods",
            name,
        )

        assert result.returncode == 1
        output = _full_output(result, output_path, "stderr")
        assert set(output) == ENVELOPE_KEYS
        assert output["ok"] is False
        assert output["status"] == "failed"
        assert output["data"] is None
        assert output["errors"] == [
            f"unknown aggregation method '{name}'; choose from "
            "numeric_mean, weighted_mean, median, quantile, prf"
        ]
        assert json.loads(output_path.read_text()) == output

    def test_removed_positive_class_option_is_an_argparse_error(
        self, sample_data, tmp_path
    ):
        """The runner has no ``--positive-label`` option.

        Argparse rejects the option with exit code 2. The runner writes no envelope.
        """
        output_path = tmp_path / "result.json"
        result = _run_evaluation(
            sample_data,
            output_path,
            "Accuracy",
            "--positive-label",
            "a",
        )

        assert result.returncode == 2
        assert result.stdout == ""
        assert "usage:" in result.stderr
        assert "unrecognized arguments: --positive-label a" in result.stderr
        assert not output_path.exists()


class TestMatcherOption:
    """``--matcher`` and its options select the item path of each prf entry.

    ``--class-key`` and ``--value-key`` apply to every list entry and
    ``--thresholds`` to every prf entry.
    """

    PRF_LABELS = ("--aggregation-methods", "prf", "--matcher", "labels")

    @pytest.mark.parametrize(
        ("metrics", "matcher_args", "message"),
        [
            pytest.param(
                "Accuracy",
                ["--aggregation-methods", "prf"],
                "prf requires --matcher labels or list",
                id="prf-without-matcher",
            ),
            pytest.param(
                "Accuracy",
                ["--aggregation-methods", "numeric_mean", "--matcher", "labels"],
                "--matcher labels and list require --aggregation-methods prf",
                id="labels-without-prf",
            ),
            pytest.param(
                "Accuracy",
                ["--thresholds", "0.5"],
                "--thresholds requires --aggregation-methods prf",
                id="thresholds-without-prf",
            ),
            pytest.param(
                "Accuracy",
                ["--aggregation-methods", "prf", "--matcher", "bogus"],
                "unknown matcher 'bogus'; choose from whole, labels, list",
                id="unknown-matcher",
            ),
            pytest.param(
                "Accuracy,TextExactMatch",
                ["--aggregation-methods", "prf", "--matcher", "labels,labels,labels"],
                "matcher count (3) must match metric count (2)",
                id="count-mismatch",
            ),
            pytest.param(
                "Accuracy",
                [*PRF_LABELS, "--class-key", "type"],
                "--class-key requires --matcher list",
                id="class-key-without-list",
            ),
            pytest.param(
                "Accuracy",
                [*PRF_LABELS, "--value-key", "text"],
                "--value-key requires --matcher list",
                id="value-key-without-list",
            ),
            pytest.param(
                "Accuracy",
                [*PRF_LABELS, "--thresholds", "0.5,x"],
                "--thresholds must be comma-separated numbers, got '0.5,x'",
                id="thresholds-not-numbers",
            ),
            pytest.param(
                "Accuracy",
                [*PRF_LABELS, "--thresholds", "0"],
                "thresholds must be greater than 0.0 and at most 1.0",
                id="threshold-out-of-range",
            ),
        ],
    )
    def test_invalid_matcher_options_fail_before_metric_execution(
        self, sample_data, tmp_path, metrics, matcher_args, message
    ):
        """An invalid matcher, key or threshold option is the pre-execution envelope."""
        output_path = tmp_path / "result.json"
        result = _run_evaluation(sample_data, output_path, metrics, *matcher_args)

        assert result.returncode == 1
        assert result.stdout == ""
        output = _full_output(result, output_path, "stderr")
        assert set(output) == ENVELOPE_KEYS
        assert output["ok"] is False
        assert output["status"] == "failed"
        assert output["data"] is None
        assert output["errors"] == [message]
        assert json.loads(output_path.read_text()) == output

    def test_list_matcher_pairs_the_items_of_each_class_at_each_threshold(
        self, tmp_path
    ):
        """Only entities of one type can pair, and each threshold has its own counts.

        Jane Smith/Smith scores 0.5 and Acme/Acme Corp about 0.444. At 0.5 only
        the ``per`` pair counts, so the macro F1 is 0.5; at 0.8 no pair counts.
        The sample score is the F1 of the sample at the first threshold.
        """
        records_path = tmp_path / "entities.json"
        records_path.write_text(
            json.dumps(
                [
                    {
                        "id": "1",
                        "pred": [
                            {"type": "PER", "text": "Jane Smith"},
                            {"type": "ORG", "text": "Acme"},
                        ],
                        "gold": [
                            {"type": "PER", "text": "Smith"},
                            {"type": "ORG", "text": "Acme Corp"},
                        ],
                    }
                ]
            )
        )
        output_path = tmp_path / "result.json"
        result = _run_evaluation(
            records_path,
            output_path,
            "TextFuzzyMatch",
            "--aggregation-methods",
            "prf",
            "--matcher",
            "list",
            "--class-key",
            "type",
            "--value-key",
            "text",
            "--thresholds",
            "0.5,0.8",
            "--sample-id-key",
            "id",
        )
        assert result.returncode == 0, result.stderr
        output = _full_output(result, output_path)
        assert output["status"] == "completed"
        values = output["data"]["results"]["TextFuzzyMatch"]
        assert values["threshold"] == 0.5
        assert values["f1"] == 0.5
        assert [row["threshold"] for row in values["by_threshold"]] == [0.5, 0.8]
        assert [row["f1"] for row in values["by_threshold"]] == [0.5, 0.0]
        (entry,) = output["data"]["samples"]["TextFuzzyMatch"]
        assert entry["sample_id"] == "1"
        assert entry["score"] == 0.5
        assert entry["matches"] == {
            "org": {
                "scores": [[pytest.approx(0.4444444444444444)]],
                "n_pred": 1,
                "n_ref": 1,
            },
            "per": {"scores": [[0.5]], "n_pred": 1, "n_ref": 1},
        }
        assert prf(_sample_results([entry]), thresholds=(0.5, 0.8)).values == values
        assert json.loads(output_path.read_text()) == output

    def test_each_prf_entry_keeps_its_own_matcher(self, tmp_path):
        """Two prf entries with different matchers get one configuration each.

        TextExactMatch with labels has one class per label, and only paris
        pairs: the macro F1 is 0.2. TextFuzzyMatch with list has the one class
        ``*``, where Paris/paris and Rome/Roma pair: the F1 is 2/3.
        """
        records_path = tmp_path / "records.json"
        records_path.write_text(
            json.dumps(
                [
                    {"pred": ["Paris", "Lyon"], "gold": ["paris", "Nice"]},
                    {"pred": ["Rome"], "gold": ["Roma"]},
                ]
            )
        )
        output_path = tmp_path / "result.json"

        result = _run_evaluation(
            records_path,
            output_path,
            "TextExactMatch,TextFuzzyMatch",
            "--aggregation-methods",
            "prf",
            "--matcher",
            "labels,list",
        )

        assert result.returncode == 0, result.stderr
        results = _full_output(result, output_path)["data"]["results"]
        exact, fuzzy = results["TextExactMatch"], results["TextFuzzyMatch"]
        assert sorted(exact["per_class"]) == ["lyon", "nice", "paris", "roma", "rome"]
        assert exact["f1"] == pytest.approx(0.2)
        assert list(fuzzy["per_class"]) == ["*"]
        assert fuzzy["f1"] == pytest.approx(2 / 3)

    def test_an_llm_metric_rejects_an_item_matcher_before_any_judge_call(
        self, tmp_path, monkeypatch
    ):
        """An LLM metric that is not an LLMBatchMetric fails before its first sample.

        The run builds no client. The profile does not exist: a judge call
        would build the client and fail the first sample with another message.
        """
        for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("AWS_PROFILE", "no-such-profile")
        monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "missing-config"))
        monkeypatch.setenv(
            "AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "missing-credentials")
        )
        records_path = tmp_path / "records.json"
        records_path.write_text(
            json.dumps([{"pred": ["Paris"], "gold": ["Paris"], "q": "Name a city"}])
        )
        output_path = tmp_path / "result.json"
        metric = "LLMCompletenessWithGroundTruth"

        result = _run_evaluation(
            records_path,
            output_path,
            metric,
            "--prompt-key",
            "q",
            "--aggregation-methods",
            "prf",
            "--matcher",
            "list",
        )

        assert result.returncode == 1, result.stderr
        output = _full_output(result, output_path)
        assert output["status"] == "failed"
        message = f"{metric} accepts only the whole matcher"
        assert output["data"]["results"] == {metric: {"error": message}}
        assert output["data"]["samples"] == {}
        assert output["errors"] == [f"{metric}: {message}"]


class TestPrfExamples:
    """The prf aggregator with the labels matcher on the repository's example files."""

    def test_intent_example_reports_the_prf_object(self, tmp_path):
        """The intent example gives the prf object of its four classes.

        Records 7 and 9 have the wrong intent, so they score 0.0. Accuracy is
        the match rate of the ten records.
        """
        data_path = REPO_ROOT / "tests" / "examples" / "f1_intent_classification.json"
        keys = {"pred_key": "predicted_intent", "ground_truth_key": "true_intent"}
        output_path = tmp_path / "result.json"
        result = _run_evaluation(
            data_path,
            output_path,
            "Accuracy",
            "--aggregation-methods",
            "prf",
            "--matcher",
            "labels",
            "--sample-id-key",
            "id",
            **keys,
        )
        assert result.returncode == 0, result.stderr
        output = _full_output(result, output_path)
        assert output["status"] == "completed"
        values = output["data"]["results"]["Accuracy"]
        assert set(values) == {"threshold", "precision", "recall", "f1", "per_class"}
        assert values["threshold"] == 0.5
        assert values["precision"] == pytest.approx(0.8125)
        assert values["recall"] == pytest.approx(0.8333333333333333)
        assert values["f1"] == pytest.approx(0.780952380952381)
        assert list(values["per_class"]) == ["billing", "other", "refund", "shipping"]

        entries = output["data"]["samples"]["Accuracy"]
        assert [entry["sample_id"] for entry in entries] == [
            str(index) for index in range(1, 11)
        ]
        assert all(set(entry) == SAMPLE_ENTRY_KEYS for entry in entries)
        wrong = {
            entry["sample_id"]: entry["matches"]
            for entry in entries
            if entry["score"] == 0.0
        }
        # The two labels of a wrong record are in different classes, so they
        # cannot pair.
        assert wrong == {
            "7": {
                "other": {"scores": [[]], "n_pred": 1, "n_ref": 0},
                "shipping": {"scores": [], "n_pred": 0, "n_ref": 1},
            },
            "9": {
                "billing": {"scores": [[]], "n_pred": 1, "n_ref": 0},
                "refund": {"scores": [], "n_pred": 0, "n_ref": 1},
            },
        }
        assert prf(_sample_results(entries)).values == values
        assert json.loads(output_path.read_text()) == output

        # A run names each metric once: a second exact metric gives the prf
        # object next to the Accuracy match rate.
        both_path = tmp_path / "both.json"
        both = _run_evaluation(
            data_path,
            both_path,
            "Accuracy,TextExactMatch",
            "--aggregation-methods",
            ",prf",
            "--matcher",
            ",labels",
            "--sample-id-key",
            "id",
            **keys,
        )
        assert both.returncode == 0, both.stderr
        results = _full_output(both, both_path)["data"]["results"]
        assert results["Accuracy"] == pytest.approx(0.8)
        assert results["TextExactMatch"] == values


class TestIdpInvoiceExample:
    """The invoice example: one field per run, gold values in their own file."""

    # Per field: the metrics the README documents and, for each, the final
    # score and the per-sample scores of the records that score below 1.0.
    FIELDS = {
        "invoice_number": {
            "TextExactMatch": (0.875, {"3": 0.0}),
            "RawExactMatch": (0.625, {"2": 0.0, "3": 0.0, "8": 0.0}),
        },
        "vendor": {
            "TextExactMatch": (0.75, {"4": 0.0, "6": 0.0}),
            "TextFuzzyMatch": (0.8674948240165632, {"4": 17 / 21, "6": 0.1304347826}),
        },
        "total": {
            "NumericExactMatch": (0.75, {"4": 0.0, "7": 0.0}),
            "TextExactMatch": (0.5, {"2": 0.0, "4": 0.0, "5": 0.0, "7": 0.0}),
        },
        "due_date": {"TextExactMatch": (0.625, {"2": 0.0, "4": 0.0, "5": 0.0})},
        "line_items": {
            "ListHungarianMatch": (0.8333333333333334, {"3": 2 / 3, "8": 0.0}),
            "ListExactMatch": (0.625, {"3": 0.0, "4": 0.0, "8": 0.0}),
        },
    }

    @pytest.mark.parametrize("field", sorted(FIELDS))
    def test_field_scores_match_the_readme(self, tmp_path, field):
        expected = self.FIELDS[field]
        output_path = tmp_path / "result.json"
        result = _run_evaluation(
            REPO_ROOT / "tests" / "examples" / "idp_invoice_extraction.json",
            output_path,
            ",".join(expected),
            "--ground-truth-path",
            str(REPO_ROOT / "tests" / "examples" / "idp_invoice_gold.json"),
            "--sample-id-key",
            "id",
            pred_key=field,
            ground_truth_key=field,
        )

        assert result.returncode == 0, result.stderr
        output = _full_output(result, output_path)
        assert output["status"] == "completed"
        for metric, (final, below_one) in expected.items():
            assert output["data"]["results"][metric] == pytest.approx(final), metric
            entries = output["data"]["samples"][metric]
            assert [entry["sample_id"] for entry in entries] == [
                str(index) for index in range(1, 9)
            ]
            assert {
                entry["sample_id"]: entry["score"]
                for entry in entries
                if entry["score"] < 1.0
            } == pytest.approx(below_one), metric


class TestKeyLookup:
    """One lookup rule for every key: top level first, else exactly once at any depth."""

    @staticmethod
    def _values(tmp_path, records, key):
        records_path = tmp_path / "records.json"
        records_path.write_text(json.dumps(records))
        return [
            item.get_raw_data()
            for item in DataHandler.parse_json_file(str(records_path), key)
        ]

    def test_the_top_level_key_wins_over_nested_occurrences(self, tmp_path):
        records = [
            {"pred": "A", "context": [{"pred": "doc-0"}]},
            {"pred": "B", "context": [{"pred": "doc-1"}, {"pred": "doc-2"}]},
        ]
        assert self._values(tmp_path, records, "pred") == ["A", "B"]

    def test_a_nested_only_key_is_found_when_it_occurs_once(self, tmp_path):
        records = [{"output": {"pred": "A"}}, {"result": {"inner": {"pred": "B"}}}]
        assert self._values(tmp_path, records, "pred") == ["A", "B"]

    def test_two_nested_occurrences_without_a_top_level_key_are_ambiguous(
        self, tmp_path
    ):
        records = [{"a": {"pred": "A"}, "b": {"pred": "B"}}]
        with pytest.raises(
            ValueError, match=re.escape("Duplicate key 'pred' in record 0")
        ):
            self._values(tmp_path, records, "pred")

    def test_an_absent_key_is_missing(self, tmp_path):
        with pytest.raises(ValueError, match=re.escape("Missing key 'id' in record 0")):
            self._values(tmp_path, [{"pred": "A"}], "id")


class TestMediaOnlyFileLoading:
    """A media-extension value is a path next to the JSON file; other values are text."""

    @staticmethod
    def _parse(tmp_path, values):
        records_path = tmp_path / "records.json"
        records_path.write_text(json.dumps([{"pred": value} for value in values]))
        return DataHandler.parse_json_file(str(records_path), "pred")

    def test_values_that_name_text_files_stay_strings(self, tmp_path):
        paths = []
        for name in ("notes.txt", "data.json", "readme.md"):
            path = tmp_path / name
            path.write_text("secret", encoding="utf-8")
            paths.append(str(path))
        items = self._parse(tmp_path, paths)
        assert [item.modality for item in items] == [Modality.TEXT] * 3
        assert [item.get_raw_data() for item in items] == paths

    def test_text_values_are_not_looked_up_on_disk(self, tmp_path, monkeypatch):
        """On Windows, looking up \\\\host\\share\\a would contact that server."""
        notes = tmp_path / "notes.txt"
        notes.write_text("secret", encoding="utf-8")
        values = [str(notes), r"\\host\share\notes.txt", "plain text"]
        looked_up = []

        def spy(original):
            def check(candidate):
                looked_up.append(str(candidate))
                return original(candidate)

            return check

        monkeypatch.setattr(os.path, "exists", spy(os.path.exists))
        monkeypatch.setattr(os.path, "isfile", spy(os.path.isfile))
        items = self._parse(tmp_path, values)

        assert [item.get_raw_data() for item in items] == values
        assert not set(values) & set(looked_up)

    @pytest.mark.parametrize(
        ("value", "modality"),
        [
            pytest.param("renders/missing_01.png", "image", id="relative-image"),
            pytest.param("./missing.png", "image", id="dot-relative"),
            pytest.param("media/clip.mp4", "video", id="video"),
            pytest.param("audio/take.wav", "audio", id="audio"),
        ],
    )
    def test_a_missing_media_path_fails_the_record(self, tmp_path, value, modality):
        # Before, the string became text and an LLM judge could score the path
        # as the answer, or an error metric failed with a str - str TypeError.
        expected = (
            f"record 1: {modality} file not found: {value!r} "
            f"(looked for {tmp_path / value})"
        )
        with pytest.raises(ValueError, match=re.escape(expected)):
            self._parse(tmp_path, ["plain text", value])

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param("header.png", id="bare-name"),
            pytest.param("see header.png", id="sentence"),
            pytest.param("The logo is in renders/header.png", id="sentence-with-path"),
            pytest.param("first line\nrenders/header.png", id="multi-line"),
        ],
    )
    def test_a_string_that_is_not_a_path_stays_text(self, tmp_path, value):
        """Only a whole-string path is a missing file; a name or prose is text."""
        items = self._parse(tmp_path, [value])
        assert items[0].modality is Modality.TEXT
        assert items[0].get_raw_data() == value

    def test_an_existing_file_loads_whatever_its_shape(self, tmp_path):
        """Existence wins: a bare name or a name with spaces next to the JSON file loads."""
        # A subprocess keeps PIL and the image data type out of this process.
        if importlib.util.find_spec("PIL") is None:
            pytest.skip("Pillow is not installed")
        source = REPO_ROOT / "tests" / "examples" / "rmse_images" / "reference_0.png"
        shutil.copy(source, tmp_path / "bare.png")
        shutil.copy(source, tmp_path / "my photo.png")
        records_path = tmp_path / "records.json"
        records_path.write_text(
            json.dumps([{"pred": "bare.png", "gold": "my photo.png"}])
        )
        output_path = tmp_path / "result.json"
        # Two text values cannot be subtracted; two equal images give 0.0.
        result = _run_evaluation(records_path, output_path, "MeanSquaredError")
        assert result.returncode == 0, result.stderr
        output = _full_output(result, output_path)
        assert output["data"]["results"]["MeanSquaredError"] == 0.0

    def test_a_missing_absolute_media_path_is_reported_as_given(self, tmp_path):
        value = str(tmp_path / "elsewhere" / "missing.png")
        expected = f"record 0: image file not found: {value!r} (looked for {value})"
        with pytest.raises(ValueError, match=re.escape(expected)):
            self._parse(tmp_path, [value])

    def test_relative_media_paths_resolve_from_the_json_file_folder(self, tmp_path):
        """The rule of the prompt manifest applies to record values too."""
        if importlib.util.find_spec("PIL") is None:
            pytest.skip("Pillow is not installed")
        data_dir = tmp_path / "data"
        (data_dir / "img").mkdir(parents=True)
        source = REPO_ROOT / "tests" / "examples" / "rmse_images" / "reference_0.png"
        shutil.copy(source, data_dir / "img" / "a.png")
        records_path = data_dir / "records.json"
        records_path.write_text(
            json.dumps([{"pred": "img/a.png", "gold": "img/a.png"}])
        )
        output_path = tmp_path / "result.json"
        # The working directory is the repository root, where img/a.png does
        # not exist; the path resolves from the folder of records.json.
        result = _run_evaluation(records_path, output_path, "MeanSquaredError")
        assert result.returncode == 0, result.stderr
        output = _full_output(result, output_path)
        assert output["data"]["results"]["MeanSquaredError"] == 0.0

    def test_the_image_example_runs_from_any_working_directory(self, tmp_path):
        if importlib.util.find_spec("PIL") is None:
            pytest.skip("Pillow is not installed")
        output_path = tmp_path / "result.json"
        result = _run_metrics(
            [
                "--execution-output-path",
                str(REPO_ROOT / "tests" / "examples" / "rmse_image_renders.json"),
                "--metrics",
                "MeanAbsoluteError",
                "--pred-key",
                "generated_image",
                "--ground-truth-key",
                "reference_image",
                "--output-path",
                str(output_path),
            ],
            cwd=str(tmp_path),
        )
        assert result.returncode == 0, result.stderr
        output = _full_output(result, output_path)
        assert output["data"]["results"]["MeanAbsoluteError"] == 10.0

    def test_runner_reports_a_missing_media_path_before_any_metric(self, tmp_path):
        records_path = tmp_path / "records.json"
        records_path.write_text(
            json.dumps([{"pred": "renders/missing_01.png", "gold": "x"}])
        )
        output_path = tmp_path / "result.json"
        result = _run_evaluation(records_path, output_path, "LLMHelpfulness,Accuracy")
        assert result.returncode == 1
        output = _full_output(result, output_path, "stderr")
        assert output["status"] == "failed"
        assert output["data"] is None
        assert output["errors"] == [
            "record 0: image file not found: 'renders/missing_01.png' "
            f"(looked for {tmp_path / 'renders' / 'missing_01.png'})"
        ]

    def test_runner_loads_a_value_that_names_an_existing_image(self, tmp_path):
        # A subprocess keeps PIL and the image data type out of this process.
        if importlib.util.find_spec("PIL") is None:
            pytest.skip("Pillow is not installed")
        image_path = str(
            REPO_ROOT / "tests" / "examples" / "rmse_images" / "reference_0.png"
        )
        records_path = tmp_path / "records.json"
        records_path.write_text(json.dumps([{"pred": image_path, "gold": image_path}]))
        output_path = tmp_path / "result.json"

        result = _run_evaluation(records_path, output_path, "MeanSquaredError")

        # Two text values cannot be subtracted; two equal images give 0.0.
        assert result.returncode == 0, result.stderr
        output = _full_output(result, output_path)
        assert output["data"]["results"]["MeanSquaredError"] == 0.0

    def test_an_image_in_another_pillow_format_is_rejected(self, tmp_path):
        """Only JPEG, PNG, BMP, TIFF, WebP and GIF open, whatever the extension."""
        # A subprocess keeps PIL and the image data type out of this process.
        if importlib.util.find_spec("PIL") is None:
            pytest.skip("Pillow is not installed")
        # A 2x2 PPM image, which Pillow can decode, under an image extension.
        (tmp_path / "scan.png").write_bytes(b"P6\n2 2\n255\n" + bytes(range(12)))
        records_path = tmp_path / "records.json"
        records_path.write_text(json.dumps([{"pred": "scan.png", "gold": "scan.png"}]))
        output_path = tmp_path / "result.json"

        result = _run_evaluation(records_path, output_path, "MeanAbsoluteError")

        assert result.returncode == 1
        output = _full_output(result, output_path, "stderr")
        (message,) = output["errors"]
        assert message.startswith("cannot identify image file")
        assert "scan.png" in message

    def test_an_image_over_64_million_pixels_stops_the_load(self, tmp_path):
        """The header is checked before any pixel is decoded."""
        # A subprocess keeps PIL and the image data type out of this process.
        if importlib.util.find_spec("PIL") is None:
            pytest.skip("Pillow is not installed")
        width, height = 8001, 8000
        # A blank 1-bit grayscale PNG: each row is a filter byte and 1001 bytes.
        rows = (b"\x00" * (1 + (width + 7) // 8)) * height
        chunks = [
            (b"IHDR", struct.pack(">IIBBBBB", width, height, 1, 0, 0, 0, 0)),
            (b"IDAT", zlib.compress(rows)),
            (b"IEND", b""),
        ]
        (tmp_path / "huge.png").write_bytes(
            b"\x89PNG\r\n\x1a\n"
            + b"".join(
                struct.pack(">I", len(payload))
                + tag
                + payload
                + struct.pack(">I", zlib.crc32(tag + payload))
                for tag, payload in chunks
            )
        )
        records_path = tmp_path / "records.json"
        records_path.write_text(json.dumps([{"pred": "huge.png", "gold": "huge.png"}]))
        output_path = tmp_path / "result.json"

        result = _run_evaluation(records_path, output_path, "MeanAbsoluteError")

        assert result.returncode == 1
        output = _full_output(result, output_path, "stderr")
        assert output["errors"] == [
            f"cannot load the 8001x8000 image {str(tmp_path / 'huge.png')!r}; an "
            "image has at most 64,000,000 pixels (8000x8000)"
        ]

    def test_text_data_keeps_a_string_that_names_a_file(self, tmp_path):
        path = tmp_path / "notes.txt"
        path.write_text("secret", encoding="utf-8")

        text = TextData(str(path))

        assert text.get_raw_data() == str(path)
        assert text.get_data() == str(path)

    def test_text_data_from_file_reads_the_content(self, tmp_path):
        path = tmp_path / "notes.txt"
        path.write_text("alpha beta", encoding="utf-8")

        text = TextData.from_file(str(path))

        assert text.get_raw_data() == "alpha beta"
        assert list(text.get_data()) == ["alpha", "beta"]

    def test_runner_never_reads_a_text_file_that_a_value_names(self, tmp_path):
        (tmp_path / "notes.txt").write_text("secret", encoding="utf-8")
        records_path = tmp_path / "records.json"
        records_path.write_text(
            json.dumps([{"pred": "notes.txt", "gold": "notes.txt"}])
        )
        output_path = tmp_path / "result.json"

        result = _run_metrics(
            [
                "--execution-output-path",
                str(records_path),
                "--metrics",
                "TextExactMatch,Accuracy",
                "--pred-key",
                "pred",
                "--ground-truth-key",
                "gold",
                "--aggregation-methods",
                ",prf",
                "--matcher",
                ",labels",
                "--output-path",
                str(output_path),
            ],
            cwd=str(tmp_path),
        )

        assert result.returncode == 0, result.stderr
        output = _full_output(result, output_path)
        assert output["data"]["results"]["TextExactMatch"] == 1.0
        entry = output["data"]["samples"]["Accuracy"][0]
        assert entry["matches"] == {
            "notes.txt": {"scores": [[1.0]], "n_pred": 1, "n_ref": 1}
        }
        assert "secret" not in result.stdout
        assert "secret" not in output_path.read_text()


class TestDataRootOption:
    """``--data-root`` confines the files that the data files name to one folder."""

    @staticmethod
    def _errors(result, output_path):
        assert result.returncode == 1
        output = _full_output(result, output_path, "stderr")
        assert output["data"] is None
        return output["errors"]

    @pytest.mark.parametrize("exists", [True, False], ids=["existing", "missing"])
    def test_a_record_value_outside_the_root_stops_the_run(self, tmp_path, exists):
        """The check comes first, so the error does not tell whether the file exists."""
        root = tmp_path / "data"
        root.mkdir()
        if exists:
            (tmp_path / "outside.png").write_bytes(b"never read")
        records_path = root / "records.json"
        records_path.write_text(json.dumps([{"pred": "../outside.png", "gold": "x"}]))
        output_path = tmp_path / "result.json"

        result = _run_evaluation(
            records_path, output_path, "Accuracy", "--data-root", str(root)
        )

        assert self._errors(result, output_path) == [
            f"record 0: '../outside.png' is outside --data-root {root}"
        ]

    def test_a_symbolic_link_out_of_the_root_stops_the_run(self, tmp_path):
        root = tmp_path / "data"
        root.mkdir()
        (tmp_path / "outside.png").write_bytes(b"never read")
        (root / "inside.png").symlink_to(tmp_path / "outside.png")
        records_path = root / "records.json"
        records_path.write_text(json.dumps([{"pred": "inside.png", "gold": "x"}]))
        output_path = tmp_path / "result.json"

        result = _run_evaluation(
            records_path, output_path, "Accuracy", "--data-root", str(root)
        )

        assert self._errors(result, output_path) == [
            f"record 0: 'inside.png' is outside --data-root {root}"
        ]

    @pytest.mark.parametrize(
        "value", [r"\\host\share\a.png", "//host/share/a.png"], ids=["\\\\", "//"]
    )
    def test_a_windows_network_path_fails_before_it_is_resolved(self, value):
        """Resolving a path on another host's share would contact that host.

        Windows paths cannot be made on POSIX, so the check runs on pure ones;
        a PureWindowsPath has no resolve(), so reaching it fails the test.
        """
        root = PureWindowsPath("C:/data")
        expected = f"record 0: {value!r} is outside --data-root {root}"
        with pytest.raises(ValueError, match=re.escape(expected)):
            DataHandler._inside_root(PureWindowsPath(value), value, 0, root, root)

    @pytest.mark.skipif(os.name == "nt", reason="// starts a network path there")
    def test_a_posix_path_that_starts_with_two_slashes_loads(self, tmp_path):
        root = (tmp_path / "data").resolve()
        root.mkdir()
        (root / "inside.txt").write_text("inside", encoding="utf-8")
        references = root / "references.json"
        references.write_text(json.dumps(["/" + str(root / "inside.txt")]))

        items = DataHandler.create_data(str(references), data_root=root)

        assert [item.get_raw_data() for item in items] == ["inside"]

    def test_without_a_root_a_path_outside_the_folder_logs_one_warning(
        self, tmp_path, caplog
    ):
        """The file still loads; the warning names its folder once."""
        folder = (tmp_path / "data").resolve()
        folder.mkdir()
        (folder / "inside.txt").write_text("inside", encoding="utf-8")
        (tmp_path / "notes.txt").write_text("outside", encoding="utf-8")
        references = folder / "references.json"
        references.write_text(json.dumps(["inside.txt", "../notes.txt"]))

        with caplog.at_level(logging.WARNING, "metric_library.core.data_handler"):
            loads = [DataHandler.create_data(str(references)) for _ in range(2)]

        for items in loads:
            assert [item.get_raw_data() for item in items] == ["inside", "outside"]
        assert [record.getMessage() for record in caplog.records] == [
            f"Data in {folder} names files outside that folder, such as "
            "'../notes.txt'; set --data-root to confine the files that data "
            "files name"
        ]

    def test_a_prompt_manifest_entry_outside_the_root_stops_the_run(self, tmp_path):
        root = tmp_path / "data"
        root.mkdir()
        (tmp_path / "notes.txt").write_text("never read", encoding="utf-8")
        manifest_path = root / "prompts.json"
        manifest_path.write_text(json.dumps(["../notes.txt"]))
        records_path = root / "records.json"
        records_path.write_text(json.dumps([{"pred": "A", "gold": "A"}]))
        output_path = tmp_path / "result.json"

        result = _run_evaluation(
            records_path,
            output_path,
            "Accuracy",
            "--prompt-path",
            str(manifest_path),
            "--data-root",
            str(root),
        )

        assert self._errors(result, output_path) == [
            f"record 0: '../notes.txt' is outside --data-root {root}"
        ]

    def test_a_keyless_reference_entry_outside_the_root_stops_the_run(self, tmp_path):
        root = tmp_path / "data"
        root.mkdir()
        (tmp_path / "notes.txt").write_text("never read", encoding="utf-8")
        gold_path = root / "references.json"
        gold_path.write_text(json.dumps(["../notes.txt"]))
        records_path = root / "records.json"
        records_path.write_text(json.dumps([{"pred": "A"}]))
        output_path = tmp_path / "result.json"

        result = _run_metrics(
            [
                "--execution-output-path",
                str(records_path),
                "--metrics",
                "Accuracy",
                "--pred-key",
                "pred",
                "--ground-truth-path",
                str(gold_path),
                "--data-root",
                str(root),
                "--output-path",
                str(output_path),
            ]
        )

        assert self._errors(result, output_path) == [
            f"record 0: '../notes.txt' is outside --data-root {root}"
        ]

    def test_files_inside_the_root_load_from_a_data_file_outside_it(self, tmp_path):
        """The path options are not checked; the paths that the data names are."""
        # A subprocess keeps PIL and the image data type out of this process.
        if importlib.util.find_spec("PIL") is None:
            pytest.skip("Pillow is not installed")
        root = tmp_path / "data"
        (root / "img").mkdir(parents=True)
        source = REPO_ROOT / "tests" / "examples" / "rmse_images" / "reference_0.png"
        shutil.copy(source, root / "img" / "a.png")
        records_path = tmp_path / "records.json"
        records_path.write_text(
            json.dumps([{"pred": "data/img/a.png", "gold": "data/img/../img/a.png"}])
        )
        output_path = tmp_path / "result.json"

        result = _run_evaluation(
            records_path, output_path, "MeanSquaredError", "--data-root", str(root)
        )

        assert result.returncode == 0, result.stderr
        output = _full_output(result, output_path)
        assert output["data"]["results"]["MeanSquaredError"] == 0.0

    def test_the_root_must_be_an_existing_directory(self, sample_data, tmp_path):
        missing = tmp_path / "missing"
        output_path = tmp_path / "result.json"

        result = _run_evaluation(
            sample_data, output_path, "Accuracy", "--data-root", str(missing)
        )

        assert self._errors(result, output_path) == [
            f"--data-root must be an existing directory: {str(missing)!r}"
        ]


class TestJudgeImageFormats:
    """Every image reaches the LLM judge in a format the Converse API reads."""

    @staticmethod
    def _run_judge(tmp_path, images):
        """Judge one record per ``{name: (mode, pages)}`` image with LLMCorrectness.

        The images are 3x2 files saved by Pillow; ``(mode, pages, (width,
        height), noise)`` sets another size, and ``noise`` fills the pixels with
        random bytes, which PNG cannot compress. A stand-in for the Bedrock
        runtime logs the format and bytes of every image block the judge is
        sent, and ``saves.log`` gets the format of every image the runner
        encodes. Returns the run and the logged blocks, in record order.
        """
        # A subprocess keeps PIL and the image data type out of this process.
        if importlib.util.find_spec("PIL") is None:
            pytest.skip("Pillow is not installed")
        save_images = textwrap.dedent("""\
            import json
            import os
            import sys

            from PIL import Image

            for path, (mode, pages, *rest) in json.loads(sys.argv[1]).items():
                (width, height), noise = rest or ((3, 2), False)

                def frame():
                    if noise:
                        pixels = os.urandom(width * height * len(mode))
                        return Image.frombytes(mode, (width, height), pixels)
                    return Image.new(mode, (width, height))

                frames = [frame() for _ in range(pages)]
                frames[0].save(path, save_all=pages > 1, append_images=frames[1:])
            """)
        paths = {str(tmp_path / name): spec for name, spec in images.items()}
        # No shell; the script and the image paths come from the test.
        subprocess.run(  # nosec B603 # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit
            [sys.executable, "-c", save_images, json.dumps(paths)], check=True
        )
        plugin_dir = tmp_path / "plugin"
        plugin_dir.mkdir()
        (plugin_dir / "custom_metrics.py").write_text(
            textwrap.dedent("""\
            import base64
            import json
            import os
            from unittest.mock import Mock

            import PIL.Image

            import metric_library.core.bedrock_client as bedrock_client

            save = PIL.Image.Image.save


            def logged_save(image, fp, format=None, **params):
                # Each save in the runner is one conversion of a judge image.
                with open(os.environ["SAVE_LOG"], "a", encoding="utf-8") as log:
                    print(format, file=log)
                return save(image, fp, format, **params)


            PIL.Image.Image.save = logged_save


            def converse(**kwargs):
                with open(os.environ["JUDGE_LOG"], "a", encoding="utf-8") as log:
                    for block in kwargs["messages"][0]["content"]:
                        if "image" in block:
                            image = block["image"]
                            data = base64.b64encode(image["source"]["bytes"]).decode()
                            print(json.dumps([image["format"], data]), file=log)
                tool_use = {
                    "toolUseId": "t1",
                    "name": "submit_evaluation",
                    "input": {"reasoning": "ok", "score": 1.0},
                }
                return {"output": {"message": {"content": [{"toolUse": tool_use}]}}}


            runtime = Mock(spec=["converse"])
            runtime.converse.side_effect = converse
            bedrock_client.boto3.Session = lambda *args, region_name=None, **kw: Mock(
                region_name=region_name or "us-east-1",
                client=Mock(return_value=runtime),
            )
            """)
        )
        records_path = tmp_path / "records.json"
        records_path.write_text(
            json.dumps(
                [
                    {"task": "Describe the image.", "pred": name, "gold": "A picture."}
                    for name in images
                ]
            )
        )
        log_path = tmp_path / "judge.log"
        result = _run_metrics(
            [
                "--execution-output-path",
                str(records_path),
                "--metrics",
                "LLMCorrectness",
                "--pred-key",
                "pred",
                "--ground-truth-key",
                "gold",
                "--prompt-key",
                "task",
                "--output-path",
                str(tmp_path / "result.json"),
                "--custom-metrics-path",
                str(plugin_dir / "custom_metrics.py"),
            ],
            cwd=str(tmp_path),
            env={"JUDGE_LOG": str(log_path), "SAVE_LOG": str(tmp_path / "saves.log")},
        )
        lines = log_path.read_text().splitlines() if log_path.exists() else []
        blocks = [
            (image_format, base64.b64decode(data))
            for image_format, data in map(json.loads, lines)
        ]
        return result, blocks

    def test_bmp_and_tiff_are_sent_as_png_and_other_formats_unchanged(self, tmp_path):
        images = {
            "photo.png": ("RGB", 1),
            "photo.jpg": ("RGB", 1),
            "photo.gif": ("RGB", 1),
            "photo.webp": ("RGB", 1),
            "scan.bmp": ("RGB", 1),
            "scan.tiff": ("CMYK", 1),
            "scan.tif": ("1", 1),
        }

        result, blocks = self._run_judge(tmp_path, images)

        assert result.returncode == 0, result.stderr
        files = {name: (tmp_path / name).read_bytes() for name in images}
        assert blocks[:4] == [
            ("png", files["photo.png"]),
            ("jpeg", files["photo.jpg"]),
            ("gif", files["photo.gif"]),
            ("webp", files["photo.webp"]),
        ]
        # A PNG signature, then the IHDR chunk whose first fields are the size.
        assert [
            (image_format, data[:8], struct.unpack(">II", data[16:24]))
            for image_format, data in blocks[4:]
        ] == [("png", b"\x89PNG\r\n\x1a\n", (3, 2))] * 3

    @pytest.mark.parametrize(
        ("image", "pattern"),
        [
            (
                ("L", 2),
                "cannot send a 2-page TIFF image to the LLM judge; split it into "
                "one image per page",
            ),
            (
                ("RGB", 1, (8001, 2), False),
                "cannot send a 8001x2 TIFF image to the LLM judge; it accepts at "
                "most 8000 pixels per side",
            ),
            (
                ("RGB", 1, (1150, 1150), True),
                r"cannot send a TIFF image of [\d,]+ bytes as PNG to the LLM judge; "
                r"it accepts at most 3,932,160 bytes \(3\.75 MB\) per image",
            ),
        ],
        ids=["two pages", "8001 pixels wide", "over 3.75 MB as PNG"],
    )
    def test_an_image_the_judge_cannot_take_fails_before_the_first_call(
        self, tmp_path, image, pattern
    ):
        result, blocks = self._run_judge(
            tmp_path, {"photo.png": ("RGB", 1), "scan.tiff": image}
        )

        assert result.returncode == 1
        output = _full_output(result, tmp_path / "result.json")
        (message,) = output["errors"]
        assert re.fullmatch(f"LLMCorrectness: prediction 1: {pattern}", message)
        # The image of record 1 is fine, but no record reaches the judge.
        assert blocks == []

    def test_an_image_sent_as_it_is_counts_its_file_bytes(self, tmp_path):
        result, blocks = self._run_judge(
            tmp_path, {"noise.png": ("RGB", 1, (1150, 1150), True)}
        )

        size = (tmp_path / "noise.png").stat().st_size
        output = _full_output(result, tmp_path / "result.json")
        assert output["errors"] == [
            f"LLMCorrectness: prediction 0: cannot send a PNG image of {size:,} "
            "bytes to the LLM judge; it accepts at most 3,932,160 bytes (3.75 MB) "
            "per image"
        ]
        assert blocks == []

    def test_a_converted_image_is_encoded_once(self, tmp_path):
        """Input validation checks the PNG of a BMP, and the request reuses it."""
        result, blocks = self._run_judge(tmp_path, {"scan.bmp": ("RGB", 1)})

        assert result.returncode == 0, result.stderr
        assert [image_format for image_format, _ in blocks] == ["png"]
        assert (tmp_path / "saves.log").read_text().splitlines() == ["PNG"]

    def test_images_at_the_limits_reach_the_judge(self, tmp_path):
        """8000 pixels on a side pass, and a BMP counts the bytes of its PNG."""
        images = {
            "wide.png": ("RGB", 1, (8000, 2), False),
            "large.bmp": ("RGB", 1, (1200, 1200), False),
        }

        result, blocks = self._run_judge(tmp_path, images)

        assert result.returncode == 0, result.stderr
        assert (tmp_path / "large.bmp").stat().st_size > 3_932_160
        assert [image_format for image_format, _ in blocks] == ["png", "png"]
        assert blocks[0][1] == (tmp_path / "wide.png").read_bytes()
        assert len(blocks[1][1]) < 3_932_160


class TestAwsOptions:
    """``--model-id`` and ``--aws-profile`` reach every Bedrock client of the run."""

    # Removed for each test, so the real environment and ~/.aws never leak in
    # and no credential provider makes a network call.
    AWS_VARIABLES = (
        "AWS_PROFILE",
        "AWS_DEFAULT_PROFILE",
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_ROLE_ARN",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
        "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
        "AWS_CONTAINER_CREDENTIALS_FULL_URI",
        "BEDROCK_MODEL_ID",
    )

    @pytest.fixture()
    def aws_run(self, tmp_path, monkeypatch):
        """Isolate the AWS configuration and return a runner call on one record."""
        config_path = tmp_path / "aws-config"
        config_path.write_text(
            textwrap.dedent("""\
            [profile p1]
            region = ap-southeast-2
            aws_access_key_id = AKIAP1EXAMPLE
            aws_secret_access_key = p1-secret

            [profile p2]
            region = eu-west-1
            aws_access_key_id = AKIAP2EXAMPLE
            aws_secret_access_key = p2-secret
            """)
        )
        for name in self.AWS_VARIABLES:
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("AWS_CONFIG_FILE", str(config_path))
        monkeypatch.setenv(
            "AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "missing-credentials")
        )
        monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")

        plugin_dir = tmp_path / "plugin"
        plugin_dir.mkdir()
        (plugin_dir / "custom_metrics.py").write_text(
            textwrap.dedent("""\
            from metric_library.core.base import Metric
            from metric_library.core.registry import MetricRegistry

            @MetricRegistry.register
            class ClientProbe(Metric):
                def compute(self, sample, client):
                    raise ValueError(
                        f"model={client.model_id} profile={client.profile} "
                        f"region={client.region}"
                    )
            """)
        )
        records_path = tmp_path / "records.json"
        records_path.write_text(json.dumps([{"pred": "a", "gold": "a"}]))
        output_path = tmp_path / "result.json"

        def run(metrics, *options):
            result = _run_metrics(
                [
                    "--execution-output-path",
                    str(records_path),
                    "--metrics",
                    metrics,
                    "--pred-key",
                    "pred",
                    "--ground-truth-key",
                    "gold",
                    *options,
                    "--output-path",
                    str(output_path),
                    "--custom-metrics-path",
                    str(plugin_dir / "custom_metrics.py"),
                ],
                cwd=str(tmp_path),
            )
            return result, json.loads(result.stdout)

        return run

    @pytest.mark.parametrize(
        ("environment", "options", "expected"),
        [
            pytest.param(
                {},
                [],
                "model=global.anthropic.claude-sonnet-4-6 profile=None region=us-east-1",
                id="defaults",
            ),
            pytest.param(
                {"BEDROCK_MODEL_ID": "m-env", "AWS_PROFILE": "p2"},
                [],
                "model=m-env profile=p2 region=eu-west-1",
                id="environment",
            ),
            pytest.param(
                {"BEDROCK_MODEL_ID": "m-env", "AWS_PROFILE": "p2"},
                ["--model-id", "m-cli", "--aws-profile", "p1"],
                "model=m-cli profile=p1 region=ap-southeast-2",
                id="options-win",
            ),
        ],
    )
    def test_judge_client_settings_come_from_the_switches_then_the_environment(
        self, aws_run, monkeypatch, environment, options, expected
    ):
        """The region shows which profile the session used."""
        for name, value in environment.items():
            monkeypatch.setenv(name, value)

        result, output = aws_run("ClientProbe", *options)

        assert result.returncode == 1, result.stderr
        assert output["data"]["results"]["ClientProbe"] == {
            "error": f"sample 1: {expected}"
        }

    def test_aws_profile_reaches_the_semantic_embedding_client(self, aws_run):
        """Ignoring the option would give 'Unable to locate credentials' instead."""
        result, output = aws_run(
            "ItemSemanticMatch,Accuracy", "--aws-profile", "no-such"
        )

        assert result.returncode == 2, result.stderr
        results = output["data"]["results"]
        assert results["ItemSemanticMatch"] == {
            "error": "The config profile (no-such) could not be found"
        }
        assert results["Accuracy"] == 1.0


class TestCustomMetricsLoading:
    """The runner loads custom metrics only from --custom-metrics-path, after it
    parses its arguments."""

    PLUGIN = """\
        from metric_library.core.base import Metric
        from metric_library.core.registry import MetricRegistry

        @MetricRegistry.register
        class AlwaysOne(Metric):
            def compute(self, sample, client):
                return 1.0
        """
    # A custom_metrics.py whose code must not run.
    TRAP = 'raise RuntimeError("a custom_metrics.py that no option names ran")\n'

    @staticmethod
    def _evaluation_args(data_path, output_path, metrics, *options):
        return [
            "--execution-output-path",
            str(data_path),
            "--metrics",
            metrics,
            "--pred-key",
            "pred",
            "--ground-truth-key",
            "gold",
            *options,
            "--output-path",
            str(output_path),
        ]

    def test_the_path_option_loads_that_file_and_skips_the_import(
        self, sample_data, tmp_path
    ):
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        (work_dir / "custom_metrics.py").write_text(self.TRAP)
        plugin_path = tmp_path / "plugins" / "my_metrics.py"
        plugin_path.parent.mkdir()
        plugin_path.write_text(textwrap.dedent(self.PLUGIN))
        output_path = tmp_path / "result.json"

        result = _run_metrics(
            self._evaluation_args(
                sample_data,
                output_path,
                "Accuracy,AlwaysOne",
                "--custom-metrics-path",
                str(plugin_path),
            ),
            cwd=str(work_dir),
        )

        assert result.returncode == 0, result.stderr
        output = _full_output(result, output_path)
        assert output["data"]["results"]["AlwaysOne"] == 1.0

    def test_the_path_option_adds_the_plugin_metrics_to_the_list(self, tmp_path):
        plugin_path = tmp_path / "my_metrics.py"
        plugin_path.write_text(textwrap.dedent(self.PLUGIN))

        result = _run_metrics(
            [
                "--list-metrics",
                "--metrics",
                "AlwaysOne",
                "--custom-metrics-path",
                str(plugin_path),
            ],
            cwd=str(tmp_path),
        )

        assert result.returncode == 0, result.stderr
        assert [entry["name"] for entry in json.loads(result.stdout)] == ["AlwaysOne"]

    def test_help_does_not_run_the_custom_metrics_module(self, tmp_path):
        (tmp_path / "custom_metrics.py").write_text(self.TRAP)

        result = _run_metrics(["--help"], cwd=str(tmp_path))

        assert result.returncode == 0, result.stderr
        assert "--custom-metrics-path" in result.stdout

    @pytest.mark.parametrize("command", ["evaluation", "list"])
    @pytest.mark.parametrize("folder", ["working-directory", "pythonpath"])
    def test_without_the_path_option_no_custom_metrics_module_runs(
        self, sample_data, tmp_path, folder, command
    ):
        plugin_dir = tmp_path / "plugins"
        plugin_dir.mkdir()
        (plugin_dir / "custom_metrics.py").write_text(self.TRAP)
        output_path = tmp_path / "result.json"
        if command == "evaluation":
            args = self._evaluation_args(sample_data, output_path, "Accuracy")
        else:
            args = ["--list-metrics", "--metrics", "Accuracy"]

        result = _run_metrics(
            args,
            extra_pythonpath=str(plugin_dir) if folder == "pythonpath" else None,
            cwd=str(plugin_dir if folder == "working-directory" else tmp_path),
        )

        assert result.returncode == 0, result.stderr

    def test_a_plugin_that_fails_to_import_gives_an_error_envelope(
        self, sample_data, tmp_path
    ):
        plugin_path = tmp_path / "my_metrics.py"
        plugin_path.write_text("import no_such_module_for_this_test\n")
        output_path = tmp_path / "result.json"

        result = _run_metrics(
            self._evaluation_args(
                sample_data,
                output_path,
                "Accuracy",
                "--custom-metrics-path",
                str(plugin_path),
            ),
            cwd=str(tmp_path),
        )

        assert result.returncode == 1
        output = _full_output(result, output_path, "stderr")
        assert output["errors"] == [
            f"cannot load the custom metrics module {str(plugin_path)!r}: "
            "No module named 'no_such_module_for_this_test'"
        ]


class TestCLI:
    """Verify CLI argument handling."""

    def test_help_flag(self):
        result = _run_metrics(["--help"])
        assert result.returncode == 0
        help_text = " ".join(result.stdout.split())
        assert "execution-output-path" in help_text
        assert "--prompt-key" in help_text
        assert "prompt records." in help_text
        assert "array of existing prompt file paths resolved relative" in help_text
        assert "--model-id" in help_text
        assert "--aws-profile" in help_text

    def test_missing_required_args(self):
        result = _run_metrics([])
        assert result.returncode != 0

    def test_abbreviated_option_names_are_rejected(self, sample_data, tmp_path):
        """An abbreviated option name is an argparse error, not a runner envelope."""
        output_path = tmp_path / "result.json"
        result = _run_metrics(
            [
                "--execution-output-path",
                str(sample_data),
                "--metrics",
                "Accuracy",
                "--pred-key",
                "pred",
                "--ground-truth-key",
                "gold",
                "--aggregation-method",
                "median",
                "--output-path",
                str(output_path),
            ]
        )

        assert result.returncode == 2
        assert "unrecognized arguments: --aggregation-method" in result.stderr
        assert "usage:" in result.stderr
        assert result.stdout == ""
        assert not output_path.exists()


class TestOutputFile:
    """The envelope file is readable by its owner only and replaces what was there."""

    @pytest.mark.parametrize("existing", [False, True], ids=["new", "replaced"])
    def test_the_envelope_file_has_mode_0600(self, sample_data, tmp_path, existing):
        output_path = tmp_path / "out" / "result.json"
        if existing:
            output_path.parent.mkdir()
            output_path.write_text("old")
            output_path.chmod(0o644)

        result = _run_evaluation(sample_data, output_path, "Accuracy")

        assert result.returncode == 0, result.stderr
        assert _full_output(result, output_path)["status"] == "completed"
        assert output_path.stat().st_mode & 0o777 == 0o600
        assert [path.name for path in output_path.parent.iterdir()] == ["result.json"]

    def test_a_symbolic_link_at_the_output_path_is_replaced(
        self, sample_data, tmp_path
    ):
        """The link's target, which may be any file of the user, is not written."""
        target = tmp_path / "target.txt"
        target.write_text("keep")
        output_path = tmp_path / "result.json"
        output_path.symlink_to(target)

        result = _run_evaluation(sample_data, output_path, "Accuracy")

        assert result.returncode == 0, result.stderr
        assert target.read_text() == "keep"
        assert not output_path.is_symlink()
        assert _full_output(result, output_path)["status"] == "completed"

    def test_dev_null_takes_the_envelope(self, sample_data):
        result = _run_evaluation(sample_data, os.devnull, "Accuracy")

        assert result.returncode == 0, result.stderr
        printed = json.loads(result.stdout)
        assert printed["status"] == "completed"
        assert printed["output_path"] == os.devnull

    def test_a_named_pipe_is_written_in_place(self, sample_data, tmp_path):
        """A reader of the pipe, such as a process substitution, gets the envelope."""
        pipe = tmp_path / "envelope.pipe"
        os.mkfifo(pipe)
        received = []
        reader = threading.Thread(
            target=lambda: received.append(pipe.read_text()), daemon=True
        )
        reader.start()

        result = _run_evaluation(sample_data, pipe, "Accuracy")
        reader.join(timeout=30)

        assert result.returncode == 0, result.stderr
        assert stat.S_ISFIFO(pipe.stat().st_mode)
        assert json.loads(received[0])["status"] == "completed"

    def test_dev_stdout_takes_the_envelope_when_it_is_a_file(
        self, sample_data, tmp_path
    ):
        """--output-path /dev/stdout > out.json: /dev/stdout points to a file."""
        captured = tmp_path / "out.json"
        with captured.open("w") as stdout:
            result = _run_metrics(
                [
                    "--execution-output-path",
                    str(sample_data),
                    "--metrics",
                    "Accuracy",
                    "--pred-key",
                    "pred",
                    "--ground-truth-key",
                    "gold",
                    "--output-path",
                    "/dev/stdout",
                ],
                stdout=stdout,
            )

        assert result.returncode == 0, result.stderr
        assert '"status": "completed"' in captured.read_text()

    def test_a_link_to_a_device_is_replaced_not_followed(self, sample_data, tmp_path):
        """A link planted at the path, here to /dev/null, gets no envelope."""
        output_path = tmp_path / "result.json"
        output_path.symlink_to(os.devnull)

        result = _run_evaluation(sample_data, output_path, "Accuracy")

        assert result.returncode == 0, result.stderr
        assert not output_path.is_symlink()
        assert _full_output(result, output_path)["status"] == "completed"


class TestStdoutOption:
    """The runner prints a summary by default and the full envelope on request."""

    def test_default_stdout_omits_samples_and_names_the_output_file(
        self, sample_data, tmp_path
    ):
        # Agents read the shell output, which their tools truncate; the
        # per-sample list grows with the record count and lives in the file.
        output_path = tmp_path / "result.json"
        result = _run_evaluation(sample_data, output_path, "Accuracy,TextExactMatch")
        assert result.returncode == 0, result.stderr
        printed = json.loads(result.stdout)
        assert set(printed) == STDOUT_KEYS
        assert printed["output_path"] == str(output_path)
        assert set(printed["data"]) == {"results"}
        assert printed["data"]["results"] == {
            "Accuracy": pytest.approx(2 / 3),
            "TextExactMatch": pytest.approx(2 / 3),
        }
        full = json.loads(output_path.read_text())
        assert set(full) == ENVELOPE_KEYS
        assert set(full["data"]) == {"results", "samples"}
        assert len(full["data"]["samples"]["Accuracy"]) == 3
        assert printed == _summary_of(full, output_path)

    def test_stdout_full_prints_the_file_envelope(self, sample_data, tmp_path):
        output_path = tmp_path / "result.json"
        result = _run_evaluation(
            sample_data, output_path, "Accuracy", "--stdout", "full"
        )
        assert result.returncode == 0, result.stderr
        printed = json.loads(result.stdout)
        assert set(printed) == ENVELOPE_KEYS
        assert printed == json.loads(output_path.read_text())
        assert len(printed["data"]["samples"]["Accuracy"]) == 3

    def test_pre_execution_error_summary_names_the_output_file(
        self, sample_data, tmp_path
    ):
        output_path = tmp_path / "result.json"
        result = _run_evaluation(
            sample_data, output_path, "Accuracy", pred_key="missing"
        )
        assert result.returncode == 1
        printed = json.loads(result.stderr)
        assert set(printed) == STDOUT_KEYS
        assert printed["data"] is None
        assert printed["output_path"] == str(output_path)
        assert printed["errors"] == [
            f"Missing key 'missing' in record 0 of {sample_data}"
        ]
        assert result.stdout == ""

    def test_an_unwritten_output_file_is_not_named(self, sample_data, tmp_path):
        # The parent of the output path is a regular file, so no envelope is
        # written; the summary must not point the reader at a missing file.
        blocker = tmp_path / "blocker"
        blocker.write_text("")
        output_path = blocker / "result.json"
        result = _run_evaluation(sample_data, output_path, "Accuracy")
        assert result.returncode == 1
        printed = json.loads(result.stderr)
        assert set(printed) == STDOUT_KEYS
        assert printed["output_path"] is None
        assert printed["errors"][-1].startswith(
            f"Could not write output to {output_path}: "
        )
        assert not output_path.exists()

    def test_summary_stays_small_as_records_grow(self, tmp_path):
        records_path = tmp_path / "big.json"
        records_path.write_text(
            json.dumps([{"pred": "a", "gold": "a"} for _ in range(500)])
        )
        output_path = tmp_path / "result.json"
        result = _run_evaluation(records_path, output_path, "Accuracy,TextExactMatch")
        assert result.returncode == 0, result.stderr
        assert len(result.stdout) < 1000
        assert len(output_path.read_text()) > 50_000


class TestOmitReasoningOption:
    """``--omit-reasoning`` keeps the judge's text out of ``data.samples``."""

    # A stand-in for the Bedrock runtime: every judge call scores 1.0 and
    # explains it, so the run needs no AWS access.
    PLUGIN = """\
        from unittest.mock import Mock

        import metric_library.core.bedrock_client as bedrock_client


        def converse(**kwargs):
            tool_use = {
                "toolUseId": "t1",
                "name": "submit_evaluation",
                "input": {"reasoning": "The judge's own words.", "score": 1.0},
            }
            return {"output": {"message": {"content": [{"toolUse": tool_use}]}}}


        runtime = Mock(spec=["converse"])
        runtime.converse.side_effect = converse
        bedrock_client.boto3.Session = lambda *args, region_name=None, **kw: Mock(
            region_name=region_name or "us-east-1",
            client=Mock(return_value=runtime),
        )
        """

    def test_the_reasoning_is_null_and_the_scores_are_unchanged(self, tmp_path):
        plugin_dir = tmp_path / "plugin"
        plugin_dir.mkdir()
        (plugin_dir / "custom_metrics.py").write_text(textwrap.dedent(self.PLUGIN))
        records_path = tmp_path / "records.json"
        records_path.write_text(
            json.dumps(
                [{"task": "Capital of France?", "pred": "Paris", "gold": "Paris"}]
            )
        )
        outputs = []
        for extra in ([], ["--omit-reasoning"]):
            output_path = tmp_path / f"result{len(extra)}.json"
            result = _run_metrics(
                [
                    "--execution-output-path",
                    str(records_path),
                    "--metrics",
                    "LLMCorrectness",
                    "--pred-key",
                    "pred",
                    "--ground-truth-key",
                    "gold",
                    "--prompt-key",
                    "task",
                    *extra,
                    "--output-path",
                    str(output_path),
                    "--custom-metrics-path",
                    str(plugin_dir / "custom_metrics.py"),
                ],
                cwd=str(tmp_path),
            )
            assert result.returncode == 0, result.stderr
            outputs.append(_full_output(result, output_path))

        kept, omitted = outputs
        assert kept["data"]["samples"]["LLMCorrectness"][0]["reasoning"] == (
            "The judge's own words."
        )
        assert omitted["data"]["samples"]["LLMCorrectness"][0]["reasoning"] is None
        assert (
            omitted["data"]["results"]
            == kept["data"]["results"]
            == {"LLMCorrectness": 1.0}
        )
        assert "The judge's own words." not in (tmp_path / "result1.json").read_text()


class TestEnvelopeMeta:
    """``meta`` in the file envelope records how the envelope was produced."""

    def test_a_deterministic_run_records_its_metric_configuration(
        self, sample_data, tmp_path
    ):
        output_path = tmp_path / "result.json"
        before = datetime.now(timezone.utc).replace(microsecond=0)

        result = _run_evaluation(
            sample_data,
            output_path,
            "Accuracy,TextExactMatch,TextFuzzyMatch",
            "--aggregation-methods",
            ",prf,quantile",
            "--matcher",
            ",labels,",
            "--thresholds",
            "0.5,0.8",
            "--quantile",
            "0.25",
        )

        assert result.returncode == 0, result.stderr
        meta = _full_output(result, output_path)["meta"]
        assert set(meta) == {"library_version", "started_at", "judge", "metrics"}
        assert isinstance(meta["library_version"], str)
        started_at = datetime.fromisoformat(meta["started_at"])
        assert started_at.utcoffset() == timedelta(0)
        assert before <= started_at <= datetime.now(timezone.utc)
        # No metric of the run built a judge client.
        assert meta["judge"] is None
        assert meta["metrics"] == {
            "Accuracy": {"aggregation_method": "numeric_mean", "matcher": "whole"},
            "TextExactMatch": {
                "aggregation_method": "prf",
                "matcher": "labels",
                "thresholds": [0.5, 0.8],
            },
            "TextFuzzyMatch": {
                "aggregation_method": "quantile",
                "matcher": "whole",
                "quantile": 0.25,
            },
        }

    def test_an_llm_run_records_the_judge_and_the_prompt_digest(self, tmp_path):
        plugin_dir = tmp_path / "plugin"
        plugin_dir.mkdir()
        (plugin_dir / "custom_metrics.py").write_text(
            textwrap.dedent(TestOmitReasoningOption.PLUGIN)
        )
        records_path = tmp_path / "records.json"
        records_path.write_text(
            json.dumps(
                [{"task": "Capital of France?", "pred": "Paris", "gold": "Paris"}]
            )
        )
        output_path = tmp_path / "result.json"

        result = _run_metrics(
            [
                "--execution-output-path",
                str(records_path),
                "--metrics",
                "LLMCorrectness,Accuracy",
                "--pred-key",
                "pred",
                "--ground-truth-key",
                "gold",
                "--prompt-key",
                "task",
                "--model-id",
                "test-model",
                "--output-path",
                str(output_path),
                "--custom-metrics-path",
                str(plugin_dir / "custom_metrics.py"),
            ],
            cwd=str(tmp_path),
            env={"AWS_REGION": "eu-west-1"},
        )

        assert result.returncode == 0, result.stderr
        meta = _full_output(result, output_path)["meta"]
        assert meta["judge"] == {"model_id": "test-model", "region": "eu-west-1"}
        digest = meta["metrics"]["LLMCorrectness"]["prompt_sha256"]
        assert re.fullmatch("[0-9a-f]{64}", digest)
        assert meta["metrics"] == {
            "LLMCorrectness": {
                "aggregation_method": "numeric_mean",
                "matcher": "whole",
                "prompt_sha256": digest,
            },
            "Accuracy": {"aggregation_method": "numeric_mean", "matcher": "whole"},
        }

    # Judge metrics that share the role, rubric and response schema and differ
    # in one other piece of fixed request text, or in nothing.
    DIGEST_PLUGIN = """\

        from metric_library.core.data import Modality
        from metric_library.core.llm_metric import LLMBatchMetric, LLMMetric
        from metric_library.core.multiplicity import Multiplicity
        from metric_library.core.registry import MetricRegistry

        RUBRIC = "Score 1.0 when the candidate gives the ground truth's answer."


        @MetricRegistry.register
        class DigestBase(LLMMetric):
            def rubric(self):
                return RUBRIC


        @MetricRegistry.register
        class DigestCopy(DigestBase):
            pass


        @MetricRegistry.register
        class DigestGuided(DigestBase):
            modality_guidance = {Modality.IMAGE: "Compare the images closely."}


        @MetricRegistry.register
        class DigestAlternatives(DigestBase):
            reference_multiplicity = Multiplicity.ANY_ACCEPTABLE


        @MetricRegistry.register
        class DigestBatch(LLMBatchMetric):
            def rubric(self):
                return RUBRIC
        """

    def test_the_prompt_digest_covers_the_fixed_text_around_the_data(self, tmp_path):
        plugin_dir = tmp_path / "plugin"
        plugin_dir.mkdir()
        (plugin_dir / "custom_metrics.py").write_text(
            textwrap.dedent(TestOmitReasoningOption.PLUGIN)
            + textwrap.dedent(self.DIGEST_PLUGIN)
        )
        records_path = tmp_path / "records.json"
        records_path.write_text(json.dumps([{"pred": "Paris", "gold": "Paris"}]))
        output_path = tmp_path / "result.json"
        names = ["DigestBase", "DigestCopy", "DigestGuided"]
        names += ["DigestAlternatives", "DigestBatch"]

        result = _run_metrics(
            [
                "--execution-output-path",
                str(records_path),
                "--metrics",
                ",".join(names),
                "--pred-key",
                "pred",
                "--ground-truth-key",
                "gold",
                "--output-path",
                str(output_path),
                "--custom-metrics-path",
                str(plugin_dir / "custom_metrics.py"),
            ],
            cwd=str(tmp_path),
            env={"AWS_REGION": "eu-west-1"},
        )

        assert result.returncode == 0, result.stderr
        metrics = _full_output(result, output_path)["meta"]["metrics"]
        digests = [metrics[name]["prompt_sha256"] for name in names]
        assert digests[1] == digests[0]
        assert len(set(digests)) == 4

    def test_an_error_envelope_records_what_was_known_when_the_run_stopped(
        self, sample_data, tmp_path
    ):
        output_path = tmp_path / "result.json"

        result = _run_evaluation(
            sample_data, output_path, "Accuracy", pred_key="missing"
        )

        assert result.returncode == 1
        output = _full_output(result, output_path, "stderr")
        assert output["data"] is None
        meta = output["meta"]
        assert set(meta) == {"library_version", "started_at", "judge", "metrics"}
        assert meta["judge"] is None
        assert meta["metrics"] == {}

    @pytest.mark.parametrize(
        ("metadata_dir", "linked", "expected"),
        [
            pytest.param(
                "metric_library-9.9.9.dist-info", True, "9.9.9", id="wheel-copy-runs"
            ),
            pytest.param(
                "metric_library-9.9.9.dist-info",
                False,
                "unknown",
                id="another-copy-runs",
            ),
            # A build leaves an .egg-info, with no RECORD, in the source tree.
            pytest.param("metric_library.egg-info", True, "unknown", id="egg-info"),
        ],
    )
    def test_the_version_is_that_of_the_installed_copy_only_when_it_runs(
        self, sample_data, tmp_path, metadata_dir, linked, expected
    ):
        """A source checkout next to an older install does not report its version."""
        site = tmp_path / "site"
        metadata = site / metadata_dir
        metadata.mkdir(parents=True)
        version = "Metadata-Version: 2.1\nName: metric-library\nVersion: 9.9.9\n"
        if metadata_dir.endswith(".dist-info"):
            (metadata / "METADATA").write_text(version)
            (metadata / "RECORD").write_text("metric_library/__init__.py,,\n")
        else:
            (metadata / "PKG-INFO").write_text(version)
        if linked:
            # The installed copy is the package that runs.
            (site / "metric_library").symlink_to(REPO_ROOT / "metric_library")
        output_path = tmp_path / "result.json"

        result = _run_metrics(
            [
                "--execution-output-path",
                str(sample_data),
                "--metrics",
                "Accuracy",
                "--pred-key",
                "pred",
                "--ground-truth-key",
                "gold",
                "--output-path",
                str(output_path),
            ],
            extra_pythonpath=str(site),
            cwd=str(tmp_path),
        )

        assert result.returncode == 0, result.stderr
        assert _full_output(result, output_path)["meta"]["library_version"] == expected


class TestListMetrics:
    """``--list-metrics`` describes the registered metrics without a dataset."""

    def test_json_lists_every_registered_metric_from_class_attributes(self):
        # A bogus profile proves no metric is instantiated: the semantic match
        # metrics build a Bedrock client on construction and would fail.
        result = _run_metrics(
            ["--list-metrics"], env={"AWS_PROFILE": "no-such-profile"}
        )
        assert result.returncode == 0, result.stderr
        entries = json.loads(result.stdout)
        assert len(entries) >= 46
        assert [entry["name"] for entry in entries] == sorted(
            (entry["name"] for entry in entries), key=str.lower
        )
        by_name = {entry["name"]: entry for entry in entries}
        assert set(by_name["Accuracy"]) == {
            "name",
            "kind",
            "modalities",
            "reference",
            "prompt",
            "context",
            "default_aggregation_method",
            "description",
        }
        assert by_name["Accuracy"]["kind"] == "deterministic"
        assert by_name["Accuracy"]["modalities"] == ["text"]
        assert by_name["TextExactMatch"]["modalities"] == ["text"]
        assert by_name["MeanSquaredError"]["modalities"] == ["audio", "image", "video"]
        assert by_name["LLMCorrectness"]["kind"] == "llm"
        assert by_name["LLMCorrectness"]["modalities"] == ["text", "image"]
        assert by_name["LLMCorrectness"]["prompt"] == "required"
        assert by_name["Accuracy"]["default_aggregation_method"] == "numeric_mean"
        assert by_name["Accuracy"]["description"] == "Exact-match rate of label sets."

    def test_metrics_filters_and_orders_the_list(self):
        result = _run_metrics(
            ["--list-metrics", "--metrics", "MeanSquaredError, accuracy"]
        )
        assert result.returncode == 0, result.stderr
        assert [e["name"] for e in json.loads(result.stdout)] == [
            "MeanSquaredError",
            "Accuracy",
        ]

    def test_table_format_has_a_header_and_one_row_per_metric(self):
        result = _run_metrics(
            [
                "--list-metrics",
                "--format",
                "table",
                "--metrics",
                "Accuracy,LLMCorrectness",
            ]
        )
        assert result.returncode == 0, result.stderr
        lines = result.stdout.rstrip("\n").split("\n")
        assert len(lines) == 3
        assert lines[0].split() == [
            "Metric",
            "Kind",
            "Modalities",
            "Reference",
            "Prompt",
            "Context",
            "Default",
            "aggregator",
            "Description",
        ]
        assert lines[1].split()[:7] == [
            "Accuracy",
            "deterministic",
            "text",
            "required",
            "omit",
            "omit",
            "numeric_mean",
        ]
        assert lines[2].split()[:7] == [
            "LLMCorrectness",
            "llm",
            "text,image",
            "required",
            "required",
            "optional",
            "numeric_mean",
        ]

    def test_custom_metrics_are_listed(self, tmp_path):
        custom_dir = tmp_path / "custom"
        custom_dir.mkdir()
        (custom_dir / "custom_metrics.py").write_text(
            textwrap.dedent("""\
            from metric_library.core.base import Metric
            from metric_library.core.registry import MetricRegistry
            from metric_library.core.data import Modality

            @MetricRegistry.register
            class AlwaysOne(Metric):
                \"\"\"Scores every sample 1.0.\"\"\"

                supported_data_types = [Modality.TEXT, Modality.IMAGE]
                default_aggregation_method = "median"

                def compute(self, sample, client) -> float:
                    return 1.0

            @MetricRegistry.register
            class Undocumented(Metric):
                supported_data_types = [Modality.TEXT]

                def compute(self, sample, client) -> float:
                    return 1.0
        """)
        )
        result = _run_metrics(
            [
                "--list-metrics",
                "--metrics",
                "AlwaysOne,Undocumented",
                "--custom-metrics-path",
                str(custom_dir / "custom_metrics.py"),
            ],
            cwd=str(tmp_path),
        )
        assert result.returncode == 0, result.stderr
        # A metric without a docstring has no description; it is not described
        # by the docstring of Metric.
        assert json.loads(result.stdout) == [
            {
                "name": "AlwaysOne",
                "kind": "deterministic",
                "modalities": ["text", "image"],
                "reference": "required",
                "prompt": "omit",
                "context": "omit",
                "default_aggregation_method": "median",
                "description": "Scores every sample 1.0.",
            },
            {
                "name": "Undocumented",
                "kind": "deterministic",
                "modalities": ["text"],
                "reference": "required",
                "prompt": "omit",
                "context": "omit",
                "default_aggregation_method": "numeric_mean",
                "description": "",
            },
        ]

    @pytest.mark.parametrize(
        ("args", "message"),
        [
            pytest.param(
                ["--list-metrics", "--metrics", "Nope"],
                "Metric 'Nope' not found",
                id="unknown-metric",
            ),
            pytest.param(
                ["--format", "table"],
                "--format requires --list-metrics",
                id="format-without-list",
            ),
            pytest.param(
                ["--format=table"],
                "--format requires --list-metrics",
                id="format-equals-without-list",
            ),
            pytest.param(
                ["--metrics", "Accuracy"],
                "the following arguments are required: --execution-output-path, "
                "--pred-key, --output-path",
                id="data-arguments-still-required",
            ),
            pytest.param(
                [
                    "--execution-output-path",
                    "data.json",
                    "--metrics",
                    "Accuracy",
                    "--output-path",
                    "out.json",
                ],
                "the following arguments are required: --pred-key",
                id="pred-key-required",
            ),
        ],
    )
    def test_argument_errors_use_argparse_conventions(self, args, message):
        result = _run_metrics(args)
        assert result.returncode == 2
        assert "usage:" in result.stderr
        assert message in result.stderr
        assert result.stdout == ""


class TestCustomMetricsNoOp:
    """Verify the no-op custom_metrics.py at the repo root."""

    def test_file_exists(self):
        assert (REPO_ROOT / "custom_metrics.py").exists()

    def test_is_valid_python(self):
        source = (REPO_ROOT / "custom_metrics.py").read_text()
        try:
            ast.parse(source)
        except SyntaxError as e:
            pytest.fail(f"custom_metrics.py has syntax error: {e}")

    def test_is_noop(self):
        """The stub should contain only a module docstring."""
        source = (REPO_ROOT / "custom_metrics.py").read_text()
        tree = ast.parse(source)
        for node in ast.iter_child_nodes(tree):
            assert isinstance(node, ast.Expr), (
                f"Stub should only contain a docstring, found {type(node).__name__}"
            )

    def test_loads_with_the_path_option(self, sample_data, tmp_path):
        output_path = tmp_path / "result.json"
        result = _run_metrics(
            [
                "--execution-output-path",
                str(sample_data),
                "--metrics",
                "Accuracy",
                "--pred-key",
                "pred",
                "--ground-truth-key",
                "gold",
                "--custom-metrics-path",
                str(REPO_ROOT / "custom_metrics.py"),
                "--output-path",
                str(output_path),
            ],
            cwd=str(tmp_path),
        )
        assert result.returncode == 0, result.stderr


class TestArrayFieldsAreMultiItemRoles:
    """A JSON array in a record field carries several items in one role."""

    def test_label_set_metrics_read_an_array_as_one_set(self, tmp_path):
        records_path = tmp_path / "records.json"
        records_path.write_text(
            json.dumps(
                [
                    {"pred": ["a", "b"], "gold": ["b", "a"]},
                    {"pred": "c", "gold": "c"},
                ]
            )
        )
        output_path = tmp_path / "result.json"

        result = _run_evaluation(records_path, output_path, "Accuracy")

        assert result.returncode == 0, result.stderr
        output = _full_output(result, output_path)
        assert output["data"]["results"]["Accuracy"] == 1.0
        entries = output["data"]["samples"]["Accuracy"]
        assert [entry["score"] for entry in entries] == [1.0, 1.0]
        assert all(set(entry) == SAMPLE_ENTRY_KEYS for entry in entries)

    def test_aligned_error_metrics_pair_the_images_of_an_array(self, tmp_path):
        """Image paths in an array pair by position; each pair's error is reported.

        Solid-color images differ by a constant, so each pair's MAE is that
        constant: 10 and 20 for the two pairs of the first record, 5 for the one
        pair of the second.
        """
        # A subprocess keeps PIL and the image data type out of this process.
        if importlib.util.find_spec("PIL") is None:
            pytest.skip("Pillow is not installed")
        for name, level in (("p10", 10), ("p0", 0), ("p20", 20), ("p5", 5)):
            (tmp_path / f"{name}.png").write_bytes(_png_bytes(rgb=(level,) * 3))
        records_path = tmp_path / "records.json"
        records_path.write_text(
            json.dumps(
                [
                    {"pred": ["p10.png", "p0.png"], "gold": ["p0.png", "p20.png"]},
                    {"pred": "p5.png", "gold": "p0.png"},
                ]
            )
        )
        output_path = tmp_path / "result.json"

        result = _run_evaluation(records_path, output_path, "MeanAbsoluteError")

        assert result.returncode == 0, result.stderr
        output = _full_output(result, output_path)
        assert output["data"]["results"]["MeanAbsoluteError"] == 10.0
        entries = output["data"]["samples"]["MeanAbsoluteError"]
        assert [entry["score"] for entry in entries] == [15.0, 5.0]
        assert [entry["item_scores"] for entry in entries] == [[10.0, 20.0], None]
        assert all(set(entry) == SAMPLE_ENTRY_KEYS for entry in entries)

    def test_unequal_aligned_counts_fail_the_metric_before_scoring(self, tmp_path):
        # A subprocess keeps PIL and the image data type out of this process.
        if importlib.util.find_spec("PIL") is None:
            pytest.skip("Pillow is not installed")
        for name, level in (("p0", 0), ("p10", 10), ("p20", 20)):
            (tmp_path / f"{name}.png").write_bytes(_png_bytes(rgb=(level,) * 3))
        records_path = tmp_path / "records.json"
        records_path.write_text(
            json.dumps(
                [
                    {
                        "pred": ["p0.png", "p10.png", "p20.png"],
                        "gold": ["p0.png", "p10.png"],
                    }
                ]
            )
        )
        output_path = tmp_path / "result.json"

        result = _run_evaluation(records_path, output_path, "MeanAbsoluteError")

        assert result.returncode == 1, result.stderr
        output = json.loads(result.stdout)
        assert output["status"] == "failed"
        assert output["data"]["results"]["MeanAbsoluteError"] == {
            "error": (
                "prediction item count (3) must match reference item count (2) "
                "for sample 0 of MeanAbsoluteError"
            )
        }

    def test_an_image_path_inside_an_array_loads_as_an_image_item(self, tmp_path):
        # A subprocess keeps PIL and the image data type out of this process.
        if importlib.util.find_spec("PIL") is None:
            pytest.skip("Pillow is not installed")
        custom_dir = tmp_path / "custom"
        custom_dir.mkdir()
        (custom_dir / "custom_metrics.py").write_text(
            textwrap.dedent(
                """\
                from metric_library.core.aggregation import SampleOutput
                from metric_library.core.base import Metric
                from metric_library.core.data import Modality
                from metric_library.core.multiplicity import Multiplicity
                from metric_library.core.registry import MetricRegistry

                @MetricRegistry.register
                class PartModalities(Metric):
                    prediction_multiplicity = Multiplicity.COMPOSITE

                    def compute(self, sample, client):
                        # One item score per part: 1.0 for an image, 0.0 for text.
                        return SampleOutput(
                            score=float(len(sample.predictions)),
                            details={
                                "item_scores": [
                                    1.0 if item.modality is Modality.IMAGE else 0.0
                                    for item in sample.predictions
                                ]
                            },
                        )
                """
            )
        )
        image_path = tmp_path / "shot.png"
        image_path.write_bytes(_png_bytes())
        records_path = tmp_path / "records.json"
        records_path.write_text(
            json.dumps([{"pred": ["A red pixel.", str(image_path)], "gold": "x"}])
        )
        output_path = tmp_path / "result.json"

        result = _run_metrics(
            [
                "--execution-output-path",
                str(records_path),
                "--metrics",
                "PartModalities",
                "--pred-key",
                "pred",
                "--ground-truth-key",
                "gold",
                "--output-path",
                str(output_path),
                "--custom-metrics-path",
                str(custom_dir / "custom_metrics.py"),
            ],
            cwd=str(tmp_path),
        )

        assert result.returncode == 0, result.stderr
        output = _full_output(result, output_path)
        assert output["data"]["results"]["PartModalities"] == 2.0
        assert output["data"]["samples"]["PartModalities"][0]["item_scores"] == [
            0.0,
            1.0,
        ]

    def test_an_array_sample_id_is_rejected(self, tmp_path):
        records_path = tmp_path / "records.json"
        records_path.write_text(json.dumps([{"pred": "a", "gold": "a", "id": ["x"]}]))
        output_path = tmp_path / "result.json"

        result = _run_evaluation(
            records_path, output_path, "Accuracy", "--sample-id-key", "id"
        )

        assert result.returncode == 1
        output = json.loads(result.stderr)
        assert output["errors"] == [
            "sample id in record 0 under key 'id' must be a scalar, not an array"
        ]


class TestMultimodalExamples:
    """The multimodal example datasets load their items from the record folder.

    A plugin metric stands in for the judge and reports how many items of each
    modality every role carries, so the run needs neither Bedrock nor a fixed
    working directory. The judge runs documented in tests/examples/README.md
    are not reproduced here.
    """

    PLUGIN = """\
        from metric_library.core.base import Metric
        from metric_library.core.data import Modality
        from metric_library.core.input_policy import InputPolicy
        from metric_library.core.multiplicity import Multiplicity
        from metric_library.core.registry import MetricRegistry

        @MetricRegistry.register
        class RoleShapes(Metric):
            prompt_policy = InputPolicy.REQUIRED
            prediction_multiplicity = Multiplicity.COMPOSITE
            reference_multiplicity = Multiplicity.COMPOSITE

            def compute(self, sample, client):
                # One digit per role: how many items, how many of them images.
                code = ""
                for items in (sample.prompts, sample.predictions, sample.references):
                    images = sum(item.modality is Modality.IMAGE for item in items)
                    code += f"{len(items)}{images}"
                return float(code)
        """

    @pytest.mark.parametrize(
        ("dataset", "keys", "expected_code"),
        [
            pytest.param(
                "llm_correctness_vqa.json",
                ("answer", "reference_answer", "question"),
                211010.0,
                id="vqa: text+image question, text answer, text reference",
            ),
            pytest.param(
                "llm_correctness_product_cutout.json",
                ("output", "reference_output", "request"),
                212121.0,
                id="cutout: text+image in every role",
            ),
        ],
    )
    def test_every_record_loads_its_text_and_image_items(
        self, tmp_path, dataset, keys, expected_code
    ):
        # A subprocess keeps PIL and the image data type out of this process.
        if importlib.util.find_spec("PIL") is None:
            pytest.skip("Pillow is not installed")
        custom_dir = tmp_path / "custom"
        custom_dir.mkdir()
        (custom_dir / "custom_metrics.py").write_text(textwrap.dedent(self.PLUGIN))
        pred_key, ground_truth_key, prompt_key = keys
        output_path = tmp_path / "result.json"

        result = _run_metrics(
            [
                "--execution-output-path",
                str(REPO_ROOT / "tests" / "examples" / dataset),
                "--metrics",
                "RoleShapes",
                "--pred-key",
                pred_key,
                "--ground-truth-key",
                ground_truth_key,
                "--prompt-key",
                prompt_key,
                "--sample-id-key",
                "id",
                "--output-path",
                str(output_path),
                "--custom-metrics-path",
                str(custom_dir / "custom_metrics.py"),
            ],
            cwd=str(tmp_path),
        )

        assert result.returncode == 0, result.stderr
        output = _full_output(result, output_path)
        entries = output["data"]["samples"]["RoleShapes"]
        assert len(entries) == 10
        assert {entry["score"] for entry in entries} == {expected_code}
        assert [entry["status"] for entry in entries] == ["completed"] * 10


class TestRagExamples:
    """The RAG example datasets give every sample its retrieved passages as context.

    A plugin metric stands in for the judge and reports how many context items
    each sample carries and how many of them are images, so the run needs no
    Bedrock. The judge runs documented in tests/examples/README.md are not
    reproduced here.
    """

    PLUGIN = """\
        from metric_library.core.base import Metric
        from metric_library.core.data import Modality
        from metric_library.core.input_policy import InputPolicy
        from metric_library.core.registry import MetricRegistry

        @MetricRegistry.register
        class ContextShapes(Metric):
            reference_policy = InputPolicy.OMIT
            prompt_policy = InputPolicy.REQUIRED
            context_policy = InputPolicy.REQUIRED

            def compute(self, sample, client):
                # Two digits: how many context items, how many of them images.
                images = sum(item.modality is Modality.IMAGE for item in sample.contexts)
                return float(f"{len(sample.contexts)}{images}")
        """

    def test_every_passage_is_one_context_item(self, tmp_path):
        # A subprocess keeps PIL and the image data type out of this process.
        if importlib.util.find_spec("PIL") is None:
            pytest.skip("Pillow is not installed")
        custom_dir = tmp_path / "custom"
        custom_dir.mkdir()
        (custom_dir / "custom_metrics.py").write_text(textwrap.dedent(self.PLUGIN))
        output_path = tmp_path / "result.json"
        result = _run_metrics(
            [
                "--execution-output-path",
                str(REPO_ROOT / "tests" / "examples" / "rag_support_kb.json"),
                "--metrics",
                "ContextShapes",
                "--pred-key",
                "answer",
                "--prompt-key",
                "question",
                "--context",
                "passages,instructions",
                "--sample-id-key",
                "id",
                "--output-path",
                str(output_path),
                "--custom-metrics-path",
                str(custom_dir / "custom_metrics.py"),
            ],
            cwd=str(tmp_path),
        )
        assert result.returncode == 0, result.stderr
        output = _full_output(result, output_path)
        entries = output["data"]["samples"]["ContextShapes"]
        # Two or three passages plus the instructions; records 1, 5 and 7
        # retrieve a scanned page (PNG, TIFF, PNG) in place of a text passage.
        assert [entry["score"] for entry in entries] == [
            31.0,
            30.0,
            30.0,
            40.0,
            31.0,
            40.0,
            31.0,
            40.0,
        ]
        assert all(entry["status"] == "completed" for entry in entries)


def _png_bytes(width=2, height=2, rgb=(255, 0, 0)):
    """Encode a solid-color 8-bit RGB PNG without any optional media dependency."""

    def chunk(tag, payload):
        body = tag + payload
        return (
            struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))
        )

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    scanlines = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(scanlines))
        + chunk(b"IEND", b"")
    )
