#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Metric evaluation runner.

Loads prediction data, computes selected metrics (built-in and custom),
and writes results as a JSON envelope.

Usage:
    python -m metric_library.run_metrics \
        --execution-output-path data.json \
        --metrics "Accuracy,TextExactMatch" \
        --aggregation-methods ",prf" \
        --matcher ",labels" \
        --pred-key "prediction" \
        --ground-truth-key "answer" \
        --output-path result.json
"""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import stat
import sys
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from metric_library.core.aggregation import (
    EvaluationStatus,
    SampleStatus,
)
from metric_library.core.bedrock_client import BedrockClient
from metric_library.core.data_handler import DataHandler
from metric_library.core.evaluation import (
    EvaluationConfiguration,
    MetricEvaluationResult,
    _safe_error,
)
from metric_library.core.llm_metric import LLMBatchMetric, LLMMetric
from metric_library.core.matching import LabelSet, ListItems, WholeValue
from metric_library.core.registry import (
    MetricError,
    MetricRegistry,
    normalize_metric_name,
)

# Aggregators selectable through --aggregation-methods. weighted_mean takes
# its per-record weights from --weight-key.
CLI_AGGREGATION_METHODS = (
    "numeric_mean",
    "weighted_mean",
    "median",
    "quantile",
    "prf",
)
# Matchers selectable through --matcher. labels and list require prf.
CLI_MATCHERS = ("whole", "labels", "list")


def _load_metrics(custom_metrics_path: str | None) -> None:
    """Register the custom metrics, then the built-in metrics.

    ``custom_metrics_path`` names the plugin file, loaded as the module
    ``custom_metrics``. Without it, no custom metrics load: a
    ``custom_metrics`` module on ``sys.path``, which ``python -m`` starts with
    the working directory, is not imported. The built-in metrics register
    after the plugin, so a plugin metric named like a built-in one is replaced,
    with a warning, unless the plugin imports the built-in metrics itself. A
    plugin that fails to load raises ``RuntimeError`` naming its file.
    """
    if custom_metrics_path is not None:
        origin = repr(custom_metrics_path)
        try:
            spec = importlib.util.spec_from_file_location(
                "custom_metrics", custom_metrics_path
            )
            if spec is None or spec.loader is None:
                raise ImportError("not a Python module file")
            module = importlib.util.module_from_spec(spec)
            # The plugin can import itself by name while it runs.
            sys.modules["custom_metrics"] = module
            spec.loader.exec_module(module)
        except Exception as error:
            sys.modules.pop("custom_metrics", None)
            raise RuntimeError(
                f"cannot load the custom metrics module {origin}: {_safe_error(error)}"
            ) from error
    import metric_library.metrics  # noqa: F401 - registers the built-in metrics


def _load_data(path, extract_key=None, data_root=None):
    """Load a path, extracting one value per JSON record when a key is set.

    With ``data_root``, every file that the data names must lie inside it.
    """
    if extract_key is not None:
        return DataHandler.parse_json_file(path, extract_key, data_root)
    return DataHandler.create_data(path, data_root=data_root)


def _load_prompt_data(path, prompt_key=None, data_root=None):
    """Load keyed JSON prompt records or a strict keyless prompt manifest."""
    if prompt_key in ("None", "null"):
        prompt_key = None
    if prompt_key is not None:
        return _load_data(path, prompt_key, data_root)

    manifest_path = Path(path)
    try:
        with manifest_path.open("r", encoding="utf-8") as prompt_file:
            manifest = json.load(prompt_file)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return _load_data(path, data_root=data_root)

    manifest_error = (
        "A JSON --prompt-path requires --prompt-key unless it is a non-empty "
        "array of existing prompt file paths."
    )
    if not isinstance(manifest, list) or not manifest:
        raise ValueError(manifest_error)

    prompt_data = []
    manifest_folder = manifest_path.resolve().parent
    for entry_index, entry_path in enumerate(manifest):
        if not isinstance(entry_path, str):
            raise ValueError(manifest_error)

        resolved_path = Path(entry_path)
        if not resolved_path.is_absolute():
            resolved_path = manifest_path.parent / resolved_path
        resolved_path = DataHandler._inside_root(
            resolved_path, entry_path, entry_index, data_root, manifest_folder
        )
        if not resolved_path.is_file():
            raise ValueError(
                "Prompt manifest entry "
                f"{entry_index} must be an existing regular file: {entry_path!r}"
            )
        prompt_data.append(DataHandler.assign_element(str(resolved_path)))

    return prompt_data


def _record_ids(
    execution_output_path: str,
    sample_id_key: str | None,
    prediction_count: int,
    data_root: Path | None = None,
) -> list[str]:
    """Return one record identifier per prediction, in input order.

    Without a key, records are numbered from 1. With a key, the identifiers are
    the ``str()`` of each record's value under that key in the execution
    output, checked for count, blanks and duplicates before any metric runs.
    """
    if sample_id_key is None:
        return [str(i + 1) for i in range(prediction_count)]

    values = DataHandler.parse_json_file(
        execution_output_path, sample_id_key, data_root
    )
    for i, item in enumerate(values):
        if isinstance(item, list):
            raise ValueError(
                f"sample id in record {i} under key {sample_id_key!r} must be a "
                "scalar, not an array"
            )
    ids = [str(item.get_raw_data()) for item in values]
    if len(ids) != prediction_count:
        raise ValueError(
            f"sample id count ({len(ids)}) must match prediction count "
            f"({prediction_count})"
        )
    for i, value in enumerate(ids):
        if not value.strip():
            raise ValueError(
                f"sample id in record {i} under key {sample_id_key!r} must be non-empty"
            )
    if len(set(ids)) != len(ids):
        raise ValueError(f"sample ids under key {sample_id_key!r} must be unique")
    return ids


def _weights(
    execution_output_path: str,
    weight_key: str,
    prediction_count: int,
    data_root: Path | None = None,
) -> tuple[float, ...]:
    """Return one ``weighted_mean`` weight per prediction, in input order.

    The weights are each record's value under ``weight_key`` in the execution
    output: finite positive JSON numbers, one per prediction, checked before
    any metric runs.
    """
    values = DataHandler.parse_json_file(execution_output_path, weight_key, data_root)
    if len(values) != prediction_count:
        raise ValueError(
            f"weight count ({len(values)}) must match prediction count "
            f"({prediction_count})"
        )
    weights = []
    for i, item in enumerate(values):
        if isinstance(item, list):
            raise ValueError(
                f"weight in record {i} under key {weight_key!r} must be a finite "
                "positive number, not an array"
            )
        value = item.get_raw_data()
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(
                f"weight in record {i} under key {weight_key!r} must be a finite "
                f"positive number, got {value!r}"
            )
        weights.append(float(value))
    return tuple(weights)


def _aligned_entries(
    value: str | None,
    metrics: list[str],
    noun: str,
    choices: tuple[str, ...],
) -> list[str | None]:
    """Return one entry per selected metric, ``None`` for the default.

    ``value`` is a comma-separated argument aligned with ``--metrics``, such as
    ``--aggregation-methods``: omitted or blank keeps every metric on its
    default; one entry applies to every metric; otherwise entry ``i`` applies
    to metric ``i`` and an empty entry keeps that default.
    """
    if value is None or not value.strip():
        return [None] * len(metrics)

    entries = [entry.strip() for entry in value.split(",")]
    if len(entries) == 1:
        entries = entries * len(metrics)
    elif len(entries) != len(metrics):
        raise ValueError(
            f"{noun} count ({len(entries)}) must match metric count ({len(metrics)})"
        )

    selected: list[str | None] = []
    for entry in entries:
        if not entry:
            selected.append(None)
        elif entry in choices:
            selected.append(entry)
        else:
            raise ValueError(
                f"unknown {noun} {entry!r}; choose from " + ", ".join(choices)
            )
    return selected


def _thresholds(value: str) -> tuple[float, ...]:
    """Parse the comma-separated ``--thresholds`` argument."""
    try:
        return tuple(float(entry) for entry in value.split(","))
    except ValueError:
        raise ValueError(
            f"--thresholds must be comma-separated numbers, got {value!r}"
        ) from None


def _validate_auxiliary_cardinality(predictions, prompts, context):
    """Reject supplied prompt/context lists that do not align with predictions."""
    prediction_count = len(predictions)
    if prompts and len(prompts) != prediction_count:
        raise ValueError(
            f"prompt count ({len(prompts)}) must match prediction count "
            f"({prediction_count})"
        )
    for source_index, context_source in enumerate(context):
        if len(context_source) != prediction_count:
            raise ValueError(
                f"context source {source_index} count ({len(context_source)}) must "
                f"match prediction count ({prediction_count})"
            )


def _envelope_value(
    result: MetricEvaluationResult | MetricError,
    record_ids: list[str],
) -> float | dict | MetricError:
    """Map one engine result to the envelope value.

    A completed result maps to its aggregate ``values`` object when the
    aggregator provides one (the ``prf`` aggregator), else to its finite
    float score. A failed sample is reported as a MetricError under the record
    identifier at its ordinal.
    """
    if isinstance(result, MetricError):
        return result
    if result.status is EvaluationStatus.FAILED:
        failed_sample = next(
            sample for sample in result.samples if sample.status is SampleStatus.FAILED
        )
        return MetricError(
            f"sample {record_ids[failed_sample.ordinal]}: {failed_sample.error}"
        )
    if result.aggregate.values is not None:
        return result.aggregate.values
    return result.score


def _sample_entries(
    result: MetricEvaluationResult,
    record_ids: list[str],
    omit_reasoning: bool = False,
) -> list[dict]:
    """Map the per-sample results of one metric to envelope entries, in input order.

    Each entry carries the record identifier at the sample's ordinal as
    ``sample_id``, the sample's ``matches``: for each class, the candidate
    pair scores and the item counts (``None`` with the whole matcher), and
    the sample's details: the ``item_scores`` list and the judge's
    ``reasoning`` (each ``None`` when the metric provides no such detail, and
    ``reasoning`` also with ``omit_reasoning``).
    """
    entries = []
    for sample in result.samples:
        details = sample.details or {}
        matches = None
        if sample.matches is not None:
            matches = {
                item_class: asdict(record)
                for item_class, record in sample.matches.items()
            }
        entry = {
            "sample_id": record_ids[sample.ordinal],
            "status": sample.status.value,
            "score": sample.score,
            "error": sample.error,
            "matches": matches,
            "item_scores": details.get("item_scores"),
            "reasoning": None if omit_reasoning else details.get("reasoning"),
        }
        entries.append(entry)
    return entries


def _library_version() -> str:
    """The version of the installed ``metric-library`` when it is the running code.

    ``"unknown"`` when no distribution is installed, or when the installed one
    is another copy than the code that runs, such as a source checkout on
    ``PYTHONPATH`` next to an older install, whose version would mislead. Only
    an installed wheel counts, the metadata with a ``RECORD``: the
    ``.egg-info`` that a build leaves in a source tree keeps the version of
    that build.
    """
    try:
        distribution = importlib.metadata.distribution("metric-library")
        installed = Path(distribution.locate_file("metric_library/__init__.py"))
        running = Path(__file__).resolve().with_name("__init__.py")
        if (
            distribution.read_text("RECORD") is not None
            and installed.resolve() == running
        ):
            return distribution.version
    except (importlib.metadata.PackageNotFoundError, OSError):
        pass
    return "unknown"


def _prompt_sha256(name: str) -> str | None:
    """The SHA-256 of the fixed text of an LLM metric's judge requests.

    The digest is of ``json.dumps(sections, sort_keys=True)``. The sections
    are the judge role, the rubric, the response schema, the
    ``modality_guidance`` entries, the sentences that introduce a candidate
    or a ground truth in several parts and, for an ``LLMBatchMetric``, the
    instructions and the response schema of a matrix call, so the digest
    changes when any of them changes. ``None`` for any other metric, which is
    not instantiated: a metric's constructor can build an AWS client.
    """
    metric_class = MetricRegistry.metrics.get(normalize_metric_name(name))
    if metric_class is None or not issubclass(metric_class, LLMMetric):
        return None
    try:
        metric = metric_class()
        sections = {
            "role": metric.role(),
            "rubric": metric.rubric(),
            "response_schema": metric.response_schema(),
            "modality_guidance": {
                str(modality): guidance
                for modality, guidance in metric.modality_guidance.items()
            },
            # Two items stand for any number; the words follow the multiplicity.
            "multi_part_intros": [
                metric._multiplicity_intro(tag, 2, multiplicity)
                for tag, multiplicity in (
                    ("candidate_response", metric.prediction_multiplicity),
                    ("ground_truth_response", metric.reference_multiplicity),
                )
            ],
        }
        if isinstance(metric, LLMBatchMetric):
            sections["matrix"] = {
                "instructions": metric._pair_instructions(2, 2),
                "response_schema": metric._matrix_response_schema(2, 2),
            }
        payload = json.dumps(sections, sort_keys=True)
    except Exception:
        return None
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _metric_meta(
    name: str,
    method: str | None,
    matcher: str,
    args: argparse.Namespace,
    thresholds: tuple[float, ...] | None,
) -> dict:
    """The configuration of one metric in ``meta.metrics`` of the envelope.

    ``aggregation_method`` is the metric's default when its entry is empty
    (``None`` for an unknown metric), followed by the options of that
    aggregator and of the matcher, and ``prompt_sha256`` for an LLM metric.
    """
    if method is None:
        try:
            method = MetricRegistry.list_metrics([name])[0][
                "default_aggregation_method"
            ]
        except KeyError:
            method = None
    entry = {"aggregation_method": method, "matcher": matcher}
    if method == "quantile":
        entry["quantile"] = args.quantile
    elif method == "weighted_mean":
        entry["weight_key"] = args.weight_key
    elif method == "prf":
        entry["thresholds"] = list(thresholds) if thresholds is not None else None
    if matcher == "list":
        entry["class_key"] = args.class_key
        entry["value_key"] = args.value_key
    digest = _prompt_sha256(name)
    if digest is not None:
        entry["prompt_sha256"] = digest
    return entry


def _stdout_envelope(output: dict, output_path: str | None, mode: str) -> dict:
    """Return the envelope the runner prints for ``--stdout`` ``mode``.

    ``full`` is the file envelope. ``summary`` drops ``data.samples``, which
    grows with the record count, and ``meta``, and adds ``output_path`` so the
    reader knows where the full envelope is; ``None`` when the file could not
    be written.
    """
    if mode == "full":
        return output
    data = output["data"]
    if data is not None:
        data = {key: value for key, value in data.items() if key != "samples"}
    summary = {key: value for key, value in output.items() if key != "meta"}
    return {**summary, "data": data, "output_path": output_path}


def _in_device_folder(path) -> bool:
    """Whether ``path`` is an entry of /dev or /dev/fd (/proc/<pid>/fd on Linux).

    Only the system makes entries there, such as /dev/null, /dev/stdout and the
    /dev/fd/N of a process substitution, whatever each one points to.
    """
    folder = os.path.realpath(os.path.dirname(os.path.abspath(path)))
    return folder in ("/dev", "/dev/fd") or folder.startswith("/proc/")


def _is_stream(status: os.stat_result) -> bool:
    """Whether ``status`` is a character device or a named pipe the user owns."""
    if stat.S_ISCHR(status.st_mode):
        return True
    owner = getattr(os, "getuid", None)
    return stat.S_ISFIFO(status.st_mode) and (owner is None or status.st_uid == owner())


def _write_output(path, output):
    """Persist a strict-JSON runner envelope, creating its parent when needed.

    The envelope goes to a new file with mode 0600, so only its owner can read
    it, in the folder of ``path``, and that file then replaces ``path``: an
    existing file gets mode 0600 too, even a read-only one, a symbolic link at
    ``path`` is replaced, not followed, and a failed write leaves ``path`` as
    it was. An entry of /dev or /dev/fd (``_in_device_folder``) is written in
    place, and so is a stream (``_is_stream``) that is not a symbolic link,
    opened without following one and checked again once open.
    """
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    if _in_device_folder(path):
        with open(path, "w", encoding="utf-8") as output_file:
            json.dump(output, output_file, indent=2, allow_nan=False)
        return
    try:
        at_path = os.lstat(path)
    except OSError:
        at_path = None
    if at_path is not None and _is_stream(at_path):
        # O_BINARY (Windows only) leaves the newline translation to open().
        flags = os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
        descriptor = os.open(path, flags)
        if _is_stream(os.fstat(descriptor)):
            with open(descriptor, "w", encoding="utf-8") as output_file:
                json.dump(output, output_file, indent=2, allow_nan=False)
            return
        # The path changed between the two checks: replace it as a file.
        os.close(descriptor)
    # mkstemp opens in binary mode on Windows, so open() translates newlines.
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=output_dir or "."
    )
    try:
        with open(descriptor, "w", encoding="utf-8") as output_file:
            json.dump(output, output_file, indent=2, allow_nan=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


_LIST_COLUMNS = (
    ("name", "Metric"),
    ("kind", "Kind"),
    ("modalities", "Modalities"),
    ("reference", "Reference"),
    ("prompt", "Prompt"),
    ("context", "Context"),
    ("default_aggregation_method", "Default aggregator"),
    ("description", "Description"),
)


def _format_metric_list(entries: list[dict], fmt: str) -> str:
    """Render ``MetricRegistry.list_metrics`` entries as JSON or a fixed-width table."""
    if fmt == "json":
        return json.dumps(entries, indent=2)
    rows = [
        [
            ",".join(entry[key]) if key == "modalities" else str(entry[key])
            for key, _ in _LIST_COLUMNS
        ]
        for entry in entries
    ]
    header = [label for _, label in _LIST_COLUMNS]
    widths = [
        max(len(header[i]), *(len(row[i]) for row in rows)) if rows else len(header[i])
        for i in range(len(header) - 1)
    ]
    lines = [header] + rows
    return "\n".join(
        "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(line[:-1]))
        + "  "
        + line[-1]
        for line in lines
    )


def main():
    """Main entry point for metric evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate metrics on prediction data", allow_abbrev=False
    )
    parser.add_argument(
        "--list-metrics",
        action="store_true",
        help=(
            "Print one entry per registered metric (built-in and custom): "
            "name, kind, supported modalities, input policies, default aggregator "
            "and description. --metrics filters the list. The data arguments are "
            "not needed."
        ),
    )
    parser.add_argument(
        "--format",
        choices=("json", "table"),
        help="Output format of --list-metrics (default: json).",
    )
    parser.add_argument(
        "--execution-output-path",
        help=(
            "Path to the execution output prediction file of the evaluated system, "
            "which is input of metric evaluation (required unless --list-metrics)"
        ),
    )
    parser.add_argument(
        "--ground-truth-path",
        help=(
            "Separate ground truth file: a JSON array of records for "
            "--ground-truth-key, or without the key one reference file or a "
            "JSON array of file paths that resolve from its folder"
        ),
    )
    parser.add_argument(
        "--output-path",
        help=(
            "Path to write the JSON result or error envelope of evaluation output "
            "(required unless --list-metrics)"
        ),
    )
    parser.add_argument(
        "--stdout",
        choices=("summary", "full"),
        default="summary",
        help=(
            "What the runner prints: summary (default) is the envelope without "
            "data.samples plus output_path, the file that holds the full envelope; "
            "full prints the same envelope as the file."
        ),
    )
    parser.add_argument(
        "--omit-reasoning",
        action="store_true",
        help=(
            "Write null instead of the judge's reasoning in every data.samples "
            "entry, for tools and agents that must not read the judge's text."
        ),
    )
    parser.add_argument(
        "--pred-key",
        help=(
            "Key to extract predictions from the execution output "
            "(required unless --list-metrics)"
        ),
    )
    parser.add_argument(
        "--ground-truth-key",
        help="Key to extract ground truth from the execution output",
    )
    parser.add_argument(
        "--sample-id-key",
        help=(
            "Key holding the record identifier in the execution output. "
            "Omitted: records are numbered from 1."
        ),
    )
    parser.add_argument(
        "--metrics",
        help="Comma-separated list of metrics to compute (required unless --list-metrics)",
    )
    parser.add_argument(
        "--context",
        default="",
        help="Comma-separated list of context fields from the execution output",
    )
    parser.add_argument(
        "--prompt-key",
        help=(
            "Field to extract from --prompt-path when supplied, otherwise from "
            "the execution output. Required for JSON prompt records."
        ),
    )
    parser.add_argument(
        "--prompt-path",
        help=(
            "Separate prompt source. Without --prompt-key, use one non-JSON "
            "prompt file or a JSON array of existing prompt file paths resolved "
            "relative to its manifest."
        ),
    )
    parser.add_argument(
        "--data-root",
        help=(
            "Folder that every file named inside the data files must lie in, "
            "symbolic links resolved: media paths in record values, prompt "
            "manifest entries and the entries of a keyless --ground-truth-path "
            "array. The path options themselves are not checked. Omitted: no "
            "folder confines them."
        ),
    )
    parser.add_argument(
        "--custom-metrics-path",
        help=(
            "Python file of custom metrics to load as the module custom_metrics, "
            "the only way the runner loads custom metrics. Its code runs with "
            "the runner's privileges. Omitted: only the built-in metrics are "
            "registered."
        ),
    )
    parser.add_argument(
        "--aggregation-methods",
        help=(
            "Comma-separated aggregators aligned with --metrics: numeric_mean, "
            "weighted_mean, median, quantile, prf, "
            "or empty for that metric's default. One entry applies to every metric."
        ),
    )
    parser.add_argument(
        "--weight-key",
        help=(
            "Key of the per-record weight in the execution output for every "
            "weighted_mean entry of --aggregation-methods: a finite positive "
            "number in each record. Required with a weighted_mean entry, rejected "
            "without one."
        ),
    )
    parser.add_argument(
        "--quantile",
        type=float,
        help="Quantile in [0, 1] for every quantile entry of --aggregation-methods.",
    )
    parser.add_argument(
        "--matcher",
        help=(
            "Comma-separated matchers aligned with --metrics: whole (the full "
            "value is one item; the default), labels (one item for each label) "
            "or list (one item for each element of a list), or empty for whole. "
            "One entry applies to every metric. labels and list require the prf "
            "aggregator, and prf requires labels or list."
        ),
    )
    parser.add_argument(
        "--class-key",
        help=(
            "Class field of each list element for every list entry of --matcher. "
            "Only items of the same class can pair."
        ),
    )
    parser.add_argument(
        "--value-key",
        help=(
            "Field of each list element that the metric compares, for every list "
            "entry of --matcher. Omitted: the whole element."
        ),
    )
    parser.add_argument(
        "--thresholds",
        help=(
            "Comma-separated pair-score thresholds in (0, 1] for every prf entry "
            "of --aggregation-methods: a pair with a score of a threshold or more "
            "is a true positive. The first threshold gives the score. Omitted: 0.5."
        ),
    )
    parser.add_argument(
        "--model-id",
        help=(
            "Bedrock model ID or inference profile of the LLM judges, for example "
            "global.anthropic.claude-sonnet-4-6. It must support forced tool use "
            "(Anthropic Claude 3 and later, Amazon Nova). A geographic profile "
            "(us., eu., au., jp.) must match the region. "
            "Omitted: BEDROCK_MODEL_ID, then global.anthropic.claude-sonnet-4-6. The "
            "semantic match metrics keep their embedding model."
        ),
    )
    parser.add_argument(
        "--aws-profile",
        help=(
            "AWS profile for every Bedrock call of the run: the LLM judges and the "
            "semantic match metrics. It wins over AWS_PROFILE and over credentials "
            "in the environment. Omitted: the standard boto3 credential chain."
        ),
    )
    args = parser.parse_args()

    if args.list_metrics:
        _load_metrics(args.custom_metrics_path)
        names = None
        if args.metrics is not None:
            names = [name.strip() for name in args.metrics.split(",")]
        try:
            entries = MetricRegistry.list_metrics(names)
        except KeyError as error:
            parser.error(error.args[0])
        print(_format_metric_list(entries, args.format or "json"))
        return
    if args.format is not None:
        parser.error("--format requires --list-metrics")
    missing = [
        option
        for option, value in (
            ("--execution-output-path", args.execution_output_path),
            ("--pred-key", args.pred_key),
            ("--metrics", args.metrics),
            ("--output-path", args.output_path),
        )
        if value is None
    ]
    if missing:
        parser.error(f"the following arguments are required: {', '.join(missing)}")

    # How the envelope was produced. The run fills in the judge and the metric
    # configuration as it determines them, so an error envelope carries what
    # was known when the run stopped.
    meta = {
        "library_version": _library_version(),
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "judge": None,
        "metrics": {},
    }

    def client_factory() -> BedrockClient:
        """Build a judge client and record its model and region in ``meta``.

        The engine calls the factory only when a metric first reads the client.
        """
        client = BedrockClient(model_id=args.model_id, profile=args.aws_profile)
        meta["judge"] = {"model_id": client.model_id, "region": client.region}
        return client

    try:
        _load_metrics(args.custom_metrics_path)
        MetricRegistry.shared_bedrock_profile = args.aws_profile
        data_root = None
        if args.data_root is not None:
            if not Path(os.path.expanduser(args.data_root)).is_dir():
                raise ValueError(
                    f"--data-root must be an existing directory: {args.data_root!r}"
                )
            data_root = DataHandler._resolved_root(args.data_root)
        pred_data = _load_data(args.execution_output_path, args.pred_key, data_root)

        ref_data = []
        if args.ground_truth_path is not None or args.ground_truth_key is not None:
            ground_truth_path = (
                args.ground_truth_path
                if args.ground_truth_path is not None
                else args.execution_output_path
            )
            ref_data = _load_data(ground_truth_path, args.ground_truth_key, data_root)

        prompt_data = []
        if args.prompt_path is not None:
            prompt_data = _load_prompt_data(
                args.prompt_path, args.prompt_key, data_root
            )
        elif args.prompt_key is not None:
            prompt_data = _load_data(
                args.execution_output_path, args.prompt_key, data_root
            )

        context = []
        if args.context:
            context_fields = [field.strip() for field in args.context.split(",")]
            if any(not field for field in context_fields):
                raise ValueError("Context field names must be non-empty")
            context = [
                _load_data(args.execution_output_path, field, data_root)
                for field in context_fields
            ]

        _validate_auxiliary_cardinality(pred_data, prompt_data, context)
        record_ids = _record_ids(
            args.execution_output_path, args.sample_id_key, len(pred_data), data_root
        )

        selected_metrics = [name.strip() for name in args.metrics.split(",")]
        if any(not name for name in selected_metrics):
            raise ValueError("Metric names must be non-empty")
        # One result per metric: a repeat would be merged away by name.
        first_entries: dict[str, str] = {}
        for name in selected_metrics:
            key = normalize_metric_name(name)
            if key in first_entries:
                raise ValueError(
                    "--metrics names one metric more than once: "
                    f"{first_entries[key]!r} and {name!r}"
                )
            first_entries[key] = name

        methods = _aligned_entries(
            args.aggregation_methods,
            selected_metrics,
            "aggregation method",
            CLI_AGGREGATION_METHODS,
        )
        matchers = [
            entry or "whole"
            for entry in _aligned_entries(
                args.matcher, selected_metrics, "matcher", CLI_MATCHERS
            )
        ]
        if args.quantile is not None and "quantile" not in methods:
            raise ValueError("--quantile requires --aggregation-methods quantile")
        if args.weight_key is not None and "weighted_mean" not in methods:
            raise ValueError(
                "--weight-key requires --aggregation-methods weighted_mean"
            )
        if "weighted_mean" in methods and args.weight_key is None:
            raise ValueError("weighted_mean requires --weight-key")
        for method, matcher in zip(methods, matchers):
            if method == "prf" and matcher == "whole":
                raise ValueError("prf requires --matcher labels or list")
            if method != "prf" and matcher != "whole":
                raise ValueError(
                    "--matcher labels and list require --aggregation-methods prf"
                )
        if args.thresholds is not None and "prf" not in methods:
            raise ValueError("--thresholds requires --aggregation-methods prf")
        for option, value in (
            ("--class-key", args.class_key),
            ("--value-key", args.value_key),
        ):
            if value is not None and "list" not in matchers:
                raise ValueError(f"{option} requires --matcher list")
        thresholds = None
        if args.thresholds is not None:
            thresholds = _thresholds(args.thresholds)
        weights = None
        if args.weight_key is not None:
            weights = _weights(
                args.execution_output_path, args.weight_key, len(pred_data), data_root
            )
        matcher_objects = {
            "whole": WholeValue(),
            "labels": LabelSet(),
            "list": ListItems(class_key=args.class_key, value_key=args.value_key),
        }
        # One configuration per distinct (method, matcher) entry, in first-seen
        # order. An entry without a method keeps the metric's default.
        entries = list(zip(methods, matchers))
        configurations: dict[tuple, EvaluationConfiguration] = {}
        for method, matcher in dict.fromkeys(entries):
            if method is None:
                continue
            configurations[(method, matcher)] = EvaluationConfiguration(
                aggregation_method=method,
                weights=weights if method == "weighted_mean" else None,
                quantile=args.quantile if method == "quantile" else None,
                thresholds=thresholds if method == "prf" else None,
                matcher=matcher_objects[matcher],
            )
        meta["metrics"] = {
            name: _metric_meta(name, method, matcher, args, thresholds)
            for name, (method, matcher) in zip(selected_metrics, entries)
        }

        # One batch_compute call per distinct entry; metrics on their default
        # share the call with configuration=None. Report in --metrics order.
        merged: dict[str, MetricEvaluationResult | MetricError] = {}
        for entry in dict.fromkeys(entries):
            names = [
                name
                for name, name_entry in zip(selected_metrics, entries)
                if name_entry == entry
            ]
            merged.update(
                MetricRegistry.batch_compute(
                    names,
                    pred_data,
                    ref_data,
                    context,
                    prompt_data,
                    configuration=configurations.get(entry),
                    client_factory=client_factory,
                )
            )
        results = {name: merged[name] for name in selected_metrics}
        samples = {
            name: _sample_entries(value, record_ids, args.omit_reasoning)
            for name, value in results.items()
            if isinstance(value, MetricEvaluationResult)
        }
        results = {
            name: _envelope_value(value, record_ids) for name, value in results.items()
        }
        metric_errors = [
            f"{metric_name}: {result['error']}"
            for metric_name, result in results.items()
            if isinstance(result, MetricError)
        ]
        success_count = len(results) - len(metric_errors)
        if not metric_errors:
            status = "completed"
            exit_code = 0
        elif success_count:
            status = "partial"
            exit_code = 2
        else:
            status = "failed"
            exit_code = 1

        output = {
            "ok": status == "completed",
            "status": status,
            "data": {"results": results, "samples": samples},
            "errors": metric_errors,
            "meta": meta,
        }
        _write_output(args.output_path, output)
        printed = _stdout_envelope(output, args.output_path, args.stdout)
        print(json.dumps(printed, indent=2, allow_nan=False))
        return exit_code

    except Exception as error:
        error_output = {
            "ok": False,
            "status": "failed",
            "data": None,
            "errors": [_safe_error(error)],
            "meta": meta,
        }
        written_path = args.output_path
        try:
            _write_output(args.output_path, error_output)
        except Exception as write_error:
            written_path = None
            error_output["errors"].append(
                f"Could not write output to {args.output_path}: {write_error}"
            )
        printed = _stdout_envelope(error_output, written_path, args.stdout)
        print(json.dumps(printed, indent=2, allow_nan=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
