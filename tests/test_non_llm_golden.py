# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Golden snapshot of every non-LLM metric on one-item-per-role inputs.

Each registered metric that is not an LLM judge is run through
``MetricRegistry.compute_metric`` over several fixed corpora, and the complete
engine result (every sample's status, score, error and details, plus the
aggregate) is compared field for field with ``tests/fixtures/non_llm_golden.json``.
The fixture also records which (metric, corpus) pairs the current code rejects
and with what error, whether input validation refuses the corpus up front
(``failed_ordinal`` is null) or a sample fails during scoring, so a refactor
that silently starts accepting or rejecting an input is caught as well.

The comparison is exact: scores round-trip through JSON unchanged, so no
tolerance is applied. Each (metric, corpus) entry is its own test case, so a
failing run names every changed entry. Refresh the fixture only for an
intentional behavior change:

1. Run ``python -m pytest tests/test_non_llm_golden.py`` and note the
   failing entries.
2. Regenerate the fixture::

       UPDATE_NON_LLM_GOLDEN=1 python -m pytest tests/test_non_llm_golden.py

3. Review the fixture diff and commit it with the change that caused it. The
   commit message names each changed entry and why it changed.
"""

import hashlib
import json
import math
import os
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

import metric_library.metrics  # noqa: F401 - registers built-in metrics
from metric_library.core.data.text_data import TextData
from metric_library.core.llm_metric import LLMMetric
from metric_library.core.registry import MetricRegistry

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "non_llm_golden.json"
UPDATE_ENV_VAR = "UPDATE_NON_LLM_GOLDEN"

# Fixed (prediction, reference) raw pairs per corpus. Every value becomes one
# ``TextData`` item, so a list is one list-valued item: how the CLI loaded every
# JSON array before several items per role, and still loads an empty array or
# one that holds arrays or objects.
CORPORA = {
    "text": [
        ("cat", "cat"),
        ("Dog ", "dog"),
        ("bird", "fish"),
        ("", ""),
        ("", "a"),
        (None, None),
        ("a b c", "c b a"),
        ("hello world", "hello  world"),
    ],
    "numeric_scalar": [
        (1.5, 1.0),
        (3, 3.0),
        (2.0, 4.5),
        (0, 0),
        (-1.25, 1.25),
    ],
    "label_sets": [
        (["a", "b"], ["b", "a"]),
        (["c"], ["c"]),
        (["d", "e", "f"], ["d", "e", "g"]),
        ([], []),
        (["x"], []),
        (["A", "b"], ["a", "B"]),
    ],
    "numeric_lists": [
        ([1, 2], [2, 1]),
        ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]),
        ([1, 2, 3], [1, 2]),
    ],
    "edge": [
        (None, "a"),
        ("a", None),
        ("  ", "a"),
    ],
}


def _non_llm_metric_names():
    """Registered metric class names that are not LLM judges, sorted."""
    return sorted(
        cls.__name__
        for cls in MetricRegistry.metrics.values()
        if not issubclass(cls, LLMMetric)
    )


NON_LLM_METRICS = _non_llm_metric_names()


def _deterministic_embedding(text):
    """A unit vector fixed by the text, so equal strings embed identically."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    angle = int.from_bytes(digest[:4], "big") / 2**32 * 2 * math.pi
    return [math.cos(angle), math.sin(angle)]


def _embedding_stub():
    client = Mock(spec=["get_embeddings"])
    client.get_embeddings.side_effect = lambda texts: [
        _deterministic_embedding(text) for text in texts
    ]
    return client


def _snapshot(result):
    """The JSON-ready view of one ``MetricEvaluationResult``."""
    aggregate = result.aggregate
    return {
        "aggregate": {
            "method": aggregate.method,
            "status": aggregate.status.value,
            "score": aggregate.score,
            "values": aggregate.values,
            "total_count": aggregate.total_count,
            "succeeded_count": aggregate.succeeded_count,
            "failed_count": aggregate.failed_count,
            "skipped_count": aggregate.skipped_count,
            "denominator": aggregate.denominator,
            "coverage": aggregate.coverage,
        },
        "samples": [
            {
                "sample_id": sample.sample_id,
                "ordinal": sample.ordinal,
                "status": sample.status.value,
                "score": sample.score,
                "error": sample.error,
                "details": sample.details,
            }
            for sample in result.samples
        ],
    }


def _evaluate(metric_name, pairs):
    """Run one metric over one corpus without ever constructing a judge client."""
    factory = Mock(side_effect=AssertionError("judge client must not be constructed"))
    result = MetricRegistry.compute_metric(
        metric_name,
        [TextData(prediction) for prediction, _ in pairs],
        [TextData(reference) for _, reference in pairs],
        None,
        None,
        client_factory=factory,
    )
    factory.assert_not_called()
    return result


def _build_snapshot():
    """Evaluate every non-LLM metric over every corpus with Bedrock stubbed out."""

    def forbid(*args, **kwargs):
        raise AssertionError("live AWS is forbidden in tests")

    results = {}
    excluded = {}
    saved_client = MetricRegistry._shared_bedrock_client
    MetricRegistry._shared_bedrock_client = _embedding_stub()
    try:
        with (
            patch("metric_library.core.bedrock_client.boto3.client", forbid),
            patch("metric_library.core.bedrock_client.boto3.Session", forbid),
        ):
            for metric_name in NON_LLM_METRICS:
                for corpus_name, pairs in CORPORA.items():
                    key = f"{metric_name}/{corpus_name}"
                    try:
                        result = _evaluate(metric_name, pairs)
                    except ValueError as error:
                        # Input validation rejected the whole corpus before any
                        # sample was scored (an undeclared modality, a count).
                        excluded[key] = {"failed_ordinal": None, "error": str(error)}
                        continue
                    if result.status.value == "completed":
                        results[key] = _snapshot(result)
                    else:
                        failed = next(
                            sample
                            for sample in result.samples
                            if sample.status.value == "failed"
                        )
                        excluded[key] = {
                            "failed_ordinal": failed.ordinal,
                            "error": failed.error,
                        }
    finally:
        MetricRegistry._shared_bedrock_client = saved_client

    snapshot = {
        "metrics": NON_LLM_METRICS,
        "corpora": CORPORA,
        "results": results,
        "excluded": excluded,
    }
    # Round-trip so tuples, enums and mappings take their JSON shape before
    # comparing with the stored fixture.
    return json.loads(json.dumps(snapshot, sort_keys=True, allow_nan=False))


@pytest.fixture(scope="module")
def current():
    snapshot = _build_snapshot()
    if os.environ.get(UPDATE_ENV_VAR) == "1":
        FIXTURE_PATH.parent.mkdir(exist_ok=True)
        FIXTURE_PATH.write_text(
            json.dumps(snapshot, sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return snapshot


@pytest.fixture(scope="module")
def stored():
    if not FIXTURE_PATH.is_file():
        pytest.fail(f"{FIXTURE_PATH} is missing; generate it with {UPDATE_ENV_VAR}=1")
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_non_llm_metric_roster_and_corpora_are_unchanged(current, stored):
    assert current["metrics"] == stored["metrics"]
    assert current["corpora"] == stored["corpora"]


@pytest.mark.parametrize(
    "key", [f"{metric}/{corpus}" for metric in NON_LLM_METRICS for corpus in CORPORA]
)
def test_metric_results_match_the_golden_snapshot(current, stored, key):
    assert current["results"].get(key) == stored["results"].get(key), (
        f"{key}: results changed"
    )
    assert current["excluded"].get(key) == stored["excluded"].get(key), (
        f"{key}: acceptance changed"
    )


def test_every_metric_corpus_pair_is_either_a_result_or_an_exclusion(current):
    keys = {f"{metric}/{corpus}" for metric in NON_LLM_METRICS for corpus in CORPORA}
    assert set(current["results"]) | set(current["excluded"]) == keys
    assert not set(current["results"]) & set(current["excluded"])
