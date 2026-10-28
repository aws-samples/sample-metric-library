# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Focused deterministic regression tests for ML-01 correctness repairs."""

import builtins
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, call, patch

import pytest


class TestSemanticComparison:
    """Verify semantic comparison uses embeddings without fuzzy fallbacks."""

    def test_normalizes_whitespace_and_returns_cosine_score(self):
        from metric_library.utils.comparison_functions import compare_semantic

        client = Mock(spec=["get_embeddings"])
        client.get_embeddings.side_effect = [
            [[1.0, 0.0]],
            [[0.0, 1.0]],
        ]

        result = compare_semantic("  expected\n\ttext  ", " actual\t text ", client)

        assert result == pytest.approx(0.5)
        assert client.mock_calls == [
            call.get_embeddings(["expected text"]),
            call.get_embeddings(["actual text"]),
        ]

    def test_provider_error_propagates_without_fuzzy_fallback(self, monkeypatch):
        from metric_library.utils import comparison_functions

        provider_error = RuntimeError("embedding provider failed")
        client = Mock(spec=["get_embeddings"])
        client.get_embeddings.side_effect = provider_error
        fuzzy = Mock(side_effect=AssertionError("fuzzy fallback must not run"))
        monkeypatch.setattr(comparison_functions, "compare_fuzzy", fuzzy)

        with pytest.raises(RuntimeError, match="embedding provider failed") as exc_info:
            comparison_functions.compare_semantic("expected", "actual", client)

        assert exc_info.value is provider_error
        fuzzy.assert_not_called()

    @pytest.mark.parametrize(
        "embedding_response",
        [None, [], [[]], [[1.0], [2.0]]],
    )
    def test_invalid_embedding_response_raises(self, embedding_response):
        from metric_library.utils.comparison_functions import compare_semantic

        client = Mock(spec=["get_embeddings"])
        client.get_embeddings.return_value = embedding_response

        with pytest.raises(ValueError, match="exactly one non-empty"):
            compare_semantic("expected", "actual", client)

    @pytest.mark.parametrize(
        ("expected", "actual", "score"),
        [
            (None, None, 1.0),
            (None, "actual", 0.0),
            ("expected", None, 0.0),
            (" \n\t", "\t  ", 1.0),
        ],
    )
    def test_null_and_blank_values_short_circuit(self, expected, actual, score):
        from metric_library.utils.comparison_functions import compare_semantic

        client = Mock(spec=["get_embeddings"])

        assert compare_semantic(expected, actual, client) == score
        client.get_embeddings.assert_not_called()

    def test_single_blank_value_is_rejected_before_client_call(self):
        from metric_library.utils.comparison_functions import compare_semantic

        client = Mock(spec=["get_embeddings"])

        with pytest.raises(ValueError, match="empty text"):
            compare_semantic(" \n\t", "actual", client)

        client.get_embeddings.assert_not_called()


class TestComparatorDispatch:
    """Verify numeric and Hungarian comparator setup and dispatch."""

    def test_numeric_dispatch_paths(self):
        from metric_library.utils.comparison_functions import (
            ComparatorType,
            comparator_funcs,
            compare_values,
            get_comparison_function,
        )

        assert compare_values("$1,234.00", "1234", ComparatorType.NUMERIC.value) == 1.0
        numeric_comparator = get_comparison_function(ComparatorType.NUMERIC.value)
        assert numeric_comparator("(42)", "42") == 1.0
        assert comparator_funcs[ComparatorType.NUMERIC]("9", 10) == 0.0

    def test_hungarian_exact_matches_permuted_lists(self):
        from metric_library.utils.comparison_functions import (
            ComparatorType,
            compare_hungarian,
        )

        result = compare_hungarian(
            ["alpha", "beta", "gamma"],
            ["gamma", "alpha", "beta"],
            ComparatorType.EXACT.value,
        )

        assert result == 1.0

    def test_text_hungarian_match_forwards_exact_string_and_normalization(
        self, monkeypatch
    ):
        import metric_library.metrics.match as match_module
        from metric_library.utils.comparison_functions import (
            ComparatorType,
            NormalizationType,
        )

        comparator_result = object()
        compare_hungarian = Mock(return_value=comparator_result)
        monkeypatch.setattr(match_module, "compare_hungarian", compare_hungarian)
        expected = ["Alpha!"]
        actual = [" alpha "]

        metric = match_module.TextHungarianMatch()
        result = metric.comparator(expected, actual)

        assert result is comparator_result
        compare_hungarian.assert_called_once_with(
            expected,
            actual,
            ComparatorType.EXACT.value,
            NormalizationType.TEXT.value,
        )
        assert isinstance(compare_hungarian.call_args.args[2], str)

    def test_dynamic_match_initializes_exact_hungarian_and_numeric(self):
        from metric_library.metrics.match import DynamicMatch, TextHungarianMatch
        from metric_library.utils.comparison_functions import ComparatorType

        exact_match = DynamicMatch(comparator_type=ComparatorType.EXACT.value)
        assert exact_match.comparator("Hello!", "hello") == 1.0

        hungarian_match = TextHungarianMatch()
        assert hungarian_match.comparator(["beta", "alpha"], ["alpha", "beta"]) == 1.0

        numeric_match = DynamicMatch(comparator_type=ComparatorType.NUMERIC.value)
        assert numeric_match.comparator("$1,000", "1000") == 1.0


class TestMatchMetricsCompute:
    """Match metrics must hand each whole value to the comparator through compute."""

    @staticmethod
    def _score(metric_name, prediction, reference):
        from metric_library.core.data.text_data import TextData
        from metric_library.core.registry import MetricRegistry
        import metric_library.metrics  # noqa: F401  registers the match metrics

        result = MetricRegistry.compute_metric(
            metric_name,
            [TextData(prediction)],
            [TextData(reference)],
            client_factory=Mock(spec=[]),
        )
        return result.score

    @pytest.mark.parametrize(
        "metric_name, prediction, reference, expected",
        [
            ("TextExactMatch", "cat", "cats", 0.0),
            ("TextExactMatch", "Paris", "paris!", 1.0),
            ("TextExactMatch", "", "", 1.0),
            ("TextExactMatch", "abc", "", 0.0),
            ("TextFuzzyMatch", "Colour", "color", 1 - 1 / 6),
            ("NumericExactMatch", "5", "5.0", 1.0),
            ("NumericExactMatch", 5, 5, 1.0),
            ("NumericExactMatch", "$1,234.00", 1234, 1.0),
            ("ListExactMatch", ["a", "b"], ["b", "a"], 1.0),
            ("ListExactMatch", ["a", "b"], ["a", "c"], 0.0),
            (
                "TextHungarianMatch",
                ["alpha", "beta", "gamma"],
                ["gamma", "alpha", "beta"],
                1.0,
            ),
            ("TextHungarianMatch", ["alpha", "beta"], ["beta", "delta"], 0.5),
        ],
    )
    def test_inline_values_are_compared_whole(
        self, metric_name, prediction, reference, expected
    ):
        score = self._score(metric_name, prediction, reference)

        assert score == pytest.approx(expected)

    def test_score_is_the_mean_over_pairs(self):
        from metric_library.core.data.text_data import TextData
        from metric_library.core.registry import MetricRegistry
        import metric_library.metrics  # noqa: F401

        result = MetricRegistry.compute_metric(
            "TextExactMatch",
            [TextData("foo"), TextData("bar"), TextData("baz")],
            [TextData("foo"), TextData("bar"), TextData("qux")],
            client_factory=Mock(spec=[]),
        )

        assert result.score == pytest.approx(2 / 3)

    def test_semantic_match_embeds_whole_values(self):
        from metric_library.core.registry import MetricRegistry

        client = Mock(spec=["get_embeddings"])
        client.get_embeddings.side_effect = [[[1.0, 0.0]], [[0.0, 1.0]]]

        with patch.object(
            MetricRegistry, "get_shared_bedrock_client", return_value=client
        ):
            score = self._score("ItemSemanticMatch", "hello world", "hello there")

        assert client.get_embeddings.call_args_list == [
            call(["hello world"]),
            call(["hello there"]),
        ]
        assert score == pytest.approx(0.5)


class TestJsonParsing:
    """Verify JSON parsing stays in memory and is safe for concurrent reads."""

    def test_parse_json_file_avoids_sibling_temp_file_and_is_concurrent(self, tmp_path):
        from metric_library.core.data_handler import DataHandler

        existing_text = tmp_path / "existing.txt"
        existing_text.write_text("from existing file", encoding="utf-8")
        json_path = tmp_path / "input.json"
        json_path.write_text(
            json.dumps(
                [
                    {"answer": "first"},
                    {"nested": {"answer": ["second", 3]}},
                    {"answer": str(existing_text)},
                ]
            ),
            encoding="utf-8",
        )
        temp_path = f"{json_path}.tmp"
        temp_open_attempts = []
        real_open = builtins.open

        def guarded_open(file, *args, **kwargs):
            if str(file) == temp_path:
                temp_open_attempts.append((file, args, kwargs))
                raise AssertionError(
                    "parse_json_file must not open a sibling .tmp file"
                )
            return real_open(file, *args, **kwargs)

        def raw(item):
            # A JSON array loads as one item per element; read them back as a list.
            if isinstance(item, list):
                return [part.get_raw_data() for part in item]
            return item.get_raw_data()

        def parse_raw_values():
            return [
                raw(item)
                for item in DataHandler.parse_json_file(str(json_path), "answer")
            ]

        worker_count = 8
        parse_barrier = threading.Barrier(worker_count)

        def parse_after_barrier(_):
            parse_barrier.wait()
            return parse_raw_values()

        expected = ["first", ["second", 3], str(existing_text)]
        with patch("builtins.open", side_effect=guarded_open):
            assert parse_raw_values() == expected
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                concurrent_results = list(
                    executor.map(parse_after_barrier, range(worker_count))
                )

        assert concurrent_results == [expected] * worker_count
        assert temp_open_attempts == []
        assert not (tmp_path / "input.json.tmp").exists()


class TestMetricRegistryCorrectness:
    """Verify logging and class-object containment behavior."""

    def test_batch_compute_log_resolves_metric_name(self, caplog, monkeypatch):
        from metric_library.core.registry import MetricRegistry

        compute_metric = Mock(return_value=0.75)
        monkeypatch.setattr(MetricRegistry, "compute_metric", compute_metric)

        with caplog.at_level(logging.INFO, logger="metric_library.core.registry"):
            result = MetricRegistry.batch_compute(
                ["ResolvedMetric"], predictions=[], references=[object()]
            )

        assert result == {"ResolvedMetric": 0.75}
        messages = [record.getMessage() for record in caplog.records]
        assert "Computing metric of ResolvedMetric" in messages
        assert all("{metric_name}" not in message for message in messages)

    def test_class_level_containment_normalizes_and_handles_non_strings(self):
        from metric_library.core.registry import MetricRegistry
        import metric_library.metrics  # noqa: F401

        assert "Text Exact-Match" in MetricRegistry
        assert "definitely-not-a-metric" not in MetricRegistry
        assert 123 not in MetricRegistry
