# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Smoke tests for the metric library.

Verifies core imports, factory registration, metric computation,
and that optional media dependencies are not eagerly loaded.
"""

import sys
from unittest.mock import Mock

import pytest


class TestFactoryRegistration:
    """Verify the data type factory registers correctly."""

    def test_text_registered_on_import(self):
        from metric_library.core.data import Modality, factory

        assert Modality.TEXT in factory._registry

    def test_media_types_not_registered_eagerly(self):
        from metric_library.core.data import Modality, factory

        assert Modality.IMAGE not in factory._registry
        assert Modality.AUDIO not in factory._registry
        assert Modality.VIDEO not in factory._registry

    def test_factory_create_text(self):
        from metric_library.core.data import Modality, factory

        obj = factory.create(Modality.TEXT, "hello")
        assert obj.get_raw_data() == "hello"

    def test_factory_create_reports_a_missing_media_module(self, monkeypatch):
        """A failed lazy import names the modality and the media extra."""
        from metric_library.core.data import Modality, factory

        monkeypatch.setitem(factory._MODULES, Modality.IMAGE, "no_such_module")
        if Modality.IMAGE in factory._registry:
            monkeypatch.delitem(factory._registry, Modality.IMAGE)

        with pytest.raises(ValueError, match="No data type available for") as excinfo:
            factory.create(Modality.IMAGE, "test.jpg")
        assert "media" in str(excinfo.value)
        assert "PIL" not in sys.modules

    def test_media_deps_not_imported(self):
        """cv2, PIL, and soundfile should not be in sys.modules."""
        import metric_library.metrics  # noqa: F401

        for mod in ["cv2", "PIL", "soundfile"]:
            assert mod not in sys.modules, f"{mod} was eagerly imported"


class TestMetricRegistry:
    """Verify metrics register and compute correctly."""

    def test_metrics_registered(self):
        from metric_library.core.registry import MetricRegistry
        import metric_library.metrics  # noqa: F401

        assert len(MetricRegistry.metrics) >= 46

    def test_metric_name_normalization(self):
        from metric_library.core.registry import MetricRegistry
        import metric_library.metrics  # noqa: F401

        m1 = MetricRegistry.get_metric("text_exact_match")
        m2 = MetricRegistry.get_metric("TextExactMatch")
        m3 = MetricRegistry.get_metric("text-exact-match")
        assert type(m1) is type(m2) is type(m3)

    def test_get_unknown_metric_raises(self):
        from metric_library.core.registry import MetricRegistry
        import metric_library.metrics  # noqa: F401

        with pytest.raises(KeyError, match="not found"):
            MetricRegistry.get_metric("nonexistent_metric_xyz")


class TestClassificationMetrics:
    """Verify deterministic classification metrics compute correctly."""

    @pytest.fixture()
    def text_data(self):
        from metric_library.core.data.text_data import TextData

        preds = [TextData("cat"), TextData("dog"), TextData("bird")]
        refs = [TextData("cat"), TextData("dog"), TextData("fish")]
        return preds, refs

    def test_accuracy(self, text_data):
        from metric_library.core.registry import MetricRegistry
        import metric_library.metrics  # noqa: F401

        preds, refs = text_data
        result = MetricRegistry.compute_metric(
            "Accuracy", preds, refs, client_factory=Mock(spec=[])
        )
        assert 0.0 < result.score <= 1.0

    def test_precision_recall_f1_of_the_labels(self, text_data):
        from metric_library import EvaluationConfiguration, LabelSet
        from metric_library.core.registry import MetricRegistry
        import metric_library.metrics  # noqa: F401

        preds, refs = text_data
        result = MetricRegistry.compute_metric(
            "Accuracy",
            preds,
            refs,
            configuration=EvaluationConfiguration(
                aggregation_method="prf", matcher=LabelSet()
            ),
            client_factory=Mock(spec=[]),
        )
        assert result.score == 0.5
        assert result.aggregate.method == "prf"

    def test_batch_compute(self, text_data):
        from metric_library.core.registry import MetricRegistry
        import metric_library.metrics  # noqa: F401

        preds, refs = text_data
        results = MetricRegistry.batch_compute(
            ["Accuracy", "TextExactMatch"],
            preds,
            refs,
            client_factory=Mock(spec=[]),
        )
        assert len(results) == 2
        for name in results:
            score = results[name].score
            assert isinstance(score, float), f"{name} returned {type(score)}"


class TestMatchMetrics:
    """Verify match-based metrics."""

    def test_text_exact_match(self):
        from metric_library.core.data.text_data import TextData
        from metric_library.core.registry import MetricRegistry
        import metric_library.metrics  # noqa: F401

        preds = [TextData("foo"), TextData("bar")]
        refs = [TextData("foo"), TextData("baz")]
        result = MetricRegistry.compute_metric(
            "TextExactMatch", preds, refs, client_factory=Mock(spec=[])
        )
        assert 0.0 < result.score <= 1.0

    def test_text_fuzzy_match(self):
        from metric_library.core.data.text_data import TextData
        from metric_library.core.registry import MetricRegistry
        import metric_library.metrics  # noqa: F401

        preds = [TextData("hello world"), TextData("foo bar")]
        refs = [TextData("hello world"), TextData("foo baz")]
        result = MetricRegistry.compute_metric(
            "TextFuzzyMatch", preds, refs, client_factory=Mock(spec=[])
        )
        assert result.score > 0.5


class TestDataHandler:
    """Verify DataHandler modality detection and element creation."""

    def test_detect_text_modality(self):
        from metric_library.core.data_handler import DataHandler
        from metric_library.core.data import Modality

        assert DataHandler._detect_modality("data.json") == Modality.TEXT
        assert DataHandler._detect_modality("file.txt") == Modality.TEXT

    def test_detect_image_modality(self):
        from metric_library.core.data_handler import DataHandler
        from metric_library.core.data import Modality

        assert DataHandler._detect_modality("photo.jpg") == Modality.IMAGE
        assert DataHandler._detect_modality("image.png") == Modality.IMAGE

    def test_detect_audio_modality(self):
        from metric_library.core.data_handler import DataHandler
        from metric_library.core.data import Modality

        assert DataHandler._detect_modality("sound.wav") == Modality.AUDIO
        assert DataHandler._detect_modality("track.mp3") == Modality.AUDIO

    def test_detect_video_modality(self):
        from metric_library.core.data_handler import DataHandler
        from metric_library.core.data import Modality

        assert DataHandler._detect_modality("clip.mp4") == Modality.VIDEO
        assert DataHandler._detect_modality("movie.avi") == Modality.VIDEO

    def test_assign_element_text(self):
        from metric_library.core.data_handler import DataHandler

        result = DataHandler.assign_element("hello")
        assert result.get_raw_data() == "hello"
