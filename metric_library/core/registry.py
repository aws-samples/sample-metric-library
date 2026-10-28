# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Metric registry system for managing and discovering metrics.

This module provides the central registry that manages all metrics in the library.
"""

from __future__ import annotations

import logging
import re
from functools import wraps
from typing import Callable, Dict, Type

from .base import Metric
from .data import DataType
from .input_policy import InputPolicy
from .evaluation import (
    EvaluationConfiguration,
    MetricEvaluationResult,
    _safe_error,
    evaluate_metric,
)

logger = logging.getLogger(__name__)

# Import BedrockClient for shared instance management
try:
    from .bedrock_client import BedrockClient

    _bedrock_available = True
except ImportError:
    _bedrock_available = False


class MetricError(dict):
    """Mark execution failures while preserving the legacy ``{"error": ...}`` map."""

    def __init__(self, message: str):
        super().__init__(error=message)


def normalize_metric_name(name):
    return re.sub(r"[^a-zA-Z0-9]", "", name).lower()


def normalize_name(func):
    """Decorator to normalize metric names in method arguments."""

    @wraps(func)
    def wrapper(cls, name: str, *args, **kwargs):
        normalized = normalize_metric_name(name)
        return func(cls, normalized, *args, **kwargs)

    return wrapper


class MetricRegistryMeta(type):
    """Provide class-level membership checks for the metric registry."""

    def __contains__(cls, name) -> bool:
        if not isinstance(name, str):
            return False
        return cls.is_registered(name)


class MetricRegistry(metaclass=MetricRegistryMeta):
    """
    Central registry for managing and discovering metrics.

    This class provides registration and discovery of metrics.
    """

    metrics: Dict[str, Type[Metric]] = {}

    # Shared BedrockClient instance for all semantic metrics across the system
    _shared_bedrock_client = None
    # AWS profile of the shared embedding client; None means the boto3 default
    # chain. The runner sets it from --aws-profile.
    shared_bedrock_profile: str | None = None

    @classmethod
    def get_shared_bedrock_client(cls):
        """
        Get or create a shared BedrockClient instance for semantic metrics.

        This provides a single BedrockClient instance that can be used by all semantic
        metrics across the entire metric library, reducing resource usage and connection overhead.
        Uses the Titan embedding model specifically for semantic similarity operations.
        The client uses the AWS profile in ``shared_bedrock_profile``.

        Returns:
            BedrockClient: Shared instance for semantic operations with Titan embedding model
        """
        if not _bedrock_available:
            raise RuntimeError("Bedrock client is not available")
        if cls._shared_bedrock_client is None:
            cls._shared_bedrock_client = BedrockClient(
                model_id="amazon.titan-embed-text-v2:0",
                profile=cls.shared_bedrock_profile,
            )
            logger.info(
                "Created shared BedrockClient for metric library with Titan embedding model"
            )
        return cls._shared_bedrock_client

    @classmethod
    def register(cls, metric_class: type[Metric]) -> type[Metric]:
        """Register a metric class with the registry.

        Only classes that subclass ``Metric`` are admitted; anything else raises
        ``TypeError``. A ``Metric`` subclass that still lacks ``compute`` is
        stored and fails later, at instantiation in ``get_metric``. A class
        whose normalized name is already registered by another class replaces
        it, and a warning names both classes.
        """
        if not isinstance(metric_class, type) or not issubclass(metric_class, Metric):
            raise TypeError(f"register expects a Metric subclass, got {metric_class!r}")
        normalized_name = normalize_metric_name(metric_class.__name__)
        previous = cls.metrics.get(normalized_name)
        if previous is not None and previous is not metric_class:
            logger.warning(
                "Metric %r: %s.%s replaces %s.%s",
                normalized_name,
                metric_class.__module__,
                metric_class.__qualname__,
                previous.__module__,
                previous.__qualname__,
            )
        cls.metrics[normalized_name] = metric_class
        return cls.metrics[normalized_name]

    @classmethod
    @normalize_name
    def get_metric(cls, name: str) -> Metric:
        """Get an instance of a registered metric."""
        if name not in cls.metrics:
            available = list(cls.metrics.keys())
            raise KeyError(f"Metric '{name}' not found. Available metrics: {available}")

        metric_class = cls.metrics[name]
        return metric_class()

    @classmethod
    @normalize_name
    def is_registered(cls, name: str) -> bool:
        """Check if a metric is registered, returns True if it is."""
        return name in cls.metrics

    @classmethod
    def list_metrics(cls, names: list[str] | None = None) -> list[dict]:
        """Return one description per registered metric, from class attributes only.

        Each entry carries ``name``, ``kind`` (``"llm"`` for a judge metric,
        ``"deterministic"`` otherwise), ``modalities``, the ``reference``,
        ``prompt`` and ``context`` policies, ``default_aggregation_method``
        and ``description`` (the first line of the class's own docstring, empty
        when it has none). ``names``
        filters and orders the result; an unknown name raises ``KeyError``. No
        metric is instantiated, so no AWS client is built. The runner's
        ``--list-metrics`` prints this list.
        """
        from .llm_metric import LLMMetric

        if names is None:
            classes = [cls.metrics[key] for key in sorted(cls.metrics)]
        else:
            classes = []
            for name in names:
                key = normalize_metric_name(name)
                if key not in cls.metrics:
                    raise KeyError(f"Metric '{name}' not found")
                classes.append(cls.metrics[key])
        entries = []
        for metric_class in classes:
            # The class's own docstring: ``inspect.getdoc`` would fall back to
            # the base class and describe a custom metric as "Base class for
            # all metrics."
            doc = metric_class.__doc__ or ""
            entries.append(
                {
                    "name": metric_class.__name__,
                    "kind": "llm"
                    if issubclass(metric_class, LLMMetric)
                    else "deterministic",
                    "modalities": [str(m) for m in metric_class.supported_data_types],
                    "reference": InputPolicy(metric_class.reference_policy).value,
                    "prompt": InputPolicy(metric_class.prompt_policy).value,
                    "context": InputPolicy(metric_class.context_policy).value,
                    "default_aggregation_method": (
                        metric_class.default_aggregation_method or "numeric_mean"
                    ),
                    "description": doc.strip().splitlines()[0] if doc.strip() else "",
                }
            )
        return entries

    @classmethod
    @normalize_name
    def compute_metric(
        cls,
        name: str,
        predictions: list[DataType],
        references: list[DataType],
        context: list[list[DataType]] | None = None,
        prompts: list[DataType] | None = None,
        *,
        configuration: EvaluationConfiguration | None = None,
        client_factory: Callable[[], BedrockClient] | None = None,
        sleeper: Callable[[float], None] | None = None,
    ) -> MetricEvaluationResult:
        """Evaluate one metric by name through the engine.

        Returns per-sample results and the aggregate. The registry resolves the
        name only; ``evaluate_metric`` validates the inputs, builds the client
        through ``client_factory`` and scores every sample.
        """
        metric = cls.get_metric(name)
        return evaluate_metric(
            metric,
            predictions,
            references,
            context,
            prompts,
            configuration=configuration,
            client_factory=client_factory,
            sleeper=sleeper,
        )

    @classmethod
    def batch_compute(
        cls,
        metrics: list[str],
        predictions: list[DataType],
        references: list[DataType],
        context: list[list[DataType]] | None = None,
        prompts: list[DataType] | None = None,
        *,
        configuration: EvaluationConfiguration | None = None,
        client_factory: Callable[[], BedrockClient] | None = None,
        sleeper: Callable[[float], None] | None = None,
    ) -> dict[str, MetricEvaluationResult | MetricError]:
        """Evaluate several metrics by name with one shared configuration.

        Each value is the metric's ``MetricEvaluationResult``, COMPLETED or
        FAILED, or a ``MetricError`` when ``compute_metric`` raised for that name.
        """
        results: dict[str, MetricEvaluationResult | MetricError] = {}

        for metric_name in metrics:
            try:
                logger.info("Computing metric of %s", metric_name)
                results[metric_name] = cls.compute_metric(
                    metric_name,
                    predictions,
                    references,
                    context,
                    prompts,
                    configuration=configuration,
                    client_factory=client_factory,
                    sleeper=sleeper,
                )
            except Exception as error:
                # The result carries the error; the message can quote the data.
                logger.debug("Error computing %s: %s", metric_name, error)
                results[metric_name] = MetricError(_safe_error(error))

        return results
