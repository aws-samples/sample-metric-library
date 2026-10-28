# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Custom metrics for evaluation.

Define metric classes here to extend the built-in metric library. Each class
must inherit from Metric and be decorated with @MetricRegistry.register.
The runner registers them when it is given this file, as in
``python -m metric_library.run_metrics --custom-metrics-path custom_metrics.py
...``; it loads no custom metrics without that option.

A deterministic metric subclasses ``Metric`` and implements
``compute(self, sample, client) -> float | SampleOutput``. The engine
calls ``compute`` once per sample and aggregates the returned scores into
the metric's final score. ``compute`` may return a
``SampleOutput(score, details)`` to keep per-sample details with the score. A
deterministic metric ignores ``client``. The ``sample`` carries
``sample.prediction``, ``sample.reference``, ``sample.prompt`` and
``sample.contexts``.

A metric judged by an LLM subclasses ``LLMMetric``
(``metric_library.core.llm_metric``) instead and implements ``rubric()``,
the evaluation criteria; it does not override ``compute``. ``LLMMetric``
builds the judge request from the sample's prompt, context, candidate and
ground truth, as text or as image blocks, and validates the returned
score. The subclass declares which inputs it reads (``reference_policy``,
``prompt_policy``, ``context_policy``) and may add ``modality_guidance``
for judging images. See "LLM-Based Metric" in the README.

Rules for authors:

- ``compute`` returns a finite real or a ``SampleOutput``. Any other
  value makes the sample FAILED.
- Validation and the input policies are inherited from Metric: references
  are required, prompts and context are dropped. Set ``prompt_policy`` or
  ``context_policy`` only when the metric reads ``sample.prompt`` or
  ``sample.contexts``.
- ``compute`` takes one sample, not the dataset. A metric written for the
  old ``compute(predictions, references, *args)`` signature instantiates
  but fails its first sample with ``'EvaluationSample' object is not
  iterable``.
- With the ``labels`` or ``list`` matcher and the ``prf`` aggregator, the
  engine calls ``compute`` once for each candidate pair of items:
  ``sample.prediction`` and ``sample.reference`` each hold one item, and the
  score must be from 0.0 to 1.0, where 1.0 is a full match. Each item is
  text: an image prediction or reference stops the metric before any sample
  is scored.
- Optional class attributes: ``supported_data_types`` restricts the accepted
  modalities (the default is all of them); an item of another modality in
  a role the metric reads fails validation before any sample is scored;
  ``default_aggregation_method``
  names the aggregator the engine uses when the caller sets none (the
  default is ``numeric_mean``).
- A record field that is a non-empty JSON array of scalars gives the
  sample several items in that role. With the ``whole`` matcher, a metric
  accepts exactly one item per role unless it declares
  ``prediction_multiplicity`` / ``reference_multiplicity``
  (``metric_library.core.multiplicity``); validation rejects more items
  before any sample is scored. A metric that reads several items uses
  ``sample.predictions`` and ``sample.references``. Multiplicity is for the
  whole path, and the matcher replaces it on the item path: with ``labels``
  or ``list``, the matcher reads the items and ``compute`` gets one item per
  role. A metric that declares ``ALIGNED`` (items paired by position) or
  ``ANY_ACCEPTABLE`` (alternatives) accepts only the ``whole`` matcher.

Example::

    from metric_library.core.aggregation import SampleOutput
    from metric_library.core.base import Metric
    from metric_library.core.bedrock_client import BedrockClient
    from metric_library.core.data import Modality
    from metric_library.core.evaluation import EvaluationSample
    from metric_library.core.registry import MetricRegistry

    @MetricRegistry.register
    class MyMetric(Metric):
        # 1.0 when the prediction equals its reference as a whole value, else 0.0.

        # Optional: restrict the accepted modalities (the default is all of them).
        supported_data_types = [Modality.TEXT]

        # Optional: the aggregator the engine uses when the caller sets none
        # (the default is numeric_mean).
        # default_aggregation_method = "median"

        def compute(self, sample: EvaluationSample, client: BedrockClient) -> float:
            # A deterministic metric ignores the client.
            return 1.0 if sample.prediction.get_raw_data() == sample.reference.get_raw_data() else 0.0
"""
