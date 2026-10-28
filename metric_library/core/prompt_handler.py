# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Judge-request composition for LLM metrics from their prompt sections.

``PromptHandler.build`` turns one already-aligned sample into the complete
judge request handed to ``BedrockClient.query`` as keyword arguments: the
judge role as the ``system`` prompt; the metric's rubric, with any
supplemental guidance after it, as the ``prompt`` text that opens the user
message; the ordered ``additional_content`` list of delimited data sections
that follows it; and the ``response_schema`` the judge's structured result is
forced to satisfy. The metric supplies the wording through its ``rubric`` and
its ``LLMMetric`` section hooks (``role``, ``supplemental``, ``input``,
``context``, ``prediction``, ``reference`` and ``response_schema``); the
handler owns what the sections must never have to repeat: input normalization,
policy omission, canonical section order and the transport shape.

The handler treats every ``DataType.get_data_for_LLM_message()`` result as an
opaque provider-compatible block and never inspects or rewrites it. It adds no
presence, cardinality, alignment or modality validation; those remain the
responsibility of ``Metric.validate_evaluation_inputs``, which runs before any
sample is scored.

Every LLM metric composes its request this way: ``LLMMetric._build_judge_request``
calls ``build`` with the sample's items that carry content. This module is
internal to metric implementations and is deliberately not re-exported from
``metric_library.core`` or the package root.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .data import DataType
from .input_policy import InputPolicy

if TYPE_CHECKING:
    from .llm_metric import LLMMetric


@dataclass(frozen=True, slots=True)
class SampleItems:
    """
    The role items one ``PromptHandler.build`` call actually consumes.

    Every role is a tuple, already normalized from the caller's scalar or
    sequence input and emptied when the metric's policy omits the role. This is
    what ``LLMMetric.supplemental`` receives, so a metric can shape its
    guidance around exactly the data the judge will see.
    """

    prompt: tuple[DataType, ...]
    context: tuple[DataType, ...]
    prediction: tuple[DataType, ...]
    reference: tuple[DataType, ...]


class PromptHandler:
    """
    Stateless composer of one judge request from a metric's prompt sections.

    ``build`` is the only interface. It is deterministic, performs no I/O, and
    reaches ``get_data_for_LLM_message()`` only through the metric's section
    hooks.
    """

    # Auxiliary roles paired with the metric attribute that governs whether the
    # role is consumed at all. Prediction is always consumed.
    _POLICY_ATTRIBUTES = {
        "prompt": "prompt_policy",
        "context": "context_policy",
        "reference": "reference_policy",
    }

    @staticmethod
    def build(
        metric: "LLMMetric",
        *,
        prompt: DataType | Sequence[DataType] | None = None,
        context: DataType | Sequence[DataType] | None = None,
        prediction: DataType | Sequence[DataType] | None = None,
        reference: DataType | Sequence[DataType] | None = None,
    ) -> dict:
        """
        Compose the ``BedrockClient.query`` keyword arguments for one request.

        Each role accepts ``None``, one scalar ``DataType`` or an ordered
        sequence of ``DataType`` values; a scalar is consumed exactly like its
        singleton sequence. A role whose metric policy is ``InputPolicy.OMIT``
        is emptied first: its items are never converted and never reach a
        section hook.

        The data sections are then rendered in the canonical order ``input``,
        ``context``, ``prediction``, ``reference``. The instructions precede
        them in the same user message: ``rubric()`` is the opening ``prompt``
        text and a non-empty ``supplemental`` result follows it after a blank
        line, so the judge reads the rubric and any guidance before the data.

        Args:
            metric: The metric whose policies and section hooks define the
                request.
            prompt: Items for the ``task`` role.
            context: Items for the ``context`` role.
            prediction: Items for the ``candidate_response`` role.
            reference: Items for the ``ground_truth_response`` role.

        Returns:
            A dictionary with ``system`` set to the metric's ``role()``,
            ``prompt`` set to its ``rubric()`` followed by any
            ``supplemental`` guidance, ``additional_content`` set to the
            ordered list of data blocks, and ``response_schema`` set to the
            metric's ``response_schema()``.
        """
        roles = {
            "prompt": PromptHandler._normalize(prompt),
            "context": PromptHandler._normalize(context),
            "prediction": PromptHandler._normalize(prediction),
            "reference": PromptHandler._normalize(reference),
        }
        for role, attribute in PromptHandler._POLICY_ATTRIBUTES.items():
            if getattr(metric, attribute) == InputPolicy.OMIT:
                roles[role] = ()

        sample = SampleItems(**roles)

        instructions = metric.rubric()
        extra = metric.supplemental(sample)
        if extra:
            instructions += f"\n\n{extra}"

        additional_content: list[dict] = (
            metric.input(sample.prompt)
            + metric.context(sample.context)
            + metric.prediction(sample.prediction)
            + metric.reference(sample.reference)
        )

        response_schema = metric.response_schema()
        if len(sample.prediction) > 1:
            response_schema = _with_part_scores(response_schema, len(sample.prediction))

        return {
            "prompt": instructions,
            "system": metric.role(),
            "additional_content": additional_content,
            "response_schema": response_schema,
        }

    @staticmethod
    def _normalize(
        value: DataType | Sequence[DataType] | None,
    ) -> tuple[DataType, ...]:
        """Normalize ``None``, one ``DataType`` or an ordered sequence to a tuple."""
        if value is None:
            return ()
        if isinstance(value, DataType):
            return (value,)
        return tuple(value)


def _with_part_scores(schema: dict, count: int) -> dict:
    """The metric's response schema extended with one score per candidate part.

    The judge answers one call with both: a score for each of the ``count``
    parts, in the order they were given, and the overall ``score`` the metric
    already requires. Under forced tool use the schema is the instruction the
    model reads about them, so its descriptions say what a part score measures,
    how the overall score relates to the parts, and that the reasoning covers
    each part before the whole.
    """
    properties = dict(schema.get("properties", {}))
    if "reasoning" in properties:
        properties["reasoning"] = {
            **properties["reasoning"],
            "description": (
                "Step by step reasoning for the evaluation: assess each part in "
                "order, then the response as a whole."
            ),
        }
    properties["part_scores"] = {
        "type": "array",
        "minItems": count,
        "maxItems": count,
        "items": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "description": (
            "Score each part of the candidate response on its own, by the same "
            "criteria and on the same 0.0 to 1.0 scale as the overall score. "
            "One score per part, in the order the parts were given."
        ),
    }
    if "score" in properties:
        properties["score"] = {
            **properties["score"],
            "description": (
                "Overall score for the whole candidate response, weighing how "
                "the parts relate; not necessarily the mean of part_scores."
            ),
        }
    required = list(schema.get("required", []))
    if "part_scores" not in required:
        required.append("part_scores")
    return {**schema, "properties": properties, "required": required}
