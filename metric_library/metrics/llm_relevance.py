# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

from ..core.registry import MetricRegistry
from ..core.llm_metric import LLMMetric
from ..core.input_policy import InputPolicy
from ..core.data import Modality


@MetricRegistry.register
class LLMRelevance(LLMMetric):
    """
    Uses LLM evaluation to assess relevance of predictions compared to references.

    This metric supports both text and image data, using an LLM to evaluate the
    relevance of the prediction. It returns a score between 0.0 and 1.0.

    The evaluation is done through structured JSON responses that include reasoning
    for the assessment, making the metric's decisions transparent and interpretable.
    """

    reference_policy = InputPolicy.OMIT
    prompt_policy = InputPolicy.REQUIRED
    context_policy = InputPolicy.OMIT

    modality_guidance = {
        Modality.IMAGE: (
            "For images, an image in the question is part of the question. For an "
            "image response, judge how much of what it depicts, including any "
            "visible text, addresses the question and how much is unrelated to it; "
            "rendering style matters only when the question asks for it."
        ),
    }

    def rubric(self) -> str:
        """Return the rubric for relevance assessment."""
        return """You are a helpful agent that can assess LLM response according to the given rubrics.

You are given a question and a response from LLM. Your task is to assess the relevance of the LLM response to the question, in other words, how focused the LLM response is on the given question.

Please rate the relevance of the response based on the following considerations:
- Does the response directly address the question asked?
- Is the information provided relevant to the topic?
- How much of the response is focused on the question vs. irrelevant content?

Rate the relevance on a scale from 0.0 to 1.0, where:
- 0.0: No part of the response is relevant to the question
- 0.2: An overwhelming amount of the response is irrelevant or the relevant information is not a direct answer
- 0.5: Roughly half of the response is relevant to the question
- 0.8: An overwhelming amount of the response is relevant to the question
- 1.0: Every piece of the response is relevant to the question"""
