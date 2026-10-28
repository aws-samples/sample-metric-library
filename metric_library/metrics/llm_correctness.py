# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

from ..core.data import Modality
from ..core.input_policy import InputPolicy
from ..core.llm_metric import LLMBatchMetric
from ..core.multiplicity import Multiplicity
from ..core.registry import MetricRegistry


@MetricRegistry.register
class LLMCorrectness(LLMBatchMetric):
    """
    Uses LLM evaluation to assess correctness of predictions compared to references.

    This metric supports both text and image data, using an LLM to evaluate whether
    the prediction matches the essential elements of the reference. It returns a score
    between 0.0 and 1.0 based on a direct score from LLM response

    The evaluation is done through structured JSON responses that include reasoning
    for the assessment, making the metric's decisions transparent and interpretable.

    It declares guidance for judging images, added after the rubric when the
    task, the candidate or the ground truth contains an image. It reads a
    candidate or a ground truth that arrives in several parts, such as an image
    and its description, as one whole.
    """

    reference_policy = InputPolicy.REQUIRED
    prompt_policy = InputPolicy.REQUIRED
    context_policy = InputPolicy.OPTIONAL

    modality_guidance = {
        Modality.IMAGE: (
            "For images, judge the depicted content: an image in the task is part "
            "of the task, and an image candidate or ground truth is compared by "
            "what it depicts. Translation, rotation, scale and rendering style are "
            "not important."
        ),
    }

    prediction_multiplicity = Multiplicity.COMPOSITE
    reference_multiplicity = Multiplicity.COMPOSITE

    def rubric(self) -> str:
        """Return the rubric for correctness assessment."""
        return """You are given a task, a candidate answer and a ground truth answer. Assess whether the candidate answer is a correct and accurate response to the task.

You may use the ground truth answer as a reference of what a correct answer should contain. It is okay if the candidate answer diverges if the response is short or long response, not multiple choice; if the essential points are mentioned then the candidate answer is correct.

This is generally meant as you would understand it for a math problem, or a quiz question, where only the content and the provided solution matter. Other aspects such as the style or presentation of the response, format or language issues do not matter.

Your evaluation should use the ground truth answer; the candidate response is correct even if it is missing explanations or is not truthful, as long as it aligns with the ground truth. However, it is not necessarily that the candidate response should be an exact match of the ground truth; if the essential points are mentioned, then it is correct."""
