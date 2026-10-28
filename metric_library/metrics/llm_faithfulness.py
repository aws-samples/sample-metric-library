# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

from ..core.registry import MetricRegistry
from ..core.llm_metric import LLMMetric
from ..core.input_policy import InputPolicy
from ..core.data import Modality


@MetricRegistry.register
class LLMFaithfulness(LLMMetric):
    """
    Uses LLM evaluation to assess faithfulness of predictions to the context.

    This metric looks at whether the response contains information not found in the prompt,
    that cannot be inferred easily from the prompt.
    Check prediction field / answer field in sample execution data is provided and if yes, use this metric to evaluate the faithfulness of predictions.
    Responses are graded on a 0-1 point scale.

    Bedrock Score Mapping:
    - "none is faithful": 0.0
    - "some is faithful": 0.25
    - "approximately half is faithful": 0.5
    - "most is faithful": 0.75
    - "all is faithful": 1.0

    Derived from : https://docs.aws.amazon.com/bedrock/latest/userguide/model-evaluation-type-judge-prompt-claude-sonnet37.html
    """

    reference_policy = InputPolicy.OMIT
    prompt_policy = InputPolicy.REQUIRED
    context_policy = InputPolicy.REQUIRED

    modality_guidance = {
        Modality.IMAGE: (
            "For images, the information an image in the Input shows, such as "
            "values, labels, text and objects, is part of the Input like the same "
            "information written in text; a claim that contradicts what the image "
            "shows is not faithful."
        ),
    }

    def rubric(self) -> str:
        """Return the rubric for faithfulness assessment."""
        return """You are given a task in some context (Input), and a candidate answer. Does the candidate answer contain any hallucinations or information that contradicts the information in the Input (task description and context)?

Hallucinations exist ONLY when the task asks to respond based on the context, otherwise the model is allowed to use its own knowledge to provide a response. Even if a claim is not verifiable, it is NOT a hallucination unless it (1) contradicts the context, or (2) the task demands the response to be based on the context, like in a summarization task.

Evaluate how much of the information in the answer is faithful to the available context (it is not a contradiction or hallucination).

Provide a score between 0.0 and 1.0 where:
- 0.0: none is faithful
- 0.25: some is faithful  
- 0.5: approximately half is faithful
- 0.75: most is faithful
- 1.0: all is faithful"""
