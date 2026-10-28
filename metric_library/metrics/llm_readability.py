# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

from ..core.registry import MetricRegistry
from ..core.llm_metric import LLMMetric
from ..core.input_policy import InputPolicy


@MetricRegistry.register
class LLMReadability(LLMMetric):
    """
    Uses LLM evaluation to assess readability of predictions or the answer.

    This metric looks at the model's responses and evaluates the terminological and
    linguistic complexity of the response.

    Check prediction or answer field in sample execution data is provided and if yes, evaluate the readability of the answer.

    Responses are graded and normalized to 0.0-1.0 range.

    Bedrock Score Mapping:
    - "unreadable": 0.0
    - "poor readability": 0.25
    - "fair readability": 0.5
    - "good readability": 0.75
    - "excellent readability": 1.0

    Derived from : https://docs.aws.amazon.com/bedrock/latest/userguide/model-evaluation-type-judge-prompt-claude-sonnet37.html
    """

    reference_policy = InputPolicy.OMIT
    prompt_policy = InputPolicy.OPTIONAL
    context_policy = InputPolicy.OMIT

    def rubric(self) -> str:
        """Return the rubric for readability assessment."""
        return """You are a helpful agent that can assess LLM response according to the given rubrics.

You are given an LLM response and, when available, the prompt or question that elicited it. Your task is to assess how easily the response can be read and understood.

Provide a score between 0.0 and 1.0 where:
- 0.0: unreadable - The response contains gibberish or could not be comprehended by any normal audience.
- 0.25: poor readability - The response is comprehensible, but it is full of poor readability factors that make comprehension very challenging.
- 0.5: fair readability - The response is comprehensible, but there is a mix of poor readability and good readability factors, so the average reader would need to spend some time processing the text in order to understand it.
- 0.75: good readability - Very few poor readability factors. Mostly clear, well-structured sentences. Standard vocabulary with clear context for any challenging words. Clear organization with topic sentences and supporting details. The average reader could comprehend by reading through quickly one time.
- 1.0: excellent readability - No poor readability factors. Consistently clear, concise, and varied sentence structures. Simple, widely understood vocabulary. Logical organization with smooth transitions between ideas. The average reader may be able to skim the text and understand all necessary points."""
