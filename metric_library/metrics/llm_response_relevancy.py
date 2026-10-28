# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

from ..core.registry import MetricRegistry
from ..core.llm_metric import LLMMetric
from ..core.input_policy import InputPolicy
from ..core.data import Modality


@MetricRegistry.register
class LLMResponseRelevancy(LLMMetric):
    """
    Evaluates how well the generated response addresses the specific question asked.

    This metric assesses the relevance of the response to the original question,
    considering directness, completeness, focus, and clarity of the answer in
    relation to what was specifically asked.

    Derived from https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/answer_relevance/
    """

    reference_policy = InputPolicy.OMIT
    prompt_policy = InputPolicy.REQUIRED
    context_policy = InputPolicy.OMIT

    modality_guidance = {
        Modality.IMAGE: (
            "For images, an image in the question is part of the question. For an "
            "image answer, apply the same criteria to what it depicts, including "
            "any visible text, and consider whether an image is an appropriate type "
            "of answer; rendering style matters only when the question asks for it."
        ),
    }

    def rubric(self) -> str:
        """Return the rubric for response relevancy assessment."""
        return """You are evaluating how well the given answer addresses the specific question asked.

Evaluate the answer based on these key criteria:

1. **Directness**: Does the answer directly address what was asked? Does it provide a clear response to the main question?

2. **Completeness**: Does the answer fully address all aspects and parts of the question? Are there missing components?

3. **Focus**: Does the answer stay focused on the question without including irrelevant information or going off-topic?

4. **Appropriateness**: Is the level of detail and type of response appropriate for the question asked?

Rate the response relevancy on a scale from 0.0 to 1.0:
- 1.0: Perfect relevance - directly, completely, and clearly answers the question
- 0.8-0.9: High relevance - addresses the question very well with minor issues
- 0.6-0.7: Good relevance - addresses the question adequately but with some gaps or extra content
- 0.4-0.5: Moderate relevance - partially addresses the question with notable issues
- 0.2-0.3: Low relevance - barely addresses the question or misses key aspects
- 0.0: No relevance - completely fails to address the question

Consider whether someone asking this question would be satisfied with the response provided."""
