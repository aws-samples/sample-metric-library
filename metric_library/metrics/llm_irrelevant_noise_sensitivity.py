# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

from ..core.registry import MetricRegistry
from ..core.llm_metric import LLMMetric
from ..core.input_policy import InputPolicy
from ..core.data import Modality


@MetricRegistry.register
class LLMNoiseSensitivityIrrelevant(LLMMetric):
    """
    Evaluates the system's sensitivity to completely irrelevant context information.

    This metric specifically assesses how much the generated response is negatively
    affected by contexts that are entirely irrelevant to the question being asked.
    This focuses on the model's ability to ignore completely unrelated context
    rather than filtering noise within relevant contexts.

    This metric measures the degradation in response quality when the model is
    presented with contexts that have no bearing on the question. It helps identify
    if the model can maintain response quality despite the presence of completely
    irrelevant distracting information.

    This metric is designed for RAG (Retrieval-Augmented Generation) applications
    to test robustness against retrieval errors that return completely unrelated
    contexts. Lower scores indicate better ability to ignore irrelevant contexts.

    Inspired by RAGChecker's noise_sensitivity_in_irrelevant metric from:
    https://github.com/amazon-science/RAGChecker
    """

    reference_policy = InputPolicy.OMIT
    prompt_policy = InputPolicy.REQUIRED
    context_policy = InputPolicy.REQUIRED

    modality_guidance = {
        Modality.IMAGE: (
            "For images, judge an image in the contexts by the information it "
            "shows, such as values, labels, text and objects, when deciding whether "
            "it is completely irrelevant to the question and whether the response "
            "was affected by it."
        ),
    }

    def rubric(self) -> str:
        """Return the rubric for irrelevant noise sensitivity assessment."""
        return """You are evaluating how much the generated response was negatively affected by completely irrelevant context information.

Focus specifically on contexts that are entirely unrelated to the question being asked. Analyze whether the response was degraded by the presence of this irrelevant information.

Consider these aspects:

1. **Irrelevant Context Identification**: Identify contexts that have no bearing on the question
2. **Distraction Assessment**: Determine if irrelevant contexts distracted the model from the main question
3. **Response Contamination**: Check if the response includes information from irrelevant contexts
4. **Quality Degradation**: Assess if response quality decreased due to irrelevant context presence
5. **Focus Maintenance**: Evaluate if the model maintained focus despite irrelevant distractions

Rate the sensitivity to completely irrelevant contexts on a scale from 0.0 to 1.0:
- 0.0: No negative impact - perfect immunity to irrelevant contexts
- 0.2-0.3: Minimal impact - response slightly affected but maintains core quality
- 0.4-0.5: Moderate impact - some degradation due to irrelevant context presence
- 0.6-0.7: Significant impact - response quality clearly affected by irrelevant contexts
- 0.8-1.0: High impact - response severely degraded or contaminated by irrelevant information

Lower scores are better, indicating good ability to ignore completely irrelevant contexts.
If no contexts are completely irrelevant to the question, return 0.0."""
