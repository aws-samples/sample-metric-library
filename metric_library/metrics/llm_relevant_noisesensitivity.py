# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

from ..core.registry import MetricRegistry
from ..core.llm_metric import LLMMetric
from ..core.input_policy import InputPolicy
from ..core.data import Modality


@MetricRegistry.register
class LLMNoiseSensitivityRelevant(LLMMetric):
    """
    Evaluates the system's sensitivity to noise within relevant context information.

    This metric specifically assesses how much the generated response is negatively
    affected by noise or irrelevant information mixed within contexts that are
    otherwise relevant to the question. This focuses on the model's ability to
    filter out noise within useful context rather than completely irrelevant context.

    This metric measures the degradation in response quality when relevant contexts
    contain some noisy or irrelevant information mixed in. It helps identify if
    the model can effectively extract useful information from contexts that contain
    both relevant and irrelevant content.

    This metric is designed for RAG (Retrieval-Augmented Generation) applications
    where context contains a mix of relevant and irrelevant information. Lower
    scores indicate better noise resistance within relevant contexts.

    Inspired by RAGChecker's noise_sensitivity_in_relevant metric from:
    https://github.com/amazon-science/RAGChecker
    """

    reference_policy = InputPolicy.OMIT
    prompt_policy = InputPolicy.REQUIRED
    context_policy = InputPolicy.REQUIRED

    modality_guidance = {
        Modality.IMAGE: (
            "For images, judge an image in the contexts by the information it "
            "shows, such as values, labels, text and objects, when separating its "
            "useful information from noise and deciding whether the response was "
            "affected by the noise."
        ),
    }

    def rubric(self) -> str:
        """Return the rubric for relevant noise sensitivity assessment."""
        return """You are evaluating how much the generated response was negatively affected by noise or irrelevant information within otherwise relevant contexts.

Focus specifically on contexts that contain useful information for answering the question, but also include some irrelevant details or noise. Analyze whether the response was degraded by this mixed content.

Consider these aspects:

1. **Relevant Context Identification**: Identify contexts that contain information useful for the question
2. **Noise Detection**: Within relevant contexts, identify irrelevant details, tangential information, or distracting content
3. **Impact Assessment**: Determine if the response quality was degraded by the noise within relevant contexts
4. **Focus Degradation**: Check if the model was distracted by irrelevant details within useful contexts
5. **Information Extraction**: Assess if the model could extract the relevant parts while ignoring noise

Rate the sensitivity to noise within relevant contexts on a scale from 0.0 to 1.0:
- 0.0: No negative impact - perfect filtering of noise within relevant contexts
- 0.2-0.3: Minimal impact - response slightly affected but maintains quality
- 0.4-0.5: Moderate impact - some degradation due to noise in relevant contexts
- 0.6-0.7: Significant impact - response quality clearly affected by mixed content
- 0.8-1.0: High impact - response severely degraded by noise within relevant contexts

Lower scores are better, indicating good ability to filter noise within relevant contexts.
If no relevant contexts contain noise, or if all contexts are entirely irrelevant, return 0.0."""
