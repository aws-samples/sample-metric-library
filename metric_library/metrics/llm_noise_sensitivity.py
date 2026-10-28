# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

from ..core.registry import MetricRegistry
from ..core.llm_metric import LLMMetric
from ..core.input_policy import InputPolicy
from ..core.data import Modality


@MetricRegistry.register
class LLMNoiseSensitivity(LLMMetric):
    """
    Evaluates the system's robustness to irrelevant or noisy context.

    This metric assesses how well the system can maintain answer quality
    when presented with irrelevant contexts mixed with relevant ones in the Agentic RAG scenario contain relevant information.
    Lower scores indicate better noise resistance (less sensitivity to irrelevant context).
    This metric is suitable for RAG application that provides context in the sample execution data.

    Derived from https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/noise_sensitivity/
    """

    reference_policy = InputPolicy.OMIT
    prompt_policy = InputPolicy.REQUIRED
    context_policy = InputPolicy.REQUIRED

    modality_guidance = {
        Modality.IMAGE: (
            "For images, judge an image in the contexts by the information it "
            "shows, such as values, labels, text and objects, when deciding whether "
            "it is relevant or noise and whether the answer was affected by it."
        ),
    }

    def rubric(self) -> str:
        """Return the rubric for noise sensitivity assessment."""
        return """You are evaluating how much the answer was negatively affected by irrelevant or noisy context.

Analyze whether the generated answer was degraded, confused, or distracted by irrelevant information in the contexts. Consider:

1. Does the answer include information from irrelevant contexts that shouldn't be there?
2. Is the answer less focused or coherent due to noise in the contexts?
3. Does the answer fail to prioritize relevant information over irrelevant information?
4. Is the overall quality of the answer reduced due to the presence of irrelevant context?

Rate the noise sensitivity (negative impact) on a scale from 0.0 to 1.0:
- 0.0: No negative impact from irrelevant context - perfect noise resistance
- 0.2-0.3: Minimal impact - answer slightly less focused but still accurate
- 0.4-0.5: Moderate impact - answer quality somewhat degraded by noise
- 0.6-0.7: Significant impact - answer clearly affected by irrelevant information
- 0.8-1.0: High impact - answer severely degraded or confused by noise

Note: Lower scores are better for this metric, indicating better robustness to noise.
If all contexts appear relevant to the question, return 0.0 as there's no noise to be sensitive to.
"""
