# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

from ..core.registry import MetricRegistry
from ..core.llm_metric import LLMBatchMetric
from ..core.input_policy import InputPolicy
from ..core.data import Modality


@MetricRegistry.register
class LLMFactualCorrectness(LLMBatchMetric):
    """
    Evaluates the factual accuracy of the generated response by comparing
    it against the ground truth reference.

    This metric focuses specifically on the correctness of factual claims,
    data points, and verifiable information in the response, using the
    reference as the standard for truth verification.

    Derived from https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/factual_correctness/
    """

    reference_policy = InputPolicy.REQUIRED
    prompt_policy = InputPolicy.OPTIONAL
    context_policy = InputPolicy.OPTIONAL

    modality_guidance = {
        Modality.IMAGE: (
            "For images, treat the numbers, labels, text and entities an image "
            "shows, and the relationships it depicts, as its factual claims, and "
            "compare them in the same way. Visual style is not a fact unless it "
            "encodes data, such as the colors in a chart legend."
        ),
    }

    def rubric(self) -> str:
        """Return the rubric for factual correctness assessment."""
        return """You are evaluating the factual correctness of the candidate response by comparing it to the ground truth reference.

Focus specifically on verifiable facts including:
1. **Numerical data**: Statistics, measurements, quantities, percentages, dates
2. **Names and entities**: People, places, organizations, products, events
3. **Relationships**: Cause-effect, hierarchical, temporal relationships
4. **Processes and procedures**: Steps, sequences, methodologies
5. **Scientific/technical facts**: Laws, principles, formulas, technical specifications

Compare each factual claim in the candidate response against the ground truth:
- **Correct**: Fact matches the ground truth exactly or with acceptable variation
- **Incorrect**: Fact contradicts or differs significantly from ground truth  
- **Missing**: Important facts from ground truth are omitted
- **Added**: Facts not in ground truth (evaluate if they're correct or incorrect)

Rate the factual correctness on a scale from 0.0 to 1.0:
- 1.0: All facts are correct and complete compared to ground truth
- 0.8-0.9: Most facts are correct with minor errors or omissions
- 0.6-0.7: Majority of facts are correct with some notable errors
- 0.4-0.5: About half of the facts are correct
- 0.2-0.3: Few facts are correct, significant errors present  
- 0.0: Most or all facts are incorrect or completely missing

Focus on factual accuracy rather than style, completeness of explanation, or opinion-based content."""
