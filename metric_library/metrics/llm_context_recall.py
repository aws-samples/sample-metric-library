# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

from ..core.registry import MetricRegistry
from ..core.llm_metric import LLMMetric
from ..core.input_policy import InputPolicy
from ..core.data import Modality


@MetricRegistry.register
class LLMContextRecall(LLMMetric):
    """
    Evaluates whether the retrieved contexts contain all necessary information
    to generate the expected answer.

    This metric assesses the completeness of retrieved contexts by checking if
    they cover all the key information points present in the ground truth answer.
    It identifies gaps in context coverage that might lead to incomplete responses.
    This metric is suitable for RAG application that provides context in the sample execution data.
    Check if context is provided in the sample execution data if yes, use this metric to evaluate the quality of these contexts.

    Derived from https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_recall/
    """

    reference_policy = InputPolicy.REQUIRED
    prompt_policy = InputPolicy.OPTIONAL
    context_policy = InputPolicy.REQUIRED

    modality_guidance = {
        Modality.IMAGE: (
            "For images, the information an image shows, such as values, labels, "
            "text and objects, counts like the same information written in text: an "
            "information point is covered when a context image shows it, and an "
            "image in the expected answer contributes the points it shows."
        ),
    }

    def rubric(self) -> str:
        """Return the rubric for context recall assessment."""
        return """You are evaluating whether the provided contexts contain all the necessary information to generate the expected answer.

Analyze the expected answer and identify all key information points it contains. Then determine what percentage of these information points can be found in or reasonably inferred from the retrieved contexts.

Follow these steps:
1. Break down the expected answer into distinct key information points or facts
2. For each information point, check if it's supported by the contexts
3. Consider both explicit information and reasonable inferences from the contexts
4. Calculate the coverage percentage

Rate the context coverage on a scale from 0.0 to 1.0:
- 1.0: All information points in the answer are covered by the contexts
- 0.8-0.9: Most information points are covered, minor gaps
- 0.6-0.7: Majority of information points are covered, some notable gaps
- 0.4-0.5: About half of the information points are covered
- 0.0-0.3: Few or none of the information points are covered by contexts

Focus on whether someone could generate the complete expected answer using only the information available in the provided contexts."""
