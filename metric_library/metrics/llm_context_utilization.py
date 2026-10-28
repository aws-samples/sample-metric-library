# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

from ..core.registry import MetricRegistry
from ..core.llm_metric import LLMMetric
from ..core.input_policy import InputPolicy
from ..core.data import Modality


@MetricRegistry.register
class LLMContextUtilization(LLMMetric):
    """
    Evaluates how effectively the generator utilizes the retrieved context information.

    This metric assesses whether the generated response makes appropriate use of the
    relevant information provided in the retrieved contexts. It measures the degree
    to which the generator incorporates, references, and builds upon the context
    rather than ignoring it or providing responses that could have been generated
    without the context.

    This metric is specifically designed for RAG (Retrieval-Augmented Generation)
    applications where context is provided in the sample execution data. It helps
    identify whether the generator is effectively leveraging the retrieved information
    to enhance response quality and accuracy.

    Higher scores indicate better utilization of available context information.

    Inspired by RAGChecker's context_utilization metric from:
    https://github.com/amazon-science/RAGChecker
    """

    reference_policy = InputPolicy.OMIT
    prompt_policy = InputPolicy.OPTIONAL
    context_policy = InputPolicy.REQUIRED

    modality_guidance = {
        Modality.IMAGE: (
            "For images, the information an image in the context shows, such as "
            "values, labels, text and objects, is context information the response "
            "can use, like the same information written in text."
        ),
    }

    def rubric(self) -> str:
        """Return the rubric for context utilization assessment."""
        return """You are evaluating how effectively the generated response utilizes the provided context information.

Analyze whether the response makes good use of the relevant information available in the retrieved contexts. Consider:

1. **Information Integration**: Does the response incorporate specific facts, data, or details from the context?
2. **Contextual References**: Are there clear connections between the context content and the response content?
3. **Added Value**: Does the context enhance the response quality beyond what could be generated without it?
4. **Relevance Usage**: Is relevant context information appropriately utilized while irrelevant context is ignored?
5. **Completeness**: Does the response leverage all the useful context information available?

Rate the context utilization on a scale from 0.0 to 1.0:
- 1.0: Excellent utilization - response fully leverages all relevant context information
- 0.8-0.9: Good utilization - response uses most relevant context effectively
- 0.6-0.7: Moderate utilization - response uses some context but misses opportunities
- 0.4-0.5: Poor utilization - response barely incorporates context information
- 0.0-0.3: No utilization - response appears to ignore available context

Focus on whether the generator is making effective use of the provided context to create a better, more informed response than would be possible without that context."""
