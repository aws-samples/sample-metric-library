# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

from ..core.registry import MetricRegistry
from ..core.llm_metric import LLMMetric
from ..core.input_policy import InputPolicy
from ..core.data import Modality


@MetricRegistry.register
class LLMSelfKnowledge(LLMMetric):
    """
    Evaluates the extent to which the generator relies on its internal knowledge
    versus the provided context information.

    This metric assesses whether the generated response is primarily based on the
    model's pre-trained knowledge or if it appropriately defers to and builds upon
    the retrieved context information. In RAG systems, it's crucial to understand
    when the model is using its internal knowledge versus external context.

    This metric helps identify cases where:
    1. The model ignores relevant context in favor of its internal knowledge
    2. The model appropriately combines context with its knowledge
    3. The model relies too heavily on internal knowledge when context should take precedence

    This metric is specifically designed for RAG (Retrieval-Augmented Generation)
    applications where context is provided in the sample execution data. It helps
    optimize the balance between leveraging retrieved information and model knowledge.

    Higher scores indicate greater reliance on internal knowledge (which may be
    problematic in RAG scenarios where context should take precedence).

    Inspired by RAGChecker's self_knowledge metric from:
    https://github.com/amazon-science/RAGChecker
    """

    reference_policy = InputPolicy.OMIT
    prompt_policy = InputPolicy.OPTIONAL
    context_policy = InputPolicy.REQUIRED

    modality_guidance = {
        Modality.IMAGE: (
            "For images, the information an image in the context shows, such as "
            "values, labels, text and objects, is context, not the model's own "
            "knowledge."
        ),
    }

    def rubric(self) -> str:
        """Return the rubric for self-knowledge assessment."""
        return """You are evaluating how much the generated response relies on the model's internal knowledge versus the provided context.

Analyze whether the response primarily draws from the model's pre-trained knowledge or appropriately uses the retrieved context. Consider:

1. **Context Dependence**: Could this response have been generated without the provided context?
2. **Knowledge Source**: Are the facts and information in the response coming from context or model knowledge?
3. **Context Priority**: When context conflicts with potential model knowledge, does the response follow the context?
4. **Novel Information**: Does the response include information that's clearly from the context rather than general knowledge?
5. **Appropriate Balance**: Is there a good balance between context usage and necessary background knowledge?

Rate the self-knowledge reliance on a scale from 0.0 to 1.0:
- 1.0: Entirely relies on internal knowledge - context appears to be ignored completely
- 0.8-0.9: Heavily relies on internal knowledge - minimal context integration
- 0.6-0.7: Moderately relies on internal knowledge - some context usage
- 0.4-0.5: Balanced usage - appropriate mix of context and knowledge
- 0.2-0.3: Primarily context-based - limited reliance on internal knowledge
- 0.0-0.1: Entirely context-dependent - minimal internal knowledge usage

Lower scores are generally better for RAG systems, indicating appropriate deference to retrieved context over internal model knowledge."""
