# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

from ..core.registry import MetricRegistry
from ..core.llm_metric import LLMMetric
from ..core.input_policy import InputPolicy
from ..core.data import Modality


@MetricRegistry.register
class LLMContextPrecision(LLMMetric):
    """
    Evaluates the precision of retrieved contexts by determining if each context
    was useful in arriving at the given answer.

    This metric assesses whether the retrieved contexts normally in the Agentic RAG scenario contain relevant information
    that directly contributes to generating the answer. It calculates average precision
    by evaluating each context's usefulness and considering the ranking order.
    This metric is suitable for RAG application that provides context in the sample execution data.
    Check if context is provided in the sample execution data if yes, use this metric to evaluate the quality of these contexts.

    Derived from https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/
    """

    reference_policy = InputPolicy.OMIT
    prompt_policy = InputPolicy.REQUIRED
    context_policy = InputPolicy.REQUIRED

    modality_guidance = {
        Modality.IMAGE: (
            "For images, the information an image shows, such as values, labels, "
            "text and objects, counts like the same information written in text: an "
            "image in the context was useful when the answer draws on what it "
            "shows."
        ),
    }

    def rubric(self) -> str:
        """Return the rubric for context precision assessment."""
        return """You are evaluating whether the provided context was useful in arriving at the given answer.

Given a question, answer, and context, determine if the context contains information that directly helped generate the answer.

Consider the following criteria:
1. Does the context contain facts, data, or information mentioned in the answer?
2. Does the context provide relevant background information that supports the answer?
3. Would it be difficult or impossible to generate this specific answer without this context?
4. Is the information in the context directly related to the question being asked?

Rate the usefulness on a scale from 0.0 to 1.0:
- 1.0: The context was highly useful and directly contributed to the answer
- 0.8-0.9: The context was useful with most information being relevant
- 0.6-0.7: The context was moderately useful with some relevant information
- 0.4-0.5: The context was somewhat useful but mostly irrelevant
- 0.0-0.3: The context was not useful or completely irrelevant

Focus on whether the context actually helped in generating the specific answer provided, not just whether it's related to the general topic."""
