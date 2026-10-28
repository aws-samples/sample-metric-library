# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import logging
from ..core.registry import MetricRegistry
from ..core.llm_metric import LLMBatchMetric
from ..core.input_policy import InputPolicy

logger = logging.getLogger(__name__)


@MetricRegistry.register
class LLMSemanticEvaluationIDP(LLMBatchMetric):
    """
    Advanced LLM-based semantic evaluation designed for IDP (Intelligent Document Processing) with detailed reasoning and context awareness.

    The judge decides whether a predicted attribute value extracted from a
    document matches the expected value in meaning, allowing for formatting
    variations, abbreviations, word order and alternative representations, and
    explains the decision in its reasoning. The optional prompt and context can
    carry the document class, the attribute definition or the document itself.
    """

    reference_policy = InputPolicy.REQUIRED
    prompt_policy = InputPolicy.OPTIONAL
    context_policy = InputPolicy.OPTIONAL

    def rubric(self) -> str:
        """
        Return the rubric for semantic assessment.

        The predicted value reaches the judge as the candidate response and
        the expected value as the ground truth response.
        """
        return """You are an expert evaluator that helps determine if predicted and expected values match for document attribute extraction. 
   You excel at understanding semantic equivalence beyond exact string matching.

Your role is to:
1. Analyze the meaning and context of both values
2. Consider formatting variations and alternative representations
3. Evaluate semantic equivalence rather than exact matches
4. Provide clear reasoning for your decisions

The predicted value is the candidate response and the expected value is the ground truth response. Do these values match in meaning, taking into account formatting differences, word order, abbreviations, and semantic equivalence?

Consider the following when evaluating:
- Semantic meaning and content similarity
- Formatting variations (dates, numbers, punctuation)
- Abbreviations and alternative representations
- Word order and phrasing differences
- Context-appropriate equivalence

Provide a score where:
- 1.0: the predicted and expected values match in meaning
- 0.0: they do not match in meaning
- A value in between only when the values match in part, such as a list of items of which only some match
"""
