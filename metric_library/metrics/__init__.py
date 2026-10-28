# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

from .accuracy import Accuracy
from .mean_absolute_error import MeanAbsoluteError
from .mean_squared_error import MeanSquaredError
from .root_mean_squared_error import RootMeanSquaredError
from .llm_judge import LLMJudge
from .llm_correctness import LLMCorrectness
from .llm_helpfulness import LLMHelpfulness
from .llm_relevance import LLMRelevance
from .llm_semantic_evaluation_idp import (
    LLMSemanticEvaluationIDP as LLMSemanticEvaluation,
)


# Import all match-based metrics
# This registers all dynamic combinations automatically
from .match import DynamicMatch


# RAG-specific metrics
from .llm_context_precision import LLMContextPrecision
from .llm_context_recall import LLMContextRecall
from .llm_context_entities_recall import LLMContextEntitiesRecall
from .llm_context_utilization import LLMContextUtilization
from .llm_noise_sensitivity import LLMNoiseSensitivity
from .llm_relevant_noisesensitivity import LLMNoiseSensitivityRelevant
from .llm_irrelevant_noise_sensitivity import LLMNoiseSensitivityIrrelevant
from .llm_response_relevancy import LLMResponseRelevancy
from .llm_hallucination_detection import LLMHallucinationDetection
from .llm_factual_correctness import LLMFactualCorrectness
from .llm_self_knowledge import LLMSelfKnowledge

# Bedrock built-in LLM metrics
from .llm_logical_coherence import LLMLogicalCoherence
from .llm_faithfulness import LLMFaithfulness
from .llm_following_instructions import LLMFollowingInstructions
from .llm_completeness_with_ground_truth import LLMCompletenessWithGroundTruth
from .llm_completeness_without_ground_truth import LLMCompletenessWithoutGroundTruth
from .llm_professional_style_tone import LLMProfessionalStyleTone
from .llm_readability import LLMReadability
from .llm_stereotyping import LLMStereotyping
from .llm_harmfulness import LLMHarmfulness
from .llm_refusal import LLMRefusal

__all__ = [
    "Accuracy",
    "MeanAbsoluteError",
    "MeanSquaredError",
    "RootMeanSquaredError",
    "LLMJudge",
    "LLMCorrectness",
    "LLMHelpfulness",
    "LLMRelevance",
    "LLMSemanticEvaluation",
    "DynamicMatch",
    # RAG-specific metrics
    "LLMContextPrecision",
    "LLMContextRecall",
    "LLMContextEntitiesRecall",
    "LLMContextUtilization",
    "LLMNoiseSensitivity",
    "LLMNoiseSensitivityRelevant",
    "LLMNoiseSensitivityIrrelevant",
    "LLMResponseRelevancy",
    "LLMHallucinationDetection",
    "LLMFactualCorrectness",
    "LLMSelfKnowledge",
    # Bedrock built-in LLM metrics
    "LLMLogicalCoherence",
    "LLMFaithfulness",
    "LLMFollowingInstructions",
    "LLMCompletenessWithGroundTruth",
    "LLMCompletenessWithoutGroundTruth",
    "LLMProfessionalStyleTone",
    "LLMReadability",
    "LLMStereotyping",
    "LLMHarmfulness",
    "LLMRefusal",
]
