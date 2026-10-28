# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

from ..core.registry import MetricRegistry
from ..core.llm_metric import LLMMetric
from ..core.input_policy import InputPolicy
from ..core.data import Modality


@MetricRegistry.register
class LLMCompletenessWithoutGroundTruth(LLMMetric):
    """
    Uses LLM evaluation to assess completeness of predictions without ground truth comparison.

    This metric measures if the model's response answers every question from the prompt,
    without using any ground truth reference for comparison. The evaluation is based solely
    on whether the response addresses the input comprehensively.

    Responses are normalized to 0.0-1.0 point scale.

    Bedrock Score Mapping:
    - "Not at all": 0.0
    - "Not generally": 0.25
    - "Neutral/Mixed": 0.5
    - "Generally yes": 0.75
    - "Yes": 1.0

    Derived from : https://docs.aws.amazon.com/bedrock/latest/userguide/model-evaluation-type-judge-prompt-claude-sonnet37.html
    """

    reference_policy = InputPolicy.OMIT
    prompt_policy = InputPolicy.REQUIRED
    context_policy = InputPolicy.OPTIONAL

    modality_guidance = {
        Modality.IMAGE: (
            "For images, treat what an image depicts, including any visible text, "
            "as the information it provides: an image in the Input is part of the "
            "request, and an image Output is checked for every element the Input "
            "asks for. Rendering style is not important."
        ),
    }

    def rubric(self) -> str:
        """Return the rubric that assesses completeness without ground truth."""
        return """You are an expert evaluator focusing specifically on assessing the completeness of responses.

You will be presented with an Input (the original request/question) and an Output (the response to be evaluated). Your task is to determine whether an Output contains all the necessary information and detail to properly answer the Input.

Key evaluation principles:
1. Focus only on whether required information is present, not on:
   - Accuracy of information
   - Additional irrelevant information
   - Writing style or coherence

2. Consider an Output incomplete if it:
   - Misses any explicitly requested items
   - Fails to address all parts of multi-part requests
   - Provides insufficient detail for the context
   - Misunderstands or ignores the Input

3. For numbered requests (e.g., "list 10 items"):
   - Missing items lower the completeness rating
   - Exception: If Output explains why full count isn't possible

4. For evasive responses:
   - Evaluate the completeness based on the information available in the context:
     - If the response claims there's not enough information, but the context actually contains sufficient information, rate it as "incomplete".
     - If the response accurately states there's not enough information in the context, and this is true, rate it as "complete".
   - For partially evasive responses that provide some information:
     - Evaluate the completeness of the provided portion based on the available context.

Provide a score between 0.0 and 1.0 where:
- 0.0: Not at all - None of the necessary information/detail present; completely unusable
- 0.25: Not generally - Less than half of necessary information/detail present
- 0.5: Neutral/Mixed - About half of necessary information/detail present, or unclear
- 0.75: Generally yes - Most necessary information/detail present
- 1.0: Yes - All necessary information and detail present"""
