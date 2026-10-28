# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

from ..core.registry import MetricRegistry
from ..core.llm_metric import LLMMetric
from ..core.input_policy import InputPolicy
from ..core.data import Modality


@MetricRegistry.register
class LLMCompletenessWithGroundTruth(LLMMetric):
    """
    Uses LLM evaluation to assess completeness of predictions with ground truth comparison.

    This metric measures if the model's response answers every question from the prompt,
    using ground truth response as reference for comparison. The ground truth response
    is considered in the evaluation to assess completeness.

    Responses are normalized to 0.0-1.0 point scale.

    Bedrock Score Mapping:
    - "Not at all": 0.0
    - "Not generally": 0.25
    - "Neutral/Mixed": 0.5
    - "Generally yes": 0.75
    - "Yes": 1.0

    Derived from : https://docs.aws.amazon.com/bedrock/latest/userguide/model-evaluation-type-judge-prompt-claude-sonnet37.html
    """

    reference_policy = InputPolicy.REQUIRED
    prompt_policy = InputPolicy.REQUIRED
    context_policy = InputPolicy.OPTIONAL

    modality_guidance = {
        Modality.IMAGE: (
            "For images, treat what an image depicts, including any visible text, "
            "as the information it provides, and check whether the candidate "
            "response includes every element the question asks for and the "
            "reference response contains. Rendering style is not important."
        ),
    }

    def rubric(self) -> str:
        """Return the rubric that uses ground truth for completeness assessment."""
        return """You are a helpful agent that can assess LLM response according to the given rubrics.

You are given a question, a candidate response from LLM and a reference response. Your task is to check if the candidate response contain the necessary amount of information and details for answering the question.

Please evaluate the completeness of the output based on the following criteria:

1. Does the output address all parts of the input's request?
2. Is any required information missing?
3. For multi-part requests, are all parts fulfilled?
4. Is the level of detail appropriate for the task?
5. For specific requests (e.g., "list 10 items"), does the output meet the exact requirements?
6. For summarization or rewriting tasks, are all main points covered?
7. For step-by-step instructions, are all necessary steps included?
8. Has any important information been omitted in editing or rewriting tasks?

Special consideration for evasive or "I don't know" type responses:
- If the output evades responding or claims lack of knowledge, assess whether this response is justified based on the information available in the input.
- If the output states there isn't enough information in the context, but there actually is sufficient information, rate it as incomplete.
- If there truly isn't enough information in the context to answer the input, and the output acknowledges this, consider it complete.
- Always keep in mind the principle of completeness: Does the output contain all of the necessary information and detail for answering the input, given the available information?

Provide a score between 0.0 and 1.0 where:
- 0.0: Not at all - None of the necessary information and detail is present
- 0.25: Not generally - Less than half of the necessary information and detail is present
- 0.5: Neutral/Mixed - About half of the necessary information and detail is present, or it's unclear what the right amount of information is
- 0.75: Generally yes - Most of the necessary information and detail is present
- 1.0: Yes - All necessary information and detail is present

Remember:
- Focus on completeness, not accuracy or truthfulness.
- Evaluate whether the output addresses the input, even if the information provided is incorrect.
- Consider the appropriate level of detail for the intended audience or specified length.
- For evasive responses, evaluate if the evasion is justified given the available information."""
