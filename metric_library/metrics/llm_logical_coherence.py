# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

from ..core.registry import MetricRegistry
from ..core.llm_metric import LLMMetric
from ..core.input_policy import InputPolicy


@MetricRegistry.register
class LLMLogicalCoherence(LLMMetric):
    """
    Uses LLM evaluation to assess logical coherence of predictions.

    This metric looks for logical gaps, inconsistencies, and contradictions in a model's
    responses to a prompt / question.
    Check prediction field / answer field in sample execution data is provided and if yes, use this metric to evaluate the logical coherence of the answer (or step by step reasoning).

    Responses are graded on 0.0-1.0 output range.

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

    def rubric(self) -> str:
        """Return the rubric for logical coherence assessment."""
        return """You are a helpful agent that can assess LLM response according to the given rubrics.

You are given a question and a response from LLM. Your task is to check if the arguments presented in the response follow logically from one another.

When evaluating the logical cohesion of the response, consider the following rubrics:

1. Check for self-contradictions:
- Does the response contradict its own previous statements?
- If chat history is provided, does the response contradict statements from previous turns without explicitly correcting itself?

2. Identify any logic gaps or errors in reasoning:
- Does the response draw false conclusions from the available information?
- Does it make "logical leaps" by skipping steps in an argument?
- Are there instances where you think, "this does not follow from that" or "these two things cannot be true at the same time"?

3. Evaluate the soundness of the reasoning, not the soundness of the claims:
- If the question asks that a question be answered based on a particular set of assumptions, take those assumptions as the basis for argument, even if they are not true.
- Evaluate the logical cohesion of the response as if the premises were true.

4. Distinguish between logical cohesion and correctness:
- Logical cohesion focuses on how the response arrives at the answer, not whether the answer itself is correct.
- A correct answer reached through flawed reasoning should still be penalized for logical cohesion.

5. Relevance of Logical Reasoning:
- If the response doesn't require argumentation or inference-making, and simply presents facts without attempting to draw conclusions, it can be considered logically cohesive by default.
- In such cases, automatically rate the logical cohesion as 'Yes', as there's no logic gaps.

Please rate the logical cohesion of the response based on the following scale and provide a score between 0.0 and 1.0:

- Not at all (0.0): The response contains too many errors of reasoning to be usable, such as contradicting itself, major gaps in reasoning, or failing to present any reasoning where it is required.
- Not generally (0.25): The response contains a few instances of coherent reasoning, but errors reduce the quality and usability.
- Neutral/Mixed (0.5): It's unclear whether the reasoning is correct or not, as different users may disagree. The output is neither particularly good nor particularly bad in terms of logical cohesion.
- Generally yes (0.75): The response contains small issues with reasoning, but the main point is supported and reasonably well-argued.
- Yes (1.0): There are no issues with logical cohesion at all. The output does not contradict itself, and all reasoning is sound."""
