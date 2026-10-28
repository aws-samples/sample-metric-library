# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

from ..core.registry import MetricRegistry
from ..core.llm_metric import LLMMetric
from ..core.input_policy import InputPolicy


@MetricRegistry.register
class LLMHelpfulness(LLMMetric):
    """
    Uses LLM evaluation to assess helpfulness of predictions compared to references.

    This metric supports both text and image data, using an LLM to evaluate the
    helpfulness of the prediction. It returns a score between 0.0 and 1.0 based on
    the scored LLM response.

    The evaluation is done through structured JSON responses that include reasoning
    for the assessment, making the metric's decisions transparent and interpretable.
    """

    reference_policy = InputPolicy.OMIT
    prompt_policy = InputPolicy.REQUIRED
    context_policy = InputPolicy.OPTIONAL

    def rubric(self) -> str:
        """Return the rubric for helpfulness assessment."""
        return """You are given a task and a candidate completion. Provide a holistic evaluation of how helpful the completion is taking the below factors into consideration.

Helpfulness can be seen as 'eager and thoughtful cooperation': a completion is helpful when it satisfies explicit and implicit expectations in the user's request. Often this will mean that the completion helps the user achieve the task.

When the request is not clearly a task, like a random text continuation, or an answer directly to the model, consider what the user's general motifs are for making the request.

Not all factors will be applicable for every kind of request. For the factors applicable, the more you would answer with yes, the more helpful the completion:

* Is the completion sensible, coherent, and clear given the current context, and/or what was said previously?
* If the goal is to solve a task, does the completion solve the task?
* Does the completion follow instructions, if provided?
* Does the completion respond with an appropriate genre, style, modality (text/image/code/etc)?
* Does the completion respond in a way that is appropriate for the target audience?
* Is the completion as specific or general as necessary?
* Is the completion as concise as possible or as elaborate as necessary?
* Does the completion avoid unnecessary content and formatting that would make it harder for the user to extract the information they are looking for?
* Does the completion anticipate the user's needs and implicit expectations? e.g. how to deal with toxic content, dubious facts; being sensitive to internationality
* When desirable, is the completion interesting? Is the completion likely to "catch someone's attention" or "arouse their curiosity", or is it unexpected in a positive way, witty or insightful? when not desirable, is the completion plain, sticking to a default or typical answer or format?
* For math, coding, and reasoning problems: is the solution simple, and efficient, or even elegant?
* For chat contexts: is the completion a single chatbot turn marked by an appropriate role label?

Rate the helpfulness on a scale from 0.0 to 1.0, where 0.0 is not helpful at all and 1.0 is above and beyond helpful."""
