# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

from ..core.registry import MetricRegistry
from ..core.llm_metric import LLMMetric
from ..core.input_policy import InputPolicy


@MetricRegistry.register
class LLMProfessionalStyleTone(LLMMetric):
    """
    Uses LLM evaluation to assess professional style and tone of predictions / answer.

    This metric looks at the model's responses and decides if the style, formatting,
    and tone of a response is appropriate for professional genres.

    Check prediction field / answer field in sample execution data is provided and if yes, evaluate the tone of the answer.

    Responses are graded and normalized to 0.0-1.0 range.

    Bedrock Score Mapping:
    - "not at all": 0.0
    - "not generally": 0.25
    - "neutral/mixed": 0.5
    - "generally yes": 0.75
    - "completely yes": 1.0

    Derived from : https://docs.aws.amazon.com/bedrock/latest/userguide/model-evaluation-type-judge-prompt-claude-sonnet37.html
    """

    reference_policy = InputPolicy.OMIT
    prompt_policy = InputPolicy.OPTIONAL
    context_policy = InputPolicy.OMIT

    def rubric(self) -> str:
        """Return the rubric for professional style and tone assessment."""
        return """You are a helpful agent that can assess LLM response according to the given rubrics.

You are given an LLM response and, when available, the prompt or question that elicited it. Your task is to assess the response's professional style and tone. In other words, assess whether it is written like something people might see in a company-wide memo at a corporate office.

A professional style has correct spelling and grammar, standard capitalization and punctuation, and a neutral to friendly and formal tone. A professional style is how one is expected to write in a professional setting, such as on a cover letter or a business memo.

A professional piece of text should have a neutral to slightly friendly tone, and be moderately formal. Style should be penalized if the output is silly, angry, rude. Text could even be penalized even for being overly formal.

You can ask yourself "If I read text like this in an email from my employer to a customer, would I be embarrassed for the person who wrote it?" If the answer is yes, this likely does not exemplify a professional style.

A variety of factors contribute to the professional style and tone of a response. Here is an example of text with good professional style and tone: "I am writing in regards to the meeting this morning." The following is a list of less professional versions of it with explanations about what makes the version less professional.

1. "I am writing in regards to eht meeting this morning." This example has issues in spelling as to professional style and tone: Misspelled words make the text less professional.
2. "writing in regards to the meeting this morning". This example has issues in grammar as to professional style and tone: Dropping the subject "I" makes the text less professional.
3. "i am writing in regards to the MeEtInG this morning." This example has issues in capitalization as to professional style and tone: Professional text should use standard capitalization.
4. "I am writing in regards to the meeting this morning I have a few points I'd like to follow up on". This example has issues in punctuation as to professional style and tone: Not adding periods when a sentence ends makes a run-on sentence, which is less professional.
5. "I'm hitting you up about the shindig this morning." This example has issues in word choice as to professional style and tone: "hitting you up" and "shindig" are less professional than their counterparts in the example sentence with good professional style and tone given above.
6. "In regards to the meeting this morning, I write." This example has issues in sentence construction as to professional style and tone: Moving "I write" to the end makes the text sound antiquated or silly and less suited for a professional environment
7. "Heyyy so about that meeting this morning 🙄 am i right?" This example has issues in the tone being unprofessional: It uses an informal, joking, or silly tone which makes a text less professional.

Provide a score between 0.0 and 1.0 where:
- 0.0: not at all - The response has major elements of style and/or tone that do not fit a professional setting. Almost none of it is professional.
- 0.25: not generally - The response has some elements that would fit a professional setting, but most of it does not.
- 0.5: neutral/mixed - The response is a roughly even mix of professional and unprofessional elements.
- 0.75: generally yes - The response almost entirely fits a professional setting.
- 1.0: completely yes - The response absolutely fits a professional setting. There is nothing that you would change in order to make this fit a professional setting."""
