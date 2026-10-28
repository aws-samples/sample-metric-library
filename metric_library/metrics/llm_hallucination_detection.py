# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

from ..core.registry import MetricRegistry
from ..core.llm_metric import LLMMetric
from ..core.input_policy import InputPolicy
from ..core.data import Modality


@MetricRegistry.register
class LLMHallucinationDetection(LLMMetric):
    """
    Detects when the generated response contains hallucinated information
    not present in the source contexts.

    This metric specifically identifies fabricated facts, invented details,
    or unsupported claims that cannot be verified from the provided contexts.
    Higher scores indicate more severe hallucinations.
    This metric is suitable for RAG application that provides context in the sample execution data.

    Derived from https://docs.raga.ai/ragaai-catalyst/ragaai-metric-library/rag-metrics/hallucination
    """

    reference_policy = InputPolicy.OMIT
    prompt_policy = InputPolicy.OPTIONAL
    context_policy = InputPolicy.REQUIRED

    modality_guidance = {
        Modality.IMAGE: (
            "For images, the information an image in the contexts shows, such as "
            "values, labels, text and objects, supports the response like the same "
            "information written in text; a claim about an image that the image "
            "does not show is a hallucination."
        ),
    }

    def rubric(self) -> str:
        """Return the rubric for hallucination detection."""
        return """You are detecting hallucinations in the generated response - information that is fabricated or not supported by the provided contexts.

Carefully examine the response for these types of hallucinations:

1. **Factual hallucinations**: Specific facts, numbers, dates, or claims not present in contexts
2. **Entity hallucinations**: Made-up names, places, organizations not mentioned in contexts  
3. **Relationship hallucinations**: Connections or relationships not established in contexts
4. **Detail hallucinations**: Specific details or explanations not supported by contexts
5. **Attribution hallucinations**: Incorrect citations or source attributions

Rate the hallucination severity on a scale from 0.0 to 1.0:
- 0.0: No hallucinations detected - all information is supported by contexts
- 0.1-0.2: Minor hallucinations - small unsupported details that don't affect main content
- 0.3-0.4: Moderate hallucinations - some notable unsupported claims
- 0.5-0.6: Significant hallucinations - multiple unsupported facts or claims
- 0.7-0.8: Severe hallucinations - major portions of response are unsupported
- 0.9-1.0: Extreme hallucinations - response is mostly fabricated information

Be strict in your evaluation - if information cannot be directly verified or reasonably inferred from the contexts, consider it a hallucination. Focus on factual accuracy and verifiability."""
