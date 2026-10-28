# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

from ..core.registry import MetricRegistry
from ..core.llm_metric import LLMBatchMetric
from ..core.input_policy import InputPolicy
from ..core.data import Modality


@MetricRegistry.register
class LLMJudge(LLMBatchMetric):
    """
    Uses an LLM to compare two inputs and evaluate their similarity or quality.

    LLMJudge queries a large language model to assess the similarity between prediction
    and reference data, returning a score between 0.0 and 1.0. This metric is flexible
    and can handle various types of data by leveraging the LLM's multimodal capabilities.
    """

    reference_policy = InputPolicy.REQUIRED
    prompt_policy = InputPolicy.OPTIONAL
    context_policy = InputPolicy.OPTIONAL

    modality_guidance = {
        Modality.IMAGE: (
            "For images, compare what each input depicts, including any visible "
            "text and how the content is laid out. When one input is an image and "
            "the other is text, compare what the image depicts with what the text "
            "describes. Differences in resolution, compression or file format are "
            "not important."
        ),
    }

    def rubric(self) -> str:
        """Return the rubric for similarity assessment."""
        return """You are evaluating the similarity between two inputs.
        
Rate the similarity between the two inputs on a scale from 0.0 to 1.0,
where 0.0 is completely different and 1.0 is perfect/identical.

Consider the following when evaluating:
- Content similarity and semantic meaning
- Structural similarity (if applicable)
- Overall quality and relevance"""
