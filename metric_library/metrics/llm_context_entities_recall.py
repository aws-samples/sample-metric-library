# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

from ..core.registry import MetricRegistry
from ..core.llm_metric import LLMMetric
from ..core.input_policy import InputPolicy
from ..core.data import Modality


@MetricRegistry.register
class LLMContextEntitiesRecall(LLMMetric):
    """
    Evaluates the recall of named entities between the retrieved contexts
    and the ground truth answer.

    This metric focuses specifically on named entities (people, places, organizations,
    dates, etc.) mentioned in the ground truth answer and determines what percentage
    of these entities are present in the retrieved contexts.
    This metric is suitable for RAG application that provides context in the sample execution data.

    Derived from https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_entities_recall
    """

    reference_policy = InputPolicy.REQUIRED
    prompt_policy = InputPolicy.OPTIONAL
    context_policy = InputPolicy.REQUIRED

    modality_guidance = {
        Modality.IMAGE: (
            "For images, an entity is mentioned in an image, whether the image is "
            "in the ground truth answer or in a retrieved context, when the image "
            "shows its name, for example in a label, a caption or other visible "
            "text."
        ),
    }

    def rubric(self) -> str:
        """Return the rubric for context entities recall assessment."""
        return """You are evaluating the recall of named entities between the retrieved contexts and the ground truth answer.

Named entities include: people, places, organizations, dates, times, monetary amounts, percentages, products, events, and other proper nouns or specific identifiers.

Follow these steps:
1. Identify all named entities mentioned in the ground truth answer
2. Check which of these entities are present in the retrieved contexts
3. Calculate the percentage of entities from the answer that appear in the contexts

Rate the entity recall on a scale from 0.0 to 1.0:
- 1.0: All entities from the answer are found in the contexts
- 0.8-0.9: Most entities (80-90%) are found in the contexts
- 0.6-0.7: Majority of entities (60-70%) are found in the contexts  
- 0.4-0.5: About half of the entities (40-50%) are found in the contexts
- 0.0-0.3: Few or no entities (0-30%) are found in the contexts

Consider exact matches and reasonable variations (e.g., "John Smith" vs "J. Smith", "United States" vs "US"). Focus only on factual entities, not common words or concepts.

If no named entities are present in the ground truth answer, return a score of 1.0."""
