# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import ast
import re
import logging
import math
from enum import Enum
from typing import List, Union, TypeVar, Optional
from ..core.bedrock_client import BedrockClient

# Import munkres for Hungarian algorithm implementation
from munkres import Munkres, make_cost_matrix

logger = logging.getLogger(__name__)


# ============================================================================
# TYPE DEFINITIONS
# ============================================================================

# TypeVar constrained to the specific types we support for comparison
ComparableValue = TypeVar("ComparableValue", str, int, float, list, dict)


# TypeVar for scalar operations
ScalarType = TypeVar("ScalarType", str, int, float)


# ============================================================================
# ENUMS FOR NORMALIZERS AND COMPARATORS
# ============================================================================


class NormalizationType(Enum):
    """Enumeration of available normalization types."""

    RAW = "raw"
    TEXT = "text"
    NUMERIC = "numeric"
    LIST = "list"
    ITEM = "item"


class ComparatorType(Enum):
    """Enumeration of available comparator types."""

    EXACT = "exact"
    FUZZY = "fuzzy"
    HUNGARIAN = "hungarian"
    SEMANTIC = "semantic"
    NUMERIC = "numeric"


# ============================================================================
# NORMALIZATION FUNCTIONS
# ============================================================================


def normalize_item(item: ComparableValue) -> str:
    """Convert item to comparable format, handling unhashable types."""
    if isinstance(item, dict):
        return str(sorted(item.items()))
    elif isinstance(item, list):
        return str(item)
    else:
        return item


def label_set(value: object) -> set[str]:
    """Build the label set of one raw value.

    ``None`` is the empty set; a list is the set of the labels of its items;
    any other value is a set of at most one label. Each item goes through
    ``normalize_item`` and then ``str(...).strip().lower()``. An integral
    float is the label of its integer, so ``1.0`` and ``1`` are one label;
    the string ``"1.0"`` is not parsed and stays ``"1.0"``. An item that is
    ``None``, or whose normalized text is empty, adds no label.
    """
    if value is None:
        return set()
    items = value if isinstance(value, list) else [value]
    labels = set()
    for item in items:
        if item is None:
            continue
        if isinstance(item, float) and item.is_integer():
            item = int(item)
        label = str(normalize_item(item)).strip().lower()
        if not label:
            continue
        labels.add(label)
    return labels


def normalize_numeric(value: ScalarType) -> float:
    """
    Normalize a numeric value by removing currency symbols and commas.

    Args:
        value: Input value to normalize

    Returns:
        Normalized float value

    Raises:
        ValueError: If value cannot be converted to numeric
    """
    if isinstance(value, (int, float)):
        return float(value)

    if not isinstance(value, str):
        value = str(value)

    # Remove currency symbols, commas, parentheses
    value = value.replace("$", "").replace(",", "").replace("(", "").replace(")", "")

    try:
        return float(value)
    except (ValueError, TypeError):
        raise ValueError(f"Cannot convert {value} to numeric value")


def normalize_text(value: str) -> str:
    """
    Normalize text by removing punctuation and standardizing whitespace.

    Args:
        value: Input text to normalize

    Returns:
        Normalized text string
    """
    if not isinstance(value, str):
        value = str(value)

    # Replace punctuation and extra whitespace
    text = re.sub(r"[^\w\s]", "", value)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def normalize_list(value: Union[str, list, None]) -> List[str]:
    """
    Normalize a list by converting all elements to strings and sorting.

    Args:
        value: Input value to normalize as list

    Returns:
        Normalized list of strings
    """
    if value is None:
        return []

    # If already a list, return normalized version
    if isinstance(value, list):
        return sorted([str(item) for item in value])

    # Try to convert a string representation of a list to a list
    if isinstance(value, str) and value.startswith("[") and value.endswith("]"):
        try:
            parsed_list = ast.literal_eval(value)
            if isinstance(parsed_list, list):
                return sorted([str(item) for item in parsed_list])
        except (ValueError, SyntaxError):
            pass

    # Default: treat as a single value list
    return [str(value)]


def normalize_raw(value: ComparableValue) -> ComparableValue:
    """
    No normalization - pass through the value as-is.

    Args:
        value: Input value

    Returns:
        The input value unchanged
    """
    return value


# Normalization function registry
normalizer_funcs = {
    NormalizationType.RAW: normalize_raw,
    NormalizationType.TEXT: normalize_text,
    NormalizationType.NUMERIC: normalize_numeric,
    NormalizationType.LIST: normalize_list,
    NormalizationType.ITEM: normalize_item,
}


def apply_normalization(
    value: ComparableValue, normalization_type: str = NormalizationType.RAW.value
) -> ComparableValue:
    """
    Apply the specified normalization to a value.

    Args:
        value: Input value to normalize
        normalization_type: Type of normalization to apply

    Returns:
        Normalized value
    """
    # Convert string to enum for lookup
    try:
        normalization_enum = NormalizationType(normalization_type)
    except ValueError:
        raise ValueError(f"Unknown normalization type: {normalization_type}")

    normalizer = normalizer_funcs[normalization_enum]
    return normalizer(value)


# ============================================================================
# COMPARISON FUNCTIONS
# ============================================================================


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity score between 0.0 and 1.0
    """
    if not vec1 or not vec2:
        return 0.0

    # Calculate dot product
    dot_product = sum(a * b for a, b in zip(vec1, vec2))

    # Calculate magnitudes
    magnitude_a = math.sqrt(sum(a * a for a in vec1))
    magnitude_b = math.sqrt(sum(b * b for b in vec2))

    # Avoid division by zero
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    # Calculate cosine similarity
    similarity = dot_product / (magnitude_a * magnitude_b)

    # Ensure the result is between 0 and 1
    return max(0.0, min(1.0, (similarity + 1) / 2))


def _generate_embedding(text: str, bedrock_client: BedrockClient) -> List[float]:
    """Generate an embedding for non-empty, whitespace-normalized text."""
    normalized_text = " ".join(text.split())
    if not normalized_text:
        raise ValueError("Cannot generate an embedding for empty text")

    embeddings = bedrock_client.get_embeddings([normalized_text])
    if not embeddings or len(embeddings) != 1 or not embeddings[0]:
        raise ValueError("Expected exactly one non-empty embedding vector")

    return embeddings[0]


def compare_semantic(
    expected: ComparableValue, actual: ComparableValue, bedrock_client: BedrockClient
) -> float:
    """
    Compare two values using semantic embedding similarity.

    Args:
        expected: Expected value
        actual: Actual value
        bedrock_client: Client used to generate embeddings

    Returns:
        Similarity score between 0.0 and 1.0
    """
    null_result = null_handling(expected, actual)
    if null_result is not None:
        return null_result

    logger.debug("Generating embeddings for semantic comparison")
    expected_embedding = _generate_embedding(str(expected), bedrock_client)
    actual_embedding = _generate_embedding(str(actual), bedrock_client)

    similarity = cosine_similarity(expected_embedding, actual_embedding)
    logger.debug(f"Semantic similarity score: {similarity:.4f}")
    return similarity


def null_handling(
    expected: Optional[ComparableValue], actual: Optional[ComparableValue]
) -> Optional[float]:
    """
    To check and to compare if either expected or actual value has None

    Args:
        expected: Expected value
        actual: Actual value

    Returns:
        1.0 if they both are None or empty string and 0.0 if one of them is None. Otherwise return None.
    """
    if expected is None and actual is None:
        return 1.0

    if expected is None or actual is None:
        return 0.0

    # Check if both values are empty strings
    if (
        isinstance(expected, str)
        and not expected.strip()
        and isinstance(actual, str)
        and not actual.strip()
    ):
        return 1.0
    return None


def strip_punctuation_space(text: str) -> str:
    """
    Strip punctuation and standardize whitespace in text.

    Args:
        text: Input text to process

    Returns:
        Processed text with punctuation removed and whitespace standardized
    """
    if not isinstance(text, str):
        text = str(text)

    # Replace punctuation and extra whitespace
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def compare_exact(
    expected: ComparableValue,
    actual: ComparableValue,
    normalization_type: str = NormalizationType.TEXT.value,
) -> float:
    """
    Compare two values for exact match after normalization.

    Args:
        expected: Expected value
        actual: Actual value
        normalization_type: Type of normalization to apply before comparison

    Returns:
        1.0 if values match exactly after normalization, 0.0 otherwise
    """
    if null_handling(expected, actual) is not None:
        return null_handling(expected, actual)

    # Apply specified normalization
    expected_norm = apply_normalization(expected, normalization_type)
    actual_norm = apply_normalization(actual, normalization_type)

    return 1.0 if expected_norm == actual_norm else 0.0


def fuzz_score(s1: str, s2: str) -> float:
    """
    Calculate fuzzy match score between two strings using Levenshtein distance.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Similarity score between 0.0 and 1.0
    """
    # Normalize inputs
    s1 = strip_punctuation_space(s1)
    s2 = strip_punctuation_space(s2)

    # Perfect match
    if s1 == s2:
        return 1.0

    # Edge cases
    if not s1 or not s2:
        return 0.0

    # Calculate Levenshtein distance
    len_s1, len_s2 = len(s1), len(s2)
    d = [[0 for _ in range(len_s2 + 1)] for _ in range(len_s1 + 1)]

    for i in range(len_s1 + 1):
        d[i][0] = i
    for j in range(len_s2 + 1):
        d[0][j] = j

    for i in range(1, len_s1 + 1):
        for j in range(1, len_s2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            d[i][j] = min(
                d[i - 1][j] + 1,  # deletion
                d[i][j - 1] + 1,  # insertion
                d[i - 1][j - 1] + cost,  # substitution
            )

    # Convert to similarity score (1.0 for identical, approaching 0.0 for very different)
    max_len = max(len_s1, len_s2)
    return 1.0 - (d[len_s1][len_s2] / max_len if max_len > 0 else 0.0)


def compare_fuzzy(
    expected: ComparableValue,
    actual: ComparableValue,
    normalization_type: str = NormalizationType.TEXT.value,
) -> float:
    """
    Compare two values using fuzzy string matching.

    Args:
        expected: Expected value
        actual: Actual value
        normalization_type: Type of normalization to apply before comparison

    Returns:
        Similarity score between 0.0 and 1.0
    """
    if null_handling(expected, actual) is not None:
        return null_handling(expected, actual)

    # Apply specified normalization
    expected_norm = apply_normalization(expected, normalization_type)
    actual_norm = apply_normalization(actual, normalization_type)

    # Calculate fuzzy similarity score
    return fuzz_score(str(expected_norm), str(actual_norm))


def compare_numeric(expected: ScalarType, actual: ScalarType) -> float:
    """
    Compare two values for exact numeric match after normalization.

    Args:
        expected: Expected value
        actual: Actual value

    Returns:
        1.0 if values match exactly after numeric normalization, 0.0 otherwise
    """
    if null_handling(expected, actual) is not None:
        return null_handling(expected, actual)

    try:
        # Try numeric comparison first
        expected_num = normalize_numeric(expected)
        actual_num = normalize_numeric(actual)
        return 1.0 if expected_num == actual_num else 0.0
    except ValueError:
        # Fall back to exact string comparison if numeric conversion fails
        return compare_exact(expected, actual)


def compare_values(
    value1: ComparableValue,
    value2: ComparableValue,
    comparator_type: str = ComparatorType.EXACT.value,
    normalization_type: str = NormalizationType.TEXT.value,
) -> float:
    """
    Compare two values based on the configured comparator type.

    Args:
        value1: First value to compare
        value2: Second value to compare
        comparator_type: Type of comparison to use
        normalization_type: Type of normalization to apply before comparison

    Returns:
        Similarity score between 0.0 and 1.0
    """
    if comparator_type == ComparatorType.EXACT.value:
        return compare_exact(value1, value2, normalization_type)
    elif comparator_type == ComparatorType.FUZZY.value:
        return compare_fuzzy(value1, value2, normalization_type)
    elif comparator_type == ComparatorType.NUMERIC.value:
        return compare_numeric(value1, value2)
    else:
        # Default to exact comparison
        return compare_exact(value1, value2, normalization_type)


def compare_hungarian(
    expected: ComparableValue,
    actual: ComparableValue,
    comparator_type: str = ComparatorType.EXACT.value,
    normalization_type: str = NormalizationType.TEXT.value,
) -> float:
    """
    Compare lists using Hungarian algorithm for maximum bipartite matching.

    Args:
        expected: Expected list or value
        actual: Actual list or value
        comparator_type: Type of comparison to use for matching
        normalization_type: Type of normalization to apply before comparison

    Returns:
        Average similarity score from optimal matching
    """
    expected_list = convert_to_list(expected)
    actual_list = convert_to_list(actual)

    # Handle simple case with single values
    if len(expected_list) == 1 and len(actual_list) == 1:
        score = compare_values(
            expected_list[0], actual_list[0], comparator_type, normalization_type
        )
        return score

    # Empty lists edge case
    if not expected_list and not actual_list:
        return 1.0
    if not expected_list or not actual_list:
        return 0.0

    # Create similarity matrix for Hungarian algorithm
    matrix = [[0 for _ in range(len(actual_list))] for _ in range(len(expected_list))]

    # Fill matrix with comparison scores
    for i, exp_val in enumerate(expected_list):
        for j, act_val in enumerate(actual_list):
            matrix[i][j] = compare_values(
                exp_val, act_val, comparator_type, normalization_type
            )

    # Convert to cost matrix (Hungarian algorithm minimizes cost)
    cost_matrix = make_cost_matrix(matrix, lambda x: 1 - x)

    # Compute the optimal assignment
    m = Munkres()
    indexes = m.compute(cost_matrix)

    # Calculate average score from optimal matching
    matches = [(i, j, matrix[i][j]) for i, j in indexes]
    avg_score = sum(score for _, _, score in matches) / len(matches) if matches else 0.0
    return avg_score


# ============================================================================
# COMPARISON FUNCTION REGISTRY
# ============================================================================

# Comparator function registry
comparator_funcs = {
    ComparatorType.EXACT: compare_exact,
    ComparatorType.FUZZY: compare_fuzzy,
    ComparatorType.HUNGARIAN: compare_hungarian,
    ComparatorType.SEMANTIC: compare_semantic,
    ComparatorType.NUMERIC: compare_numeric,
}

# Define available comparators and normalizers
COMPARATORS = [comp_type.value for comp_type in ComparatorType]


def get_comparison_function(
    comparator_type: str,
    normalization_type: str = NormalizationType.TEXT.value,
    **kwargs,
):
    """
    Get a comparison function configured with the specified comparator and normalization.

    Args:
        comparator_type: Type of comparison ('exact', 'fuzzy', 'hungarian', 'semantic', 'numeric')
        normalization_type: Type of normalization to apply
        **kwargs: Additional arguments for specific comparators (e.g., bedrock_client for semantic)

    Returns:
        Configured comparison function
    """
    # Convert string to enum for comparison
    try:
        comparator_enum = ComparatorType(comparator_type)
    except ValueError:
        raise ValueError(f"Unknown comparator type: {comparator_type}")

    if comparator_enum == ComparatorType.EXACT:
        return lambda x, y: compare_exact(x, y, normalization_type)
    elif comparator_enum == ComparatorType.FUZZY:
        return lambda x, y: compare_fuzzy(x, y, normalization_type)
    elif comparator_enum == ComparatorType.HUNGARIAN:
        return lambda x, y: compare_hungarian(
            x, y, ComparatorType.EXACT.value, normalization_type
        )
    elif comparator_enum == ComparatorType.NUMERIC:
        return compare_numeric
    elif comparator_enum == ComparatorType.SEMANTIC:
        bedrock_client = kwargs.get("bedrock_client")
        if not bedrock_client:
            raise ValueError("Semantic comparison requires bedrock_client parameter")
        return lambda x, y: compare_semantic(x, y, bedrock_client)
    else:
        raise ValueError(f"Unknown comparator type: {comparator_type}")


def convert_to_list(value: Optional[ComparableValue]) -> List[str]:
    """
    Convert a value to a list.

    Args:
        value: Input value to convert (string, list, or other)

    Returns:
        List of strings
    """
    if value is None:
        return []

    # If already a list, return as is
    if isinstance(value, list):
        return [str(item) for item in value]

    # Try to convert a string representation of a list to a list
    if isinstance(value, str) and value.startswith("[") and value.endswith("]"):
        try:
            parsed_list = ast.literal_eval(value)
            if isinstance(parsed_list, list):
                return [str(item) for item in parsed_list]
        except (ValueError, SyntaxError):
            pass
    # Default: treat as a single value
    return [str(value)]
