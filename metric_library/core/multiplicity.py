# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""How a metric reads a role that arrives as several items."""

from enum import Enum


class Multiplicity(Enum):
    """
    How many items a role may carry per sample, and how they relate.

    A metric declares one value per role (``prediction_multiplicity``,
    ``reference_multiplicity``): how one ``compute`` call reads several items
    of that role. Multiplicity is for the whole path, and the matcher replaces
    it on the item path. With the ``whole`` matcher, the engine enforces the
    count rule before any sample is scored, and LLM metrics composing through
    ``PromptHandler`` turn the value into one introductory sentence when a role
    carries more than one item. With the ``labels`` or ``list`` matcher, the
    matcher reads the items of each role instead, and ``compute`` gets one item
    per role. The matchers pair items by score and count every unpaired item,
    so a metric that declares ``ALIGNED`` or ``ANY_ACCEPTABLE`` accepts only
    the ``whole`` matcher.
    """

    SINGLE = "single"
    """Exactly one item: the metric scores one value per role."""

    ALIGNED = "aligned"
    """Several items paired by position with the other role; equal counts."""

    COMPOSITE = "composite"
    """Several items are parts of one whole and are judged together."""

    ANY_ACCEPTABLE = "any_acceptable"
    """Any one of several items is acceptable on its own."""
