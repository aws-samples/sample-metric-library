# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Matchers: the layer before the metric.

A matcher gives the (class, item) entries of one prediction or reference
value. With ``WholeValue``, the default, the engine scores each sample as it
is. With an item matcher (``LabelSet`` or ``ListItems``), the engine scores
each candidate pair of items of one class, and the ``prf`` aggregator pairs
the items at each threshold. The item matchers read text: each item is a
``TextData``, and the engine stops a metric before the first sample when a
prediction or reference item is not text.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..utils.comparison_functions import label_set
from .data import DataType
from .data.text_data import TextData


class Matcher(ABC):
    """Give the items of one prediction or reference value."""

    @abstractmethod
    def items(self, value: DataType | None) -> list[tuple[str, DataType | None]]:
        """Return the (class, item) entries of one value."""


@dataclass(frozen=True)
class WholeValue(Matcher):
    """The full value is the only item, also when it is empty.

    This is the default matcher. The engine scores the sample as it is: one
    ``compute`` call whose score is the sample score, with no match record.
    """

    def items(self, value: DataType | None) -> list[tuple[str, DataType | None]]:
        return [("*", value)]


@dataclass(frozen=True)
class LabelSet(Matcher):
    """One item for each label of the value; the label is also its class.

    The labels follow the rule of ``Accuracy``: a string is one label, a list
    is a set of labels, and ``None``, a blank string and an empty list have no
    labels. Labels are trimmed and lowercased, and an integral float such as
    1.0 is the label 1. Only equal labels can pair.
    """

    def items(self, value: DataType | None) -> list[tuple[str, DataType]]:
        raw = None if value is None else value.get_raw_data()
        return [(label, TextData(label)) for label in sorted(label_set(raw))]


@dataclass(frozen=True)
class ListItems(Matcher):
    """One item for each element of a list value.

    A value that is not a list is a one-element list, so ``"Rome"`` and
    ``["Rome"]`` are the same value, as they are when a record loads.
    ``class_key`` names the class field of each element, and the class follows
    the label rule of ``LabelSet``; without it, every element is in the class
    ``"*"``. ``value_key`` names the field that the metric compares; without
    it, the item is the whole element. ``None`` and an empty list have no
    items. An element without a named field fails the sample.
    """

    class_key: str | None = None
    value_key: str | None = None

    def items(self, value: DataType | None) -> list[tuple[str, DataType]]:
        raw = None if value is None else value.get_raw_data()
        if raw is None:
            return []
        entries = []
        for index, element in enumerate(raw if isinstance(raw, list) else [raw]):
            item_class = "*"
            if self.class_key is not None:
                labels = (
                    label_set(element.get(self.class_key))
                    if isinstance(element, dict)
                    else set()
                )
                if len(labels) != 1:
                    raise ValueError(
                        f"item {index} needs a non-empty {self.class_key!r} field"
                    )
                (item_class,) = labels
            if self.value_key is not None:
                if not isinstance(element, dict) or self.value_key not in element:
                    raise ValueError(f"item {index} has no {self.value_key!r} field")
                element = element[self.value_key]
            entries.append((item_class, TextData(element)))
        return entries


__all__ = ["LabelSet", "ListItems", "Matcher", "WholeValue"]
