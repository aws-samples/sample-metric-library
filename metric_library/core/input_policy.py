# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Declarative input contracts for metric inputs (references, prompts, context)."""

from enum import Enum


class InputPolicy(str, Enum):
    """
    How a metric treats one of its auxiliary inputs.

    Metrics declare a policy per input (``reference_policy``, ``prompt_policy``,
    ``context_policy``) as class attributes. The base class enforces the contract
    before any model invocation.

    ``str`` is mixed in so members compare equal to, and hash like, their plain
    string values (``InputPolicy.OMIT == "omit"``).

    Use ``InputPolicy(value)`` to coerce and validate an arbitrary value; it
    raises ``ValueError`` for anything outside the declared members.
    """

    REQUIRED = "required"
    """The input must be supplied, aligned with predictions, and non-empty."""

    OPTIONAL = "optional"
    """The input may be supplied; if it is, it must align with predictions."""

    OMIT = "omit"
    """The input is never consumed, even if a caller supplies it."""

    def __str__(self) -> str:
        return self.value
