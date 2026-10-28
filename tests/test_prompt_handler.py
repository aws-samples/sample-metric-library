# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Focused behavioral tests for the production ``PromptHandler.build`` interface.

Every test drives ``PromptHandler.build`` with a small unregistered ``LLMMetric``
subclass and role items, then asserts on the complete returned query keyword
arguments (``system``, ``prompt``, ``additional_content`` and
``response_schema``) and on the conversion effects observed on the items. Real
``TextData`` values and a real PNG file rendered through the production image
block cover the content blocks; a small in-memory ``DataType`` double with a
deliberately non-text block is used only where conversion counts or opaque
block shapes matter.
"""

import struct
import zlib

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from metric_library.core.data import DataType, Modality
from metric_library.core.data.text_data import TextData
from metric_library.core.input_policy import InputPolicy
from metric_library.core.llm_metric import (
    JUDGE_RESPONSE_SCHEMA,
    LLMMetric,
    Multiplicity,
)
from metric_library.core.prompt_handler import PromptHandler

# The single golden copies of the default judge role and the two-part response
# schema live with the LLMCorrectness request golden.
from test_llm_correctness_prompt_handler import JUDGE_ROLE, TWO_PART_SCHEMA

# A rubric that is deliberately not the LLMCorrectness rubric, so the tests
# prove the metric handed to the handler, not a built-in one, supplies the text.
RUBRIC = "Judge whether the candidate answer states the same facts as the reference."

# Canonical rendering order and the delimiter tag each role emits.
ROLES = ("prompt", "context", "prediction", "reference")
TAGS = {
    "prompt": "task",
    "context": "context",
    "prediction": "candidate_response",
    "reference": "ground_truth_response",
}

# Roles governed by a metric policy; prediction is always rendered.
AUXILIARY_ROLES = ("prompt", "context", "reference")

# Roles whose default section introduces several items, with the ``LLMMetric``
# default multiplicity for each. Prompt and context items are never introduced.
DEFAULT_MULTIPLICITY = {
    "prediction": Multiplicity.COMPOSITE,
    "reference": Multiplicity.COMPOSITE,
}

# The complete set of ``BedrockClient.query`` keyword arguments a request carries.
REQUEST_KEYS = {"prompt", "system", "additional_content", "response_schema"}


def _metric(
    rubric=RUBRIC,
    *,
    prompt=InputPolicy.REQUIRED,
    context=InputPolicy.OPTIONAL,
    reference=InputPolicy.REQUIRED,
    supported=None,
    **attributes,
):
    """An unregistered ``LLMMetric`` whose rubric and role policies are the arguments.

    The class is built on the fly and never registered, so the tests prove the
    handler reads whatever metric it is handed. The default policies render every
    role that receives items, no section hook is overridden, and the metric keeps
    the ``LLMMetric`` defaults of accepting text and image data, introducing
    multi-part predictions and references as composites and declaring no
    modality guidance.

    ``supported`` narrows ``supported_data_types``. Any further keyword becomes a
    class attribute, which is how a test overrides a section hook such as
    ``role``, ``prediction`` or ``supplemental`` with a plain function, or a
    declaration such as ``reference_multiplicity`` or ``modality_guidance``. The
    ``context`` and ``reference`` hooks share their names with the policy
    keywords, so they cannot be overridden this way.
    """
    namespace = {
        "rubric": lambda self: rubric,
        "prompt_policy": prompt,
        "context_policy": context,
        "reference_policy": reference,
        **attributes,
    }
    if supported is not None:
        namespace["supported_data_types"] = list(supported)
    return type("SectionHookMetric", (LLMMetric,), namespace)()


class OpaqueData(DataType):
    """In-memory ``DataType`` returning one arbitrary block and counting conversions.

    The block intentionally looks like neither a text nor an image content
    block, so a test can prove the handler appends whatever
    ``get_data_for_LLM_message`` returns without inspecting or rewriting it. Its
    modality is text, so it keeps bare delimiters.
    """

    def __init__(self, name):
        self.block = {"opaque": {"name": name, "payload": [name, len(name)]}}
        self.conversions = 0
        super().__init__(f"opaque::{name}")

    def preprocess(self, path):
        self.raw_data = path
        self.data = path

    @property
    def modality(self):
        return Modality.TEXT

    def get_data_for_LLM_message(self):
        self.conversions += 1
        return self.block


class OpaqueImageData(OpaqueData):
    """``OpaqueData`` whose modality is image.

    It keeps the opaque block and the conversion counter, so a test can prove an
    image-modality item was or was not converted without any media dependency.
    """

    @property
    def modality(self):
        return Modality.IMAGE


class PngFileData(DataType):
    """Image-modality ``DataType`` over a PNG file, rendered by the base-class block.

    ``ImageData`` sends a file in a format Converse reads as this base-class
    block, so this yields its exact block for the same file while keeping
    Pillow out of the test process: the smoke tests assert that optional media
    dependencies are never imported or registered eagerly.
    """

    def preprocess(self, path):
        with open(path, "rb") as handle:
            self.raw_data = handle.read()
        self.data = self.raw_data

    @property
    def modality(self):
        return Modality.IMAGE


def _png_bytes(width=2, height=2, rgb=(255, 0, 0)):
    """Encode a solid-color 8-bit RGB PNG without any optional media dependency."""

    def chunk(tag, payload):
        body = tag + payload
        return (
            struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))
        )

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    scanlines = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(scanlines))
        + chunk(b"IEND", b"")
    )


def _open(tag, index=None, count=None, modality=Modality.TEXT):
    """The opening delimiter block for item ``index`` of ``count`` rendered in ``tag``.

    One text item keeps the bare delimiter; several items are numbered with
    ``part``/``of`` attributes, and a non-text item names its modality after them.
    """
    attributes = ""
    if count is not None and count > 1:
        attributes += f' part="{index + 1}" of="{count}"'
    if modality is not Modality.TEXT:
        attributes += f' modality="{modality}"'
    return {"text": f"\n<{tag}{attributes}>"}


def _close(tag):
    """The closing delimiter block for ``tag``."""
    return {"text": f"</{tag}>"}


COUNTERPART = {"prediction": "ground truth response", "reference": "candidate response"}


def _intro(role, count, multiplicity):
    """The sentence introducing ``count`` items of ``role``; ``None`` under SINGLE."""
    noun = TAGS[role].replace("_", " ")
    if multiplicity in (None, Multiplicity.SINGLE):
        return None
    if multiplicity is Multiplicity.COMPOSITE:
        return {
            "text": (
                f"\nThe {noun} consists of {count} parts that together form one "
                "response."
            )
        }
    if multiplicity is Multiplicity.ALIGNED:
        return {
            "text": (
                f"\nThe {noun} has {count} parts; part i corresponds to part i "
                f"of the {COUNTERPART[role]}."
            )
        }
    return {
        "text": (
            f"\nThere are {count} alternative {noun}s. "
            "The candidate is correct if it matches any one of them."
        )
    }


_DEFAULT = object()


def _section(role, *blocks, multiplicity=_DEFAULT):
    """The blocks one role renders for text-item ``blocks`` in that order.

    Several items in a role with a multiplicity are introduced once first; then
    every item is one open/item/close triple. ``multiplicity`` defaults to the
    ``LLMMetric`` default for the role.
    """
    if multiplicity is _DEFAULT:
        multiplicity = DEFAULT_MULTIPLICITY.get(role)
    tag = TAGS[role]
    content = []
    intro = _intro(role, len(blocks), multiplicity) if len(blocks) > 1 else None
    if intro is not None:
        content.append(intro)
    for index, block in enumerate(blocks):
        content.extend([_open(tag, index, len(blocks)), block, _close(tag)])
    return content


def _text_block(value):
    return {"text": value}


def _texts(request):
    """Every string the request carries: system, prompt and each text block."""
    return [request["system"], request["prompt"]] + [
        block["text"] for block in request["additional_content"] if "text" in block
    ]


def _text_items():
    """One distinct ``TextData`` value per role, keyed by role name."""
    return {role: TextData(f"{role}-text") for role in ROLES}


def _opaque_items(exclude=()):
    """One distinct ``OpaqueData`` value per role, keyed by role name."""
    return {role: OpaqueData(f"{role}-kept") for role in ROLES if role not in exclude}


def _expected_opaque_content(items):
    """The canonical delimited sections for a role-keyed mapping of doubles."""
    content = []
    for role in ROLES:
        if role in items:
            content.extend(_section(role, items[role].block))
    return content


# The complete content for one text item per role.
ONE_ITEM_CONTENT = [
    _open("task"),
    _text_block("prompt-text"),
    _close("task"),
    _open("context"),
    _text_block("context-text"),
    _close("context"),
    _open("candidate_response"),
    _text_block("prediction-text"),
    _close("candidate_response"),
    _open("ground_truth_response"),
    _text_block("reference-text"),
    _close("ground_truth_response"),
]


@pytest.fixture
def png_image(tmp_path):
    """A real two-pixel PNG file loaded as image-modality data, plus its bytes."""
    path = tmp_path / "tiny.png"
    path.write_bytes(_png_bytes())
    return PngFileData(str(path)), path.read_bytes()


class TestRequestComposition:
    """System is the judge role, prompt the bare rubric, content only the data.

    The schema, not prose, shapes the result.
    """

    def test_request_is_role_system_rubric_prompt_schema_and_data_content(self):
        request = PromptHandler.build(_metric(), **_text_items())

        assert set(request) == REQUEST_KEYS
        assert request["system"] == JUDGE_ROLE
        assert request["prompt"] == RUBRIC
        assert request["response_schema"] == JUDGE_RESPONSE_SCHEMA
        assert request["additional_content"] == ONE_ITEM_CONTENT

    def test_rubric_whitespace_is_preserved_verbatim(self):
        rubric = "  \n\tScore the answer.\n\n  Trailing whitespace stays.  \n"

        request = PromptHandler.build(_metric(rubric), prediction=TextData("candidate"))

        assert request["prompt"] == rubric

    def test_system_prompt_and_schema_do_not_depend_on_supplied_items(self):
        empty = PromptHandler.build(_metric())
        populated = PromptHandler.build(_metric(), **_text_items())

        assert empty["system"] == populated["system"] == JUDGE_ROLE
        assert empty["prompt"] == populated["prompt"] == RUBRIC
        assert empty["response_schema"] == populated["response_schema"]
        assert empty["additional_content"] == []

    def test_response_schema_hook_result_is_forwarded_unchanged(self):
        schema = {"type": "object", "properties": {"verdict": {"type": "string"}}}
        metric = _metric(response_schema=lambda self: schema)

        request = PromptHandler.build(metric, **_text_items())

        assert request["response_schema"] is schema
        assert request["additional_content"] == ONE_ITEM_CONTENT


class TestCanonicalRoleOrder:
    """Rendered roles appear in prompt, context, prediction, reference order."""

    def test_one_text_item_per_role_yields_four_delimited_sections_in_order(self):
        content = PromptHandler.build(_metric(), **_text_items())["additional_content"]

        assert len(content) == 12
        assert content == ONE_ITEM_CONTENT
        assert content[::3] == [_open(TAGS[role]) for role in ROLES]
        assert content[2::3] == [_close(TAGS[role]) for role in ROLES]

    def test_keyword_supply_order_does_not_change_rendered_order(self):
        items = _text_items()

        canonical = PromptHandler.build(_metric(), **items)
        reversed_supply = PromptHandler.build(
            _metric(), **{role: items[role] for role in reversed(ROLES)}
        )

        assert reversed_supply == canonical
        assert canonical["additional_content"][::3] == [
            _open(TAGS[role]) for role in ROLES
        ]

    def test_two_items_per_role_render_role_by_role_without_interleaving(self):
        items = {
            role: [OpaqueData(f"{role}-0"), OpaqueData(f"{role}-1")] for role in ROLES
        }

        content = PromptHandler.build(_metric(), **items)["additional_content"]

        expected = []
        for role in ROLES:
            expected.extend(_section(role, items[role][0].block, items[role][1].block))
        assert content == expected
        assert [
            block["text"]
            for block in content
            if block.get("text", "").startswith("\n<")
        ] == [
            f'\n<{TAGS[role]} part="{position}" of="2">'
            for role in ROLES
            for position in (1, 2)
        ]
        assert all(item.conversions == 1 for role in ROLES for item in items[role])


class TestScalarAndSequenceRendering:
    """Every item is one delimited triple, converts once, and keeps supplied order."""

    @pytest.mark.parametrize("role", ROLES)
    def test_one_item_yields_one_triple_wrapping_its_own_block(self, role):
        item = OpaqueData(f"{role}-only")

        content = PromptHandler.build(_metric(), **{role: item})["additional_content"]

        assert content == [_open(TAGS[role]), item.block, _close(TAGS[role])]
        assert content[1] is item.block
        assert item.conversions == 1

    @pytest.mark.parametrize("role", ROLES)
    def test_scalar_renders_identically_to_its_singleton_sequence(self, role):
        item = TextData(f"{role}-scalar")

        scalar = PromptHandler.build(_metric(), **{role: item})
        as_list = PromptHandler.build(_metric(), **{role: [item]})
        as_tuple = PromptHandler.build(_metric(), **{role: (item,)})

        assert scalar == as_list == as_tuple
        assert scalar["additional_content"] == [
            _open(TAGS[role]),
            _text_block(f"{role}-scalar"),
            _close(TAGS[role]),
        ]

    @pytest.mark.parametrize("role", ROLES)
    def test_two_items_yield_numbered_triples_in_supplied_order(self, role):
        first = OpaqueData(f"{role}-first")
        second = OpaqueData(f"{role}-second")
        tag = TAGS[role]

        forward = PromptHandler.build(_metric(), **{role: [first, second]})

        assert forward["additional_content"] == _section(
            role, first.block, second.block
        )
        assert forward["additional_content"][-6:] == [
            {"text": f'\n<{tag} part="1" of="2">'},
            first.block,
            _close(tag),
            {"text": f'\n<{tag} part="2" of="2">'},
            second.block,
            _close(tag),
        ]
        # Only prediction and reference carry a multiplicity, so only they are
        # introduced; prompt and context items are just numbered.
        assert (len(forward["additional_content"]) == 7) is (
            role in DEFAULT_MULTIPLICITY
        )
        assert (first.conversions, second.conversions) == (1, 1)

        backward = PromptHandler.build(_metric(), **{role: [second, first]})

        assert backward["additional_content"] == _section(
            role, second.block, first.block
        )


class TestMixedModalityRole:
    """Text and image items in one role keep their real blocks and supplied order."""

    @pytest.mark.parametrize("role", ROLES)
    def test_text_then_image_appends_both_real_blocks_unchanged_in_order(
        self, role, png_image
    ):
        image, png_bytes = png_image
        text = TextData(f"{role}-caption")
        tag = TAGS[role]

        content = PromptHandler.build(_metric(), **{role: [text, image]})[
            "additional_content"
        ]

        triples = [
            {"text": f'\n<{tag} part="1" of="2">'},
            _text_block(f"{role}-caption"),
            _close(tag),
            {"text": f'\n<{tag} part="2" of="2" modality="image">'},
            {"image": {"format": "png", "source": {"bytes": png_bytes}}},
            _close(tag),
        ]
        if role in DEFAULT_MULTIPLICITY:
            triples.insert(0, _intro(role, 2, DEFAULT_MULTIPLICITY[role]))
        assert content == triples
        assert content[-5] == text.get_data_for_LLM_message()
        assert content[-2] == image.get_data_for_LLM_message()
        assert content[-3] == _open(tag, 1, 2, modality=Modality.IMAGE)


class TestPolicySuppression:
    """OMIT roles and empty OPTIONAL roles contribute nothing and convert nothing."""

    @pytest.mark.parametrize("shape", ["scalar", "sequence"])
    @pytest.mark.parametrize("role", AUXILIARY_ROLES)
    def test_omit_role_with_items_renders_nothing_and_never_converts(self, role, shape):
        metric = _metric(**{role: InputPolicy.OMIT})
        kept = _opaque_items(exclude=(role,))
        omitted = [OpaqueData(f"{role}-omitted-0"), OpaqueData(f"{role}-omitted-1")]
        supplied = omitted[0] if shape == "scalar" else omitted

        content = PromptHandler.build(metric, **kept, **{role: supplied})[
            "additional_content"
        ]

        assert content == _expected_opaque_content(kept)
        assert not any(TAGS[role] in block.get("text", "") for block in content)
        assert all(item.conversions == 0 for item in omitted)
        assert all(item.conversions == 1 for item in kept.values())

    @pytest.mark.parametrize(
        "empty", [None, [], ()], ids=["none", "empty-list", "empty-tuple"]
    )
    @pytest.mark.parametrize("role", AUXILIARY_ROLES)
    def test_optional_role_without_items_renders_nothing(self, role, empty):
        metric = _metric(**{role: InputPolicy.OPTIONAL})
        kept = _opaque_items(exclude=(role,))

        content = PromptHandler.build(metric, **kept, **{role: empty})[
            "additional_content"
        ]

        assert content == _expected_opaque_content(kept)
        assert _open(TAGS[role]) not in content
        assert all(item.conversions == 1 for item in kept.values())


class TestAlternateMetric:
    """A different, unregistered metric renders through the same build method."""

    def test_distinct_rubric_with_context_omitted_renders_remaining_roles(self):
        rubric = "Rate whether the response is faithful to its sources."
        metric = _metric(
            rubric,
            reference=InputPolicy.REQUIRED,
            prompt=InputPolicy.REQUIRED,
            context=InputPolicy.OMIT,
        )
        items = _text_items()

        request = PromptHandler.build(metric, **items)

        assert request["prompt"] == rubric
        assert RUBRIC not in request["prompt"]
        assert request["system"] == JUDGE_ROLE
        assert request["response_schema"] == JUDGE_RESPONSE_SCHEMA
        assert request["additional_content"] == [
            *_section("prompt", _text_block("prompt-text")),
            *_section("prediction", _text_block("prediction-text")),
            *_section("reference", _text_block("reference-text")),
        ]

    def test_changing_only_the_rubric_changes_only_the_prompt(self):
        items = _text_items()

        base = PromptHandler.build(_metric(), **items)
        other = PromptHandler.build(_metric("A different rubric."), **items)

        assert base["prompt"] == RUBRIC
        assert other["prompt"] == "A different rubric."
        assert {key: other[key] for key in REQUEST_KEYS - {"prompt"}} == {
            key: base[key] for key in REQUEST_KEYS - {"prompt"}
        }


class TestSingleNonTextItemDelimiter:
    """One non-text item keeps an unnumbered delimiter that names its modality."""

    @pytest.mark.parametrize("role", ROLES)
    def test_one_image_item_yields_a_modality_delimiter_without_numbering(
        self, role, png_image
    ):
        image, png_bytes = png_image
        tag = TAGS[role]

        content = PromptHandler.build(_metric(), **{role: image})["additional_content"]

        assert content == [
            {"text": f'\n<{tag} modality="image">'},
            {"image": {"format": "png", "source": {"bytes": png_bytes}}},
            _close(tag),
        ]
        assert content[0] == _open(tag, modality=Modality.IMAGE)
        assert content[1] == image.get_data_for_LLM_message()


class TestImageFormatNormalization:
    """The image block's ``format`` is the Bedrock name, whatever the file is called."""

    @pytest.mark.parametrize(
        ("file_name", "expected_format"),
        [
            ("photo.jpg", "jpeg"),
            ("photo.JPG", "jpeg"),
            ("photo.jpeg", "jpeg"),
            ("photo.PNG", "png"),
            ("photo.webp", "webp"),
        ],
    )
    def test_file_extension_becomes_the_bedrock_format_name(
        self, tmp_path, file_name, expected_format
    ):
        path = tmp_path / file_name
        path.write_bytes(_png_bytes())
        image = PngFileData(str(path))

        content = PromptHandler.build(_metric(), prediction=image)["additional_content"]

        assert content[1] == {
            "image": {"format": expected_format, "source": {"bytes": _png_bytes()}}
        }


class TestOmittedRoles:
    """An omitted role's items are never converted; the other roles render normally."""

    @pytest.mark.parametrize("role", AUXILIARY_ROLES)
    def test_omitted_role_is_dropped_and_other_roles_render_normally(self, role):
        metric = _metric(supported=[Modality.TEXT], **{role: InputPolicy.OMIT})
        kept = _opaque_items(exclude=(role,))
        omitted = OpaqueImageData(f"{role}-omitted")

        content = PromptHandler.build(metric, **kept, **{role: omitted})[
            "additional_content"
        ]

        assert content == _expected_opaque_content(kept)
        assert omitted.conversions == 0
        assert all(item.conversions == 1 for item in kept.values())


class TestMultiplicityIntro:
    """Several items in a role with a multiplicity are introduced once, in its words."""

    def test_composite_prediction_parts_are_introduced_as_one_response(self):
        parts = [TextData("part-a"), TextData("part-b")]
        intro = {
            "text": (
                "\nThe candidate response consists of 2 parts that together form "
                "one response."
            )
        }

        content = PromptHandler.build(
            _metric(), **{**_text_items(), "prediction": parts}
        )["additional_content"]

        assert content == [
            *_section("prompt", _text_block("prompt-text")),
            *_section("context", _text_block("context-text")),
            intro,
            _open("candidate_response", 0, 2),
            _text_block("part-a"),
            _close("candidate_response"),
            _open("candidate_response", 1, 2),
            _text_block("part-b"),
            _close("candidate_response"),
            *_section("reference", _text_block("reference-text")),
        ]
        assert content.count(intro) == 1
        assert content[content.index(intro) - 1] == _close("context")

    def test_any_acceptable_references_are_introduced_as_alternatives(self):
        metric = _metric(reference_multiplicity=Multiplicity.ANY_ACCEPTABLE)
        truths = [TextData("truth-a"), TextData("truth-b")]
        intro = {
            "text": (
                "\nThere are 2 alternative ground truth responses. "
                "The candidate is correct if it matches any one of them."
            )
        }

        content = PromptHandler.build(metric, **{**_text_items(), "reference": truths})[
            "additional_content"
        ]

        assert content[-7:] == [
            intro,
            _open("ground_truth_response", 0, 2),
            _text_block("truth-a"),
            _close("ground_truth_response"),
            _open("ground_truth_response", 1, 2),
            _text_block("truth-b"),
            _close("ground_truth_response"),
        ]
        assert content == [
            *ONE_ITEM_CONTENT[:9],
            *_section(
                "reference",
                _text_block("truth-a"),
                _text_block("truth-b"),
                multiplicity=Multiplicity.ANY_ACCEPTABLE,
            ),
        ]
        assert content.count(intro) == 1
        assert not any("consists of" in block.get("text", "") for block in content)

    @pytest.mark.parametrize("multiplicity", list(Multiplicity), ids=str)
    def test_one_item_per_role_is_never_introduced(self, multiplicity):
        metric = _metric(
            prediction_multiplicity=multiplicity,
            reference_multiplicity=multiplicity,
        )

        content = PromptHandler.build(metric, **_text_items())["additional_content"]

        assert content == ONE_ITEM_CONTENT


# Guidance a test metric declares per modality.
IMAGE_HINT = "Judge what the image depicts, not how it is rendered."
TEXT_HINT = "Judge the text on its facts alone."


class TestModalityGuidance:
    """Declared guidance follows the rubric for the modalities of the roles in focus.

    The roles in focus are the candidate and every role the metric requires. The
    guidance joins the prompt text after a blank line, ahead of the data; an
    item in an optional role adds none.
    """

    def test_an_image_candidate_appends_the_guidance_to_the_rubric(self, png_image):
        image, _ = png_image
        items = {**_text_items(), "prediction": image}
        metric = _metric(modality_guidance={Modality.IMAGE: IMAGE_HINT})

        plain = PromptHandler.build(_metric(), **items)
        request = PromptHandler.build(metric, **items)

        assert plain["prompt"] == RUBRIC
        assert request["prompt"] == f"{RUBRIC}\n\n{IMAGE_HINT}"
        assert request["additional_content"] == plain["additional_content"]
        assert request["additional_content"][-1] == _close("ground_truth_response")

    @pytest.mark.parametrize(
        ("policy", "guided"),
        [(InputPolicy.REQUIRED, True), (InputPolicy.OPTIONAL, False)],
        ids=["required", "optional"],
    )
    @pytest.mark.parametrize("role", AUXILIARY_ROLES)
    def test_an_image_adds_the_guidance_only_in_a_required_role(
        self, png_image, role, policy, guided
    ):
        image, _ = png_image
        items = {**_text_items(), role: image}
        metric = _metric(
            modality_guidance={Modality.IMAGE: IMAGE_HINT}, **{role: policy}
        )

        request = PromptHandler.build(metric, **items)

        assert request["prompt"] == (f"{RUBRIC}\n\n{IMAGE_HINT}" if guided else RUBRIC)
        assert image.get_data_for_LLM_message() in request["additional_content"]

    def test_text_only_sample_renders_no_guidance(self):
        metric = _metric(modality_guidance={Modality.IMAGE: IMAGE_HINT})

        request = PromptHandler.build(metric, **_text_items())

        assert request["prompt"] == RUBRIC
        assert request["additional_content"] == ONE_ITEM_CONTENT
        assert IMAGE_HINT not in "".join(_texts(request))

    def test_guidance_for_several_present_modalities_is_joined_in_modality_order(
        self, png_image
    ):
        image, _ = png_image
        # Declared image-first to prove enum order, not declaration order, wins.
        metric = _metric(
            modality_guidance={Modality.IMAGE: IMAGE_HINT, Modality.TEXT: TEXT_HINT}
        )

        request = PromptHandler.build(metric, **{**_text_items(), "reference": image})

        assert request["prompt"] == f"{RUBRIC}\n\n{TEXT_HINT}\n{IMAGE_HINT}"
        assert request["additional_content"][-1] == _close("ground_truth_response")


# Guidance a test metric returns from ``supplemental``.
SUPPLEMENTAL_GUIDANCE = "Weigh every part of the candidate before scoring."


class TestSupplementalHook:
    """``supplemental`` guidance follows the rubric after a blank line, if non-empty."""

    def test_non_empty_supplemental_follows_the_rubric_after_a_blank_line(self):
        items = _text_items()
        metric = _metric(supplemental=lambda self, sample: SUPPLEMENTAL_GUIDANCE)

        default = PromptHandler.build(_metric(), **items)
        request = PromptHandler.build(metric, **items)

        assert request["prompt"] == f"{RUBRIC}\n\n{SUPPLEMENTAL_GUIDANCE}"
        assert request["additional_content"] == default["additional_content"]
        assert request["system"] == default["system"] == JUDGE_ROLE
        assert request["response_schema"] == JUDGE_RESPONSE_SCHEMA

    def test_empty_supplemental_adds_nothing(self):
        items = _text_items()
        metric = _metric(supplemental=lambda self, sample: "")

        default = PromptHandler.build(_metric(), **items)
        request = PromptHandler.build(metric, **items)

        assert request == default
        assert request["prompt"] == RUBRIC
        assert len(request["additional_content"]) == 12
        assert request["additional_content"][-1] == _close("ground_truth_response")

    @pytest.mark.parametrize(
        ("context_policy", "consumed"),
        [(InputPolicy.OPTIONAL, True), (InputPolicy.OMIT, False)],
        ids=["context-consumed", "context-omitted"],
    )
    def test_supplemental_receives_only_the_consumed_items(
        self, png_image, context_policy, consumed
    ):
        image, _ = png_image
        seen = []

        def supplemental(self, sample):
            seen.append(sample)
            return ""

        metric = _metric(context=context_policy, supplemental=supplemental)
        items = {**_text_items(), "context": image}

        content = PromptHandler.build(metric, **items)["additional_content"]

        assert len(seen) == 1
        sample = seen[0]
        assert sample.prompt == (items["prompt"],)
        assert sample.prediction == (items["prediction"],)
        assert sample.reference == (items["reference"],)
        assert sample.context == ((image,) if consumed else ())
        assert (_open("context", modality=Modality.IMAGE) in content) is consumed


class TestRoleHook:
    """Overriding ``role`` replaces the system prompt and nothing else."""

    def test_role_override_replaces_the_system_prompt_and_leaves_the_rest_unchanged(
        self,
    ):
        role_text = "You are a strict grader of short factual answers."
        items = _text_items()
        metric = _metric(role=lambda self: role_text)

        default = PromptHandler.build(_metric(), **items)
        request = PromptHandler.build(metric, **items)

        assert default["system"] == JUDGE_ROLE
        assert request["system"] == role_text
        assert {key: request[key] for key in REQUEST_KEYS - {"system"}} == {
            key: default[key] for key in REQUEST_KEYS - {"system"}
        }


# The block a test metric prepends to its candidate section.
INTRO_BLOCK = {"text": "\nThe candidate answer follows; judge it as one whole."}


class TestSectionOverrideComposition:
    """A section override built on the default rendering slots into the canonical order."""

    def test_prediction_intro_lands_after_context_and_before_the_candidate_delimiter(
        self,
    ):
        def prediction(self, items):
            # The explicit spelling of ``super().prediction(items)`` for a hook
            # that is attached to the class as a plain function.
            return [INTRO_BLOCK, *LLMMetric.prediction(self, items)]

        metric = _metric(prediction=prediction)
        items = {
            **_text_items(),
            "context": [TextData("context-a"), TextData("context-b")],
        }

        default = PromptHandler.build(_metric(), **items)["additional_content"]
        content = PromptHandler.build(metric, **items)["additional_content"]

        assert content == [
            *_section("prompt", _text_block("prompt-text")),
            *_section("context", _text_block("context-a"), _text_block("context-b")),
            INTRO_BLOCK,
            *_section("prediction", _text_block("prediction-text")),
            *_section("reference", _text_block("reference-text")),
        ]
        intro_at = content.index(INTRO_BLOCK)
        assert content[intro_at - 1] == _close("context")
        assert content[intro_at + 1] == _open("candidate_response")
        assert content[:intro_at] + content[intro_at + 1 :] == default


# Per-role input shape for the randomized rendering test: ``None``, one scalar
# item, or a ``list``/``tuple`` of zero to four items.
ROLE_SHAPES = st.one_of(
    st.none(),
    st.just("scalar"),
    st.tuples(st.sampled_from([list, tuple]), st.integers(min_value=0, max_value=4)),
)


def _materialize_role(role, shape):
    """Fresh, distinct ``OpaqueData`` items for ``role`` plus the value to supply."""
    if shape is None:
        return [], None
    if shape == "scalar":
        item = OpaqueData(f"{role}-0")
        return [item], item
    container, length = shape
    items = [OpaqueData(f"{role}-{index}") for index in range(length)]
    return items, container(items)


class TestRenderingProperty:
    """Any role shapes, policies and multiplicities render as canonical sections."""

    @settings(max_examples=100)
    @given(
        rubric=st.text(),
        shapes=st.fixed_dictionaries({role: ROLE_SHAPES for role in ROLES}),
        policies=st.fixed_dictionaries(
            {role: st.sampled_from(InputPolicy) for role in AUXILIARY_ROLES}
        ),
        multiplicities=st.fixed_dictionaries(
            {role: st.sampled_from(Multiplicity) for role in DEFAULT_MULTIPLICITY}
        ),
    )
    def test_any_role_shapes_policies_and_multiplicities_render_canonically(
        self, rubric, shapes, policies, multiplicities
    ):
        metric = _metric(
            rubric,
            **policies,
            **{f"{role}_multiplicity": value for role, value in multiplicities.items()},
        )
        items = {}
        supplied = {}
        for role in ROLES:
            items[role], supplied[role] = _materialize_role(role, shapes[role])

        request = PromptHandler.build(metric, **supplied)

        # Canonical reference rendering: walk the roles in canonical order, skip
        # OMIT roles entirely, introduce a role that carries several items and
        # has a multiplicity, then emit one open/item/close triple per item in
        # supplied order, numbering the delimiters of a role with several items.
        # Prediction has no policy and always renders.
        expected = []
        rendered = []
        omitted = []
        for role in ROLES:
            if policies.get(role) == InputPolicy.OMIT:
                omitted.extend(items[role])
                continue
            count = len(items[role])
            intro = _intro(role, count, multiplicities.get(role)) if count > 1 else None
            if intro is not None:
                expected.append(intro)
            for index, item in enumerate(items[role]):
                expected.extend(
                    [_open(TAGS[role], index, count), item.block, _close(TAGS[role])]
                )
                rendered.append(item)

        assert set(request) == REQUEST_KEYS
        assert request["system"] == JUDGE_ROLE
        assert request["prompt"] == rubric
        candidate_parts = len(items["prediction"])
        if candidate_parts > 1:
            # The two-part golden schema, sized to this candidate's parts.
            part_scores = {
                **TWO_PART_SCHEMA["properties"]["part_scores"],
                "minItems": candidate_parts,
                "maxItems": candidate_parts,
            }
            assert request["response_schema"] == {
                **TWO_PART_SCHEMA,
                "properties": {
                    **TWO_PART_SCHEMA["properties"],
                    "part_scores": part_scores,
                },
            }
        else:
            assert request["response_schema"] == JUDGE_RESPONSE_SCHEMA
        assert request["additional_content"] == expected
        data_blocks = [
            block for block in request["additional_content"] if "opaque" in block
        ]
        assert all(
            block is item.block
            for block, item in zip(data_blocks, rendered, strict=True)
        )
        assert all(item.conversions == 1 for item in rendered)
        assert all(item.conversions == 0 for item in omitted)

        # A scalar role renders exactly like its singleton sequence.
        as_singletons = {
            role: [value] if shapes[role] == "scalar" else value
            for role, value in supplied.items()
        }
        assert PromptHandler.build(metric, **as_singletons) == request
