# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Behavioral tests for ``BedrockClient`` against a mocked Bedrock Runtime client.

``query`` is exercised through the exact Converse request it builds and the
result it extracts from the Converse response; ``get_embeddings`` is only
checked to still go through ``invoke_model``. No live AWS is ever reached.
"""

import io
import json
import logging
from unittest.mock import Mock, call

import pytest

from metric_library.core.bedrock_client import BedrockClient

# A judge result schema of the kind a forced tool call would be given.
EVALUATION_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "required": ["reasoning", "score"],
}

JUDGE_RESULT = {"reasoning": "ok", "score": 0.5}


@pytest.fixture
def runtime(monkeypatch):
    """The mocked ``bedrock-runtime`` client every ``BedrockClient`` here wraps.

    ``BedrockClient`` builds one ``boto3.Session`` and asks it for the runtime
    client, so the session is stubbed to hand back this mock; a bare
    ``boto3.client`` call would bypass the session and is forbidden.
    """
    client = Mock(spec=["converse", "invoke_model"])

    def fail(*args, **kwargs):
        raise AssertionError("live AWS is forbidden in tests")

    def session(*args, region_name=None, **kwargs):
        return Mock(region_name=region_name, client=Mock(return_value=client))

    monkeypatch.delenv("AWS_PROFILE", raising=False)
    monkeypatch.setattr("metric_library.core.bedrock_client.boto3.Session", session)
    monkeypatch.setattr("metric_library.core.bedrock_client.boto3.client", fail)
    return client


@pytest.fixture
def bedrock(runtime):
    return BedrockClient(model_id="test-model", region="us-east-1")


def _converse_response(*blocks):
    """A Converse response whose assistant message carries exactly ``blocks``."""
    return {"output": {"message": {"role": "assistant", "content": list(blocks)}}}


def _text_response(text):
    return _converse_response({"text": text})


class TestConverseRequest:
    """``query`` sends one user message through ``converse`` with explicit settings."""

    def test_prompt_only_query_sends_the_minimal_request_and_returns_text(
        self, runtime, bedrock
    ):
        runtime.converse.return_value = _text_response("world")

        result = bedrock.query("hello")

        assert result == "world"
        assert runtime.converse.call_args_list == [
            call(
                modelId="test-model",
                messages=[{"role": "user", "content": [{"text": "hello"}]}],
                inferenceConfig={"maxTokens": 4096, "temperature": 0.0},
            )
        ]
        assert set(runtime.converse.call_args.kwargs) == {
            "modelId",
            "messages",
            "inferenceConfig",
        }

    def test_additional_content_blocks_follow_the_prompt_unchanged(
        self, runtime, bedrock
    ):
        runtime.converse.return_value = _text_response("ok")
        png_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        text_block = {"text": "\nCandidate Response (image) :"}
        image_block = {"image": {"format": "png", "source": {"bytes": png_bytes}}}

        bedrock.query("hello", additional_content=[text_block, image_block])

        content = runtime.converse.call_args.kwargs["messages"][0]["content"]
        assert content == [{"text": "hello"}, text_block, image_block]
        assert content[1] is text_block
        assert content[2] is image_block
        assert content[2]["image"]["source"]["bytes"] is png_bytes

    def test_without_prompt_the_user_content_is_exactly_the_additional_blocks(
        self, runtime, bedrock
    ):
        runtime.converse.return_value = _text_response("ok")
        blocks = [{"text": "\nCandidate Response :"}, {"text": "candidate-0"}]

        bedrock.query(prompt=None, additional_content=blocks)

        assert runtime.converse.call_args.kwargs["messages"] == [
            {"role": "user", "content": blocks}
        ]

    def test_system_prompt_is_sent_as_one_system_text_block(self, runtime, bedrock):
        runtime.converse.return_value = _text_response("ok")

        bedrock.query("hello", system="be strict")

        kwargs = runtime.converse.call_args.kwargs
        assert kwargs["system"] == [{"text": "be strict"}]
        assert kwargs["messages"] == [{"role": "user", "content": [{"text": "hello"}]}]
        assert "toolConfig" not in kwargs

    def test_response_schema_forces_the_evaluation_tool(self, runtime, bedrock):
        runtime.converse.return_value = _converse_response(
            {
                "toolUse": {
                    "toolUseId": "t1",
                    "name": "submit_evaluation",
                    "input": JUDGE_RESULT,
                }
            }
        )

        bedrock.query("hello", response_schema=EVALUATION_SCHEMA)

        assert runtime.converse.call_args.kwargs["toolConfig"] == {
            "tools": [
                {
                    "toolSpec": {
                        "name": "submit_evaluation",
                        "description": "Submit the evaluation result.",
                        "inputSchema": {"json": EVALUATION_SCHEMA},
                    }
                }
            ],
            "toolChoice": {"tool": {"name": "submit_evaluation"}},
        }


class TestConverseResponse:
    """``query`` extracts text, parsed JSON or the forced tool input from the reply."""

    def test_text_is_taken_from_the_first_text_block(self, runtime, bedrock):
        runtime.converse.return_value = _converse_response(
            {"reasoningContent": {"reasoningText": {"text": "thinking"}}},
            {"text": "first"},
            {"text": "second"},
        )

        assert bedrock.query("hello") == "first"

    @pytest.mark.parametrize(
        "text",
        [
            'Here is the result:\n```json\n{"reasoning": "ok", "score": 0.5}\n```\n',
            '```\n{"reasoning": "ok", "score": 0.5}\n```',
            '{"reasoning": "ok", "score": 0.5}',
        ],
        ids=["json-fence", "bare-fence", "unfenced"],
    )
    def test_parse_json_returns_the_parsed_object(self, runtime, bedrock, text):
        runtime.converse.return_value = _text_response(text)

        assert bedrock.query("hello", parse_json=True) == JUDGE_RESULT

    def test_response_schema_returns_the_tool_input(self, runtime, bedrock):
        runtime.converse.return_value = _converse_response(
            {
                "toolUse": {
                    "toolUseId": "t1",
                    "name": "submit_evaluation",
                    "input": JUDGE_RESULT,
                }
            }
        )

        result = bedrock.query("hello", response_schema=EVALUATION_SCHEMA)

        assert result == JUDGE_RESULT

    def test_response_schema_without_a_tool_call_raises(self, runtime, bedrock):
        runtime.converse.return_value = _text_response(json.dumps(JUDGE_RESULT))

        with pytest.raises(ValueError) as exc_info:
            bedrock.query("hello", response_schema=EVALUATION_SCHEMA)

        assert str(exc_info.value) == (
            "Judge response did not include the evaluation tool call"
        )

    def test_a_tool_call_cut_at_the_token_limit_raises(self, runtime, bedrock):
        """The partial tool input of a truncated reply is not returned."""
        runtime.converse.return_value = {
            **_converse_response(
                {
                    "toolUse": {
                        "toolUseId": "t1",
                        "name": "submit_evaluation",
                        "input": {"reasoning": "The candidate"},
                    }
                }
            ),
            "stopReason": "max_tokens",
        }

        with pytest.raises(ValueError) as exc_info:
            bedrock.query("hello", max_tokens=8192, response_schema=EVALUATION_SCHEMA)

        assert str(exc_info.value) == (
            "Judge response stopped at the limit of 8192 output tokens before "
            "the evaluation tool call was complete"
        )

    def test_empty_response_content_raises(self, runtime, bedrock):
        runtime.converse.return_value = _converse_response()

        with pytest.raises(ValueError) as exc_info:
            bedrock.query("hello")

        assert str(exc_info.value) == "No content in response"

    def test_provider_exception_propagates_unchanged(self, runtime, bedrock, caplog):
        caplog.set_level(logging.DEBUG, logger="metric_library.core.bedrock_client")
        provider_error = RuntimeError("provider failed")
        runtime.converse.side_effect = provider_error

        with pytest.raises(RuntimeError) as exc_info:
            bedrock.query("hello")

        assert exc_info.value is provider_error
        # The caller reports the error; stderr gets no copy of its message.
        assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


class TestEmbeddings:
    """``get_embeddings`` stays on ``invoke_model`` with the Titan request body."""

    def test_get_embeddings_still_uses_invoke_model(self, runtime, bedrock):
        runtime.invoke_model.return_value = {
            "body": io.BytesIO(json.dumps({"embedding": [0.1, 0.2]}).encode("utf-8"))
        }

        assert bedrock.get_embeddings(["hello"]) == [[0.1, 0.2]]
        assert runtime.invoke_model.call_args_list == [
            call(modelId="test-model", body=json.dumps({"inputText": "hello"}))
        ]
        runtime.converse.assert_not_called()
