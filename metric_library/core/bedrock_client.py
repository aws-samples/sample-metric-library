# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Bedrock client for LLM-based metrics.

This module provides a simple wrapper around AWS Bedrock for making LLM calls.
``query`` sends judge requests through the Bedrock Runtime Converse API, so the
content it accepts and builds consists of Converse ``ContentBlock`` dictionaries
(``{"text": ...}``, ``{"image": {...}}``) that boto3 serializes itself, image
bytes included. ``get_embeddings`` still uses ``invoke_model`` with the Titan
embeddings request body.
"""

import json
import boto3
from botocore.config import Config
import os
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

# Default judge model: the global cross-Region inference profile, which
# accepts requests from every commercial source region. A geographic
# profile (us., eu., au., jp.) works only from its own regions.
DEFAULT_MODEL_ID = "global.anthropic.claude-sonnet-4-6"

# Region used only when neither the arguments, the environment nor the
# selected profile name one.
DEFAULT_REGION = "us-east-1"


class BedrockClient:
    """
    Client for interacting with AWS Bedrock models.

    This client provides a simple interface for querying LLMs through AWS Bedrock,
    with support for text and image inputs, JSON parsing, and embeddings.

    Credentials come from the standard boto3 chain: an explicit ``profile``
    argument, else ``AWS_ACCESS_KEY_ID`` / ``AWS_SECRET_ACCESS_KEY`` /
    ``AWS_SESSION_TOKEN``, else ``AWS_PROFILE``, the shared config and
    credentials files (static keys, SSO, assumed roles, credential_process),
    then container and instance roles.

    Region priority (highest to lowest): the ``region`` argument,
    ``AWS_REGION``, ``AWS_DEFAULT_REGION``, the selected profile's ``region``,
    then ``DEFAULT_REGION``.

    Model priority: the ``model_id`` argument, ``BEDROCK_MODEL_ID``, then
    ``DEFAULT_MODEL_ID``. A geographic inference profile (``us.``, ``eu.``,
    ``au.``, ``jp.``) must match the client's region; the default is a global
    profile, so it works from any region.
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        region: Optional[str] = None,
        profile: Optional[str] = None,
    ):
        """
        Initialize the Bedrock client.

        Args:
            model_id: The Bedrock model ID to use (defaults to env var BEDROCK_MODEL_ID or DEFAULT_MODEL_ID)
            region: AWS region for Bedrock (defaults to AWS_REGION, AWS_DEFAULT_REGION,
                the profile's region, then DEFAULT_REGION)
            profile: AWS profile name. When omitted boto3 applies its default
                chain, in which environment credentials take precedence over
                AWS_PROFILE. An explicit name here makes that profile win.
        """
        self.model_id = model_id or os.getenv("BEDROCK_MODEL_ID") or DEFAULT_MODEL_ID
        self.profile = profile or os.getenv("AWS_PROFILE")

        # One session for every configuration. ``profile_name=None`` lets
        # boto3 read AWS_PROFILE itself, which keeps the standard precedence
        # of environment keys over a profile. boto3 reads AWS_DEFAULT_REGION
        # and the profile's region on its own but not AWS_REGION, so that one
        # is passed through here.
        session = boto3.Session(
            profile_name=profile,
            region_name=region or os.getenv("AWS_REGION") or None,
        )
        self.region = session.region_name or DEFAULT_REGION
        # A judge reply has at most 8,192 output tokens (4,096 outside the
        # matrix calls), which Claude Sonnet 4.6 wrote in 47 seconds, so the
        # client waits up to 600 seconds for a reply: botocore waits 60, and
        # the 3,600 that AWS advises for Claude 4 is for far longer outputs.
        # The engine retries on top of boto3, so boto3 sends a request at most
        # 3 times, the count of its standard mode, set here so that
        # AWS_MAX_ATTEMPTS and AWS_RETRY_MODE cannot raise it. A stalled
        # connection then holds one judge call for at most 600 s x 3 per
        # engine attempt, 90 minutes with the engine's 3 default attempts.
        self.client = session.client(
            "bedrock-runtime",
            region_name=self.region,
            config=Config(
                read_timeout=600,
                retries={"mode": "standard", "total_max_attempts": 3},
            ),
        )

    def query(
        self,
        prompt: Optional[str] = None,
        additional_content: Optional[List[Dict]] = None,
        parse_json: bool = False,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        *,
        system: Optional[str] = None,
        response_schema: Optional[Dict] = None,
    ) -> Any:
        """
        Query the LLM through the Converse API with one user message.

        Args:
            prompt: Optional opening text, sent as the first ``{"text": ...}``
                content block of the user message
            additional_content: Optional Converse ``ContentBlock`` dictionaries
                (``{"text": ...}``, ``{"image": {...}}``), appended after the
                prompt block exactly as given
            parse_json: Whether to parse the text response as JSON; ignored when
                ``response_schema`` is given
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            system: Optional system prompt, sent as one ``{"text": ...}`` system
                block
            response_schema: Optional JSON schema for the result. When given, the
                model is forced to call a ``submit_evaluation`` tool whose input
                schema is this schema, and that tool call's input dictionary is
                returned instead of response text

        Returns:
            str or dict: Response text, parsed JSON, or the forced tool call's
            input dictionary

        Raises:
            ValueError: If the response carries no text block, or no tool call
                when ``response_schema`` is given
        """
        # Build the single user message's content
        content: List[Dict] = []

        # Add prompt as first text block
        if prompt is not None:
            content.append({"text": prompt})

        # Add additional content if provided
        if additional_content:
            content.extend(additional_content)

        # Prepare the Converse request
        kwargs: Dict[str, Any] = {
            "modelId": self.model_id,
            "messages": [{"role": "user", "content": content}],
            "inferenceConfig": {"maxTokens": max_tokens, "temperature": temperature},
        }
        if system is not None:
            kwargs["system"] = [{"text": system}]
        if response_schema is not None:
            kwargs["toolConfig"] = {
                "tools": [
                    {
                        "toolSpec": {
                            "name": "submit_evaluation",
                            "description": "Submit the evaluation result.",
                            "inputSchema": {"json": response_schema},
                        }
                    }
                ],
                "toolChoice": {"tool": {"name": "submit_evaluation"}},
            }

        try:
            # Invoke the model
            response = self.client.converse(**kwargs)
            blocks = response["output"]["message"]["content"]

            # A forced tool call carries the result as the tool input
            if response_schema is not None:
                # A reply cut at the token limit holds a partial tool input.
                if response.get("stopReason") == "max_tokens":
                    raise ValueError(
                        "Judge response stopped at the limit of "
                        f"{max_tokens} output tokens before the evaluation "
                        "tool call was complete"
                    )
                for block in blocks:
                    if "toolUse" in block:
                        return block["toolUse"]["input"]
                raise ValueError(
                    "Judge response did not include the evaluation tool call"
                )

            # Extract text from the first text block
            response_text = next(
                (block["text"] for block in blocks if "text" in block), None
            )
            if response_text is None:
                raise ValueError("No content in response")

            # Parse JSON if requested
            if parse_json:
                # Try to extract JSON from markdown code blocks
                if "```json" in response_text:
                    json_start = response_text.find("```json") + 7
                    json_end = response_text.find("```", json_start)
                    response_text = response_text[json_start:json_end].strip()
                elif "```" in response_text:
                    json_start = response_text.find("```") + 3
                    json_end = response_text.find("```", json_start)
                    response_text = response_text[json_start:json_end].strip()

                return json.loads(response_text)

            return response_text

        except Exception as e:
            # The caller reports the error; the message can quote the request.
            logger.debug("Error querying Bedrock: %s", e)
            raise

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        embeddings = []

        for text in texts:
            request_body = {"inputText": text}

            try:
                response = self.client.invoke_model(
                    modelId=self.model_id, body=json.dumps(request_body)
                )

                response_body = json.loads(response["body"].read())
                embeddings.append(response_body["embedding"])

            except Exception as e:
                # The caller reports the error; the message can quote the text.
                logger.debug("Error getting embeddings: %s", e)
                raise

        return embeddings
