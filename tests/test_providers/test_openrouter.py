"""Tests for the OpenRouter provider."""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest

from custom_components.ai_agent_ha.providers import ProviderRegistry

# Import to trigger registration
from custom_components.ai_agent_ha.providers import openrouter as openrouter_module  # noqa: F401
from custom_components.ai_agent_ha.providers.openrouter import OpenRouterProvider

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant


class TestOpenRouterProviderRegistration:
    """Tests for OpenRouter provider registration."""

    def test_registered_in_registry(self) -> None:
        """Test that 'openrouter' is in available_providers()."""
        available = ProviderRegistry.available_providers()
        assert "openrouter" in available


class TestOpenRouterProviderSupportsTools:
    """Tests for OpenRouter provider tool support."""

    def test_supports_tools(self) -> None:
        """Test that OpenRouter provider returns True for supports_tools."""
        mock_hass = MagicMock()
        config = {"token": "sk-or-test-key"}

        provider = OpenRouterProvider(mock_hass, config)

        assert provider.supports_tools is True


class TestOpenRouterProviderApiUrl:
    """Tests for OpenRouter provider API URL."""

    def test_api_url(self) -> None:
        """Test that api_url returns the correct OpenRouter endpoint."""
        mock_hass = MagicMock()
        config = {"token": "sk-or-test-key"}

        provider = OpenRouterProvider(mock_hass, config)

        assert provider.api_url == "https://openrouter.ai/api/v1/chat/completions"


class TestOpenRouterProviderBuildHeaders:
    """Tests for OpenRouter provider header building."""

    def test_build_headers(self) -> None:
        """Test Authorization, HTTP-Referer, and X-Title headers."""
        mock_hass = MagicMock()
        config = {"token": "sk-or-test-key-12345"}

        provider = OpenRouterProvider(mock_hass, config)
        headers = provider._build_headers()

        assert headers["Authorization"] == "Bearer sk-or-test-key-12345"
        assert headers["Content-Type"] == "application/json"
        assert headers["HTTP-Referer"] == "https://home-assistant.io"
        assert headers["X-Title"] == "AI Agent HA"


class TestOpenRouterProviderBuildPayload:
    """Tests for OpenRouter provider payload building."""

    def test_build_payload(self) -> None:
        """Test that model and messages are in payload (OpenAI-compatible format)."""
        mock_hass = MagicMock()
        config = {"token": "sk-or-test-key", "model": "openai/gpt-4"}

        provider = OpenRouterProvider(mock_hass, config)
        messages = [{"role": "user", "content": "Hello"}]
        payload = provider._build_payload(messages)

        assert payload["model"] == "openai/gpt-4"
        assert payload["messages"] == messages

    def test_build_payload_default_model(self) -> None:
        """Test that default model is used when not specified."""
        mock_hass = MagicMock()
        config = {"token": "sk-or-test-key"}

        provider = OpenRouterProvider(mock_hass, config)
        messages = [{"role": "user", "content": "Hello"}]
        payload = provider._build_payload(messages)

        assert payload["model"] == "openai/gpt-4"
        assert payload["messages"] == messages

    def test_build_payload_with_tools(self) -> None:
        """Test that tools are included when passed."""
        mock_hass = MagicMock()
        config = {"token": "sk-or-test-key", "model": "openai/gpt-4"}

        provider = OpenRouterProvider(mock_hass, config)
        messages = [{"role": "user", "content": "Turn on the lights"}]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "turn_on_light",
                    "description": "Turn on a light",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "entity_id": {"type": "string"}
                        },
                        "required": ["entity_id"],
                    },
                },
            }
        ]

        payload = provider._build_payload(messages, tools=tools)

        assert payload["model"] == "openai/gpt-4"
        assert payload["messages"] == messages
        assert payload["tools"] == tools


class TestOpenRouterProviderExtractResponse:
    """Tests for OpenRouter provider response extraction."""

    def test_extract_response(self) -> None:
        """Test extraction from choices[0].message.content (OpenAI-compatible)."""
        mock_hass = MagicMock()
        config = {"token": "sk-or-test-key"}

        provider = OpenRouterProvider(mock_hass, config)
        response_data = {
            "id": "gen-123",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello! How can I help you today?",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25},
        }

        result = provider._extract_response(response_data)

        assert result == "Hello! How can I help you today?"

    def test_extract_response_with_tool_calls(self) -> None:
        """Test handling of tool_calls in response."""
        mock_hass = MagicMock()
        config = {"token": "sk-or-test-key"}

        provider = OpenRouterProvider(mock_hass, config)
        response_data = {
            "id": "gen-123",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_abc123",
                                "type": "function",
                                "function": {
                                    "name": "turn_on_light",
                                    "arguments": '{"entity_id": "light.living_room"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }

        result = provider._extract_response(response_data)

        # When tool_calls are present, return a JSON string with the tool calls
        parsed_result = json.loads(result)
        assert "tool_calls" in parsed_result
        assert len(parsed_result["tool_calls"]) == 1
        assert parsed_result["tool_calls"][0]["function"]["name"] == "turn_on_light"

    def test_extract_response_empty_content(self) -> None:
        """Test handling of empty content in response."""
        mock_hass = MagicMock()
        config = {"token": "sk-or-test-key"}

        provider = OpenRouterProvider(mock_hass, config)
        response_data = {
            "id": "gen-123",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                    },
                }
            ],
        }

        result = provider._extract_response(response_data)

        assert result == ""

    def test_extract_response_no_choices(self) -> None:
        """Test handling of response with no choices."""
        mock_hass = MagicMock()
        config = {"token": "sk-or-test-key"}

        provider = OpenRouterProvider(mock_hass, config)
        response_data = {
            "id": "gen-123",
            "choices": [],
        }

        result = provider._extract_response(response_data)

        assert result == ""
