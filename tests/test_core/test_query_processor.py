"""Tests for QueryProcessor."""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.ai_agent_ha.core.query_processor import (
    INVISIBLE_CHARS,
    QueryProcessor,
)


class MockProvider:
    """Mock AI provider for testing."""

    def __init__(self, response: str = "Test response") -> None:
        """Initialize with a canned response."""
        self._response = response
        self.get_response = AsyncMock(return_value=response)
        self.supports_tools = True

    async def get_response_async(
        self, messages: list[dict[str, Any]], **kwargs: Any
    ) -> str:
        """Get a mock response."""
        return await self.get_response(messages, **kwargs)


class TestSanitizeQuery:
    """Tests for _sanitize_query method."""

    def test_sanitize_query_removes_invisible_chars(self) -> None:
        """Test that invisible characters (BOM, zero-width) are removed."""
        provider = MockProvider()
        processor = QueryProcessor(provider)

        # Build query with various invisible characters
        query_with_invisibles = (
            "\ufeffHello\u200bWorld\u200c!\u200d\u2060"  # BOM, ZW-space, ZWNJ, ZWJ, WJ
        )
        result = processor._sanitize_query(query_with_invisibles)

        assert result == "HelloWorld!"
        # Verify no invisible chars remain
        for char in INVISIBLE_CHARS:
            assert char not in result

    def test_sanitize_query_truncates_long_query(self) -> None:
        """Test that long queries are truncated to max_length."""
        provider = MockProvider()
        processor = QueryProcessor(provider, max_query_length=50)

        long_query = "a" * 100
        result = processor._sanitize_query(long_query)

        assert len(result) == 50
        assert result == "a" * 50

    def test_sanitize_query_truncates_with_custom_max_length(self) -> None:
        """Test truncation with explicit max_length parameter."""
        provider = MockProvider()
        processor = QueryProcessor(provider, max_query_length=1000)

        long_query = "b" * 200
        result = processor._sanitize_query(long_query, max_length=30)

        assert len(result) == 30
        assert result == "b" * 30

    def test_sanitize_query_strips_whitespace(self) -> None:
        """Test that leading/trailing whitespace is stripped."""
        provider = MockProvider()
        processor = QueryProcessor(provider)

        query_with_whitespace = "   Hello World   \n\t"
        result = processor._sanitize_query(query_with_whitespace)

        assert result == "Hello World"

    def test_sanitize_query_handles_empty_string(self) -> None:
        """Test that empty strings are handled gracefully."""
        provider = MockProvider()
        processor = QueryProcessor(provider)

        result = processor._sanitize_query("")
        assert result == ""

        result = processor._sanitize_query("   ")
        assert result == ""


class TestBuildMessages:
    """Tests for _build_messages method."""

    def test_build_messages_with_history(self) -> None:
        """Test that history is combined with new query."""
        provider = MockProvider()
        processor = QueryProcessor(provider)

        history = [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"},
        ]
        query = "New question"

        result = processor._build_messages(query, history)

        assert len(result) == 3
        assert result[0] == {"role": "user", "content": "Previous question"}
        assert result[1] == {"role": "assistant", "content": "Previous answer"}
        assert result[2] == {"role": "user", "content": "New question"}

    def test_build_messages_with_system_prompt(self) -> None:
        """Test that system prompt is added first."""
        provider = MockProvider()
        processor = QueryProcessor(provider)

        history = [{"role": "user", "content": "Previous question"}]
        query = "New question"
        system_prompt = "You are a helpful assistant."

        result = processor._build_messages(query, history, system_prompt=system_prompt)

        assert len(result) == 3
        assert result[0] == {"role": "system", "content": "You are a helpful assistant."}
        assert result[1] == {"role": "user", "content": "Previous question"}
        assert result[2] == {"role": "user", "content": "New question"}

    def test_build_messages_without_history(self) -> None:
        """Test building messages with no history."""
        provider = MockProvider()
        processor = QueryProcessor(provider)

        query = "Hello"
        result = processor._build_messages(query, [])

        assert len(result) == 1
        assert result[0] == {"role": "user", "content": "Hello"}

    def test_build_messages_with_system_prompt_no_history(self) -> None:
        """Test system prompt with empty history."""
        provider = MockProvider()
        processor = QueryProcessor(provider)

        query = "Hello"
        result = processor._build_messages(query, [], system_prompt="Be concise.")

        assert len(result) == 2
        assert result[0] == {"role": "system", "content": "Be concise."}
        assert result[1] == {"role": "user", "content": "Hello"}


class TestProcess:
    """Tests for the process method."""

    @pytest.mark.asyncio
    async def test_process_simple_query(self) -> None:
        """Test processing a simple query returns success with response."""
        provider = MockProvider(response="Hello! How can I help?")
        processor = QueryProcessor(provider)

        result = await processor.process(query="Hi there", messages=[])

        assert result["success"] is True
        assert result["response"] == "Hello! How can I help?"
        assert "messages" in result
        provider.get_response.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_empty_query_fails(self) -> None:
        """Test that empty query returns an error."""
        provider = MockProvider()
        processor = QueryProcessor(provider)

        result = await processor.process(query="", messages=[])

        assert result["success"] is False
        assert "error" in result
        provider.get_response.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_whitespace_only_query_fails(self) -> None:
        """Test that whitespace-only query returns an error."""
        provider = MockProvider()
        processor = QueryProcessor(provider)

        result = await processor.process(query="   \n\t  ", messages=[])

        assert result["success"] is False
        assert "error" in result
        provider.get_response.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_with_tools(self) -> None:
        """Test that tools are passed to provider."""
        provider = MockProvider(response="Using a tool")
        processor = QueryProcessor(provider)

        tools = [
            {"name": "get_weather", "description": "Get weather info"},
            {"name": "search", "description": "Search the web"},
        ]

        result = await processor.process(query="What's the weather?", messages=[], tools=tools)

        assert result["success"] is True
        provider.get_response.assert_called_once()
        # Verify tools were passed to get_response
        call_kwargs = provider.get_response.call_args.kwargs
        assert call_kwargs.get("tools") == tools

    @pytest.mark.asyncio
    async def test_process_with_system_prompt(self) -> None:
        """Test that system prompt is included in messages."""
        provider = MockProvider(response="I am helpful.")
        processor = QueryProcessor(provider)

        result = await processor.process(
            query="Hello",
            messages=[],
            system_prompt="You are a helpful assistant.",
        )

        assert result["success"] is True
        # Verify the messages passed to provider include system prompt
        call_args = provider.get_response.call_args.args[0]
        assert call_args[0]["role"] == "system"
        assert call_args[0]["content"] == "You are a helpful assistant."

    @pytest.mark.asyncio
    async def test_process_with_history(self) -> None:
        """Test that conversation history is preserved."""
        provider = MockProvider(response="Continuing conversation")
        processor = QueryProcessor(provider)

        history = [
            {"role": "user", "content": "First message"},
            {"role": "assistant", "content": "First response"},
        ]

        result = await processor.process(query="Second message", messages=history)

        assert result["success"] is True
        # Verify the messages passed include history
        call_args = provider.get_response.call_args.args[0]
        assert len(call_args) == 3
        assert call_args[0]["content"] == "First message"
        assert call_args[1]["content"] == "First response"
        assert call_args[2]["content"] == "Second message"

    @pytest.mark.asyncio
    async def test_process_returns_updated_messages(self) -> None:
        """Test that process returns the updated message list."""
        provider = MockProvider(response="Response text")
        processor = QueryProcessor(provider)

        result = await processor.process(query="Hello", messages=[])

        assert "messages" in result
        messages = result["messages"]
        # Should have user message + assistant response
        assert len(messages) == 2
        assert messages[0] == {"role": "user", "content": "Hello"}
        assert messages[1] == {"role": "assistant", "content": "Response text"}

    @pytest.mark.asyncio
    async def test_process_handles_provider_error(self) -> None:
        """Test that provider exceptions are handled gracefully."""
        provider = MockProvider()
        provider.get_response = AsyncMock(side_effect=Exception("API Error"))
        processor = QueryProcessor(provider)

        result = await processor.process(query="Hello", messages=[])

        assert result["success"] is False
        assert "error" in result
        assert "API Error" in result["error"]
