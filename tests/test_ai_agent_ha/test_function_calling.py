"""Tests for function calling module - TDD approach.

These tests define the expected behavior of the function_calling module
for native LLM function calling across multiple providers.
"""

import json
import os
import sys

import pytest

# Add the custom_components/ai_agent_ha directory directly to path
# This allows importing function_calling without triggering __init__.py
# which requires homeassistant module
_module_path = os.path.join(
    os.path.dirname(__file__), "..", "..", "custom_components", "ai_agent_ha"
)
sys.path.insert(0, _module_path)

# Direct import from the module file
from function_calling import (
    FunctionCall,
    FunctionCallHandler,
    ToolSchemaConverter,
    UnexpectedToolCallHandler,
)


class TestFunctionCallDataclass:
    """Test FunctionCall dataclass."""

    def test_function_call_creation(self):
        """FunctionCall should store id, name, and arguments."""
        fc = FunctionCall(
            id="call_123", name="get_weather", arguments={"location": "Paris"}
        )

        assert fc.id == "call_123"
        assert fc.name == "get_weather"
        assert fc.arguments == {"location": "Paris"}


# Tests for ToolSchemaConverter - TODO 1
class TestToolSchemaConverter:
    """Test ToolSchemaConverter - converts Tool metadata to provider formats."""

    @pytest.fixture
    def mock_tool(self):
        """Create a mock tool for testing."""
        from dataclasses import dataclass
        from typing import ClassVar, List

        # Import ToolParameter from the tools module
        _tools_path = os.path.join(_module_path, "tools")
        sys.path.insert(0, _tools_path)
        from base import ToolParameter

        @dataclass
        class MockTool:
            """Mock tool for testing schema conversion."""

            id: ClassVar[str] = "get_weather"
            description: ClassVar[str] = "Get current weather for a location"
            parameters: ClassVar[List[ToolParameter]] = [
                ToolParameter(
                    name="location",
                    type="string",
                    description="City and country, e.g. Paris, France",
                    required=True,
                ),
                ToolParameter(
                    name="units",
                    type="string",
                    description="Temperature units",
                    required=False,
                    default="celsius",
                    enum=["celsius", "fahrenheit"],
                ),
            ]

        return MockTool()

    @pytest.fixture
    def mock_tool_simple(self):
        """Create a simple mock tool without optional params."""
        from dataclasses import dataclass
        from typing import ClassVar, List

        _tools_path = os.path.join(_module_path, "tools")
        sys.path.insert(0, _tools_path)
        from base import ToolParameter

        @dataclass
        class MockToolSimple:
            """Simple mock tool for testing."""

            id: ClassVar[str] = "get_entities"
            description: ClassVar[str] = "Get all entities in a domain"
            parameters: ClassVar[List[ToolParameter]] = [
                ToolParameter(
                    name="domain",
                    type="str",
                    description="Entity domain like 'light' or 'switch'",
                    required=True,
                ),
            ]

        return MockToolSimple()

    # ===== OpenAI Format Tests =====

    def test_to_openai_format_basic_structure(self, mock_tool):
        """OpenAI format should have type='function' and function object."""
        result = ToolSchemaConverter.to_openai_format([mock_tool])

        assert len(result) == 1
        tool = result[0]
        assert tool["type"] == "function"
        assert "function" in tool
        assert tool["function"]["name"] == "get_weather"
        assert tool["function"]["description"] == "Get current weather for a location"

    def test_to_openai_format_parameters(self, mock_tool):
        """OpenAI format should have proper parameter schema."""
        result = ToolSchemaConverter.to_openai_format([mock_tool])

        params = result[0]["function"]["parameters"]
        assert params["type"] == "object"
        assert "properties" in params
        assert "location" in params["properties"]
        assert "units" in params["properties"]

        # Required params
        assert "required" in params
        assert "location" in params["required"]
        assert "units" not in params["required"]

    def test_to_openai_format_type_mapping(self, mock_tool_simple):
        """OpenAI format should map 'str' to 'string'."""
        result = ToolSchemaConverter.to_openai_format([mock_tool_simple])

        props = result[0]["function"]["parameters"]["properties"]
        assert props["domain"]["type"] == "string"

    def test_to_openai_format_enum_values(self, mock_tool):
        """OpenAI format should include enum values."""
        result = ToolSchemaConverter.to_openai_format([mock_tool])

        units_prop = result[0]["function"]["parameters"]["properties"]["units"]
        assert "enum" in units_prop
        assert units_prop["enum"] == ["celsius", "fahrenheit"]

    def test_to_openai_format_multiple_tools(self, mock_tool, mock_tool_simple):
        """Should handle multiple tools."""
        result = ToolSchemaConverter.to_openai_format([mock_tool, mock_tool_simple])

        assert len(result) == 2
        names = [t["function"]["name"] for t in result]
        assert "get_weather" in names
        assert "get_entities" in names

    # ===== Anthropic Format Tests =====

    def test_to_anthropic_format_basic_structure(self, mock_tool):
        """Anthropic format should have name, description, and input_schema."""
        result = ToolSchemaConverter.to_anthropic_format([mock_tool])

        assert len(result) == 1
        tool = result[0]
        assert tool["name"] == "get_weather"
        assert tool["description"] == "Get current weather for a location"
        assert "input_schema" in tool

    def test_to_anthropic_format_input_schema(self, mock_tool):
        """Anthropic format should have proper input_schema."""
        result = ToolSchemaConverter.to_anthropic_format([mock_tool])

        schema = result[0]["input_schema"]
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "location" in schema["properties"]
        assert "required" in schema

    def test_to_anthropic_format_type_mapping(self, mock_tool_simple):
        """Anthropic format should map types correctly."""
        result = ToolSchemaConverter.to_anthropic_format([mock_tool_simple])

        props = result[0]["input_schema"]["properties"]
        assert props["domain"]["type"] == "string"

    # ===== Gemini Format Tests =====

    def test_to_gemini_format_basic_structure(self, mock_tool):
        """Gemini format should have functionDeclarations array."""
        result = ToolSchemaConverter.to_gemini_format([mock_tool])

        assert len(result) == 1
        tool = result[0]
        assert "functionDeclarations" in tool

        func_decl = tool["functionDeclarations"][0]
        assert func_decl["name"] == "get_weather"
        assert func_decl["description"] == "Get current weather for a location"

    def test_to_gemini_format_parameters(self, mock_tool):
        """Gemini format should have parameters with UPPERCASE types."""
        result = ToolSchemaConverter.to_gemini_format([mock_tool])

        params = result[0]["functionDeclarations"][0]["parameters"]
        assert params["type"] == "OBJECT"
        assert "properties" in params

    def test_to_gemini_format_type_mapping(self, mock_tool_simple):
        """Gemini format should map types to UPPERCASE."""
        result = ToolSchemaConverter.to_gemini_format([mock_tool_simple])

        props = result[0]["functionDeclarations"][0]["parameters"]["properties"]
        assert props["domain"]["type"] == "STRING"

    def test_to_gemini_format_required_params(self, mock_tool):
        """Gemini format should include required array."""
        result = ToolSchemaConverter.to_gemini_format([mock_tool])

        params = result[0]["functionDeclarations"][0]["parameters"]
        assert "required" in params
        assert "location" in params["required"]


# Tests for FunctionCallHandler - TODO 3
class TestFunctionCallHandler:
    """Test FunctionCallHandler - parses function calls from responses."""

    # ===== OpenAI Response Parsing =====

    @pytest.fixture
    def openai_response_with_function_call(self):
        """OpenAI response containing a function call."""
        return {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call_abc123",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "Paris, France"}',
                                },
                            }
                        ]
                    }
                }
            ]
        }

    @pytest.fixture
    def openai_response_text_only(self):
        """OpenAI response with text only (no function call)."""
        return {"choices": [{"message": {"content": "The weather is nice today."}}]}

    @pytest.fixture
    def openai_response_multiple_calls(self):
        """OpenAI response with multiple parallel function calls."""
        return {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "Paris"}',
                                },
                            },
                            {
                                "id": "call_2",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "London"}',
                                },
                            },
                        ]
                    }
                }
            ]
        }

    def test_parse_openai_response_extracts_function_call(
        self, openai_response_with_function_call
    ):
        """Should extract function name and arguments from OpenAI response."""
        result = FunctionCallHandler.parse_openai_response(
            openai_response_with_function_call
        )

        assert result is not None
        assert len(result) == 1
        fc = result[0]
        assert fc.id == "call_abc123"
        assert fc.name == "get_weather"
        assert fc.arguments == {"location": "Paris, France"}

    def test_parse_openai_response_returns_none_for_text(
        self, openai_response_text_only
    ):
        """Should return None when no function call present."""
        result = FunctionCallHandler.parse_openai_response(openai_response_text_only)
        assert result is None

    def test_parse_openai_response_handles_multiple_calls(
        self, openai_response_multiple_calls
    ):
        """Should parse multiple parallel function calls."""
        result = FunctionCallHandler.parse_openai_response(
            openai_response_multiple_calls
        )

        assert result is not None
        assert len(result) == 2
        assert result[0].id == "call_1"
        assert result[1].id == "call_2"

    def test_is_function_call_openai_true(self, openai_response_with_function_call):
        """Should detect function call in OpenAI response."""
        assert (
            FunctionCallHandler.is_function_call(
                openai_response_with_function_call, "openai"
            )
            is True
        )

    def test_is_function_call_openai_false(self, openai_response_text_only):
        """Should return False for text-only OpenAI response."""
        assert (
            FunctionCallHandler.is_function_call(openai_response_text_only, "openai")
            is False
        )

    # ===== Anthropic Response Parsing =====

    @pytest.fixture
    def anthropic_response_with_tool_use(self):
        """Anthropic response containing a tool_use block."""
        return {
            "content": [
                {"type": "text", "text": "Let me check the weather for you."},
                {
                    "type": "tool_use",
                    "id": "toolu_abc123",
                    "name": "get_weather",
                    "input": {"location": "Paris, France"},
                },
            ]
        }

    @pytest.fixture
    def anthropic_response_text_only(self):
        """Anthropic response with text only."""
        return {"content": [{"type": "text", "text": "Hello, how can I help?"}]}

    def test_parse_anthropic_response_extracts_tool_use(
        self, anthropic_response_with_tool_use
    ):
        """Should extract tool_use from Anthropic response."""
        result = FunctionCallHandler.parse_anthropic_response(
            anthropic_response_with_tool_use
        )

        assert result is not None
        assert len(result) == 1
        fc = result[0]
        assert fc.id == "toolu_abc123"
        assert fc.name == "get_weather"
        assert fc.arguments == {"location": "Paris, France"}

    def test_parse_anthropic_response_returns_none_for_text(
        self, anthropic_response_text_only
    ):
        """Should return None when no tool_use present."""
        result = FunctionCallHandler.parse_anthropic_response(
            anthropic_response_text_only
        )
        assert result is None

    def test_is_function_call_anthropic_true(self, anthropic_response_with_tool_use):
        """Should detect tool_use in Anthropic response."""
        assert (
            FunctionCallHandler.is_function_call(
                anthropic_response_with_tool_use, "anthropic"
            )
            is True
        )

    def test_is_function_call_anthropic_false(self, anthropic_response_text_only):
        """Should return False for text-only Anthropic response."""
        assert (
            FunctionCallHandler.is_function_call(
                anthropic_response_text_only, "anthropic"
            )
            is False
        )

    # ===== Gemini Response Parsing =====

    @pytest.fixture
    def gemini_response_with_function_call(self):
        """Gemini response containing a functionCall."""
        return {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "get_weather",
                                    "args": {"location": "Paris, France"},
                                }
                            }
                        ]
                    }
                }
            ]
        }

    @pytest.fixture
    def gemini_response_text_only(self):
        """Gemini response with text only."""
        return {
            "candidates": [{"content": {"parts": [{"text": "The weather is sunny."}]}}]
        }

    def test_parse_gemini_response_extracts_function_call(
        self, gemini_response_with_function_call
    ):
        """Should extract functionCall from Gemini response."""
        result = FunctionCallHandler.parse_gemini_response(
            gemini_response_with_function_call
        )

        assert result is not None
        assert len(result) == 1
        fc = result[0]
        assert fc.name == "get_weather"
        assert fc.arguments == {"location": "Paris, France"}

    def test_parse_gemini_response_returns_none_for_text(
        self, gemini_response_text_only
    ):
        """Should return None when no functionCall present."""
        result = FunctionCallHandler.parse_gemini_response(gemini_response_text_only)
        assert result is None

    def test_is_function_call_gemini_true(self, gemini_response_with_function_call):
        """Should detect functionCall in Gemini response."""
        assert (
            FunctionCallHandler.is_function_call(
                gemini_response_with_function_call, "gemini"
            )
            is True
        )

    def test_is_function_call_gemini_false(self, gemini_response_text_only):
        """Should return False for text-only Gemini response."""
        assert (
            FunctionCallHandler.is_function_call(gemini_response_text_only, "gemini")
            is False
        )


# Tests for UnexpectedToolCallHandler will be added in TODO 5
class TestUnexpectedToolCallHandler:
    """Test UnexpectedToolCallHandler - handles Gemini's UNEXPECTED_TOOL_CALL."""

    pass  # Tests will be added in TODO 5
