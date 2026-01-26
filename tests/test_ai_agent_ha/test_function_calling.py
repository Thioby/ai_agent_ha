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


# Tests for ToolSchemaConverter will be added in TODO 1
class TestToolSchemaConverter:
    """Test ToolSchemaConverter - converts Tool metadata to provider formats."""

    pass  # Tests will be added in TODO 1


# Tests for FunctionCallHandler will be added in TODO 3
class TestFunctionCallHandler:
    """Test FunctionCallHandler - parses function calls from responses."""

    pass  # Tests will be added in TODO 3


# Tests for UnexpectedToolCallHandler will be added in TODO 5
class TestUnexpectedToolCallHandler:
    """Test UnexpectedToolCallHandler - handles Gemini's UNEXPECTED_TOOL_CALL."""

    pass  # Tests will be added in TODO 5
