"""Comprehensive tests for AI Agent HA tools.

Tests cover:
- Tool base classes and registry
- WebFetch tool functionality
- WebSearch tool functionality
- Error handling and edge cases
- HTML to Markdown conversion
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import sys
import os

# Add parent directory to path for imports
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "custom_components", "ai_agent_ha")
)

from tools.base import (
    Tool,
    ToolCategory,
    ToolExecutionError,
    ToolParameter,
    ToolRegistry,
    ToolResult,
)
from tools.webfetch import (
    WebFetchTool,
    convert_html_to_markdown,
    extract_text_from_html,
    get_accept_header,
    MAX_RESPONSE_SIZE,
    DEFAULT_TIMEOUT,
    MAX_TIMEOUT,
)
from tools.websearch import WebSearchTool


class TestToolParameter:
    """Tests for ToolParameter class."""

    def test_required_parameter_validation(self):
        """Test required parameter validation."""
        param = ToolParameter(
            name="url",
            type="string",
            description="The URL",
            required=True,
        )

        assert param.validate("https://example.com") is True
        assert param.validate(None) is False
        assert param.validate("") is True  # Empty string is valid

    def test_optional_parameter_validation(self):
        """Test optional parameter validation."""
        param = ToolParameter(
            name="format",
            type="string",
            description="Output format",
            required=False,
            default="markdown",
        )

        assert param.validate("text") is True
        assert param.validate(None) is True

    def test_enum_validation(self):
        """Test enum parameter validation."""
        param = ToolParameter(
            name="format",
            type="string",
            description="Output format",
            required=True,
            enum=["markdown", "text", "html"],
        )

        assert param.validate("markdown") is True
        assert param.validate("text") is True
        assert param.validate("invalid") is False

    def test_type_validation(self):
        """Test type validation."""
        int_param = ToolParameter(
            name="count", type="integer", description="Count", required=True
        )
        assert int_param.validate(10) is True
        assert int_param.validate("10") is False

        str_param = ToolParameter(
            name="name", type="string", description="Name", required=True
        )
        assert str_param.validate("test") is True
        assert str_param.validate(123) is False

        bool_param = ToolParameter(
            name="flag", type="boolean", description="Flag", required=True
        )
        assert bool_param.validate(True) is True
        assert bool_param.validate("true") is False


class TestToolResult:
    """Tests for ToolResult class."""

    def test_success_result(self):
        """Test successful result creation."""
        result = ToolResult(
            output="Content here",
            success=True,
            title="Test Result",
            metadata={"key": "value"},
        )

        assert result.output == "Content here"
        assert result.success is True
        assert result.title == "Test Result"
        assert result.metadata == {"key": "value"}
        assert result.error is None

    def test_error_result(self):
        """Test error result creation."""
        result = ToolResult(
            output="",
            success=False,
            error="Something went wrong",
        )

        assert result.success is False
        assert result.error == "Something went wrong"

    def test_to_dict(self):
        """Test serialization to dictionary."""
        result = ToolResult(
            output="test",
            metadata={"count": 5},
            title="Title",
            success=True,
        )

        d = result.to_dict()
        assert d["output"] == "test"
        assert d["metadata"] == {"count": 5}
        assert d["title"] == "Title"
        assert d["success"] is True


class TestToolRegistry:
    """Tests for ToolRegistry class."""

    def setup_method(self):
        """Clear registry before each test."""
        ToolRegistry.clear()

    def test_register_tool(self):
        """Test tool registration."""

        @ToolRegistry.register
        class TestTool(Tool):
            id = "test_tool"
            description = "A test tool"

            async def execute(self, **params):
                return ToolResult(output="test", metadata={})

        assert "test_tool" in ToolRegistry._tools
        assert ToolRegistry.get_tool_class("test_tool") is TestTool

    def test_get_tool_instance(self):
        """Test getting tool instance."""

        @ToolRegistry.register
        class TestTool(Tool):
            id = "instance_test"
            description = "Test"

            async def execute(self, **params):
                return ToolResult(output="test", metadata={})

        tool = ToolRegistry.get_tool("instance_test")
        assert tool is not None
        assert isinstance(tool, TestTool)

    def test_list_tools(self):
        """Test listing tools."""

        @ToolRegistry.register
        class Tool1(Tool):
            id = "tool1"
            description = "Tool 1"

            async def execute(self, **params):
                return ToolResult(output="1", metadata={})

        @ToolRegistry.register
        class Tool2(Tool):
            id = "tool2"
            description = "Tool 2"

            async def execute(self, **params):
                return ToolResult(output="2", metadata={})

        tools = ToolRegistry.list_tools()
        assert len(tools) == 2
        ids = [t["id"] for t in tools]
        assert "tool1" in ids
        assert "tool2" in ids

    @pytest.mark.asyncio
    async def test_execute_tool(self):
        """Test executing a tool through the registry."""

        @ToolRegistry.register
        class EchoTool(Tool):
            id = "echo"
            description = "Echo tool"
            parameters = [
                ToolParameter(
                    name="message", type="string", description="Message", required=True
                )
            ]

            async def execute(self, message: str, **params):
                return ToolResult(output=message, metadata={"length": len(message)})

        result = await ToolRegistry.execute_tool("echo", {"message": "hello"})
        assert result.output == "hello"
        assert result.metadata["length"] == 5

    @pytest.mark.asyncio
    async def test_execute_nonexistent_tool(self):
        """Test executing a tool that doesn't exist."""
        with pytest.raises(ToolExecutionError) as exc_info:
            await ToolRegistry.execute_tool("nonexistent", {})

        assert "not found" in str(exc_info.value)


class TestHTMLConversion:
    """Tests for HTML conversion functions."""

    def test_extract_text_basic(self):
        """Test basic text extraction."""
        html = "<p>Hello <strong>World</strong>!</p>"
        text = extract_text_from_html(html)
        assert "Hello" in text
        assert "World" in text

    def test_extract_text_removes_scripts(self):
        """Test that script tags are removed."""
        html = """
        <html>
        <head><script>alert('bad')</script></head>
        <body><p>Content</p></body>
        </html>
        """
        text = extract_text_from_html(html)
        assert "alert" not in text
        assert "Content" in text

    def test_extract_text_removes_styles(self):
        """Test that style tags are removed."""
        html = """
        <style>.red { color: red; }</style>
        <p class="red">Content</p>
        """
        text = extract_text_from_html(html)
        assert "color" not in text
        assert "Content" in text

    def test_markdown_headers(self):
        """Test header conversion to markdown."""
        html = "<h1>Title</h1><h2>Subtitle</h2><h3>Section</h3>"
        md = convert_html_to_markdown(html)
        assert "# Title" in md
        assert "## Subtitle" in md
        assert "### Section" in md

    def test_markdown_links(self):
        """Test link conversion to markdown."""
        html = '<a href="https://example.com">Example</a>'
        md = convert_html_to_markdown(html)
        assert "[Example](https://example.com)" in md

    def test_markdown_bold(self):
        """Test bold text conversion."""
        html = "<strong>bold</strong> and <b>also bold</b>"
        md = convert_html_to_markdown(html)
        assert "**bold**" in md
        assert "**also bold**" in md

    def test_markdown_italic(self):
        """Test italic text conversion."""
        html = "<em>italic</em> and <i>also italic</i>"
        md = convert_html_to_markdown(html)
        assert "*italic*" in md
        assert "*also italic*" in md

    def test_markdown_code_inline(self):
        """Test inline code conversion."""
        html = "<code>const x = 1;</code>"
        md = convert_html_to_markdown(html)
        assert "`const x = 1;`" in md

    def test_markdown_lists(self):
        """Test list conversion."""
        html = "<ul><li>Item 1</li><li>Item 2</li></ul>"
        md = convert_html_to_markdown(html)
        assert "- Item 1" in md
        assert "- Item 2" in md


class TestAcceptHeader:
    """Tests for Accept header generation."""

    def test_markdown_header(self):
        """Test markdown Accept header."""
        header = get_accept_header("markdown")
        assert "text/markdown" in header
        assert "text/html" in header

    def test_text_header(self):
        """Test text Accept header."""
        header = get_accept_header("text")
        assert "text/plain" in header

    def test_html_header(self):
        """Test HTML Accept header."""
        header = get_accept_header("html")
        assert "text/html" in header


class TestWebFetchTool:
    """Tests for WebFetchTool."""

    def setup_method(self):
        """Setup test fixtures."""
        self.tool = WebFetchTool()

    def test_tool_registration(self):
        """Test that WebFetchTool is registered."""
        tool_class = ToolRegistry.get_tool_class("web_fetch")
        assert tool_class is WebFetchTool

    def test_tool_metadata(self):
        """Test tool metadata."""
        assert self.tool.id == "web_fetch"
        assert self.tool.category == ToolCategory.WEB
        assert len(self.tool.parameters) == 3

    @pytest.mark.asyncio
    async def test_invalid_url_scheme(self):
        """Test that non-HTTP URLs are rejected."""
        result = await self.tool.execute(url="ftp://example.com")
        assert result.success is False
        assert "http" in result.error.lower()

    @pytest.mark.asyncio
    async def test_valid_fetch_html(self):
        """Test fetching HTML content."""
        html_content = "<html><body><h1>Test</h1><p>Content</p></body></html>"

        with patch("aiohttp.ClientSession") as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.headers = {"content-type": "text/html"}
            mock_response.charset = "utf-8"
            mock_response.read = AsyncMock(return_value=html_content.encode())

            mock_context = AsyncMock()
            mock_context.__aenter__.return_value = mock_response
            mock_context.__aexit__.return_value = None

            mock_session_instance = MagicMock()
            mock_session_instance.get.return_value = mock_context
            mock_session_instance.__aenter__ = AsyncMock(
                return_value=mock_session_instance
            )
            mock_session_instance.__aexit__ = AsyncMock(return_value=None)
            mock_session.return_value = mock_session_instance

            result = await self.tool.execute(url="https://example.com", format="html")

            assert result.success is True
            assert "<h1>Test</h1>" in result.output

    @pytest.mark.asyncio
    async def test_fetch_converts_to_markdown(self):
        """Test that HTML is converted to markdown when requested."""
        html_content = "<html><body><h1>Title</h1><p>Paragraph</p></body></html>"

        with patch("aiohttp.ClientSession") as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.headers = {"content-type": "text/html"}
            mock_response.charset = "utf-8"
            mock_response.read = AsyncMock(return_value=html_content.encode())

            mock_context = AsyncMock()
            mock_context.__aenter__.return_value = mock_response
            mock_context.__aexit__.return_value = None

            mock_session_instance = MagicMock()
            mock_session_instance.get.return_value = mock_context
            mock_session_instance.__aenter__ = AsyncMock(
                return_value=mock_session_instance
            )
            mock_session_instance.__aexit__ = AsyncMock(return_value=None)
            mock_session.return_value = mock_session_instance

            result = await self.tool.execute(
                url="https://example.com", format="markdown"
            )

            assert result.success is True
            assert "# Title" in result.output

    @pytest.mark.asyncio
    async def test_fetch_error_status(self):
        """Test handling of error HTTP status."""
        with patch("aiohttp.ClientSession") as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 404

            mock_context = AsyncMock()
            mock_context.__aenter__.return_value = mock_response
            mock_context.__aexit__.return_value = None

            mock_session_instance = MagicMock()
            mock_session_instance.get.return_value = mock_context
            mock_session_instance.__aenter__ = AsyncMock(
                return_value=mock_session_instance
            )
            mock_session_instance.__aexit__ = AsyncMock(return_value=None)
            mock_session.return_value = mock_session_instance

            result = await self.tool.execute(url="https://example.com/notfound")

            assert result.success is False
            assert "404" in result.error

    @pytest.mark.asyncio
    async def test_fetch_timeout(self):
        """Test handling of request timeout."""
        with patch("aiohttp.ClientSession") as mock_session:
            mock_session_instance = MagicMock()
            mock_session_instance.get.side_effect = asyncio.TimeoutError()
            mock_session_instance.__aenter__ = AsyncMock(
                return_value=mock_session_instance
            )
            mock_session_instance.__aexit__ = AsyncMock(return_value=None)
            mock_session.return_value = mock_session_instance

            result = await self.tool.execute(url="https://slow.example.com", timeout=1)

            assert result.success is False
            assert "timeout" in result.error.lower()

    @pytest.mark.asyncio
    async def test_fetch_respects_max_size(self):
        """Test that oversized responses are rejected."""
        with patch("aiohttp.ClientSession") as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.headers = {"content-length": str(MAX_RESPONSE_SIZE + 1)}

            mock_context = AsyncMock()
            mock_context.__aenter__.return_value = mock_response
            mock_context.__aexit__.return_value = None

            mock_session_instance = MagicMock()
            mock_session_instance.get.return_value = mock_context
            mock_session_instance.__aenter__ = AsyncMock(
                return_value=mock_session_instance
            )
            mock_session_instance.__aexit__ = AsyncMock(return_value=None)
            mock_session.return_value = mock_session_instance

            result = await self.tool.execute(url="https://example.com/large")

            assert result.success is False
            assert "large" in result.error.lower() or "5MB" in result.error

    def test_parameter_validation(self):
        """Test parameter validation."""
        # Missing required URL
        errors = self.tool.validate_parameters({})
        assert len(errors) > 0
        assert "url" in errors[0].lower()

        # Invalid format enum
        errors = self.tool.validate_parameters(
            {"url": "https://example.com", "format": "invalid"}
        )
        assert len(errors) > 0

        # Valid parameters
        errors = self.tool.validate_parameters(
            {"url": "https://example.com", "format": "markdown"}
        )
        assert len(errors) == 0


class TestWebSearchTool:
    """Tests for WebSearchTool."""

    def setup_method(self):
        """Setup test fixtures."""
        self.tool = WebSearchTool()

    def test_tool_registration(self):
        """Test that WebSearchTool is registered."""
        tool_class = ToolRegistry.get_tool_class("web_search")
        assert tool_class is WebSearchTool

    def test_tool_metadata(self):
        """Test tool metadata."""
        assert self.tool.id == "web_search"
        assert self.tool.category == ToolCategory.WEB
        assert len(self.tool.parameters) == 5

    @pytest.mark.asyncio
    async def test_successful_search(self):
        """Test successful web search."""
        mock_response_data = {
            "jsonrpc": "2.0",
            "result": {"content": [{"type": "text", "text": "Search result content"}]},
        }

        with patch("aiohttp.ClientSession") as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.text = AsyncMock(
                return_value=f"data: {json.dumps(mock_response_data)}"
            )

            mock_context = AsyncMock()
            mock_context.__aenter__.return_value = mock_response
            mock_context.__aexit__.return_value = None

            mock_session_instance = MagicMock()
            mock_session_instance.post.return_value = mock_context
            mock_session_instance.__aenter__ = AsyncMock(
                return_value=mock_session_instance
            )
            mock_session_instance.__aexit__ = AsyncMock(return_value=None)
            mock_session.return_value = mock_session_instance

            result = await self.tool.execute(query="test query")

            assert result.success is True
            assert "Search result content" in result.output

    @pytest.mark.asyncio
    async def test_search_no_results(self):
        """Test search with no results."""
        mock_response_data = {"jsonrpc": "2.0", "result": {"content": []}}

        with patch("aiohttp.ClientSession") as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.text = AsyncMock(
                return_value=f"data: {json.dumps(mock_response_data)}"
            )

            mock_context = AsyncMock()
            mock_context.__aenter__.return_value = mock_response
            mock_context.__aexit__.return_value = None

            mock_session_instance = MagicMock()
            mock_session_instance.post.return_value = mock_context
            mock_session_instance.__aenter__ = AsyncMock(
                return_value=mock_session_instance
            )
            mock_session_instance.__aexit__ = AsyncMock(return_value=None)
            mock_session.return_value = mock_session_instance

            result = await self.tool.execute(query="obscure query")

            assert result.success is False
            assert "No" in result.output or result.error is not None

    @pytest.mark.asyncio
    async def test_search_api_error(self):
        """Test handling of API errors."""
        with patch("aiohttp.ClientSession") as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 500
            mock_response.text = AsyncMock(return_value="Internal Server Error")

            mock_context = AsyncMock()
            mock_context.__aenter__.return_value = mock_response
            mock_context.__aexit__.return_value = None

            mock_session_instance = MagicMock()
            mock_session_instance.post.return_value = mock_context
            mock_session_instance.__aenter__ = AsyncMock(
                return_value=mock_session_instance
            )
            mock_session_instance.__aexit__ = AsyncMock(return_value=None)
            mock_session.return_value = mock_session_instance

            result = await self.tool.execute(query="test")

            assert result.success is False
            assert "500" in result.error

    @pytest.mark.asyncio
    async def test_search_timeout(self):
        """Test search timeout handling."""
        with patch("aiohttp.ClientSession") as mock_session:
            mock_session_instance = MagicMock()
            mock_session_instance.post.side_effect = asyncio.TimeoutError()
            mock_session_instance.__aenter__ = AsyncMock(
                return_value=mock_session_instance
            )
            mock_session_instance.__aexit__ = AsyncMock(return_value=None)
            mock_session.return_value = mock_session_instance

            result = await self.tool.execute(query="test")

            assert result.success is False
            assert "timeout" in result.error.lower()

    def test_sse_parsing(self):
        """Test SSE response parsing."""
        valid_sse = (
            'data: {"result": {"content": [{"type": "text", "text": "Result"}]}}'
        )
        content = self.tool._parse_sse_response(valid_sse)
        assert content == "Result"

        # Invalid JSON
        invalid_sse = "data: not json"
        content = self.tool._parse_sse_response(invalid_sse)
        assert content is None

        # Empty response
        empty_sse = ""
        content = self.tool._parse_sse_response(empty_sse)
        assert content is None

    def test_parameter_validation(self):
        """Test parameter validation."""
        # Missing required query
        errors = self.tool.validate_parameters({})
        assert len(errors) > 0

        # Invalid type enum
        errors = self.tool.validate_parameters({"query": "test", "type": "invalid"})
        assert len(errors) > 0

        # Valid parameters
        errors = self.tool.validate_parameters({"query": "test", "type": "auto"})
        assert len(errors) == 0


class TestToolSystemPrompt:
    """Tests for system prompt generation."""

    def setup_method(self):
        """Clear and re-register tools."""
        ToolRegistry.clear()
        # Re-import to trigger registration
        from tools import webfetch, websearch

    def test_system_prompt_includes_tools(self):
        """Test that system prompt includes all tools."""
        prompt = ToolRegistry.get_system_prompt()

        assert "web_fetch" in prompt
        assert "web_search" in prompt

    def test_system_prompt_format(self):
        """Test system prompt format."""
        prompt = ToolRegistry.get_system_prompt()

        # Should include tool descriptions
        assert "Fetch content" in prompt or "fetch" in prompt.lower()
        assert "Search" in prompt or "search" in prompt.lower()


class TestContext7Tools:
    """Tests for Context7 tools."""

    def test_context7_resolve_registration(self):
        """Test that Context7ResolveTool is registered."""
        from tools.context7 import Context7ResolveTool

        tool_class = ToolRegistry.get_tool_class("context7_resolve")
        assert tool_class is Context7ResolveTool

    def test_context7_docs_registration(self):
        """Test that Context7DocsTool is registered."""
        from tools.context7 import Context7DocsTool

        tool_class = ToolRegistry.get_tool_class("context7_docs")
        assert tool_class is Context7DocsTool

    def test_context7_resolve_metadata(self):
        """Test context7_resolve tool metadata."""
        tool = ToolRegistry.get_tool("context7_resolve")
        assert tool.id == "context7_resolve"
        assert tool.category == ToolCategory.WEB
        assert len(tool.parameters) == 2

        param_names = [p.name for p in tool.parameters]
        assert "library_name" in param_names
        assert "query" in param_names

    def test_context7_docs_metadata(self):
        """Test context7_docs tool metadata."""
        tool = ToolRegistry.get_tool("context7_docs")
        assert tool.id == "context7_docs"
        assert tool.category == ToolCategory.WEB
        assert len(tool.parameters) == 3

        param_names = [p.name for p in tool.parameters]
        assert "library_id" in param_names
        assert "query" in param_names
        assert "topic" in param_names

    @pytest.mark.asyncio
    async def test_context7_docs_invalid_library_id(self):
        """Test that invalid library IDs are rejected."""
        tool = ToolRegistry.get_tool("context7_docs")

        # Library ID must start with /
        result = await tool.execute(
            library_id="facebook/react",  # Missing leading /
            query="hooks",
        )

        assert result.success is False
        assert "Invalid library ID" in result.error

    @pytest.mark.asyncio
    async def test_context7_resolve_success(self):
        """Test successful library resolution."""
        from tools.context7 import Context7ResolveTool

        tool = Context7ResolveTool()

        mock_response_data = {
            "jsonrpc": "2.0",
            "result": {
                "content": [{"type": "text", "text": "Library ID: /facebook/react"}]
            },
        }

        with patch("aiohttp.ClientSession") as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.text = AsyncMock(
                return_value=f"data: {json.dumps(mock_response_data)}"
            )

            mock_context = AsyncMock()
            mock_context.__aenter__.return_value = mock_response
            mock_context.__aexit__.return_value = None

            mock_session_instance = MagicMock()
            mock_session_instance.post.return_value = mock_context
            mock_session_instance.__aenter__ = AsyncMock(
                return_value=mock_session_instance
            )
            mock_session_instance.__aexit__ = AsyncMock(return_value=None)
            mock_session.return_value = mock_session_instance

            result = await tool.execute(library_name="react", query="hooks")

            assert result.success is True
            assert (
                "react" in result.output.lower() or "/facebook/react" in result.output
            )

    @pytest.mark.asyncio
    async def test_context7_docs_success(self):
        """Test successful documentation retrieval."""
        from tools.context7 import Context7DocsTool

        tool = Context7DocsTool()

        mock_response_data = {
            "jsonrpc": "2.0",
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": "# React Hooks\n\nuseEffect is used for side effects...",
                    }
                ]
            },
        }

        with patch("aiohttp.ClientSession") as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.text = AsyncMock(
                return_value=f"data: {json.dumps(mock_response_data)}"
            )

            mock_context = AsyncMock()
            mock_context.__aenter__.return_value = mock_response
            mock_context.__aexit__.return_value = None

            mock_session_instance = MagicMock()
            mock_session_instance.post.return_value = mock_context
            mock_session_instance.__aenter__ = AsyncMock(
                return_value=mock_session_instance
            )
            mock_session_instance.__aexit__ = AsyncMock(return_value=None)
            mock_session.return_value = mock_session_instance

            result = await tool.execute(library_id="/facebook/react", query="useEffect")

            assert result.success is True
            assert "useEffect" in result.output or "Hooks" in result.output

    @pytest.mark.asyncio
    async def test_context7_timeout(self):
        """Test timeout handling."""
        from tools.context7 import Context7ResolveTool

        tool = Context7ResolveTool()

        with patch("aiohttp.ClientSession") as mock_session:
            mock_session_instance = MagicMock()
            mock_session_instance.post.side_effect = asyncio.TimeoutError()
            mock_session_instance.__aenter__ = AsyncMock(
                return_value=mock_session_instance
            )
            mock_session_instance.__aexit__ = AsyncMock(return_value=None)
            mock_session.return_value = mock_session_instance

            result = await tool.execute(library_name="react", query="hooks")

            assert result.success is False
            assert "timeout" in result.error.lower()

    @pytest.mark.asyncio
    async def test_context7_api_error(self):
        """Test API error handling."""
        from tools.context7 import Context7ResolveTool

        tool = Context7ResolveTool()

        with patch("aiohttp.ClientSession") as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 500
            mock_response.text = AsyncMock(return_value="Internal Server Error")

            mock_context = AsyncMock()
            mock_context.__aenter__.return_value = mock_response
            mock_context.__aexit__.return_value = None

            mock_session_instance = MagicMock()
            mock_session_instance.post.return_value = mock_context
            mock_session_instance.__aenter__ = AsyncMock(
                return_value=mock_session_instance
            )
            mock_session_instance.__aexit__ = AsyncMock(return_value=None)
            mock_session.return_value = mock_session_instance

            result = await tool.execute(library_name="react", query="hooks")

            assert result.success is False
            assert "500" in result.error

    def test_context7_parameter_validation(self):
        """Test parameter validation."""
        tool = ToolRegistry.get_tool("context7_resolve")

        # Missing required parameters
        errors = tool.validate_parameters({})
        assert len(errors) > 0

        # Valid parameters
        errors = tool.validate_parameters({"library_name": "react", "query": "hooks"})
        assert len(errors) == 0

    def test_context7_in_system_prompt(self):
        """Test that context7 tools appear in system prompt."""
        # Re-import to ensure registration
        from tools import context7

        prompt = ToolRegistry.get_system_prompt()
        assert "context7" in prompt.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
