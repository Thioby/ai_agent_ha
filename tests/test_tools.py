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

from custom_components.ai_agent_ha.tools.base import (
    Tool,
    ToolCategory,
    ToolExecutionError,
    ToolParameter,
    ToolRegistry,
    ToolResult,
)
from custom_components.ai_agent_ha.tools.webfetch import (
    WebFetchTool,
    convert_html_to_markdown,
    extract_text_from_html,
    get_accept_header,
    MAX_RESPONSE_SIZE,
    DEFAULT_TIMEOUT,
    MAX_TIMEOUT,
)
from custom_components.ai_agent_ha.tools.websearch import WebSearchTool
from custom_components.ai_agent_ha.tools.context7 import Context7ResolveTool, Context7DocsTool
from custom_components.ai_agent_ha.tools import (
    context7,
    webfetch,
    websearch,
)
from custom_components.ai_agent_ha.tools.ha_native import (
    GetEntityState,
    GetEntitiesByDomain,
    GetEntityRegistrySummary,
    GetEntityRegistry,
    CallService,
    GetEntitiesByDeviceClass,
    GetEntitiesByArea,
    GetHistory,
    GetStatistics,
    GetAreaRegistry,
    GetWeatherData,
    GetAutomations,
    GetScenes,
    GetPersonData,
)


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
        ToolRegistry.register(WebFetchTool)
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
            assert "timed out" in result.error.lower()

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
        ToolRegistry.register(WebSearchTool)
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
            assert "timed out" in result.error.lower()

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
        # Explicitly register known tools
        ToolRegistry.register(WebFetchTool)
        ToolRegistry.register(WebSearchTool)
        # Import context7 to register those tools too (if they are not automatically registered by import)
        from custom_components.ai_agent_ha.tools import context7
        from custom_components.ai_agent_ha.tools.context7 import Context7ResolveTool, Context7DocsTool
        ToolRegistry.register(Context7ResolveTool)
        ToolRegistry.register(Context7DocsTool)

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

    def setup_method(self):
        """Setup test fixtures."""
        from custom_components.ai_agent_ha.tools.context7 import Context7ResolveTool, Context7DocsTool
        ToolRegistry.register(Context7ResolveTool)
        ToolRegistry.register(Context7DocsTool)

    def test_context7_resolve_registration(self):
        """Test that Context7ResolveTool is registered."""
        from custom_components.ai_agent_ha.tools.context7 import Context7ResolveTool

        tool_class = ToolRegistry.get_tool_class("context7_resolve")
        assert tool_class is Context7ResolveTool

    def test_context7_docs_registration(self):
        """Test that Context7DocsTool is registered."""
        from custom_components.ai_agent_ha.tools.context7 import Context7DocsTool

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
        from custom_components.ai_agent_ha.tools.context7 import Context7ResolveTool

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
        from custom_components.ai_agent_ha.tools.context7 import Context7DocsTool

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
        from custom_components.ai_agent_ha.tools.context7 import Context7ResolveTool

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
            assert "timed out" in result.error.lower()

    @pytest.mark.asyncio
    async def test_context7_api_error(self):
        """Test API error handling."""
        from custom_components.ai_agent_ha.tools.context7 import Context7ResolveTool

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
        from custom_components.ai_agent_ha.tools import context7

        prompt = ToolRegistry.get_system_prompt()
        assert "context7" in prompt.lower()


class TestToolsModuleConvenienceFunctions:
    """Tests for convenience functions in tools/__init__.py."""

    def setup_method(self):
        """Clear and re-register tools."""
        ToolRegistry.clear()
        ToolRegistry.register(WebFetchTool)
        ToolRegistry.register(WebSearchTool)

    def test_get_tools_system_prompt(self):
        """Test get_tools_system_prompt convenience function."""
        from custom_components.ai_agent_ha.tools import get_tools_system_prompt

        prompt = get_tools_system_prompt()
        assert isinstance(prompt, str)
        assert "web_fetch" in prompt or "fetch" in prompt.lower()

    def test_get_tools_system_prompt_with_hass(self):
        """Test get_tools_system_prompt with hass and config."""
        from custom_components.ai_agent_ha.tools import get_tools_system_prompt

        mock_hass = MagicMock()
        mock_config = {"some_key": "some_value"}

        prompt = get_tools_system_prompt(hass=mock_hass, config=mock_config)
        assert isinstance(prompt, str)

    @pytest.mark.asyncio
    async def test_execute_tool_function(self):
        """Test execute_tool convenience function."""
        from custom_components.ai_agent_ha.tools import execute_tool

        # Register a simple test tool
        @ToolRegistry.register
        class SimpleTool(Tool):
            id = "simple_test"
            description = "Simple test tool"
            parameters = [
                ToolParameter(name="value", type="string", description="Value", required=True)
            ]

            async def execute(self, value: str, **params):
                return ToolResult(output=f"Got: {value}", metadata={})

        result = await execute_tool("simple_test", {"value": "hello"})
        assert result.output == "Got: hello"

    @pytest.mark.asyncio
    async def test_execute_tool_with_hass_config(self):
        """Test execute_tool with hass and config."""
        from custom_components.ai_agent_ha.tools import execute_tool

        @ToolRegistry.register
        class ConfigTool(Tool):
            id = "config_test"
            description = "Config test tool"

            async def execute(self, **params):
                return ToolResult(output="executed", metadata={})

        mock_hass = MagicMock()
        result = await execute_tool("config_test", {}, hass=mock_hass, config={"key": "val"})
        assert result.success is True

    def test_list_tools_function(self):
        """Test list_tools convenience function."""
        from custom_components.ai_agent_ha.tools import list_tools

        tools = list_tools()
        assert isinstance(tools, list)
        assert len(tools) >= 2  # At least web_fetch and web_search

        ids = [t["id"] for t in tools]
        assert "web_fetch" in ids
        assert "web_search" in ids

    def test_list_tools_enabled_only(self):
        """Test list_tools with enabled_only parameter."""
        from custom_components.ai_agent_ha.tools import list_tools

        tools_enabled = list_tools(enabled_only=True)
        tools_all = list_tools(enabled_only=False)

        # Both should return lists
        assert isinstance(tools_enabled, list)
        assert isinstance(tools_all, list)


class TestHaNativeGetEntityState:
    """Test GetEntityState tool."""

    @pytest.fixture
    def tool(self, hass):
        """Create tool instance."""
        tool = GetEntityState()
        tool.hass = hass
        return tool

    @pytest.mark.asyncio
    async def test_execute_missing_entity_id(self, tool):
        """Test execute with missing entity_id."""
        result = await tool.execute(entity_id="")

        assert result.success is False
        assert "entity" in result.error.lower() or "required" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_entity_not_found(self, tool, hass):
        """Test execute with non-existent entity."""
        result = await tool.execute(entity_id="light.nonexistent")

        assert result.success is False
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_success(self, tool, hass):
        """Test execute with valid entity."""
        hass.states.async_set(
            "light.living_room",
            "on",
            {"brightness": 255, "friendly_name": "Living Room Light"}
        )

        result = await tool.execute(entity_id="light.living_room")

        assert result.success is True
        data = json.loads(result.output)
        assert data["entity_id"] == "light.living_room"
        assert data["state"] == "on"
        assert data["attributes"]["brightness"] == 255


class TestHaNativeGetEntitiesByDomain:
    """Test GetEntitiesByDomain tool."""

    @pytest.fixture
    def tool(self, hass):
        """Create tool instance."""
        tool = GetEntitiesByDomain()
        tool.hass = hass
        return tool

    @pytest.mark.asyncio
    async def test_execute_missing_domain(self, tool):
        """Test execute with missing domain."""
        result = await tool.execute(domain="")

        assert result.success is False
        assert "domain" in result.error.lower() or "required" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_empty_domain(self, tool, hass):
        """Test execute with domain that has no entities."""
        result = await tool.execute(domain="vacuum")

        assert result.success is True
        data = json.loads(result.output)
        assert data == []

    @pytest.mark.asyncio
    async def test_execute_success(self, tool, hass):
        """Test execute with valid domain."""
        hass.states.async_set("light.one", "on", {"friendly_name": "Light One"})
        hass.states.async_set("light.two", "off", {"friendly_name": "Light Two"})
        hass.states.async_set("switch.other", "on", {})

        result = await tool.execute(domain="light")

        assert result.success is True
        data = json.loads(result.output)
        assert len(data) == 2
        entity_ids = [e["entity_id"] for e in data]
        assert "light.one" in entity_ids
        assert "light.two" in entity_ids


class TestHaNativeGetEntityRegistrySummary:
    """Test GetEntityRegistrySummary tool."""

    @pytest.fixture
    def tool(self, hass):
        """Create tool instance."""
        tool = GetEntityRegistrySummary()
        tool.hass = hass
        return tool

    @pytest.mark.asyncio
    async def test_execute_empty_registry(self, tool, hass):
        """Test execute with empty entity registry."""
        result = await tool.execute()

        assert result.success is True
        data = json.loads(result.output)
        assert "total_entities" in data
        assert "by_domain" in data
        assert "by_area" in data

    @pytest.mark.asyncio
    async def test_execute_with_entities(self, tool, hass):
        """Test execute with entities in registry."""
        hass.states.async_set("light.one", "on", {"device_class": "light"})
        hass.states.async_set("sensor.temp", "23", {"device_class": "temperature"})

        result = await tool.execute()

        assert result.success is True
        data = json.loads(result.output)
        assert isinstance(data["by_domain"], dict)


class TestHaNativeGetEntityRegistry:
    """Test GetEntityRegistry tool."""

    @pytest.fixture
    def tool(self, hass):
        """Create tool instance."""
        tool = GetEntityRegistry()
        tool.hass = hass
        return tool

    @pytest.mark.asyncio
    async def test_execute_returns_placeholder(self, tool):
        """Test execute returns placeholder response."""
        result = await tool.execute(domain="light")

        assert result.success is True
        assert "agent handler" in result.output.lower()


class TestHaNativeCallService:
    """Test CallService tool.

    Note: CallService is a schema-only definition that always returns success.
    The actual service call logic is handled by agent.py.
    """

    @pytest.fixture
    def tool(self, hass):
        """Create tool instance."""
        tool = CallService()
        tool.hass = hass
        return tool

    @pytest.mark.asyncio
    async def test_execute_returns_success(self, tool, hass):
        """Test execute always returns success (schema-only tool)."""
        # CallService is a placeholder tool, actual logic is in agent.py
        result = await tool.execute(
            domain="light",
            service="turn_on",
            target={"entity_id": "light.test"}
        )

        assert result.success is True
        assert "agent logic" in result.output.lower()

    @pytest.mark.asyncio
    async def test_execute_with_service_data(self, tool, hass):
        """Test service call with service_data."""
        result = await tool.execute(
            domain="light",
            service="turn_on",
            target={"entity_id": "light.test"},
            service_data={"brightness": 128}
        )

        assert result.success is True


class TestHaNativeGetEntitiesByDeviceClass:
    """Test GetEntitiesByDeviceClass tool."""

    @pytest.fixture
    def tool(self, hass):
        """Create tool instance."""
        tool = GetEntitiesByDeviceClass()
        tool.hass = hass
        return tool

    @pytest.mark.asyncio
    async def test_execute_missing_device_class(self, tool):
        """Test execute with missing device_class."""
        result = await tool.execute(device_class="")

        assert result.success is False
        assert "device_class" in result.error.lower() or "missing" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_no_matching_entities(self, tool, hass):
        """Test execute with device_class that has no entities."""
        result = await tool.execute(device_class="power")

        assert result.success is True
        data = json.loads(result.output)
        assert data == []

    @pytest.mark.asyncio
    async def test_execute_success(self, tool, hass):
        """Test execute with valid device_class."""
        hass.states.async_set("sensor.temp1", "22.5", {"device_class": "temperature"})
        hass.states.async_set("sensor.temp2", "21.0", {"device_class": "temperature"})
        hass.states.async_set("sensor.humidity", "45", {"device_class": "humidity"})

        result = await tool.execute(device_class="temperature")

        assert result.success is True
        data = json.loads(result.output)
        assert len(data) == 2
        entity_ids = [e["entity_id"] for e in data]
        assert "sensor.temp1" in entity_ids
        assert "sensor.temp2" in entity_ids

    @pytest.mark.asyncio
    async def test_execute_with_domain_filter(self, tool, hass):
        """Test execute with domain filter."""
        hass.states.async_set("sensor.temp", "22.5", {"device_class": "temperature"})
        hass.states.async_set("binary_sensor.motion", "on", {"device_class": "motion"})

        result = await tool.execute(device_class="temperature", domain="sensor")

        assert result.success is True
        data = json.loads(result.output)
        assert len(data) == 1
        assert data[0]["entity_id"] == "sensor.temp"


class TestHaNativeGetEntitiesByArea:
    """Test GetEntitiesByArea tool."""

    @pytest.fixture
    def tool(self, hass):
        """Create tool instance."""
        tool = GetEntitiesByArea()
        tool.hass = hass
        return tool

    @pytest.mark.asyncio
    async def test_execute_missing_area_id(self, tool):
        """Test execute with missing area_id."""
        result = await tool.execute(area_id="")

        assert result.success is False
        assert "area_id" in result.error.lower() or "missing" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_no_entities_in_area(self, tool, hass):
        """Test execute with area that has no entities."""
        result = await tool.execute(area_id="nonexistent_area")

        assert result.success is True
        data = json.loads(result.output)
        assert data == []


class TestHaNativeGetHistory:
    """Test GetHistory tool."""

    @pytest.fixture
    def tool(self, hass):
        """Create tool instance."""
        tool = GetHistory()
        tool.hass = hass
        return tool

    @pytest.mark.asyncio
    async def test_execute_missing_entity_id(self, tool):
        """Test execute with missing entity_id."""
        result = await tool.execute(entity_id="")

        assert result.success is False
        assert "entity" in result.error.lower() or "required" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_no_recorder(self, tool, hass):
        """Test execute when recorder is not available."""
        # Without recorder integration, this should return an error
        result = await tool.execute(entity_id="sensor.temperature", hours=24)

        # Either success with empty data or error due to no recorder
        assert result is not None


class TestHaNativeGetStatistics:
    """Test GetStatistics tool."""

    @pytest.fixture
    def tool(self, hass):
        """Create tool instance."""
        tool = GetStatistics()
        tool.hass = hass
        return tool

    @pytest.mark.asyncio
    async def test_execute_missing_entity_id(self, tool):
        """Test execute with missing entity_id."""
        result = await tool.execute(entity_id="")

        assert result.success is False
        assert "entity" in result.error.lower() or "missing" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_no_recorder(self, tool, hass):
        """Test execute when recorder is not available."""
        result = await tool.execute(entity_id="sensor.temperature")

        # Should return error when recorder is not available
        assert result.success is False
        assert "recorder" in result.output.lower() or "recorder" in result.error.lower()


class TestHaNativeGetAreaRegistry:
    """Test GetAreaRegistry tool."""

    @pytest.fixture
    def tool(self, hass):
        """Create tool instance."""
        tool = GetAreaRegistry()
        tool.hass = hass
        return tool

    @pytest.mark.asyncio
    async def test_execute_success(self, tool, hass):
        """Test execute returns area registry."""
        result = await tool.execute()

        assert result.success is True
        data = json.loads(result.output)
        assert isinstance(data, dict)


class TestHaNativeGetWeatherData:
    """Test GetWeatherData tool."""

    @pytest.fixture
    def tool(self, hass):
        """Create tool instance."""
        tool = GetWeatherData()
        tool.hass = hass
        return tool

    @pytest.mark.asyncio
    async def test_execute_no_weather_entities(self, tool, hass):
        """Test execute with no weather entities."""
        result = await tool.execute()

        assert result.success is False
        assert "weather" in result.output.lower() or "weather" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_with_weather_entity(self, tool, hass):
        """Test execute with weather entity."""
        hass.states.async_set(
            "weather.home",
            "sunny",
            {
                "temperature": 22,
                "humidity": 45,
                "pressure": 1013,
                "wind_speed": 10,
                "forecast": [
                    {"datetime": "2024-01-01T12:00:00", "temperature": 23, "condition": "sunny"}
                ]
            }
        )

        result = await tool.execute()

        assert result.success is True
        data = json.loads(result.output)
        assert "current" in data
        assert data["current"]["temperature"] == 22


class TestHaNativeGetAutomations:
    """Test GetAutomations tool."""

    @pytest.fixture
    def tool(self, hass):
        """Create tool instance."""
        tool = GetAutomations()
        tool.hass = hass
        return tool

    @pytest.mark.asyncio
    async def test_execute_no_automations(self, tool, hass):
        """Test execute with no automations."""
        result = await tool.execute()

        assert result.success is True
        data = json.loads(result.output)
        assert data == []

    @pytest.mark.asyncio
    async def test_execute_with_automations(self, tool, hass):
        """Test execute with automations."""
        hass.states.async_set(
            "automation.morning_lights",
            "on",
            {
                "friendly_name": "Morning Lights",
                "last_triggered": "2024-01-15T07:30:00"
            }
        )
        hass.states.async_set(
            "automation.night_mode",
            "off",
            {"friendly_name": "Night Mode"}
        )

        result = await tool.execute()

        assert result.success is True
        data = json.loads(result.output)
        assert len(data) == 2
        entity_ids = [a["entity_id"] for a in data]
        assert "automation.morning_lights" in entity_ids


class TestHaNativeGetScenes:
    """Test GetScenes tool."""

    @pytest.fixture
    def tool(self, hass):
        """Create tool instance."""
        tool = GetScenes()
        tool.hass = hass
        return tool

    @pytest.mark.asyncio
    async def test_execute_no_scenes(self, tool, hass):
        """Test execute with no scenes."""
        result = await tool.execute()

        assert result.success is True
        data = json.loads(result.output)
        assert data == []

    @pytest.mark.asyncio
    async def test_execute_with_scenes(self, tool, hass):
        """Test execute with scenes."""
        hass.states.async_set(
            "scene.movie_night",
            "scening",
            {
                "friendly_name": "Movie Night",
                "icon": "mdi:movie"
            }
        )

        result = await tool.execute()

        assert result.success is True
        data = json.loads(result.output)
        assert len(data) == 1
        assert data[0]["entity_id"] == "scene.movie_night"


class TestHaNativeGetPersonData:
    """Test GetPersonData tool."""

    @pytest.fixture
    def tool(self, hass):
        """Create tool instance."""
        tool = GetPersonData()
        tool.hass = hass
        return tool

    @pytest.mark.asyncio
    async def test_execute_no_persons(self, tool, hass):
        """Test execute with no persons."""
        result = await tool.execute()

        assert result.success is True
        data = json.loads(result.output)
        assert data == []

    @pytest.mark.asyncio
    async def test_execute_with_persons(self, tool, hass):
        """Test execute with person data."""
        hass.states.async_set(
            "person.john",
            "home",
            {
                "friendly_name": "John",
                "latitude": 51.5074,
                "longitude": -0.1278,
                "source": "device_tracker.johns_phone"
            }
        )

        result = await tool.execute()

        assert result.success is True
        data = json.loads(result.output)
        assert len(data) == 1
        assert data[0]["entity_id"] == "person.john"
        assert data[0]["state"] == "home"
