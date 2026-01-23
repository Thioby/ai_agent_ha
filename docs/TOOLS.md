# AI Agent HA Tools Reference

## TOOL_SYSTEM_ARCHITECTURE
```
Location: custom_components/ai_agent_ha/tools/
Structure:
  base.py      - Tool, ToolRegistry, ToolResult, ToolParameter, ToolExecutionError
  webfetch.py  - WebFetchTool (fetches URLs)
  websearch.py - WebSearchTool (searches web via Exa AI)
  __init__.py  - Package exports, convenience functions
```

## ADDING_NEW_TOOL
```python
# File: tools/mytool.py
from .base import Tool, ToolRegistry, ToolResult, ToolParameter, ToolCategory

@ToolRegistry.register
class MyTool(Tool):
    id = "my_tool"  # REQUIRED: unique identifier
    description = "What this tool does"  # REQUIRED
    category = ToolCategory.WEB  # or DATA, HOME_ASSISTANT, UTILITY
    
    parameters = [
        ToolParameter(
            name="param1",
            type="string",  # string, integer, boolean, list, dict
            description="Parameter description",
            required=True,
            default=None,
            enum=["option1", "option2"],  # optional
        ),
    ]
    
    async def execute(self, param1: str, **kwargs) -> ToolResult:
        # Implementation
        return ToolResult(
            output="result content",
            success=True,
            title="Result Title",
            metadata={"key": "value"},
        )
```

Then add import in `tools/__init__.py`:
```python
from . import mytool
```

## TOOL_REGISTRATION_IN_AGENT
```
File: agent.py

1. Import at top:
   from .tools import ToolRegistry, get_tools_system_prompt, execute_tool

2. Add to SYSTEM_PROMPT (both regular and LOCAL versions):
   - Tool description in command list
   - Example JSON format

3. Add to data_request_types list (~line 3107):
   "my_tool",

4. Add dispatch handler (~line 3226):
   elif request_type == "my_tool":
       result = await execute_tool(
           "my_tool",
           {"param1": parameters.get("param1")},
           hass=self.hass,
           config=self.config,
       )
       if result.success:
           data = {"output": result.output, "metadata": result.metadata}
       else:
           data = {"error": result.error or "Tool failed"}
```

## WEB_FETCH_TOOL
```
ID: web_fetch
Category: WEB
Purpose: Fetch content from URLs, convert HTML to markdown/text

Parameters:
  url (string, required): URL starting with http:// or https://
  format (string, optional): "markdown" | "text" | "html", default="markdown"
  timeout (integer, optional): seconds, default=30, max=120

Returns:
  success: {content, title, metadata}
  error: {error: string}

Example request:
{
  "request_type": "web_fetch",
  "parameters": {
    "url": "https://example.com",
    "format": "markdown"
  }
}

Behavior:
- 5MB max response size
- Spoofs Chrome User-Agent
- Converts HTML to markdown using regex-based conversion
- Removes script/style tags before conversion
- Handles encoding from response charset
```

## WEB_SEARCH_TOOL
```
ID: web_search
Category: WEB
Purpose: Search web using Exa AI MCP endpoint

Parameters:
  query (string, required): search query
  num_results (integer, optional): default=8
  type (string, optional): "auto" | "fast" | "deep", default="auto"
  livecrawl (string, optional): "fallback" | "preferred", default="fallback"
  context_max_chars (integer, optional): default=10000

Returns:
  success: {results, title, metadata}
  error: {error: string}

Example request:
{
  "request_type": "web_search",
  "parameters": {
    "query": "home automation 2025",
    "num_results": 5,
    "type": "auto"
  }
}

API Details:
- Endpoint: https://mcp.exa.ai/mcp
- Protocol: JSON-RPC 2.0 over HTTP POST
- Response: SSE format (Server-Sent Events)
- Timeout: 25 seconds
```

## TOOLRESULT_STRUCTURE
```python
@dataclass
class ToolResult:
    output: str           # Main content
    metadata: Dict        # Additional data, default={}
    title: Optional[str]  # Result title
    success: bool         # True=success, False=error
    error: Optional[str]  # Error message if failed

    def to_dict() -> Dict  # JSON serialization
```

## TOOLREGISTRY_API
```python
# Register tool (decorator)
@ToolRegistry.register
class MyTool(Tool): ...

# Get tool class
ToolRegistry.get_tool_class("tool_id") -> Type[Tool]

# Get tool instance (cached)
ToolRegistry.get_tool("tool_id", hass=None, config=None) -> Tool

# Get all tools
ToolRegistry.get_all_tools(hass, config, enabled_only=True) -> List[Tool]

# Generate system prompt
ToolRegistry.get_system_prompt(hass, config, enabled_only=True) -> str

# Execute tool
await ToolRegistry.execute_tool("tool_id", params, hass, config) -> ToolResult

# List tool metadata
ToolRegistry.list_tools(enabled_only=True) -> List[Dict]

# Clear registry (testing)
ToolRegistry.clear() -> None
```

## CONVENIENCE_FUNCTIONS
```python
# In tools/__init__.py

get_tools_system_prompt(hass=None, config=None) -> str
# Wrapper for ToolRegistry.get_system_prompt()

await execute_tool(tool_id, params, hass=None, config=None) -> ToolResult
# Wrapper for ToolRegistry.execute_tool()

list_tools(enabled_only=True) -> list
# Wrapper for ToolRegistry.list_tools()
```

## ERROR_HANDLING
```python
class ToolExecutionError(Exception):
    tool_id: str           # Which tool failed
    details: Dict          # Additional error context

# Raise in tools:
raise ToolExecutionError("message", tool_id="my_tool", details={"key": "value"})

# Handle in agent:
try:
    result = await execute_tool(...)
except ToolExecutionError as e:
    data = {"error": str(e)}
```

## TESTING_TOOLS
```
File: tests/test_tools.py

Test Categories:
- ToolParameter validation
- ToolResult serialization
- ToolRegistry registration/execution
- HTML conversion (extract_text, convert_markdown)
- WebFetchTool (mock aiohttp)
- WebSearchTool (mock Exa API)
- System prompt generation

Run:
  pytest tests/test_tools.py -v
```

## FILE_LOCATIONS
```
custom_components/ai_agent_ha/
├── tools/
│   ├── __init__.py    # Package, exports, convenience functions
│   ├── base.py        # Tool, ToolRegistry, ToolResult, ToolParameter
│   ├── webfetch.py    # WebFetchTool, HTML conversion functions
│   └── websearch.py   # WebSearchTool, Exa API client
├── agent.py           # Integration: imports, system prompt, dispatch
└── ...

tests/
└── test_tools.py      # Comprehensive tool tests
```

## OPENCODE_COMPATIBILITY
```
Implementation mirrors OpenCode (opencode/packages/opencode/src/tool/):
- Same API endpoints (Exa AI)
- Same request/response format (JSON-RPC, SSE)
- Same constants (5MB limit, 30s timeout, 120s max)
- Same User-Agent spoofing
- Similar HTML→Markdown conversion (TurndownService equivalent)
```
