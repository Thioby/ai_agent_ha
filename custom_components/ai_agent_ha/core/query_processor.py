"""Query processor for AI agent interactions."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from ..function_calling import FunctionCallHandler, FunctionCall
from ..tools.base import ToolRegistry
from .response_parser import ResponseParser

if TYPE_CHECKING:
    from ..providers.registry import AIProvider

_LOGGER = logging.getLogger(__name__)

# Invisible characters that should be stripped from queries
INVISIBLE_CHARS = [
    "\ufeff",  # BOM (Byte Order Mark)
    "\u200b",  # Zero-width space
    "\u200c",  # Zero-width non-joiner
    "\u200d",  # Zero-width joiner
    "\u2060",  # Word joiner
]


class QueryProcessor:
    """Processes user queries for AI providers.

    This class handles query sanitization, message building, and
    coordinating with AI providers to generate responses.
    """

    def __init__(
        self,
        provider: AIProvider,
        max_iterations: int = 10,
        max_query_length: int = 1000,
    ) -> None:
        """Initialize the query processor.

        Args:
            provider: The AI provider to use for generating responses.
            max_iterations: Maximum tool call iterations (for future use).
            max_query_length: Maximum allowed query length after sanitization.
        """
        self.provider = provider
        self.max_iterations = max_iterations
        self.max_query_length = max_query_length
        self.response_parser = ResponseParser()

    def _sanitize_query(self, query: str, max_length: int | None = None) -> str:
        """Sanitize a user query by removing invisible characters.

        Args:
            query: The raw user query.
            max_length: Optional maximum length override. If not provided,
                uses self.max_query_length.

        Returns:
            The sanitized query string.
        """
        # Remove invisible characters
        sanitized = query
        for char in INVISIBLE_CHARS:
            sanitized = sanitized.replace(char, "")

        # Strip leading/trailing whitespace
        sanitized = sanitized.strip()

        # Truncate to max length
        effective_max = max_length if max_length is not None else self.max_query_length
        if len(sanitized) > effective_max:
            sanitized = sanitized[:effective_max]

        return sanitized

    def _build_messages(
        self,
        query: str,
        history: list[dict[str, Any]],
        system_prompt: str | None = None,
        rag_context: str | None = None,
    ) -> list[dict[str, Any]]:
        """Build the message list for the AI provider.

        Args:
            query: The current user query.
            history: Previous conversation messages.
            system_prompt: Optional system message to prepend.
            rag_context: Optional RAG context to include.

        Returns:
            List of message dictionaries ready for the provider.
        """
        messages: list[dict[str, Any]] = []

        # Build system prompt with RAG context if available
        final_system_prompt = system_prompt or ""

        if rag_context:
            rag_section = (
                "\n\n--- SUGGESTED ENTITIES ---\n"
                f"{rag_context}\n"
                "--- END SUGGESTIONS ---\n\n"
                "These are suggestions based on your query. Use available tools "
                "(get_entities_by_domain, get_state, etc.) to find other entities if needed."
            )
            final_system_prompt = (
                final_system_prompt + rag_section
                if final_system_prompt
                else rag_section
            )
            _LOGGER.info(
                "RAG context added to system prompt (%d chars)", len(rag_context)
            )
            _LOGGER.debug("RAG context FULL: %s", rag_context)
            _LOGGER.debug(
                "Final system prompt length: %d chars", len(final_system_prompt)
            )

        # Add system prompt first if we have one
        if final_system_prompt:
            messages.append({"role": "system", "content": final_system_prompt})

        # Add conversation history
        messages.extend(history)

        # Add the current user query
        messages.append({"role": "user", "content": query})

        return messages

    def _detect_function_call(self, response_text: str) -> list[FunctionCall] | None:
        """Detect and parse function calls from response text.

        Args:
            response_text: The text response from the AI provider.

        Returns:
            List of FunctionCall objects if found, None otherwise.
        """
        # Parse the text to JSON/Dict
        parsed_result = self.response_parser.parse(response_text)

        if parsed_result["type"] == "text":
            return None

        content = parsed_result["content"]
        if not isinstance(content, dict):
            return None

        # Try FunctionCallHandler for standard provider formats
        # We try strict formats first

        # OpenAI format
        fc_openai = (
            FunctionCallHandler.parse_openai_response(
                {"choices": [{"message": {"tool_calls": content.get("tool_calls")}}]}
            )
            if "tool_calls" in content
            else None
        )
        if fc_openai:
            return fc_openai

        # Gemini format (simulated based on typical JSON output)
        if "functionCall" in content:
            return [
                FunctionCall(
                    id=f"gemini_{content['functionCall'].get('name', '')}",
                    name=content["functionCall"].get("name", ""),
                    arguments=content["functionCall"].get("args", {}),
                )
            ]

        # Anthropic format: {"tool_use": {"id": ..., "name": ..., "input": ...}}
        if "tool_use" in content:
            tool_use = content["tool_use"]
            return [
                FunctionCall(
                    id=tool_use.get("id", ""),
                    name=tool_use.get("name", ""),
                    arguments=tool_use.get("input", {}),
                )
            ]

        # Fallback: Check for simplified/direct JSON format often used in custom implementations
        # e.g. {"function": "name", "parameters": {}} or {"name": "name", "arguments": {}}
        name = content.get("function") or content.get("name") or content.get("tool")
        args = (
            content.get("parameters") or content.get("arguments") or content.get("args")
        )

        if name and isinstance(name, str) and isinstance(args, dict):
            return [FunctionCall(id=name, name=name, arguments=args)]

        # Check for list of tool calls in 'tool_calls' key directly
        tool_calls_list = content.get("tool_calls")
        if isinstance(tool_calls_list, list):
            result = []
            for tc in tool_calls_list:
                # Handle OpenAI-style inside the list
                func = tc.get("function", {})
                if func:
                    result.append(
                        FunctionCall(
                            id=tc.get("id", ""),
                            name=func.get("name", ""),
                            arguments=func.get("arguments", {})
                            if isinstance(func.get("arguments"), dict)
                            else json.loads(func.get("arguments", "{}")),
                        )
                    )
            if result:
                return result

        return None

    async def process(
        self,
        query: str,
        messages: list[dict[str, Any]],
        system_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Process a user query and return the AI response.

        Args:
            query: The user's query text.
            messages: Conversation history (previous messages).
            system_prompt: Optional system message for context.
            tools: Optional list of tools available to the AI.
            **kwargs: Additional arguments passed to the provider.

        Returns:
            Dict with:
                - success: True if successful, False otherwise.
                - response: The AI response text (on success).
                - messages: Updated message list including response (on success).
                - error: Error message (on failure).
        """
        # Sanitize the query
        sanitized_query = self._sanitize_query(query)

        # Check for empty query
        if not sanitized_query:
            return {
                "success": False,
                "error": "Query is empty or contains only whitespace",
            }

        # Extract RAG context from kwargs
        rag_context = kwargs.pop("rag_context", None)

        # Build the message list with RAG context
        built_messages = self._build_messages(
            sanitized_query,
            messages,
            system_prompt=system_prompt,
            rag_context=rag_context,
        )

        hass = kwargs.get("hass")
        current_iteration = 0

        try:
            while current_iteration < self.max_iterations:
                # Call the provider
                provider_kwargs: dict[str, Any] = {**kwargs}
                if tools is not None:
                    provider_kwargs["tools"] = tools

                response_text = await self.provider.get_response(
                    built_messages, **provider_kwargs
                )

                # Detect function call
                function_calls = self._detect_function_call(response_text)

                if not function_calls:
                    # No function call, return the response
                    # Build the updated message list with the response
                    updated_messages = list(built_messages)
                    # Remove system prompt from returned messages if present
                    if (
                        system_prompt
                        and updated_messages
                        and updated_messages[0].get("role") == "system"
                    ):
                        updated_messages = updated_messages[1:]

                    updated_messages.append(
                        {"role": "assistant", "content": response_text}
                    )

                    return {
                        "success": True,
                        "response": response_text,
                        "messages": updated_messages,
                    }

                # Handle function calls
                _LOGGER.info("Detected function calls: %s", function_calls)

                # Append assistant's message (the tool call)
                built_messages.append({"role": "assistant", "content": response_text})

                for fc in function_calls:
                    try:
                        result = await ToolRegistry.execute_tool(
                            tool_id=fc.name, params=fc.arguments, hass=hass
                        )
                        result_str = json.dumps(result.to_dict())
                        built_messages.append(
                            {
                                "role": "function",
                                "name": fc.name,
                                "tool_use_id": fc.id,  # For Anthropic compatibility
                                "content": result_str,
                            }
                        )
                    except Exception as e:
                        error_msg = json.dumps({"error": str(e), "tool": fc.name})
                        built_messages.append(
                            {
                                "role": "function",
                                "name": fc.name,
                                "tool_use_id": fc.id,  # For Anthropic compatibility
                                "content": error_msg,
                            }
                        )

                current_iteration += 1

            # Max iterations reached
            return {
                "success": False,
                "error": "Maximum iterations reached without final response",
            }

        except Exception as e:
            _LOGGER.error("Error processing query: %s", str(e))
            return {
                "success": False,
                "error": str(e),
            }
