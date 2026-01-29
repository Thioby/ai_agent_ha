"""Query processor for AI agent interactions."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

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
    ) -> list[dict[str, Any]]:
        """Build the message list for the AI provider.

        Args:
            query: The current user query.
            history: Previous conversation messages.
            system_prompt: Optional system message to prepend.

        Returns:
            List of message dictionaries ready for the provider.
        """
        messages: list[dict[str, Any]] = []

        # Add system prompt first if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Add conversation history
        messages.extend(history)

        # Add the current user query
        messages.append({"role": "user", "content": query})

        return messages

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

        # Build the message list
        built_messages = self._build_messages(
            sanitized_query, messages, system_prompt=system_prompt
        )

        try:
            # Call the provider
            provider_kwargs: dict[str, Any] = {**kwargs}
            if tools is not None:
                provider_kwargs["tools"] = tools

            response = await self.provider.get_response(built_messages, **provider_kwargs)

            # Build the updated message list with the response
            updated_messages = list(built_messages)
            # Remove system prompt from returned messages if present
            if system_prompt and updated_messages and updated_messages[0].get("role") == "system":
                updated_messages = updated_messages[1:]

            updated_messages.append({"role": "assistant", "content": response})

            return {
                "success": True,
                "response": response,
                "messages": updated_messages,
            }

        except Exception as e:
            _LOGGER.error("Error processing query: %s", str(e))
            return {
                "success": False,
                "error": str(e),
            }
