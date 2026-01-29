"""Utility functions for AI Agent HA."""
from __future__ import annotations

from typing import Any


# Sensitive field patterns (case-insensitive)
SENSITIVE_PATTERNS = {
    "token",
    "key",
    "password",
    "secret",
    "credential",
    "auth",
    "authorization",
    "api_key",
    "apikey",
    "llama_token",
    "openai_token",
    "gemini_token",
    "anthropic_token",
    "openrouter_token",
    "alter_token",
    "zai_token",
    "groq_token",
}


def sanitize_for_logging(data: Any, mask: str = "***REDACTED***") -> Any:
    """Sanitize sensitive data for safe logging.

    Recursively masks sensitive fields like API keys, tokens, passwords, etc.
    This prevents accidental exposure of credentials in debug logs.

    Args:
        data: The data structure to sanitize (dict, list, str, etc.)
        mask: The string to use for masking sensitive values

    Returns:
        A sanitized copy of the data with sensitive fields masked

    Example:
        >>> config = {"openai_token": "sk-abc123", "ai_provider": "openai"}
        >>> sanitize_for_logging(config)
        {"openai_token": "***REDACTED***", "ai_provider": "openai"}
    """
    if isinstance(data, dict):
        sanitized = {}
        for key, value in data.items():
            # Check if key matches any sensitive pattern
            key_lower = str(key).lower()
            is_sensitive = any(pattern in key_lower for pattern in SENSITIVE_PATTERNS)

            if is_sensitive:
                sanitized[key] = mask
            else:
                # Recursively sanitize nested structures
                sanitized[key] = sanitize_for_logging(value, mask)
        return sanitized

    elif isinstance(data, list):
        return [sanitize_for_logging(item, mask) for item in data]

    elif isinstance(data, tuple):
        return tuple(sanitize_for_logging(item, mask) for item in data)

    else:
        # Primitive types (str, int, bool, etc.) - return as-is
        return data
