"""Error handling module for AI Agent self-healing behavior.

This module provides error classification and retry tracking to enable
the AI agent to self-correct when tools fail, similar to OpenCode/Claude Code.

Classes:
    ErrorType: Enum for error classification (TRANSIENT vs LOGIC)
    ErrorClassifier: Classifies errors to determine retry strategy
    RetryTracker: Tracks retry attempts per tool call

Functions:
    format_error_for_llm: Formats error messages for LLM consumption
"""

import json
import logging
import re
from enum import Enum
from typing import Any, Dict, Tuple

_LOGGER = logging.getLogger(__name__)


class ErrorType(Enum):
    """Classification of error types for retry strategy.

    TRANSIENT: Temporary errors that may resolve on retry (network, rate limits)
    LOGIC: Errors requiring different approach (invalid params, not found)
    """

    TRANSIENT = "transient"
    LOGIC = "logic"


class ErrorClassifier:
    """Classifies errors to determine appropriate retry strategy.

    TRANSIENT errors (auto-retry with backoff):
        - Network timeouts
        - Connection resets
        - Rate limits
        - HTTP 503/504 errors
        - Temporary unavailability

    LOGIC errors (feed to LLM for self-correction):
        - Entity/resource not found
        - Invalid parameters
        - Permission denied
        - Malformed requests
    """

    # Patterns indicating transient errors (case-insensitive)
    TRANSIENT_PATTERNS = [
        r"timeout",
        r"timed?\s*out",
        r"connection",
        r"econnreset",
        r"rate\s*limit",
        r"too\s*many\s*requests",
        r"503",
        r"504",
        r"temporarily\s*unavailable",
        r"service\s*unavailable",
        r"try\s*again\s*later",
        r"network\s*error",
        r"socket\s*error",
    ]

    # Patterns indicating logic errors (case-insensitive)
    LOGIC_PATTERNS = [
        r"not\s*found",
        r"does\s*not\s*exist",
        r"invalid",
        r"malformed",
        r"permission\s*denied",
        r"unauthorized",
        r"forbidden",
        r"bad\s*request",
        r"missing\s*required",
        r"unknown\s*entity",
        r"no\s*such",
    ]

    @classmethod
    def classify(cls, error: str) -> Tuple[ErrorType, bool]:
        """Classify an error message to determine retry strategy.

        Args:
            error: The error message string to classify

        Returns:
            Tuple of (ErrorType, is_retryable)
            - TRANSIENT errors are retryable (auto-retry with backoff)
            - LOGIC errors are not retryable (feed to LLM)
        """
        if not error:
            return ErrorType.LOGIC, False

        error_lower = error.lower()

        # Check for transient patterns first (they take priority)
        for pattern in cls.TRANSIENT_PATTERNS:
            if re.search(pattern, error_lower):
                _LOGGER.debug(f"Classified as TRANSIENT: {error[:100]}")
                return ErrorType.TRANSIENT, True

        # Check for logic patterns
        for pattern in cls.LOGIC_PATTERNS:
            if re.search(pattern, error_lower):
                _LOGGER.debug(f"Classified as LOGIC: {error[:100]}")
                return ErrorType.LOGIC, False

        # Default to LOGIC (let LLM decide what to do)
        _LOGGER.debug(f"Classified as LOGIC (default): {error[:100]}")
        return ErrorType.LOGIC, False


class RetryTracker:
    """Tracks retry attempts for a specific tool call.

    Provides:
        - Attempt counting
        - Max retry enforcement
        - Exponential backoff calculation
        - Reset capability

    Usage:
        tracker = RetryTracker(max_attempts=3)
        while tracker.can_retry():
            tracker.increment()
            backoff = tracker.get_backoff()
            # ... attempt operation with backoff delay
    """

    def __init__(self, max_attempts: int = 3):
        """Initialize retry tracker.

        Args:
            max_attempts: Maximum number of retry attempts allowed
        """
        self.max_attempts = max_attempts
        self.current_attempt = 0

    def can_retry(self) -> bool:
        """Check if more retry attempts are allowed.

        Returns:
            True if current_attempt < max_attempts
        """
        return self.current_attempt < self.max_attempts

    def increment(self) -> None:
        """Increment the attempt counter."""
        self.current_attempt += 1
        _LOGGER.debug(f"Retry attempt {self.current_attempt}/{self.max_attempts}")

    def get_backoff(self) -> int:
        """Get the backoff delay in seconds for current attempt.

        Uses simple linear backoff: attempt_number seconds.

        Returns:
            Backoff delay in seconds (1, 2, 3, ...)
        """
        return max(1, self.current_attempt)

    def reset(self) -> None:
        """Reset the tracker to initial state."""
        self.current_attempt = 0
        _LOGGER.debug("Retry tracker reset")


def format_error_for_llm(
    error: str,
    request_type: str,
    parameters: Dict[str, Any],
    attempt: int,
    max_attempts: int,
    max_error_length: int = 500,
) -> str:
    """Format an error message for LLM consumption.

    Creates a structured JSON message that the LLM can parse to understand
    what went wrong and how to self-correct.

    Args:
        error: The original error message
        request_type: The tool/request type that failed
        parameters: The parameters that were used
        attempt: Current attempt number (1-based)
        max_attempts: Maximum attempts allowed
        max_error_length: Maximum length for error message (default 500)

    Returns:
        JSON string with structured error information for LLM
    """
    # Truncate long error messages
    truncated_error = error
    if len(error) > max_error_length:
        truncated_error = error[:max_error_length] + "... (truncated)"

    # Truncate long parameter values
    safe_params = {}
    for key, value in parameters.items():
        str_value = str(value)
        if len(str_value) > 100:
            safe_params[key] = str_value[:100] + "..."
        else:
            safe_params[key] = value

    error_info = {
        "tool_error": True,
        "error": truncated_error,
        "failed_request": request_type,
        "failed_parameters": safe_params,
        "attempt": attempt,
        "max_attempts": max_attempts,
        "hint": "Please analyze this error and try a different approach. "
        "Consider using alternative tools or parameters.",
    }

    return json.dumps(error_info, ensure_ascii=False)
