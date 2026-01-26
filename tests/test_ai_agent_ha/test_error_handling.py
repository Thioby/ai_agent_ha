"""Tests for error handling module - TDD RED phase.

These tests define the expected behavior of the error_handler module.
"""

import json
import os
import sys

import pytest

# Add the custom_components/ai_agent_ha directory directly to path
# This allows importing error_handler without triggering __init__.py
# which requires homeassistant module
_module_path = os.path.join(
    os.path.dirname(__file__), "..", "..", "custom_components", "ai_agent_ha"
)
sys.path.insert(0, _module_path)

# Direct import from the module file
from error_handler import (
    ErrorClassifier,
    ErrorType,
    RetryTracker,
    format_error_for_llm,
)


class TestErrorClassifier:
    """Test error classification logic."""

    def test_classify_network_timeout_as_transient(self):
        """Network timeout errors should be classified as TRANSIENT."""
        error_msg = "Connection timeout while fetching data"
        error_type, is_retryable = ErrorClassifier.classify(error_msg)

        assert error_type == ErrorType.TRANSIENT
        assert is_retryable is True

    def test_classify_connection_error_as_transient(self):
        """Connection errors should be classified as TRANSIENT."""
        error_msg = "ECONNRESET: Connection reset by peer"
        error_type, is_retryable = ErrorClassifier.classify(error_msg)

        assert error_type == ErrorType.TRANSIENT
        assert is_retryable is True

    def test_classify_rate_limit_as_transient(self):
        """Rate limit errors should be classified as TRANSIENT."""
        error_msg = "Rate limit exceeded. Please try again later."
        error_type, is_retryable = ErrorClassifier.classify(error_msg)

        assert error_type == ErrorType.TRANSIENT
        assert is_retryable is True

    def test_classify_503_as_transient(self):
        """HTTP 503 errors should be classified as TRANSIENT."""
        error_msg = "Service temporarily unavailable (503)"
        error_type, is_retryable = ErrorClassifier.classify(error_msg)

        assert error_type == ErrorType.TRANSIENT
        assert is_retryable is True

    def test_classify_entity_not_found_as_logic(self):
        """Entity not found errors should be classified as LOGIC."""
        error_msg = "Entity light.living_room_xyz not found"
        error_type, is_retryable = ErrorClassifier.classify(error_msg)

        assert error_type == ErrorType.LOGIC
        assert is_retryable is False

    def test_classify_invalid_parameters_as_logic(self):
        """Invalid parameter errors should be classified as LOGIC."""
        error_msg = "Invalid parameter 'brightness': expected int, got string"
        error_type, is_retryable = ErrorClassifier.classify(error_msg)

        assert error_type == ErrorType.LOGIC
        assert is_retryable is False

    def test_classify_permission_denied_as_logic(self):
        """Permission denied errors should be classified as LOGIC."""
        error_msg = "Permission denied: cannot access entity"
        error_type, is_retryable = ErrorClassifier.classify(error_msg)

        assert error_type == ErrorType.LOGIC
        assert is_retryable is False

    def test_classify_does_not_exist_as_logic(self):
        """'Does not exist' errors should be classified as LOGIC."""
        error_msg = "Area 'kitchen_2' does not exist"
        error_type, is_retryable = ErrorClassifier.classify(error_msg)

        assert error_type == ErrorType.LOGIC
        assert is_retryable is False

    def test_classify_unknown_error_as_logic(self):
        """Unknown errors should default to LOGIC (let LLM decide)."""
        error_msg = "Something unexpected happened"
        error_type, is_retryable = ErrorClassifier.classify(error_msg)

        assert error_type == ErrorType.LOGIC
        assert is_retryable is False


class TestRetryTracker:
    """Test retry tracking logic."""

    def test_retry_tracker_initial_state(self):
        """New tracker should allow retries."""
        tracker = RetryTracker(max_attempts=3)

        assert tracker.can_retry() is True
        assert tracker.current_attempt == 0

    def test_retry_tracker_increments(self):
        """Tracker should increment attempt count."""
        tracker = RetryTracker(max_attempts=3)

        tracker.increment()
        assert tracker.current_attempt == 1
        assert tracker.can_retry() is True

        tracker.increment()
        assert tracker.current_attempt == 2
        assert tracker.can_retry() is True

        tracker.increment()
        assert tracker.current_attempt == 3
        assert tracker.can_retry() is False

    def test_retry_tracker_max_reached(self):
        """Tracker should return False when max attempts reached."""
        tracker = RetryTracker(max_attempts=3)

        # Use up all attempts
        for _ in range(3):
            tracker.increment()

        assert tracker.can_retry() is False
        assert tracker.current_attempt == 3

    def test_retry_tracker_backoff(self):
        """Tracker should provide increasing backoff times."""
        tracker = RetryTracker(max_attempts=3)

        # First attempt - 1 second backoff
        tracker.increment()
        assert tracker.get_backoff() == 1

        # Second attempt - 2 seconds backoff
        tracker.increment()
        assert tracker.get_backoff() == 2

        # Third attempt - 3 seconds backoff
        tracker.increment()
        assert tracker.get_backoff() == 3

    def test_retry_tracker_reset(self):
        """Tracker should reset to initial state."""
        tracker = RetryTracker(max_attempts=3)

        tracker.increment()
        tracker.increment()
        assert tracker.current_attempt == 2

        tracker.reset()
        assert tracker.current_attempt == 0
        assert tracker.can_retry() is True


class TestFormatErrorForLLM:
    """Test error message formatting for LLM consumption."""

    def test_format_error_basic(self):
        """Format basic error message."""
        result = format_error_for_llm(
            error="Entity not found",
            request_type="get_entity_state",
            parameters={"entity_id": "light.test"},
            attempt=1,
            max_attempts=3,
        )

        assert "tool_error" in result
        assert "Entity not found" in result
        assert "get_entity_state" in result
        assert "attempt" in result.lower() or "1" in result

    def test_format_error_truncates_long_messages(self):
        """Long error messages should be truncated to 500 chars."""
        long_error = "A" * 1000
        result = format_error_for_llm(
            error=long_error,
            request_type="web_fetch",
            parameters={"url": "https://example.com"},
            attempt=1,
            max_attempts=3,
        )

        # Error message in result should be truncated (500 + "... (truncated)")
        # Total JSON will be larger due to metadata, but error itself should be capped
        parsed = json.loads(result)
        assert len(parsed["error"]) <= 520  # 500 + "... (truncated)"
        assert "truncated" in parsed["error"]

    def test_format_error_includes_hint(self):
        """Formatted error should include hint for LLM."""
        result = format_error_for_llm(
            error="Entity not found",
            request_type="get_entity_state",
            parameters={"entity_id": "light.test"},
            attempt=1,
            max_attempts=3,
        )

        # Should include some guidance for the LLM
        assert (
            "try" in result.lower()
            or "alternative" in result.lower()
            or "approach" in result.lower()
        )

    def test_format_error_valid_json_structure(self):
        """Formatted error should be valid JSON or contain structured info."""
        result = format_error_for_llm(
            error="Test error",
            request_type="test_tool",
            parameters={"param": "value"},
            attempt=2,
            max_attempts=3,
        )

        # Should be parseable as JSON
        try:
            parsed = json.loads(result)
            assert "tool_error" in parsed or "error" in parsed
        except json.JSONDecodeError:
            # If not JSON, should still contain key info
            assert "test_tool" in result.lower() or "error" in result.lower()
