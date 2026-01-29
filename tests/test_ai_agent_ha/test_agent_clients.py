"""Tests for AI client implementations and security utilities."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from custom_components.ai_agent_ha.agent import (
    LocalClient,
    OpenAIClient,
    GeminiClient,
    AnthropicClient,
    OpenRouterClient,
    LlamaClient,
    AlterClient,
    ZaiClient,
    sanitize_for_logging,
)


class TestSanitizeForLogging:
    """Test sanitize_for_logging security function."""

    def test_masks_simple_tokens(self):
        """Test masking simple dict with tokens."""
        data = {
            "openai_token": "sk-abc123secret",
            "ai_provider": "openai",
        }
        result = sanitize_for_logging(data)
        assert result["openai_token"] == "***REDACTED***"
        assert result["ai_provider"] == "openai"

    def test_masks_all_token_types(self):
        """Test masking all known token patterns."""
        data = {
            "llama_token": "secret1",
            "gemini_token": "secret2",
            "anthropic_token": "secret3",
            "openrouter_token": "secret4",
            "alter_token": "secret5",
            "zai_token": "secret6",
            "api_key": "secret7",
            "password": "secret8",
            "secret": "secret9",
            "auth": "secret10",
        }
        result = sanitize_for_logging(data)
        for key in data:
            assert result[key] == "***REDACTED***"

    def test_masks_nested_dict(self):
        """Test masking nested dict structures."""
        data = {
            "config": {
                "openai_token": "secret",
                "model": "gpt-4",
            },
            "provider": "openai",
        }
        result = sanitize_for_logging(data)
        assert result["config"]["openai_token"] == "***REDACTED***"
        assert result["config"]["model"] == "gpt-4"
        assert result["provider"] == "openai"

    def test_masks_list_items(self):
        """Test masking items in lists."""
        data = [
            {"token": "secret1"},
            {"name": "test"},
        ]
        result = sanitize_for_logging(data)
        assert result[0]["token"] == "***REDACTED***"
        assert result[1]["name"] == "test"

    def test_masks_tuple_items(self):
        """Test masking items in tuples."""
        data = (
            {"password": "secret"},
            {"value": "public"},
        )
        result = sanitize_for_logging(data)
        assert isinstance(result, tuple)
        assert result[0]["password"] == "***REDACTED***"
        assert result[1]["value"] == "public"

    def test_case_insensitive_matching(self):
        """Test case-insensitive pattern matching."""
        data = {
            "TOKEN": "secret1",
            "Token": "secret2",
            "API_KEY": "secret3",
            "ApiKey": "secret4",
        }
        result = sanitize_for_logging(data)
        for key in data:
            assert result[key] == "***REDACTED***"

    def test_passes_normal_fields(self):
        """Test that non-sensitive fields pass through."""
        data = {
            "ai_provider": "openai",
            "model": "gpt-4",
            "temperature": 0.7,
            "enabled": True,
            "count": 42,
        }
        result = sanitize_for_logging(data)
        assert result == data

    def test_custom_mask(self):
        """Test custom mask string."""
        data = {"token": "secret"}
        result = sanitize_for_logging(data, mask="[HIDDEN]")
        assert result["token"] == "[HIDDEN]"

    def test_primitive_values_unchanged(self):
        """Test that primitive values are returned unchanged."""
        assert sanitize_for_logging("string") == "string"
        assert sanitize_for_logging(42) == 42
        assert sanitize_for_logging(3.14) == 3.14
        assert sanitize_for_logging(True) is True
        assert sanitize_for_logging(None) is None


class TestLocalClient:
    """Test Local AI client functionality."""

    def test_local_client_initialization(self):
        """Test LocalClient initialization."""
        client = LocalClient("http://localhost:11434/api/generate", "llama3.2")
        assert client.url == "http://localhost:11434/api/generate"
        assert client.model == "llama3.2"

        # Test without model
        client_no_model = LocalClient("http://localhost:11434/api/generate")
        assert client_no_model.model == ""

    def _create_mock_response(self, status=200, json_data=None, text_data=None):
        """Helper to create a mock aiohttp response."""
        mock_resp = MagicMock()
        mock_resp.status = status
        mock_resp.headers = {"Content-Type": "application/json"}

        if json_data is not None:
            mock_resp.text = AsyncMock(return_value=json.dumps(json_data))
        elif text_data is not None:
            mock_resp.text = AsyncMock(return_value=text_data)
        else:
            mock_resp.text = AsyncMock(return_value="")

        return mock_resp

    @pytest.mark.asyncio
    async def test_local_client_get_response_ollama_format(self):
        """Test LocalClient with Ollama response format."""
        client = LocalClient("http://localhost:11434/api/generate", "llama3.2")
        messages = [{"role": "user", "content": "Hello"}]

        mock_resp = self._create_mock_response(
            json_data={"response": "Hello! How can I help you?", "done": True}
        )

        # Create async context manager mocks
        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=MagicMock(
            post=MagicMock(return_value=MagicMock(
                __aenter__=AsyncMock(return_value=mock_resp),
                __aexit__=AsyncMock(return_value=None)
            ))
        ))
        mock_session_ctx.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session_ctx):
            result = await client.get_response(messages)

        parsed = json.loads(result)
        assert parsed["request_type"] == "final_response"
        assert parsed["response"] == "Hello! How can I help you?"

    @pytest.mark.asyncio
    async def test_local_client_get_response_openai_format(self):
        """Test LocalClient with OpenAI-like response format."""
        client = LocalClient("http://localhost:8080/v1/completions", "local-model")
        messages = [{"role": "user", "content": "Test"}]

        mock_resp = self._create_mock_response(
            json_data={
                "choices": [{"message": {"content": "Test response"}}]
            }
        )

        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=MagicMock(
            post=MagicMock(return_value=MagicMock(
                __aenter__=AsyncMock(return_value=mock_resp),
                __aexit__=AsyncMock(return_value=None)
            ))
        ))
        mock_session_ctx.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session_ctx):
            result = await client.get_response(messages)

        parsed = json.loads(result)
        assert parsed["request_type"] == "final_response"
        assert parsed["response"] == "Test response"

    @pytest.mark.asyncio
    async def test_local_client_get_response_json_response(self):
        """Test LocalClient when model returns valid JSON with request_type."""
        client = LocalClient("http://localhost:11434/api/generate", "llama3.2")
        messages = [{"role": "user", "content": "Get entities"}]

        # Model returns proper JSON response
        json_response = json.dumps({
            "request_type": "get_entities_by_domain",
            "parameters": {"domain": "light"}
        })
        mock_resp = self._create_mock_response(
            json_data={"response": json_response, "done": True}
        )

        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=MagicMock(
            post=MagicMock(return_value=MagicMock(
                __aenter__=AsyncMock(return_value=mock_resp),
                __aexit__=AsyncMock(return_value=None)
            ))
        ))
        mock_session_ctx.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session_ctx):
            result = await client.get_response(messages)

        parsed = json.loads(result)
        assert parsed["request_type"] == "get_entities_by_domain"
        assert parsed["parameters"]["domain"] == "light"

    @pytest.mark.asyncio
    async def test_local_client_get_response_http_404_with_model(self):
        """Test LocalClient handles 404 error with model not found."""
        client = LocalClient("http://localhost:11434/api/generate", "nonexistent-model")
        messages = [{"role": "user", "content": "Hello"}]

        mock_resp = self._create_mock_response(status=404, text_data="model not found")

        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=MagicMock(
            post=MagicMock(return_value=MagicMock(
                __aenter__=AsyncMock(return_value=mock_resp),
                __aexit__=AsyncMock(return_value=None)
            ))
        ))
        mock_session_ctx.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session_ctx):
            with pytest.raises(Exception) as exc_info:
                await client.get_response(messages)

        assert "nonexistent-model" in str(exc_info.value)
        assert "not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_local_client_get_response_http_404_no_model(self):
        """Test LocalClient handles 404 error without model."""
        client = LocalClient("http://localhost:11434/api/wrong-endpoint")
        messages = [{"role": "user", "content": "Hello"}]

        mock_resp = self._create_mock_response(status=404, text_data="endpoint not found")

        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=MagicMock(
            post=MagicMock(return_value=MagicMock(
                __aenter__=AsyncMock(return_value=mock_resp),
                __aexit__=AsyncMock(return_value=None)
            ))
        ))
        mock_session_ctx.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session_ctx):
            with pytest.raises(Exception) as exc_info:
                await client.get_response(messages)

        assert "endpoint not found" in str(exc_info.value).lower() or "not found" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_local_client_get_response_http_400(self):
        """Test LocalClient handles 400 bad request error."""
        client = LocalClient("http://localhost:11434/api/generate", "llama3.2")
        messages = [{"role": "user", "content": "Hello"}]

        mock_resp = self._create_mock_response(status=400, text_data="invalid request body")

        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=MagicMock(
            post=MagicMock(return_value=MagicMock(
                __aenter__=AsyncMock(return_value=mock_resp),
                __aexit__=AsyncMock(return_value=None)
            ))
        ))
        mock_session_ctx.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session_ctx):
            with pytest.raises(Exception) as exc_info:
                await client.get_response(messages)

        assert "Bad request" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_local_client_get_response_http_500(self):
        """Test LocalClient handles 500 server error."""
        client = LocalClient("http://localhost:11434/api/generate", "llama3.2")
        messages = [{"role": "user", "content": "Hello"}]

        mock_resp = self._create_mock_response(status=500, text_data="internal server error")

        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=MagicMock(
            post=MagicMock(return_value=MagicMock(
                __aenter__=AsyncMock(return_value=mock_resp),
                __aexit__=AsyncMock(return_value=None)
            ))
        ))
        mock_session_ctx.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session_ctx):
            with pytest.raises(Exception) as exc_info:
                await client.get_response(messages)

        assert "500" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_local_client_get_response_empty_response(self):
        """Test LocalClient handles empty response from Ollama."""
        client = LocalClient("http://localhost:11434/api/generate", "llama3.2")
        messages = [{"role": "user", "content": "Hello"}]

        mock_resp = self._create_mock_response(
            json_data={"response": "", "done": True}
        )

        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=MagicMock(
            post=MagicMock(return_value=MagicMock(
                __aenter__=AsyncMock(return_value=mock_resp),
                __aexit__=AsyncMock(return_value=None)
            ))
        ))
        mock_session_ctx.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session_ctx):
            result = await client.get_response(messages)

        parsed = json.loads(result)
        assert parsed["request_type"] == "final_response"
        assert "empty response" in parsed["response"].lower()

    @pytest.mark.asyncio
    async def test_local_client_get_response_model_loading(self):
        """Test LocalClient handles Ollama model loading state."""
        client = LocalClient("http://localhost:11434/api/generate", "llama3.2")
        messages = [{"role": "user", "content": "Hello"}]

        mock_resp = self._create_mock_response(
            json_data={"response": "", "done": True, "done_reason": "load"}
        )

        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=MagicMock(
            post=MagicMock(return_value=MagicMock(
                __aenter__=AsyncMock(return_value=mock_resp),
                __aexit__=AsyncMock(return_value=None)
            ))
        ))
        mock_session_ctx.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session_ctx):
            result = await client.get_response(messages)

        parsed = json.loads(result)
        assert parsed["request_type"] == "final_response"
        assert "loading" in parsed["response"].lower()

    @pytest.mark.asyncio
    async def test_local_client_get_response_not_done(self):
        """Test LocalClient handles Ollama not done state."""
        client = LocalClient("http://localhost:11434/api/generate", "llama3.2")
        messages = [{"role": "user", "content": "Hello"}]

        mock_resp = self._create_mock_response(
            json_data={"response": "", "done": False}
        )

        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=MagicMock(
            post=MagicMock(return_value=MagicMock(
                __aenter__=AsyncMock(return_value=mock_resp),
                __aexit__=AsyncMock(return_value=None)
            ))
        ))
        mock_session_ctx.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session_ctx):
            result = await client.get_response(messages)

        parsed = json.loads(result)
        assert parsed["request_type"] == "final_response"
        assert "processing" in parsed["response"].lower() or "still" in parsed["response"].lower()

    @pytest.mark.asyncio
    async def test_local_client_get_response_generic_content_field(self):
        """Test LocalClient with generic content field format."""
        client = LocalClient("http://localhost:8080/api/generate", "local-model")
        messages = [{"role": "user", "content": "Test"}]

        mock_resp = self._create_mock_response(
            json_data={"content": "Generic content response"}
        )

        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=MagicMock(
            post=MagicMock(return_value=MagicMock(
                __aenter__=AsyncMock(return_value=mock_resp),
                __aexit__=AsyncMock(return_value=None)
            ))
        ))
        mock_session_ctx.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session_ctx):
            result = await client.get_response(messages)

        parsed = json.loads(result)
        assert parsed["request_type"] == "final_response"
        assert parsed["response"] == "Generic content response"

    @pytest.mark.asyncio
    async def test_local_client_get_response_message_field(self):
        """Test LocalClient with message field format."""
        client = LocalClient("http://localhost:8080/api/generate", "local-model")
        messages = [{"role": "user", "content": "Test"}]

        mock_resp = self._create_mock_response(
            json_data={"message": {"content": "Message content response"}}
        )

        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=MagicMock(
            post=MagicMock(return_value=MagicMock(
                __aenter__=AsyncMock(return_value=mock_resp),
                __aexit__=AsyncMock(return_value=None)
            ))
        ))
        mock_session_ctx.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session_ctx):
            result = await client.get_response(messages)

        parsed = json.loads(result)
        assert parsed["request_type"] == "final_response"
        assert parsed["response"] == "Message content response"

    @pytest.mark.asyncio
    async def test_local_client_get_response_plain_text(self):
        """Test LocalClient handles plain text (non-JSON) response."""
        client = LocalClient("http://localhost:11434/api/generate", "llama3.2")
        messages = [{"role": "user", "content": "Hello"}]

        mock_resp = self._create_mock_response(text_data="Plain text response from model")

        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=MagicMock(
            post=MagicMock(return_value=MagicMock(
                __aenter__=AsyncMock(return_value=mock_resp),
                __aexit__=AsyncMock(return_value=None)
            ))
        ))
        mock_session_ctx.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session_ctx):
            result = await client.get_response(messages)

        parsed = json.loads(result)
        assert parsed["request_type"] == "final_response"
        assert parsed["response"] == "Plain text response from model"

    @pytest.mark.asyncio
    async def test_local_client_get_response_unexpected_format(self):
        """Test LocalClient handles unexpected response format."""
        client = LocalClient("http://localhost:11434/api/generate", "llama3.2")
        messages = [{"role": "user", "content": "Hello"}]

        mock_resp = self._create_mock_response(
            json_data={"unexpected_field": "some value", "other_field": 123}
        )

        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=MagicMock(
            post=MagicMock(return_value=MagicMock(
                __aenter__=AsyncMock(return_value=mock_resp),
                __aexit__=AsyncMock(return_value=None)
            ))
        ))
        mock_session_ctx.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session_ctx):
            result = await client.get_response(messages)

        parsed = json.loads(result)
        assert parsed["request_type"] == "final_response"
        assert "unexpected response format" in parsed["response"].lower()

    @pytest.mark.asyncio
    async def test_local_client_prompt_formatting(self):
        """Test LocalClient formats messages correctly in prompt."""
        client = LocalClient("http://localhost:11434/api/generate", "llama3.2")
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]

        mock_resp = self._create_mock_response(
            json_data={"response": "I'm doing well!", "done": True}
        )

        captured_payload = None

        def capture_post(*args, **kwargs):
            nonlocal captured_payload
            captured_payload = kwargs.get("json", {})
            mock_post_ctx = MagicMock()
            mock_post_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
            mock_post_ctx.__aexit__ = AsyncMock(return_value=None)
            return mock_post_ctx

        mock_session = MagicMock()
        mock_session.post = capture_post
        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_ctx.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session_ctx):
            await client.get_response(messages)

        assert captured_payload is not None
        assert "System: You are a helpful assistant" in captured_payload["prompt"]
        assert "User: Hello" in captured_payload["prompt"]
        assert "Assistant: Hi there!" in captured_payload["prompt"]
        assert "User: How are you?" in captured_payload["prompt"]
        assert captured_payload["model"] == "llama3.2"
        assert captured_payload["stream"] is False

    @pytest.mark.asyncio
    async def test_local_client_openai_text_format(self):
        """Test LocalClient with OpenAI text field format."""
        client = LocalClient("http://localhost:8080/v1/completions", "local-model")
        messages = [{"role": "user", "content": "Test"}]

        mock_resp = self._create_mock_response(
            json_data={
                "choices": [{"text": "Text field response"}]
            }
        )

        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=MagicMock(
            post=MagicMock(return_value=MagicMock(
                __aenter__=AsyncMock(return_value=mock_resp),
                __aexit__=AsyncMock(return_value=None)
            ))
        ))
        mock_session_ctx.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session_ctx):
            result = await client.get_response(messages)

        parsed = json.loads(result)
        assert parsed["request_type"] == "final_response"
        assert parsed["response"] == "Text field response"


class TestOpenAIClient:
    """Test OpenAI client functionality."""

    def test_openai_client_initialization(self):
        """Test OpenAIClient initialization."""
        client = OpenAIClient("test-token", "gpt-3.5-turbo")
        assert client.token == "test-token"
        assert client.model == "gpt-3.5-turbo"

    def test_openai_restricted_model_detection(self):
        """Test OpenAI restricted model detection."""
        # Test restricted models
        client_o3 = OpenAIClient("test-token", "o3-mini")
        assert client_o3._is_restricted_model() is True
        
        # Test unrestricted models
        client_gpt = OpenAIClient("test-token", "gpt-3.5-turbo")
        assert client_gpt._is_restricted_model() is False

    @pytest.mark.asyncio
    async def test_openai_client_invalid_token(self):
        """Test OpenAIClient with invalid token."""
        client = OpenAIClient("invalid-token", "gpt-3.5-turbo")
        
        with pytest.raises(Exception) as exc_info:
            await client.get_response([{"role": "user", "content": "test"}])
        assert "Invalid OpenAI API key format" in str(exc_info.value)


class TestGeminiClient:
    """Test Gemini client functionality."""

    def test_gemini_client_initialization(self):
        """Test GeminiClient initialization."""
        client = GeminiClient("test-token", "gemini-2.5-flash")
        assert client.token == "test-token"
        assert client.model == "gemini-2.5-flash"
        assert "gemini-2.5-flash" in client.api_url

    def test_gemini_client_strips_whitespace(self):
        """Test GeminiClient strips whitespace from token."""
        client = GeminiClient("  test-token  ", "gemini-2.5-flash")
        assert client.token == "test-token"

    def test_gemini_client_default_model(self):
        """Test GeminiClient uses default model."""
        client = GeminiClient("test-token")
        assert client.model == "gemini-2.5-flash"

    def _create_mock_response(self, status=200, json_data=None, text_data=None):
        """Helper to create a mock aiohttp response."""
        mock_resp = MagicMock()
        mock_resp.status = status
        mock_resp.headers = {"Content-Type": "application/json"}

        if json_data is not None:
            mock_resp.text = AsyncMock(return_value=json.dumps(json_data))
        elif text_data is not None:
            mock_resp.text = AsyncMock(return_value=text_data)
        else:
            mock_resp.text = AsyncMock(return_value="")

        return mock_resp

    @pytest.mark.asyncio
    async def test_gemini_client_missing_token(self):
        """Test GeminiClient raises error with missing token."""
        client = GeminiClient("", "gemini-2.5-flash")
        messages = [{"role": "user", "content": "Hello"}]

        with pytest.raises(Exception) as exc_info:
            await client.get_response(messages)

        assert "Missing Gemini API key" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_gemini_client_get_response_success(self):
        """Test GeminiClient successful response."""
        client = GeminiClient("test-token", "gemini-2.5-flash")
        messages = [{"role": "user", "content": "Hello"}]

        mock_resp = self._create_mock_response(
            json_data={
                "candidates": [{
                    "content": {
                        "parts": [{"text": "Hello! How can I help?"}]
                    },
                    "finishReason": "STOP"
                }],
                "usageMetadata": {
                    "promptTokenCount": 10,
                    "totalTokenCount": 25
                }
            }
        )

        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=MagicMock(
            post=MagicMock(return_value=MagicMock(
                __aenter__=AsyncMock(return_value=mock_resp),
                __aexit__=AsyncMock(return_value=None)
            ))
        ))
        mock_session_ctx.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session_ctx):
            result = await client.get_response(messages)

        assert result == "Hello! How can I help?"

    @pytest.mark.asyncio
    async def test_gemini_client_get_response_http_error(self):
        """Test GeminiClient handles HTTP error."""
        client = GeminiClient("test-token", "gemini-2.5-flash")
        messages = [{"role": "user", "content": "Hello"}]

        mock_resp = self._create_mock_response(status=400, text_data="Bad Request")

        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=MagicMock(
            post=MagicMock(return_value=MagicMock(
                __aenter__=AsyncMock(return_value=mock_resp),
                __aexit__=AsyncMock(return_value=None)
            ))
        ))
        mock_session_ctx.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session_ctx):
            with pytest.raises(Exception) as exc_info:
                await client.get_response(messages)

        assert "400" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_gemini_client_get_response_invalid_json(self):
        """Test GeminiClient handles invalid JSON response."""
        client = GeminiClient("test-token", "gemini-2.5-flash")
        messages = [{"role": "user", "content": "Hello"}]

        mock_resp = self._create_mock_response(text_data="not valid json")

        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=MagicMock(
            post=MagicMock(return_value=MagicMock(
                __aenter__=AsyncMock(return_value=mock_resp),
                __aexit__=AsyncMock(return_value=None)
            ))
        ))
        mock_session_ctx.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session_ctx):
            with pytest.raises(Exception) as exc_info:
                await client.get_response(messages)

        assert "Invalid JSON" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_gemini_client_message_conversion(self):
        """Test GeminiClient converts messages to Gemini format."""
        client = GeminiClient("test-token", "gemini-2.5-flash")
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
            {"role": "user", "content": "How are you?"},
        ]

        mock_resp = self._create_mock_response(
            json_data={
                "candidates": [{
                    "content": {"parts": [{"text": "I'm fine!"}]}
                }]
            }
        )

        captured_payload = None

        def capture_post(url, **kwargs):
            nonlocal captured_payload
            captured_payload = kwargs.get("json", {})
            mock_post_ctx = MagicMock()
            mock_post_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
            mock_post_ctx.__aexit__ = AsyncMock(return_value=None)
            return mock_post_ctx

        mock_session = MagicMock()
        mock_session.post = capture_post
        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_ctx.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session_ctx):
            await client.get_response(messages)

        # Check system instruction
        assert "systemInstruction" in captured_payload
        assert captured_payload["systemInstruction"]["parts"][0]["text"] == "You are helpful"

        # Check contents
        assert len(captured_payload["contents"]) == 3  # user, model, user
        assert captured_payload["contents"][0]["role"] == "user"
        assert captured_payload["contents"][1]["role"] == "model"
        assert captured_payload["contents"][2]["role"] == "user"

    @pytest.mark.asyncio
    async def test_gemini_client_missing_parts(self):
        """Test GeminiClient handles response missing parts."""
        client = GeminiClient("test-token", "gemini-2.5-flash")
        messages = [{"role": "user", "content": "Hello"}]

        mock_resp = self._create_mock_response(
            json_data={
                "candidates": [{
                    "content": {"parts": []}  # Empty parts
                }]
            }
        )

        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=MagicMock(
            post=MagicMock(return_value=MagicMock(
                __aenter__=AsyncMock(return_value=mock_resp),
                __aexit__=AsyncMock(return_value=None)
            ))
        ))
        mock_session_ctx.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session_ctx):
            result = await client.get_response(messages)

        # Should return string representation
        assert "candidates" in result

    @pytest.mark.asyncio
    async def test_gemini_client_empty_text(self):
        """Test GeminiClient handles empty text in response."""
        client = GeminiClient("test-token", "gemini-2.5-flash")
        messages = [{"role": "user", "content": "Hello"}]

        mock_resp = self._create_mock_response(
            json_data={
                "candidates": [{
                    "content": {"parts": [{"text": ""}]}
                }]
            }
        )

        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=MagicMock(
            post=MagicMock(return_value=MagicMock(
                __aenter__=AsyncMock(return_value=mock_resp),
                __aexit__=AsyncMock(return_value=None)
            ))
        ))
        mock_session_ctx.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session_ctx):
            result = await client.get_response(messages)

        assert result == ""


class TestAnthropicClient:
    """Test Anthropic client functionality."""

    def test_anthropic_client_initialization(self):
        """Test AnthropicClient initialization."""
        client = AnthropicClient("test-token", "claude-3-5-sonnet-20241022")
        assert client.token == "test-token"
        assert client.model == "claude-3-5-sonnet-20241022"
        assert client.api_url == "https://api.anthropic.com/v1/messages"

    def test_anthropic_client_default_model(self):
        """Test AnthropicClient uses default model."""
        client = AnthropicClient("test-token")
        assert client.model == "claude-sonnet-4-5-20250929"

    def _create_mock_response(self, status=200, json_data=None, text_data=None):
        """Helper to create a mock aiohttp response."""
        mock_resp = MagicMock()
        mock_resp.status = status
        mock_resp.headers = {"Content-Type": "application/json"}

        if json_data is not None:
            mock_resp.json = AsyncMock(return_value=json_data)
            mock_resp.text = AsyncMock(return_value=json.dumps(json_data))
        elif text_data is not None:
            mock_resp.text = AsyncMock(return_value=text_data)
        else:
            mock_resp.text = AsyncMock(return_value="")

        return mock_resp

    @pytest.mark.asyncio
    async def test_anthropic_client_get_response_success(self):
        """Test AnthropicClient successful response."""
        client = AnthropicClient("test-token", "claude-3-sonnet")
        messages = [{"role": "user", "content": "Hello"}]

        mock_resp = self._create_mock_response(
            json_data={
                "content": [
                    {"type": "text", "text": "Hello! How can I help?"}
                ]
            }
        )

        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=MagicMock(
            post=MagicMock(return_value=MagicMock(
                __aenter__=AsyncMock(return_value=mock_resp),
                __aexit__=AsyncMock(return_value=None)
            ))
        ))
        mock_session_ctx.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session_ctx):
            result = await client.get_response(messages)

        assert result == "Hello! How can I help?"

    @pytest.mark.asyncio
    async def test_anthropic_client_get_response_http_error(self):
        """Test AnthropicClient handles HTTP error."""
        client = AnthropicClient("test-token", "claude-3-sonnet")
        messages = [{"role": "user", "content": "Hello"}]

        mock_resp = self._create_mock_response(status=401, text_data="Unauthorized")

        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=MagicMock(
            post=MagicMock(return_value=MagicMock(
                __aenter__=AsyncMock(return_value=mock_resp),
                __aexit__=AsyncMock(return_value=None)
            ))
        ))
        mock_session_ctx.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session_ctx):
            with pytest.raises(Exception) as exc_info:
                await client.get_response(messages)

        assert "401" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_anthropic_client_message_conversion(self):
        """Test AnthropicClient converts messages correctly."""
        client = AnthropicClient("test-token", "claude-3-sonnet")
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
            {"role": "user", "content": "How are you?"},
        ]

        mock_resp = self._create_mock_response(
            json_data={
                "content": [{"type": "text", "text": "I'm fine!"}]
            }
        )

        captured_payload = None
        captured_headers = None

        def capture_post(url, **kwargs):
            nonlocal captured_payload, captured_headers
            captured_payload = kwargs.get("json", {})
            captured_headers = kwargs.get("headers", {})
            mock_post_ctx = MagicMock()
            mock_post_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
            mock_post_ctx.__aexit__ = AsyncMock(return_value=None)
            return mock_post_ctx

        mock_session = MagicMock()
        mock_session.post = capture_post
        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_ctx.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session_ctx):
            await client.get_response(messages)

        # Check system message
        assert captured_payload["system"] == "You are helpful"

        # Check messages (should exclude system)
        assert len(captured_payload["messages"]) == 3
        assert captured_payload["messages"][0]["role"] == "user"
        assert captured_payload["messages"][1]["role"] == "assistant"
        assert captured_payload["messages"][2]["role"] == "user"

        # Check headers
        assert captured_headers["x-api-key"] == "test-token"
        assert "anthropic-version" in captured_headers

    @pytest.mark.asyncio
    async def test_anthropic_client_empty_content_skipped(self):
        """Test AnthropicClient skips empty content messages."""
        client = AnthropicClient("test-token", "claude-3-sonnet")
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": ""},  # Empty, should be skipped
            {"role": "user", "content": "Question"},
        ]

        mock_resp = self._create_mock_response(
            json_data={
                "content": [{"type": "text", "text": "Answer"}]
            }
        )

        captured_payload = None

        def capture_post(url, **kwargs):
            nonlocal captured_payload
            captured_payload = kwargs.get("json", {})
            mock_post_ctx = MagicMock()
            mock_post_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
            mock_post_ctx.__aexit__ = AsyncMock(return_value=None)
            return mock_post_ctx

        mock_session = MagicMock()
        mock_session.post = capture_post
        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_ctx.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session_ctx):
            await client.get_response(messages)

        # Empty assistant message should be skipped
        assert len(captured_payload["messages"]) == 2

    @pytest.mark.asyncio
    async def test_anthropic_client_no_text_block(self):
        """Test AnthropicClient handles response without text block."""
        client = AnthropicClient("test-token", "claude-3-sonnet")
        messages = [{"role": "user", "content": "Hello"}]

        mock_resp = self._create_mock_response(
            json_data={
                "content": [{"type": "image", "data": "..."}]  # No text block
            }
        )

        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=MagicMock(
            post=MagicMock(return_value=MagicMock(
                __aenter__=AsyncMock(return_value=mock_resp),
                __aexit__=AsyncMock(return_value=None)
            ))
        ))
        mock_session_ctx.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session_ctx):
            result = await client.get_response(messages)

        # Should return string representation
        assert "content" in result


class TestOpenRouterClient:
    """Test OpenRouter client functionality."""

    def test_openrouter_client_initialization(self):
        """Test OpenRouterClient initialization."""
        client = OpenRouterClient("test-token", "openai/gpt-4o")
        assert client.token == "test-token"
        assert client.model == "openai/gpt-4o"


class TestLlamaClient:
    """Test Llama client functionality."""

    def test_llama_client_initialization(self):
        """Test LlamaClient initialization."""
        client = LlamaClient("test-token", "Llama-4-Maverick-17B-128E-Instruct-FP8")
        assert client.token == "test-token"
        assert client.model == "Llama-4-Maverick-17B-128E-Instruct-FP8"
        assert client.api_url == "https://api.llama.com/v1/chat/completions"

    def test_llama_client_default_model(self):
        """Test LlamaClient uses default model."""
        client = LlamaClient("test-token")
        assert client.model == "Llama-4-Maverick-17B-128E-Instruct-FP8"

    def _create_mock_response(self, status=200, json_data=None, text_data=None):
        """Helper to create a mock aiohttp response."""
        mock_resp = MagicMock()
        mock_resp.status = status
        mock_resp.headers = {"Content-Type": "application/json"}

        if json_data is not None:
            mock_resp.json = AsyncMock(return_value=json_data)
            mock_resp.text = AsyncMock(return_value=json.dumps(json_data))
        elif text_data is not None:
            mock_resp.text = AsyncMock(return_value=text_data)
        else:
            mock_resp.text = AsyncMock(return_value="")

        return mock_resp

    @pytest.mark.asyncio
    async def test_llama_client_get_response_success(self):
        """Test LlamaClient successful response."""
        client = LlamaClient("test-token", "llama-model")
        messages = [{"role": "user", "content": "Hello"}]

        mock_resp = self._create_mock_response(
            json_data={
                "completion_message": {
                    "content": {
                        "text": "Hello! How can I help?"
                    }
                }
            }
        )

        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=MagicMock(
            post=MagicMock(return_value=MagicMock(
                __aenter__=AsyncMock(return_value=mock_resp),
                __aexit__=AsyncMock(return_value=None)
            ))
        ))
        mock_session_ctx.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session_ctx):
            result = await client.get_response(messages)

        assert result == "Hello! How can I help?"

    @pytest.mark.asyncio
    async def test_llama_client_get_response_fallback(self):
        """Test LlamaClient fallback when expected structure missing."""
        client = LlamaClient("test-token", "llama-model")
        messages = [{"role": "user", "content": "Hello"}]

        # Response missing expected structure
        mock_resp = self._create_mock_response(
            json_data={"some_other_field": "value"}
        )

        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=MagicMock(
            post=MagicMock(return_value=MagicMock(
                __aenter__=AsyncMock(return_value=mock_resp),
                __aexit__=AsyncMock(return_value=None)
            ))
        ))
        mock_session_ctx.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session_ctx):
            result = await client.get_response(messages)

        # Should return string representation of data
        assert "some_other_field" in result

    @pytest.mark.asyncio
    async def test_llama_client_get_response_http_error(self):
        """Test LlamaClient handles HTTP error."""
        client = LlamaClient("test-token", "llama-model")
        messages = [{"role": "user", "content": "Hello"}]

        mock_resp = self._create_mock_response(status=500, text_data="Internal error")

        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=MagicMock(
            post=MagicMock(return_value=MagicMock(
                __aenter__=AsyncMock(return_value=mock_resp),
                __aexit__=AsyncMock(return_value=None)
            ))
        ))
        mock_session_ctx.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session_ctx):
            with pytest.raises(Exception) as exc_info:
                await client.get_response(messages)

        assert "500" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_llama_client_request_payload(self):
        """Test LlamaClient sends correct payload."""
        client = LlamaClient("test-token", "llama-model")
        messages = [{"role": "user", "content": "Hello"}]

        mock_resp = self._create_mock_response(
            json_data={"completion_message": {"content": {"text": "Hi"}}}
        )

        captured_payload = None
        captured_headers = None

        def capture_post(url, **kwargs):
            nonlocal captured_payload, captured_headers
            captured_payload = kwargs.get("json", {})
            captured_headers = kwargs.get("headers", {})
            mock_post_ctx = MagicMock()
            mock_post_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
            mock_post_ctx.__aexit__ = AsyncMock(return_value=None)
            return mock_post_ctx

        mock_session = MagicMock()
        mock_session.post = capture_post
        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_ctx.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session_ctx):
            await client.get_response(messages)

        assert captured_payload["model"] == "llama-model"
        assert captured_payload["messages"] == messages
        assert captured_payload["temperature"] == 0.7
        assert captured_payload["top_p"] == 0.9
        assert "Bearer test-token" in captured_headers["Authorization"]


class TestOpenAIClientGetResponse:
    """Test OpenAI client get_response functionality."""

    def _create_mock_response(self, status=200, json_data=None, text_data=None):
        """Helper to create a mock aiohttp response."""
        mock_resp = MagicMock()
        mock_resp.status = status
        mock_resp.headers = {"Content-Type": "application/json"}

        if json_data is not None:
            mock_resp.text = AsyncMock(return_value=json.dumps(json_data))
        elif text_data is not None:
            mock_resp.text = AsyncMock(return_value=text_data)
        else:
            mock_resp.text = AsyncMock(return_value="")

        return mock_resp

    @pytest.mark.asyncio
    async def test_openai_client_get_response_success(self):
        """Test OpenAIClient successful response."""
        client = OpenAIClient("sk-test-token", "gpt-4")
        messages = [{"role": "user", "content": "Hello"}]

        mock_resp = self._create_mock_response(
            json_data={
                "choices": [{"message": {"content": "Hello! How can I help?"}}]
            }
        )

        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=MagicMock(
            post=MagicMock(return_value=MagicMock(
                __aenter__=AsyncMock(return_value=mock_resp),
                __aexit__=AsyncMock(return_value=None)
            ))
        ))
        mock_session_ctx.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session_ctx):
            result = await client.get_response(messages)

        assert result == "Hello! How can I help?"

    @pytest.mark.asyncio
    async def test_openai_client_get_response_http_error(self):
        """Test OpenAIClient handles HTTP error."""
        client = OpenAIClient("sk-test-token", "gpt-4")
        messages = [{"role": "user", "content": "Hello"}]

        mock_resp = self._create_mock_response(status=401, text_data="Unauthorized")

        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=MagicMock(
            post=MagicMock(return_value=MagicMock(
                __aenter__=AsyncMock(return_value=mock_resp),
                __aexit__=AsyncMock(return_value=None)
            ))
        ))
        mock_session_ctx.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session_ctx):
            with pytest.raises(Exception) as exc_info:
                await client.get_response(messages)

        assert "401" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_openai_client_get_response_invalid_json(self):
        """Test OpenAIClient handles invalid JSON response."""
        client = OpenAIClient("sk-test-token", "gpt-4")
        messages = [{"role": "user", "content": "Hello"}]

        mock_resp = self._create_mock_response(text_data="not valid json {")

        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=MagicMock(
            post=MagicMock(return_value=MagicMock(
                __aenter__=AsyncMock(return_value=mock_resp),
                __aexit__=AsyncMock(return_value=None)
            ))
        ))
        mock_session_ctx.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session_ctx):
            with pytest.raises(Exception) as exc_info:
                await client.get_response(messages)

        assert "Invalid JSON" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_openai_client_restricted_model_no_temperature(self):
        """Test OpenAIClient doesn't send temperature for restricted models."""
        client = OpenAIClient("sk-test-token", "o3-mini")
        messages = [{"role": "user", "content": "Hello"}]

        mock_resp = self._create_mock_response(
            json_data={"choices": [{"message": {"content": "Hi"}}]}
        )

        captured_payload = None

        def capture_post(url, **kwargs):
            nonlocal captured_payload
            captured_payload = kwargs.get("json", {})
            mock_post_ctx = MagicMock()
            mock_post_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
            mock_post_ctx.__aexit__ = AsyncMock(return_value=None)
            return mock_post_ctx

        mock_session = MagicMock()
        mock_session.post = capture_post
        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_ctx.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session_ctx):
            await client.get_response(messages)

        assert "temperature" not in captured_payload
        assert "top_p" not in captured_payload

    @pytest.mark.asyncio
    async def test_openai_client_normal_model_has_temperature(self):
        """Test OpenAIClient sends temperature for normal models."""
        client = OpenAIClient("sk-test-token", "gpt-4")
        messages = [{"role": "user", "content": "Hello"}]

        mock_resp = self._create_mock_response(
            json_data={"choices": [{"message": {"content": "Hi"}}]}
        )

        captured_payload = None

        def capture_post(url, **kwargs):
            nonlocal captured_payload
            captured_payload = kwargs.get("json", {})
            mock_post_ctx = MagicMock()
            mock_post_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
            mock_post_ctx.__aexit__ = AsyncMock(return_value=None)
            return mock_post_ctx

        mock_session = MagicMock()
        mock_session.post = capture_post
        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_ctx.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session_ctx):
            await client.get_response(messages)

        assert captured_payload["temperature"] == 0.7
        assert captured_payload["top_p"] == 0.9

    @pytest.mark.asyncio
    async def test_openai_client_empty_content(self):
        """Test OpenAIClient handles empty content."""
        client = OpenAIClient("sk-test-token", "gpt-4")
        messages = [{"role": "user", "content": "Hello"}]

        mock_resp = self._create_mock_response(
            json_data={"choices": [{"message": {"content": ""}}]}
        )

        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=MagicMock(
            post=MagicMock(return_value=MagicMock(
                __aenter__=AsyncMock(return_value=mock_resp),
                __aexit__=AsyncMock(return_value=None)
            ))
        ))
        mock_session_ctx.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session_ctx):
            result = await client.get_response(messages)

        assert result == ""

    @pytest.mark.asyncio
    async def test_openai_client_missing_message_structure(self):
        """Test OpenAIClient handles missing message structure."""
        client = OpenAIClient("sk-test-token", "gpt-4")
        messages = [{"role": "user", "content": "Hello"}]

        mock_resp = self._create_mock_response(
            json_data={"choices": [{"text": "some text"}]}  # Missing message key
        )

        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=MagicMock(
            post=MagicMock(return_value=MagicMock(
                __aenter__=AsyncMock(return_value=mock_resp),
                __aexit__=AsyncMock(return_value=None)
            ))
        ))
        mock_session_ctx.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session_ctx):
            result = await client.get_response(messages)

        # Should return string representation
        assert "choices" in result


class TestOpenRouterClientGetResponse:
    """Test OpenRouter client get_response functionality."""

    def _create_mock_response(self, status=200, json_data=None, text_data=None):
        """Helper to create a mock aiohttp response."""
        mock_resp = MagicMock()
        mock_resp.status = status
        mock_resp.headers = {"Content-Type": "application/json"}

        if json_data is not None:
            mock_resp.json = AsyncMock(return_value=json_data)
            mock_resp.text = AsyncMock(return_value=json.dumps(json_data))
        elif text_data is not None:
            mock_resp.text = AsyncMock(return_value=text_data)
        else:
            mock_resp.text = AsyncMock(return_value="")

        return mock_resp

    def test_openrouter_default_model(self):
        """Test OpenRouterClient uses default model."""
        client = OpenRouterClient("test-token")
        assert client.model == "openai/gpt-4o"
        assert client.api_url == "https://openrouter.ai/api/v1/chat/completions"

    @pytest.mark.asyncio
    async def test_openrouter_client_get_response_success(self):
        """Test OpenRouterClient successful response."""
        client = OpenRouterClient("test-token", "anthropic/claude-3")
        messages = [{"role": "user", "content": "Hello"}]

        mock_resp = self._create_mock_response(
            json_data={
                "choices": [{"message": {"content": "Hello from OpenRouter!"}}]
            }
        )

        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=MagicMock(
            post=MagicMock(return_value=MagicMock(
                __aenter__=AsyncMock(return_value=mock_resp),
                __aexit__=AsyncMock(return_value=None)
            ))
        ))
        mock_session_ctx.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session_ctx):
            result = await client.get_response(messages)

        assert result == "Hello from OpenRouter!"

    @pytest.mark.asyncio
    async def test_openrouter_client_get_response_http_error(self):
        """Test OpenRouterClient handles HTTP error."""
        client = OpenRouterClient("test-token", "anthropic/claude-3")
        messages = [{"role": "user", "content": "Hello"}]

        mock_resp = self._create_mock_response(status=429, text_data="Rate limited")

        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=MagicMock(
            post=MagicMock(return_value=MagicMock(
                __aenter__=AsyncMock(return_value=mock_resp),
                __aexit__=AsyncMock(return_value=None)
            ))
        ))
        mock_session_ctx.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session_ctx):
            with pytest.raises(Exception) as exc_info:
                await client.get_response(messages)

        assert "429" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_openrouter_client_missing_choices(self):
        """Test OpenRouterClient handles response without choices."""
        client = OpenRouterClient("test-token", "anthropic/claude-3")
        messages = [{"role": "user", "content": "Hello"}]

        mock_resp = self._create_mock_response(
            json_data={"error": "something went wrong"}
        )

        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=MagicMock(
            post=MagicMock(return_value=MagicMock(
                __aenter__=AsyncMock(return_value=mock_resp),
                __aexit__=AsyncMock(return_value=None)
            ))
        ))
        mock_session_ctx.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session_ctx):
            result = await client.get_response(messages)

        # Should return string representation
        assert "error" in result

    @pytest.mark.asyncio
    async def test_openrouter_client_request_headers(self):
        """Test OpenRouterClient sends correct headers."""
        client = OpenRouterClient("test-token", "openai/gpt-4o")
        messages = [{"role": "user", "content": "Hello"}]

        mock_resp = self._create_mock_response(
            json_data={"choices": [{"message": {"content": "Hi"}}]}
        )

        captured_headers = None

        def capture_post(url, **kwargs):
            nonlocal captured_headers
            captured_headers = kwargs.get("headers", {})
            mock_post_ctx = MagicMock()
            mock_post_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
            mock_post_ctx.__aexit__ = AsyncMock(return_value=None)
            return mock_post_ctx

        mock_session = MagicMock()
        mock_session.post = capture_post
        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_ctx.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session_ctx):
            await client.get_response(messages)

        assert "Bearer test-token" in captured_headers["Authorization"]
        assert "HTTP-Referer" in captured_headers
        assert "X-Title" in captured_headers


class TestAlterClientGetResponse:
    """Test Alter client get_response functionality."""

    def _create_mock_response(self, status=200, json_data=None, text_data=None):
        """Helper to create a mock aiohttp response."""
        mock_resp = MagicMock()
        mock_resp.status = status
        mock_resp.headers = {"Content-Type": "application/json"}

        if json_data is not None:
            mock_resp.json = AsyncMock(return_value=json_data)
            mock_resp.text = AsyncMock(return_value=json.dumps(json_data))
        elif text_data is not None:
            mock_resp.text = AsyncMock(return_value=text_data)
        else:
            mock_resp.text = AsyncMock(return_value="")

        return mock_resp

    def test_alter_client_initialization(self):
        """Test AlterClient initialization."""
        client = AlterClient("test-token", "alter-model")
        assert client.token == "test-token"
        assert client.model == "alter-model"
        assert client.api_url == "https://alterhq.com/api/v1/chat/completions"

    def test_alter_client_default_model(self):
        """Test AlterClient uses empty default model."""
        client = AlterClient("test-token")
        assert client.model == ""

    @pytest.mark.asyncio
    async def test_alter_client_get_response_success(self):
        """Test AlterClient successful response."""
        client = AlterClient("test-token", "alter-model")
        messages = [{"role": "user", "content": "Hello"}]

        mock_resp = self._create_mock_response(
            json_data={
                "choices": [{"message": {"content": "Hello from Alter!"}}]
            }
        )

        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=MagicMock(
            post=MagicMock(return_value=MagicMock(
                __aenter__=AsyncMock(return_value=mock_resp),
                __aexit__=AsyncMock(return_value=None)
            ))
        ))
        mock_session_ctx.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session_ctx):
            result = await client.get_response(messages)

        assert result == "Hello from Alter!"

    @pytest.mark.asyncio
    async def test_alter_client_get_response_http_error(self):
        """Test AlterClient handles HTTP error."""
        client = AlterClient("test-token", "alter-model")
        messages = [{"role": "user", "content": "Hello"}]

        mock_resp = self._create_mock_response(status=500, text_data="Server error")

        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=MagicMock(
            post=MagicMock(return_value=MagicMock(
                __aenter__=AsyncMock(return_value=mock_resp),
                __aexit__=AsyncMock(return_value=None)
            ))
        ))
        mock_session_ctx.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session_ctx):
            with pytest.raises(Exception) as exc_info:
                await client.get_response(messages)

        assert "500" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_alter_client_missing_choices(self):
        """Test AlterClient handles response without choices."""
        client = AlterClient("test-token", "alter-model")
        messages = [{"role": "user", "content": "Hello"}]

        mock_resp = self._create_mock_response(
            json_data={"data": "unexpected format"}
        )

        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=MagicMock(
            post=MagicMock(return_value=MagicMock(
                __aenter__=AsyncMock(return_value=mock_resp),
                __aexit__=AsyncMock(return_value=None)
            ))
        ))
        mock_session_ctx.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session_ctx):
            result = await client.get_response(messages)

        assert "data" in result


class TestZaiClientGetResponse:
    """Test Zai client get_response functionality."""

    def _create_mock_response(self, status=200, json_data=None, text_data=None):
        """Helper to create a mock aiohttp response."""
        mock_resp = MagicMock()
        mock_resp.status = status
        mock_resp.headers = {"Content-Type": "application/json"}

        if json_data is not None:
            mock_resp.json = AsyncMock(return_value=json_data)
            mock_resp.text = AsyncMock(return_value=json.dumps(json_data))
        elif text_data is not None:
            mock_resp.text = AsyncMock(return_value=text_data)
        else:
            mock_resp.text = AsyncMock(return_value="")

        return mock_resp

    def test_zai_client_initialization_general(self):
        """Test ZaiClient initialization with general endpoint."""
        client = ZaiClient("test-token", "glm-4.7", "general")
        assert client.token == "test-token"
        assert client.model == "glm-4.7"
        assert client.endpoint_type == "general"
        assert "paas/v4" in client.api_url
        assert "coding" not in client.api_url

    def test_zai_client_initialization_coding(self):
        """Test ZaiClient initialization with coding endpoint."""
        client = ZaiClient("test-token", "glm-4.7", "coding")
        assert client.endpoint_type == "coding"
        assert "coding/paas/v4" in client.api_url

    @pytest.mark.asyncio
    async def test_zai_client_get_response_success(self):
        """Test ZaiClient successful response."""
        client = ZaiClient("test-token", "glm-4.7", "general")
        messages = [{"role": "user", "content": "Hello"}]

        mock_resp = self._create_mock_response(
            json_data={
                "choices": [{"message": {"content": "Hello from z.ai!"}}]
            }
        )

        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=MagicMock(
            post=MagicMock(return_value=MagicMock(
                __aenter__=AsyncMock(return_value=mock_resp),
                __aexit__=AsyncMock(return_value=None)
            ))
        ))
        mock_session_ctx.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session_ctx):
            result = await client.get_response(messages)

        assert result == "Hello from z.ai!"

    @pytest.mark.asyncio
    async def test_zai_client_get_response_http_error(self):
        """Test ZaiClient handles HTTP error."""
        client = ZaiClient("test-token", "glm-4.7", "general")
        messages = [{"role": "user", "content": "Hello"}]

        mock_resp = self._create_mock_response(status=403, text_data="Forbidden")

        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=MagicMock(
            post=MagicMock(return_value=MagicMock(
                __aenter__=AsyncMock(return_value=mock_resp),
                __aexit__=AsyncMock(return_value=None)
            ))
        ))
        mock_session_ctx.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session_ctx):
            with pytest.raises(Exception) as exc_info:
                await client.get_response(messages)

        assert "403" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_zai_client_missing_choices(self):
        """Test ZaiClient handles response without choices."""
        client = ZaiClient("test-token", "glm-4.7", "general")
        messages = [{"role": "user", "content": "Hello"}]

        mock_resp = self._create_mock_response(
            json_data={"result": "no choices here"}
        )

        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=MagicMock(
            post=MagicMock(return_value=MagicMock(
                __aenter__=AsyncMock(return_value=mock_resp),
                __aexit__=AsyncMock(return_value=None)
            ))
        ))
        mock_session_ctx.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session_ctx):
            result = await client.get_response(messages)

        assert "result" in result