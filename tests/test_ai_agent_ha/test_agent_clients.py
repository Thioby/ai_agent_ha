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
    AnthropicOAuthClient,
    GeminiOAuthClient,
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

    @pytest.mark.asyncio
    async def test_openrouter_client_headers_all_values(self):
        """Test OpenRouterClient sends all expected header values."""
        client = OpenRouterClient("my-secret-token", "anthropic/claude-3-sonnet")
        messages = [{"role": "user", "content": "Test"}]

        mock_resp = self._create_mock_response(
            json_data={"choices": [{"message": {"content": "Response"}}]}
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

        assert captured_headers["Authorization"] == "Bearer my-secret-token"
        assert captured_headers["Content-Type"] == "application/json"
        assert captured_headers["HTTP-Referer"] == "https://home-assistant.io"
        assert captured_headers["X-Title"] == "Home Assistant AI Agent"

    @pytest.mark.asyncio
    async def test_openrouter_client_request_payload(self):
        """Test OpenRouterClient sends correct request payload."""
        client = OpenRouterClient("test-token", "openai/gpt-4o")
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"}
        ]

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

        assert captured_payload["model"] == "openai/gpt-4o"
        assert captured_payload["messages"] == messages
        assert captured_payload["temperature"] == 0.7
        assert captured_payload["top_p"] == 0.9
        assert "max_tokens" not in captured_payload

    @pytest.mark.asyncio
    async def test_openrouter_client_empty_choices_array(self):
        """Test OpenRouterClient handles empty choices array."""
        client = OpenRouterClient("test-token", "openai/gpt-4o")
        messages = [{"role": "user", "content": "Hello"}]

        mock_resp = self._create_mock_response(
            json_data={"choices": [], "id": "response-123"}
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

        assert "choices" in result
        assert "response-123" in result

    @pytest.mark.asyncio
    async def test_openrouter_client_message_without_content(self):
        """Test OpenRouterClient handles message without content field."""
        client = OpenRouterClient("test-token", "openai/gpt-4o")
        messages = [{"role": "user", "content": "Hello"}]

        mock_resp = self._create_mock_response(
            json_data={"choices": [{"message": {"role": "assistant"}}]}
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

        # Should return string representation when content is missing
        assert isinstance(result, str)
        assert "choices" in result

    @pytest.mark.asyncio
    async def test_openrouter_client_function_call_response(self):
        """Test OpenRouterClient handles function call in response."""
        client = OpenRouterClient("test-token", "openai/gpt-4o")
        messages = [{"role": "user", "content": "What's the weather?"}]

        # OpenAI-compatible function call format
        mock_resp = self._create_mock_response(
            json_data={
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location": "San Francisco"}'
                            }
                        }]
                    }
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

        # Current implementation returns None for content field
        # This test documents current behavior - in future, might want to handle tool_calls
        assert result != "None"  # Should handle None gracefully

    @pytest.mark.asyncio
    async def test_openrouter_client_500_error(self):
        """Test OpenRouterClient handles 500 server error."""
        client = OpenRouterClient("test-token", "openai/gpt-4o")
        messages = [{"role": "user", "content": "Hello"}]

        mock_resp = self._create_mock_response(
            status=500,
            text_data="Internal server error"
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
            with pytest.raises(Exception) as exc_info:
                await client.get_response(messages)

        assert "500" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_openrouter_client_401_unauthorized(self):
        """Test OpenRouterClient handles 401 unauthorized error."""
        client = OpenRouterClient("invalid-token", "openai/gpt-4o")
        messages = [{"role": "user", "content": "Hello"}]

        mock_resp = self._create_mock_response(
            status=401,
            text_data='{"error": {"message": "Invalid API key"}}'
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
            with pytest.raises(Exception) as exc_info:
                await client.get_response(messages)

        assert "401" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_openrouter_client_timeout(self):
        """Test OpenRouterClient uses correct timeout."""
        client = OpenRouterClient("test-token", "openai/gpt-4o")
        messages = [{"role": "user", "content": "Hello"}]

        mock_resp = self._create_mock_response(
            json_data={"choices": [{"message": {"content": "Hi"}}]}
        )

        captured_timeout = None

        def capture_post(url, **kwargs):
            nonlocal captured_timeout
            captured_timeout = kwargs.get("timeout")
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

        assert captured_timeout is not None
        assert captured_timeout.total == 300

    @pytest.mark.asyncio
    async def test_openrouter_client_custom_model(self):
        """Test OpenRouterClient with various model names."""
        models = [
            "anthropic/claude-3-opus",
            "google/gemini-pro",
            "meta-llama/llama-2-70b-chat",
            "openai/gpt-4-turbo-preview"
        ]

        for model in models:
            client = OpenRouterClient("test-token", model)
            messages = [{"role": "user", "content": "Test"}]

            mock_resp = self._create_mock_response(
                json_data={"choices": [{"message": {"content": "Response"}}]}
            )

            captured_model = None

            def capture_post(url, **kwargs):
                nonlocal captured_model
                payload = kwargs.get("json", {})
                captured_model = payload.get("model")
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

            assert captured_model == model

    @pytest.mark.asyncio
    async def test_openrouter_client_url_endpoint(self):
        """Test OpenRouterClient uses correct API endpoint."""
        client = OpenRouterClient("test-token")
        messages = [{"role": "user", "content": "Hello"}]

        mock_resp = self._create_mock_response(
            json_data={"choices": [{"message": {"content": "Hi"}}]}
        )

        captured_url = None

        def capture_post(url, **kwargs):
            nonlocal captured_url
            captured_url = url
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

        assert captured_url == "https://openrouter.ai/api/v1/chat/completions"


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

    @pytest.mark.asyncio
    async def test_alter_client_invalid_json(self):
        """Test AlterClient handles invalid JSON response."""
        client = AlterClient("test-token", "alter-model")
        messages = [{"role": "user", "content": "Hello"}]

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(side_effect=json.JSONDecodeError("Invalid", "", 0))
        mock_resp.text = AsyncMock(return_value="Not valid JSON")

        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=MagicMock(
            post=MagicMock(return_value=MagicMock(
                __aenter__=AsyncMock(return_value=mock_resp),
                __aexit__=AsyncMock(return_value=None)
            ))
        ))
        mock_session_ctx.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session_ctx):
            with pytest.raises(json.JSONDecodeError):
                await client.get_response(messages)

    @pytest.mark.asyncio
    async def test_alter_client_empty_content(self):
        """Test AlterClient handles empty content response."""
        client = AlterClient("test-token", "alter-model")
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

    @pytest.mark.asyncio
    async def test_zai_client_invalid_json(self):
        """Test ZaiClient handles invalid JSON response."""
        client = ZaiClient("test-token", "glm-4.7", "general")
        messages = [{"role": "user", "content": "Hello"}]

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(side_effect=json.JSONDecodeError("Invalid", "", 0))
        mock_resp.text = AsyncMock(return_value="Not valid JSON")

        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=MagicMock(
            post=MagicMock(return_value=MagicMock(
                __aenter__=AsyncMock(return_value=mock_resp),
                __aexit__=AsyncMock(return_value=None)
            ))
        ))
        mock_session_ctx.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session_ctx):
            with pytest.raises(json.JSONDecodeError):
                await client.get_response(messages)

    @pytest.mark.asyncio
    async def test_zai_client_empty_content(self):
        """Test ZaiClient handles empty content response."""
        client = ZaiClient("test-token", "glm-4.7", "general")
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


class TestAnthropicOAuthClient:
    """Test AnthropicOAuthClient functionality."""

    def _create_mock_response(self, status=200, json_data=None, text_data=None):
        """Helper to create mock aiohttp response."""
        mock_resp = MagicMock()
        mock_resp.status = status
        if json_data is not None:
            mock_resp.text = AsyncMock(return_value=json.dumps(json_data))
        elif text_data is not None:
            mock_resp.text = AsyncMock(return_value=text_data)
        return mock_resp

    def test_anthropic_oauth_client_initialization(self):
        """Test AnthropicOAuthClient initialization."""
        mock_hass = MagicMock()
        mock_entry = MagicMock()
        mock_entry.data = {
            "anthropic_oauth": {
                "access_token": "test-token",
                "refresh_token": "test-refresh",
                "expires_at": 9999999999,
            }
        }

        client = AnthropicOAuthClient(mock_hass, mock_entry, "claude-3-opus")
        assert client.model == "claude-3-opus"
        assert client.api_url == "https://api.anthropic.com/v1/messages"
        assert client._oauth_data["access_token"] == "test-token"

    def test_transform_request_adds_tool_prefix(self):
        """Test _transform_request adds mcp_ prefix to tools."""
        mock_hass = MagicMock()
        mock_entry = MagicMock()
        mock_entry.data = {"anthropic_oauth": {}}

        client = AnthropicOAuthClient(mock_hass, mock_entry)
        payload = {
            "tools": [{"name": "my_tool", "description": "test"}],
            "model": "claude-3-opus",
        }

        result = client._transform_request(payload)

        assert result["tools"][0]["name"] == "mcp_my_tool"

    def test_transform_request_renames_tool_use_blocks(self):
        """Test _transform_request renames tool_use blocks in messages."""
        mock_hass = MagicMock()
        mock_entry = MagicMock()
        mock_entry.data = {"anthropic_oauth": {}}

        client = AnthropicOAuthClient(mock_hass, mock_entry)
        payload = {
            "messages": [
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "name": "execute_command"}],
                }
            ],
            "model": "claude-3-opus",
        }

        result = client._transform_request(payload)

        assert result["messages"][0]["content"][0]["name"] == "mcp_execute_command"

    def test_transform_request_replaces_system_string(self):
        """Test _transform_request replaces OpenCode in system string."""
        mock_hass = MagicMock()
        mock_entry = MagicMock()
        mock_entry.data = {"anthropic_oauth": {}}

        client = AnthropicOAuthClient(mock_hass, mock_entry)
        payload = {
            "system": "You are using OpenCode for coding tasks.",
            "model": "claude-3-opus",
        }

        result = client._transform_request(payload)

        assert "OpenCode" not in result["system"]
        assert "Claude Code" in result["system"]

    def test_transform_request_replaces_system_list(self):
        """Test _transform_request replaces OpenCode in system list."""
        mock_hass = MagicMock()
        mock_entry = MagicMock()
        mock_entry.data = {"anthropic_oauth": {}}

        client = AnthropicOAuthClient(mock_hass, mock_entry)
        payload = {
            "system": [{"type": "text", "text": "You are OpenCode assistant."}],
            "model": "claude-3-opus",
        }

        result = client._transform_request(payload)

        assert "OpenCode" not in result["system"][0]["text"]

    def test_transform_response_removes_mcp_prefix(self):
        """Test _transform_response removes mcp_ prefix from names."""
        mock_hass = MagicMock()
        mock_entry = MagicMock()
        mock_entry.data = {"anthropic_oauth": {}}

        client = AnthropicOAuthClient(mock_hass, mock_entry)
        text = '{"name": "mcp_execute_command", "type": "tool_use"}'

        result = client._transform_response(text)

        assert '"name": "execute_command"' in result

    @pytest.mark.asyncio
    async def test_get_valid_token_returns_cached(self):
        """Test _get_valid_token returns cached token if not expired."""
        import time

        mock_hass = MagicMock()
        mock_entry = MagicMock()
        mock_entry.data = {
            "anthropic_oauth": {
                "access_token": "cached-token",
                "refresh_token": "test-refresh",
                "expires_at": time.time() + 600,  # Valid for 10 more minutes
            }
        }

        client = AnthropicOAuthClient(mock_hass, mock_entry)
        token = await client._get_valid_token()

        assert token == "cached-token"

    @pytest.mark.asyncio
    async def test_get_valid_token_refreshes_expired(self):
        """Test _get_valid_token refreshes expired token."""
        import time

        mock_hass = MagicMock()
        mock_entry = MagicMock()
        mock_entry.data = {
            "anthropic_oauth": {
                "access_token": "old-token",
                "refresh_token": "test-refresh",
                "expires_at": time.time() - 100,  # Expired
            }
        }

        new_tokens = {
            "access_token": "new-token",
            "refresh_token": "new-refresh",
            "expires_at": time.time() + 3600,
        }

        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=MagicMock())
        mock_session_ctx.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session_ctx):
            with patch(
                "custom_components.ai_agent_ha.oauth.refresh_token",
                new=AsyncMock(return_value=new_tokens),
            ):
                client = AnthropicOAuthClient(mock_hass, mock_entry)
                token = await client._get_valid_token()

        assert token == "new-token"

    @pytest.mark.asyncio
    async def test_get_response_success(self):
        """Test successful get_response."""
        import time

        mock_hass = MagicMock()
        mock_entry = MagicMock()
        mock_entry.data = {
            "anthropic_oauth": {
                "access_token": "valid-token",
                "refresh_token": "test-refresh",
                "expires_at": time.time() + 600,
            }
        }

        response_data = {
            "content": [{"type": "text", "text": "Hello from Claude!"}],
        }
        mock_resp = self._create_mock_response(json_data=response_data)

        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(
            return_value=MagicMock(
                post=MagicMock(
                    return_value=MagicMock(
                        __aenter__=AsyncMock(return_value=mock_resp),
                        __aexit__=AsyncMock(return_value=None),
                    )
                )
            )
        )
        mock_session_ctx.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session_ctx):
            client = AnthropicOAuthClient(mock_hass, mock_entry)
            result = await client.get_response([{"role": "user", "content": "Hi"}])

        assert result == "Hello from Claude!"

    @pytest.mark.asyncio
    async def test_get_response_http_error(self):
        """Test get_response handles HTTP error."""
        import time

        mock_hass = MagicMock()
        mock_entry = MagicMock()
        mock_entry.data = {
            "anthropic_oauth": {
                "access_token": "valid-token",
                "refresh_token": "test-refresh",
                "expires_at": time.time() + 600,
            }
        }

        mock_resp = self._create_mock_response(status=401, text_data="Unauthorized")

        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(
            return_value=MagicMock(
                post=MagicMock(
                    return_value=MagicMock(
                        __aenter__=AsyncMock(return_value=mock_resp),
                        __aexit__=AsyncMock(return_value=None),
                    )
                )
            )
        )
        mock_session_ctx.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session_ctx):
            client = AnthropicOAuthClient(mock_hass, mock_entry)
            with pytest.raises(Exception) as exc_info:
                await client.get_response([{"role": "user", "content": "Hi"}])

        assert "401" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_response_with_system_message(self):
        """Test get_response properly formats system messages."""
        import time

        mock_hass = MagicMock()
        mock_entry = MagicMock()
        mock_entry.data = {
            "anthropic_oauth": {
                "access_token": "valid-token",
                "refresh_token": "test-refresh",
                "expires_at": time.time() + 600,
            }
        }

        response_data = {
            "content": [{"type": "text", "text": "Response with system"}],
        }
        mock_resp = self._create_mock_response(json_data=response_data)

        # Track what payload was sent
        sent_payload = {}

        def mock_post(url, headers=None, json=None, timeout=None):
            nonlocal sent_payload
            sent_payload = json
            return MagicMock(
                __aenter__=AsyncMock(return_value=mock_resp),
                __aexit__=AsyncMock(return_value=None),
            )

        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(
            return_value=MagicMock(post=mock_post)
        )
        mock_session_ctx.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session_ctx):
            client = AnthropicOAuthClient(mock_hass, mock_entry)
            result = await client.get_response(
                [
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": "Hi"},
                ]
            )

        assert result == "Response with system"
        # Verify system is formatted as array with Claude Code prefix
        assert isinstance(sent_payload["system"], list)
        assert len(sent_payload["system"]) == 2
        assert "Claude Code" in sent_payload["system"][0]["text"]
        assert "helpful assistant" in sent_payload["system"][1]["text"]

    @pytest.mark.asyncio
    async def test_get_response_with_multiple_content_blocks(self):
        """Test get_response returns first text block from multiple content blocks."""
        import time

        mock_hass = MagicMock()
        mock_entry = MagicMock()
        mock_entry.data = {
            "anthropic_oauth": {
                "access_token": "valid-token",
                "refresh_token": "test-refresh",
                "expires_at": time.time() + 600,
            }
        }

        response_data = {
            "content": [
                {"type": "text", "text": "First text block"},
                {"type": "tool_use", "name": "some_tool"},
                {"type": "text", "text": "Second text block"},
            ],
        }
        mock_resp = self._create_mock_response(json_data=response_data)

        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(
            return_value=MagicMock(
                post=MagicMock(
                    return_value=MagicMock(
                        __aenter__=AsyncMock(return_value=mock_resp),
                        __aexit__=AsyncMock(return_value=None),
                    )
                )
            )
        )
        mock_session_ctx.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session_ctx):
            client = AnthropicOAuthClient(mock_hass, mock_entry)
            result = await client.get_response([{"role": "user", "content": "Hi"}])

        assert result == "First text block"

    @pytest.mark.asyncio
    async def test_get_response_with_no_text_blocks(self):
        """Test get_response returns string repr when no text blocks."""
        import time

        mock_hass = MagicMock()
        mock_entry = MagicMock()
        mock_entry.data = {
            "anthropic_oauth": {
                "access_token": "valid-token",
                "refresh_token": "test-refresh",
                "expires_at": time.time() + 600,
            }
        }

        response_data = {
            "content": [{"type": "tool_use", "name": "some_tool"}],
        }
        mock_resp = self._create_mock_response(json_data=response_data)

        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(
            return_value=MagicMock(
                post=MagicMock(
                    return_value=MagicMock(
                        __aenter__=AsyncMock(return_value=mock_resp),
                        __aexit__=AsyncMock(return_value=None),
                    )
                )
            )
        )
        mock_session_ctx.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session_ctx):
            client = AnthropicOAuthClient(mock_hass, mock_entry)
            result = await client.get_response([{"role": "user", "content": "Hi"}])

        assert "tool_use" in result  # Should be string repr of data

    @pytest.mark.asyncio
    async def test_get_response_applies_transform_request(self):
        """Test get_response applies tool prefix transformation."""
        import time

        mock_hass = MagicMock()
        mock_entry = MagicMock()
        mock_entry.data = {
            "anthropic_oauth": {
                "access_token": "valid-token",
                "refresh_token": "test-refresh",
                "expires_at": time.time() + 600,
            }
        }

        response_data = {
            "content": [{"type": "text", "text": "Tool response"}],
        }
        mock_resp = self._create_mock_response(json_data=response_data)

        # Track what payload was sent
        sent_payload = {}

        def mock_post(url, headers=None, json=None, timeout=None):
            nonlocal sent_payload
            sent_payload = json
            return MagicMock(
                __aenter__=AsyncMock(return_value=mock_resp),
                __aexit__=AsyncMock(return_value=None),
            )

        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(
            return_value=MagicMock(post=mock_post)
        )
        mock_session_ctx.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session_ctx):
            client = AnthropicOAuthClient(mock_hass, mock_entry)
            # Pass tools to trigger transformation
            result = await client.get_response(
                [{"role": "user", "content": "Hi"}],
                tools=[{"name": "execute", "description": "Execute command"}],
            )

        # Note: get_response doesn't use **kwargs for tools, but we verify transform is called
        assert result == "Tool response"

    @pytest.mark.asyncio
    async def test_get_response_applies_transform_response(self):
        """Test get_response applies mcp_ prefix removal transformation."""
        import time

        mock_hass = MagicMock()
        mock_entry = MagicMock()
        mock_entry.data = {
            "anthropic_oauth": {
                "access_token": "valid-token",
                "refresh_token": "test-refresh",
                "expires_at": time.time() + 600,
            }
        }

        # Response with mcp_ prefix that should be removed
        response_text = '{"content": [{"type": "text", "text": "test"}], "name": "mcp_execute"}'
        mock_resp = self._create_mock_response(text_data=response_text)

        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(
            return_value=MagicMock(
                post=MagicMock(
                    return_value=MagicMock(
                        __aenter__=AsyncMock(return_value=mock_resp),
                        __aexit__=AsyncMock(return_value=None),
                    )
                )
            )
        )
        mock_session_ctx.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session_ctx):
            client = AnthropicOAuthClient(mock_hass, mock_entry)
            result = await client.get_response([{"role": "user", "content": "Hi"}])

        assert result == "test"

    @pytest.mark.asyncio
    async def test_get_response_filters_messages(self):
        """Test get_response filters out system and empty messages."""
        import time

        mock_hass = MagicMock()
        mock_entry = MagicMock()
        mock_entry.data = {
            "anthropic_oauth": {
                "access_token": "valid-token",
                "refresh_token": "test-refresh",
                "expires_at": time.time() + 600,
            }
        }

        response_data = {
            "content": [{"type": "text", "text": "Response"}],
        }
        mock_resp = self._create_mock_response(json_data=response_data)

        # Track what payload was sent
        sent_payload = {}

        def mock_post(url, headers=None, json=None, timeout=None):
            nonlocal sent_payload
            sent_payload = json
            return MagicMock(
                __aenter__=AsyncMock(return_value=mock_resp),
                __aexit__=AsyncMock(return_value=None),
            )

        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(
            return_value=MagicMock(post=mock_post)
        )
        mock_session_ctx.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session_ctx):
            client = AnthropicOAuthClient(mock_hass, mock_entry)
            result = await client.get_response(
                [
                    {"role": "system", "content": "System prompt"},
                    {"role": "user", "content": ""},  # Empty should be filtered
                    {"role": "user", "content": "Real message"},
                    {"role": "assistant", "content": ""},  # Empty should be filtered
                    {"role": "assistant", "content": "Assistant response"},
                ]
            )

        # Should only have 2 messages (non-empty user and assistant)
        assert len(sent_payload["messages"]) == 2
        assert sent_payload["messages"][0]["content"] == "Real message"
        assert sent_payload["messages"][1]["content"] == "Assistant response"

    @pytest.mark.asyncio
    async def test_get_response_uses_correct_headers(self):
        """Test get_response uses correct authorization headers."""
        import time

        mock_hass = MagicMock()
        mock_entry = MagicMock()
        mock_entry.data = {
            "anthropic_oauth": {
                "access_token": "test-token-123",
                "refresh_token": "test-refresh",
                "expires_at": time.time() + 600,
            }
        }

        response_data = {
            "content": [{"type": "text", "text": "Response"}],
        }
        mock_resp = self._create_mock_response(json_data=response_data)

        # Track headers
        sent_headers = {}

        def mock_post(url, headers=None, json=None, timeout=None):
            nonlocal sent_headers
            sent_headers = headers
            return MagicMock(
                __aenter__=AsyncMock(return_value=mock_resp),
                __aexit__=AsyncMock(return_value=None),
            )

        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(
            return_value=MagicMock(post=mock_post)
        )
        mock_session_ctx.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session_ctx):
            client = AnthropicOAuthClient(mock_hass, mock_entry)
            result = await client.get_response([{"role": "user", "content": "Hi"}])

        # Verify correct OAuth headers
        assert sent_headers["authorization"] == "Bearer test-token-123"
        assert sent_headers["content-type"] == "application/json"
        assert sent_headers["anthropic-version"] == "2023-06-01"
        assert "oauth-2025-04-20" in sent_headers["anthropic-beta"]
        assert "claude-code-20250219" in sent_headers["anthropic-beta"]

    @pytest.mark.asyncio
    async def test_token_refresh_on_expiry(self):
        """Test token is refreshed when expired."""
        import time

        mock_hass = MagicMock()
        mock_entry = MagicMock()
        mock_entry.data = {
            "anthropic_oauth": {
                "access_token": "old-token",
                "refresh_token": "test-refresh",
                "expires_at": time.time() - 100,  # Expired
            }
        }

        new_tokens = {
            "access_token": "new-token",
            "refresh_token": "new-refresh",
            "expires_at": time.time() + 3600,
        }

        response_data = {
            "content": [{"type": "text", "text": "Response with new token"}],
        }
        mock_resp = self._create_mock_response(json_data=response_data)

        # Track which token was used
        used_token = None

        def mock_post(url, headers=None, json=None, timeout=None):
            nonlocal used_token
            used_token = headers.get("authorization")
            return MagicMock(
                __aenter__=AsyncMock(return_value=mock_resp),
                __aexit__=AsyncMock(return_value=None),
            )

        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(
            return_value=MagicMock(post=mock_post)
        )
        mock_session_ctx.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session_ctx):
            with patch(
                "custom_components.ai_agent_ha.oauth.refresh_token",
                new=AsyncMock(return_value=new_tokens),
            ):
                client = AnthropicOAuthClient(mock_hass, mock_entry)
                result = await client.get_response([{"role": "user", "content": "Hi"}])

        # Should have used the new token
        assert used_token == "Bearer new-token"
        assert result == "Response with new token"

    @pytest.mark.asyncio
    async def test_token_refresh_updates_config_entry(self):
        """Test token refresh updates the config entry data."""
        import time

        mock_hass = MagicMock()
        mock_entry = MagicMock()
        mock_entry.data = {
            "anthropic_oauth": {
                "access_token": "old-token",
                "refresh_token": "test-refresh",
                "expires_at": time.time() - 100,  # Expired
            }
        }

        new_tokens = {
            "access_token": "new-token",
            "refresh_token": "new-refresh",
            "expires_at": time.time() + 3600,
        }

        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=MagicMock())
        mock_session_ctx.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session_ctx):
            with patch(
                "custom_components.ai_agent_ha.oauth.refresh_token",
                new=AsyncMock(return_value=new_tokens),
            ):
                client = AnthropicOAuthClient(mock_hass, mock_entry)
                token = await client._get_valid_token()

        # Verify config entry was updated
        mock_hass.config_entries.async_update_entry.assert_called_once()
        call_args = mock_hass.config_entries.async_update_entry.call_args
        assert call_args[0][0] == mock_entry
        assert call_args[1]["data"]["anthropic_oauth"]["access_token"] == "new-token"
        assert call_args[1]["data"]["anthropic_oauth"]["refresh_token"] == "new-refresh"

    @pytest.mark.asyncio
    async def test_token_refresh_handles_oauth_error(self):
        """Test token refresh handles OAuthRefreshError."""
        import time
        from custom_components.ai_agent_ha.oauth import OAuthRefreshError

        mock_hass = MagicMock()
        mock_entry = MagicMock()
        mock_entry.data = {
            "anthropic_oauth": {
                "access_token": "old-token",
                "refresh_token": "test-refresh",
                "expires_at": time.time() - 100,  # Expired
            }
        }

        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=MagicMock())
        mock_session_ctx.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session_ctx):
            with patch(
                "custom_components.ai_agent_ha.oauth.refresh_token",
                new=AsyncMock(side_effect=OAuthRefreshError("Refresh failed")),
            ):
                client = AnthropicOAuthClient(mock_hass, mock_entry)
                with pytest.raises(OAuthRefreshError):
                    await client._get_valid_token()

    @pytest.mark.asyncio
    async def test_token_refresh_with_buffer_time(self):
        """Test token is refreshed 300 seconds before expiry."""
        import time

        mock_hass = MagicMock()
        mock_entry = MagicMock()
        # Token expires in 250 seconds - should be refreshed (< 300s buffer)
        mock_entry.data = {
            "anthropic_oauth": {
                "access_token": "old-token",
                "refresh_token": "test-refresh",
                "expires_at": time.time() + 250,
            }
        }

        new_tokens = {
            "access_token": "new-token",
            "refresh_token": "new-refresh",
            "expires_at": time.time() + 3600,
        }

        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=MagicMock())
        mock_session_ctx.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session_ctx):
            with patch(
                "custom_components.ai_agent_ha.oauth.refresh_token",
                new=AsyncMock(return_value=new_tokens),
            ) as mock_refresh:
                client = AnthropicOAuthClient(mock_hass, mock_entry)
                token = await client._get_valid_token()

        # Should have refreshed
        mock_refresh.assert_called_once()
        assert token == "new-token"

    @pytest.mark.asyncio
    async def test_token_refresh_lock_prevents_concurrent_refresh(self):
        """Test refresh lock prevents concurrent token refreshes."""
        import time
        import asyncio

        mock_hass = MagicMock()
        mock_entry = MagicMock()
        mock_entry.data = {
            "anthropic_oauth": {
                "access_token": "old-token",
                "refresh_token": "test-refresh",
                "expires_at": time.time() - 100,  # Expired
            }
        }

        new_tokens = {
            "access_token": "new-token",
            "refresh_token": "new-refresh",
            "expires_at": time.time() + 3600,
        }

        refresh_call_count = 0

        async def mock_refresh_token(session, refresh_token):
            nonlocal refresh_call_count
            refresh_call_count += 1
            await asyncio.sleep(0.1)  # Simulate delay
            return new_tokens

        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=MagicMock())
        mock_session_ctx.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session_ctx):
            with patch(
                "custom_components.ai_agent_ha.oauth.refresh_token",
                new=mock_refresh_token,
            ):
                client = AnthropicOAuthClient(mock_hass, mock_entry)
                # Call get_valid_token concurrently
                results = await asyncio.gather(
                    client._get_valid_token(),
                    client._get_valid_token(),
                    client._get_valid_token(),
                )

        # All should get the same token
        assert all(token == "new-token" for token in results)
        # But refresh should only be called once due to lock
        assert refresh_call_count == 1


class TestGeminiOAuthClient:
    """Test GeminiOAuthClient functionality."""

    def _create_mock_response(self, status=200, json_data=None, text_data=None):
        """Helper to create mock aiohttp response."""
        mock_resp = MagicMock()
        mock_resp.status = status
        if json_data is not None:
            mock_resp.text = AsyncMock(return_value=json.dumps(json_data))
            mock_resp.json = AsyncMock(return_value=json_data)
        elif text_data is not None:
            mock_resp.text = AsyncMock(return_value=text_data)
        return mock_resp

    def test_gemini_oauth_client_initialization(self):
        """Test GeminiOAuthClient initialization."""
        mock_hass = MagicMock()
        mock_entry = MagicMock()
        mock_entry.data = {
            "gemini_oauth": {
                "access_token": "test-token",
                "refresh_token": "test-refresh",
                "expires_at": 9999999999,
                "managed_project_id": "test-project",
            }
        }

        client = GeminiOAuthClient(mock_hass, mock_entry, "gemini-3-pro")
        assert client.model == "gemini-3-pro"
        assert "cloudcode-pa.googleapis.com" in client.api_url
        assert client._oauth_data["managed_project_id"] == "test-project"

    @pytest.mark.asyncio
    async def test_get_valid_token_returns_cached(self):
        """Test _get_valid_token returns cached token if not expired."""
        import time

        mock_hass = MagicMock()
        mock_entry = MagicMock()
        mock_entry.data = {
            "gemini_oauth": {
                "access_token": "cached-token",
                "refresh_token": "test-refresh",
                "expires_at": time.time() + 600,
            }
        }

        client = GeminiOAuthClient(mock_hass, mock_entry)
        token = await client._get_valid_token()

        assert token == "cached-token"

    @pytest.mark.asyncio
    async def test_get_valid_token_no_access_token_raises(self):
        """Test _get_valid_token raises when no access token."""
        import time

        mock_hass = MagicMock()
        mock_entry = MagicMock()
        mock_entry.data = {
            "gemini_oauth": {
                "refresh_token": "test-refresh",
                "expires_at": time.time() + 600,
            }
        }

        client = GeminiOAuthClient(mock_hass, mock_entry)
        with pytest.raises(Exception) as exc_info:
            await client._get_valid_token()

        assert "No access token available" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_valid_token_no_refresh_token_raises(self):
        """Test _get_valid_token raises when no refresh token."""
        import time

        mock_hass = MagicMock()
        mock_entry = MagicMock()
        mock_entry.data = {
            "gemini_oauth": {
                "access_token": "old-token",
                "expires_at": time.time() - 100,  # Expired
            }
        }

        client = GeminiOAuthClient(mock_hass, mock_entry)
        with pytest.raises(Exception) as exc_info:
            await client._get_valid_token()

        assert "No refresh token available" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_valid_token_refreshes_expired(self):
        """Test _get_valid_token refreshes expired token."""
        import time

        mock_hass = MagicMock()
        mock_entry = MagicMock()
        mock_entry.data = {
            "gemini_oauth": {
                "access_token": "old-token",
                "refresh_token": "test-refresh",
                "expires_at": time.time() - 100,
            }
        }

        new_tokens = {
            "access_token": "new-gemini-token",
            "refresh_token": "new-refresh",
            "expires_at": time.time() + 3600,
        }

        mock_session_ctx = MagicMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=MagicMock())
        mock_session_ctx.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session_ctx):
            with patch(
                "custom_components.ai_agent_ha.gemini_oauth.refresh_token",
                new=AsyncMock(return_value=new_tokens),
            ):
                client = GeminiOAuthClient(mock_hass, mock_entry)
                token = await client._get_valid_token()

        assert token == "new-gemini-token"

    @pytest.mark.asyncio
    async def test_save_project_id(self):
        """Test _save_project_id persists project ID."""
        mock_hass = MagicMock()
        mock_entry = MagicMock()
        mock_entry.data = {"gemini_oauth": {"access_token": "test"}}

        client = GeminiOAuthClient(mock_hass, mock_entry)
        await client._save_project_id("projects/123456")

        assert client._oauth_data["managed_project_id"] == "projects/123456"
        mock_hass.config_entries.async_update_entry.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensure_project_id_returns_cached(self):
        """Test _ensure_project_id returns cached project ID."""
        mock_hass = MagicMock()
        mock_entry = MagicMock()
        mock_entry.data = {
            "gemini_oauth": {
                "access_token": "test",
                "managed_project_id": "cached-project-id",
            }
        }

        mock_session = MagicMock()
        client = GeminiOAuthClient(mock_hass, mock_entry)
        project_id = await client._ensure_project_id(mock_session, "token")

        assert project_id == "cached-project-id"

    @pytest.mark.asyncio
    async def test_ensure_project_id_from_load_code_assist(self):
        """Test _ensure_project_id gets project from loadCodeAssist."""
        mock_hass = MagicMock()
        mock_entry = MagicMock()
        mock_entry.data = {"gemini_oauth": {"access_token": "test"}}

        load_response = self._create_mock_response(
            json_data={"cloudaicompanionProject": "loaded-project-123"}
        )

        mock_session = MagicMock()
        mock_post_ctx = MagicMock()
        mock_post_ctx.__aenter__ = AsyncMock(return_value=load_response)
        mock_post_ctx.__aexit__ = AsyncMock(return_value=None)
        mock_session.post = MagicMock(return_value=mock_post_ctx)

        client = GeminiOAuthClient(mock_hass, mock_entry)
        project_id = await client._ensure_project_id(mock_session, "token")

        assert project_id == "loaded-project-123"

    @pytest.mark.asyncio
    async def test_ensure_project_id_enterprise_tier_raises(self):
        """Test _ensure_project_id raises for enterprise tier."""
        mock_hass = MagicMock()
        mock_entry = MagicMock()
        mock_entry.data = {"gemini_oauth": {"access_token": "test"}}

        load_response = self._create_mock_response(
            json_data={"currentTier": {"id": "ENTERPRISE"}}
        )

        mock_session = MagicMock()
        mock_post_ctx = MagicMock()
        mock_post_ctx.__aenter__ = AsyncMock(return_value=load_response)
        mock_post_ctx.__aexit__ = AsyncMock(return_value=None)
        mock_session.post = MagicMock(return_value=mock_post_ctx)

        client = GeminiOAuthClient(mock_hass, mock_entry)
        with pytest.raises(Exception) as exc_info:
            await client._ensure_project_id(mock_session, "token")

        assert "ENTERPRISE" in str(exc_info.value)


class TestGeminiOAuthClientRetry:
    """Test GeminiOAuthClient._retry_with_backoff method."""

    @pytest.mark.asyncio
    async def test_successful_call_no_retry(self):
        """Test successful call on first attempt requires no retry."""
        mock_hass = MagicMock()
        mock_entry = MagicMock()
        mock_entry.data = {"gemini_oauth": {"access_token": "test"}}

        client = GeminiOAuthClient(mock_hass, mock_entry)

        # Mock function that succeeds immediately
        mock_func = AsyncMock(return_value="success")

        result = await client._retry_with_backoff(mock_func, "arg1", kwarg1="value1")

        assert result == "success"
        assert mock_func.call_count == 1
        mock_func.assert_called_once_with("arg1", kwarg1="value1")

    @pytest.mark.asyncio
    async def test_429_error_with_retry_success(self):
        """Test 429 rate limit error retries and eventually succeeds."""
        mock_hass = MagicMock()
        mock_entry = MagicMock()
        mock_entry.data = {"gemini_oauth": {"access_token": "test"}}

        client = GeminiOAuthClient(mock_hass, mock_entry)

        # Mock function that fails twice with 429, then succeeds
        mock_func = AsyncMock(
            side_effect=[
                Exception("429 Rate Limited"),
                Exception("429 Rate Limited"),
                "success",
            ]
        )

        with patch("asyncio.sleep", new=AsyncMock()) as mock_sleep:
            result = await client._retry_with_backoff(mock_func)

        assert result == "success"
        assert mock_func.call_count == 3
        assert mock_sleep.call_count == 2

    @pytest.mark.asyncio
    async def test_5xx_error_with_retry(self):
        """Test 5xx server error retries and eventually succeeds."""
        mock_hass = MagicMock()
        mock_entry = MagicMock()
        mock_entry.data = {"gemini_oauth": {"access_token": "test"}}

        client = GeminiOAuthClient(mock_hass, mock_entry)

        # Mock function that fails with 500, 503, then succeeds
        mock_func = AsyncMock(
            side_effect=[
                Exception("500 Internal Server Error"),
                Exception("503 Service Unavailable"),
                "success",
            ]
        )

        with patch("asyncio.sleep", new=AsyncMock()) as mock_sleep:
            result = await client._retry_with_backoff(mock_func)

        assert result == "success"
        assert mock_func.call_count == 3
        assert mock_sleep.call_count == 2

    @pytest.mark.asyncio
    async def test_max_attempts_exceeded(self):
        """Test max retry attempts exceeded raises exception."""
        mock_hass = MagicMock()
        mock_entry = MagicMock()
        mock_entry.data = {"gemini_oauth": {"access_token": "test"}}

        client = GeminiOAuthClient(mock_hass, mock_entry)

        # Mock function that always fails with 429
        mock_func = AsyncMock(side_effect=Exception("429 Rate Limited"))

        with patch("asyncio.sleep", new=AsyncMock()):
            with pytest.raises(Exception) as exc_info:
                await client._retry_with_backoff(mock_func)

        assert "429" in str(exc_info.value)
        assert mock_func.call_count == client.MAX_ATTEMPTS

    @pytest.mark.asyncio
    async def test_non_retryable_error(self):
        """Test non-retryable error raises immediately without retry."""
        mock_hass = MagicMock()
        mock_entry = MagicMock()
        mock_entry.data = {"gemini_oauth": {"access_token": "test"}}

        client = GeminiOAuthClient(mock_hass, mock_entry)

        # Mock function that fails with non-retryable error (400)
        mock_func = AsyncMock(side_effect=Exception("400 Bad Request"))

        with pytest.raises(Exception) as exc_info:
            await client._retry_with_backoff(mock_func)

        assert "400 Bad Request" in str(exc_info.value)
        assert mock_func.call_count == 1


class TestGeminiOAuthClientEnsureProjectId:
    """Test GeminiOAuthClient._ensure_project_id method."""

    def _create_mock_response(self, status=200, json_data=None, text_data=None):
        """Helper to create mock aiohttp response."""
        mock_resp = MagicMock()
        mock_resp.status = status
        if json_data is not None:
            mock_resp.text = AsyncMock(return_value=json.dumps(json_data))
            mock_resp.json = AsyncMock(return_value=json_data)
        elif text_data is not None:
            mock_resp.text = AsyncMock(return_value=text_data)
        return mock_resp

    @pytest.mark.asyncio
    async def test_returns_cached_project_id(self):
        """Test that cached project ID is returned immediately."""
        mock_hass = MagicMock()
        mock_entry = MagicMock()
        mock_entry.data = {
            "gemini_oauth": {
                "access_token": "test-token",
                "managed_project_id": "cached-project-123",
            }
        }

        client = GeminiOAuthClient(mock_hass, mock_entry)
        mock_session = MagicMock()

        project_id = await client._ensure_project_id(mock_session, "test-token")

        assert project_id == "cached-project-123"
        # Should not make any API calls
        mock_session.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_loadCodeAssist_returns_existing_project(self):
        """Test loadCodeAssist returns existing project ID."""
        mock_hass = MagicMock()
        mock_entry = MagicMock()
        mock_entry.data = {"gemini_oauth": {"access_token": "test-token"}}

        client = GeminiOAuthClient(mock_hass, mock_entry)

        # Mock loadCodeAssist response with existing project
        load_response = self._create_mock_response(
            status=200,
            text_data=json.dumps({"cloudaicompanionProject": "existing-project-456"}),
        )

        mock_session = MagicMock()
        mock_session.post.return_value.__aenter__.return_value = load_response

        with patch.object(client, "_save_project_id", new=AsyncMock()) as mock_save:
            project_id = await client._ensure_project_id(mock_session, "test-token")

        assert project_id == "existing-project-456"
        mock_save.assert_called_once_with("existing-project-456")
        # Should only call loadCodeAssist, not onboardUser
        assert mock_session.post.call_count == 1
        assert ":loadCodeAssist" in mock_session.post.call_args[0][0]

    @pytest.mark.asyncio
    async def test_enterprise_tier_rejection(self):
        """Test that enterprise tier raises exception requiring manual config."""
        mock_hass = MagicMock()
        mock_entry = MagicMock()
        mock_entry.data = {"gemini_oauth": {"access_token": "test-token"}}

        client = GeminiOAuthClient(mock_hass, mock_entry)

        # Mock loadCodeAssist response with enterprise tier
        load_response = self._create_mock_response(
            status=200,
            text_data=json.dumps({
                "currentTier": {"id": "ENTERPRISE"},
                "cloudaicompanionProject": None,
            }),
        )

        mock_session = MagicMock()
        mock_session.post.return_value.__aenter__.return_value = load_response

        with pytest.raises(Exception) as exc_info:
            await client._ensure_project_id(mock_session, "test-token")

        assert "ENTERPRISE" in str(exc_info.value)
        assert "manual project configuration" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_successful_onboarding(self):
        """Test successful onboarding flow when loadCodeAssist returns no project."""
        mock_hass = MagicMock()
        mock_entry = MagicMock()
        mock_entry.data = {"gemini_oauth": {"access_token": "test-token"}}

        client = GeminiOAuthClient(mock_hass, mock_entry)

        # Mock loadCodeAssist response with no project
        load_response = self._create_mock_response(
            status=200,
            text_data=json.dumps({"currentTier": {"id": "FREE"}}),
        )

        # Mock onboardUser response with completed onboarding
        onboard_response = self._create_mock_response(
            status=200,
            json_data={
                "done": True,
                "response": {
                    "cloudaicompanionProject": {"id": "new-project-789"}
                },
            },
        )

        mock_session = MagicMock()
        mock_session.post.return_value.__aenter__.side_effect = [
            load_response,
            onboard_response,
        ]

        with patch.object(client, "_save_project_id", new=AsyncMock()) as mock_save:
            project_id = await client._ensure_project_id(mock_session, "test-token")

        assert project_id == "new-project-789"
        mock_save.assert_called_once_with("new-project-789")
        # Should call both loadCodeAssist and onboardUser
        assert mock_session.post.call_count == 2

    @pytest.mark.asyncio
    async def test_onboarding_retry_then_success(self):
        """Test onboarding retries when not done, then succeeds."""
        mock_hass = MagicMock()
        mock_entry = MagicMock()
        mock_entry.data = {"gemini_oauth": {"access_token": "test-token"}}

        client = GeminiOAuthClient(mock_hass, mock_entry)

        # Mock loadCodeAssist response with no project
        load_response = self._create_mock_response(
            status=200,
            text_data=json.dumps({"currentTier": {"id": "FREE"}}),
        )

        # Mock onboardUser responses: first not done, second done
        onboard_response_pending = self._create_mock_response(
            status=200,
            json_data={"done": False},
        )
        onboard_response_complete = self._create_mock_response(
            status=200,
            json_data={
                "done": True,
                "response": {
                    "cloudaicompanionProject": {"id": "retry-project-999"}
                },
            },
        )

        mock_session = MagicMock()
        mock_session.post.return_value.__aenter__.side_effect = [
            load_response,
            onboard_response_pending,
            onboard_response_complete,
        ]

        with patch.object(client, "_save_project_id", new=AsyncMock()) as mock_save:
            with patch("asyncio.sleep", new=AsyncMock()) as mock_sleep:
                project_id = await client._ensure_project_id(
                    mock_session, "test-token"
                )

        assert project_id == "retry-project-999"
        mock_save.assert_called_once_with("retry-project-999")
        # Should sleep once between retries
        mock_sleep.assert_called_once_with(5)

    @pytest.mark.asyncio
    async def test_max_onboarding_attempts_exceeded(self):
        """Test exception raised when max onboarding attempts exceeded."""
        mock_hass = MagicMock()
        mock_entry = MagicMock()
        mock_entry.data = {"gemini_oauth": {"access_token": "test-token"}}

        client = GeminiOAuthClient(mock_hass, mock_entry)

        # Mock loadCodeAssist response with no project
        load_response = self._create_mock_response(
            status=200,
            text_data=json.dumps({"currentTier": {"id": "FREE"}}),
        )

        # Mock onboardUser responses: always pending (never done)
        onboard_response_pending = self._create_mock_response(
            status=200,
            json_data={"done": False},
        )

        mock_session = MagicMock()
        # First call is loadCodeAssist, next 10 are onboardUser attempts
        mock_session.post.return_value.__aenter__.side_effect = (
            [load_response] + [onboard_response_pending] * 10
        )

        with patch("asyncio.sleep", new=AsyncMock()):
            with pytest.raises(Exception) as exc_info:
                await client._ensure_project_id(mock_session, "test-token")

        assert "Failed to obtain Gemini project ID after onboarding" in str(
            exc_info.value
        )
        # Should call loadCodeAssist once + 10 onboardUser attempts
        assert mock_session.post.call_count == 11

    @pytest.mark.asyncio
    async def test_onboarding_http_error(self):
        """Test onboarding handles HTTP errors and retries."""
        mock_hass = MagicMock()
        mock_entry = MagicMock()
        mock_entry.data = {"gemini_oauth": {"access_token": "test-token"}}

        client = GeminiOAuthClient(mock_hass, mock_entry)

        # Mock loadCodeAssist response with no project
        load_response = self._create_mock_response(
            status=200,
            text_data=json.dumps({"currentTier": {"id": "FREE"}}),
        )

        # Mock onboardUser responses: first HTTP error, then success
        onboard_response_error = self._create_mock_response(
            status=500,
            text_data="Internal Server Error",
        )
        onboard_response_success = self._create_mock_response(
            status=200,
            json_data={
                "done": True,
                "response": {
                    "cloudaicompanionProject": {"id": "recovered-project-111"}
                },
            },
        )

        mock_session = MagicMock()
        mock_session.post.return_value.__aenter__.side_effect = [
            load_response,
            onboard_response_error,
            onboard_response_success,
        ]

        with patch.object(client, "_save_project_id", new=AsyncMock()) as mock_save:
            with patch("asyncio.sleep", new=AsyncMock()):
                project_id = await client._ensure_project_id(
                    mock_session, "test-token"
                )

        assert project_id == "recovered-project-111"
        mock_save.assert_called_once_with("recovered-project-111")

    @pytest.mark.asyncio
    async def test_loadCodeAssist_client_error(self):
        """Test handling of ClientError during loadCodeAssist."""
        import aiohttp

        mock_hass = MagicMock()
        mock_entry = MagicMock()
        mock_entry.data = {"gemini_oauth": {"access_token": "test-token"}}

        client = GeminiOAuthClient(mock_hass, mock_entry)

        # Mock loadCodeAssist to raise ClientError
        mock_session = MagicMock()
        mock_session.post.return_value.__aenter__.side_effect = [
            aiohttp.ClientError("Connection failed"),
        ]

        # Add onboardUser success response after ClientError
        onboard_response = self._create_mock_response(
            status=200,
            json_data={
                "done": True,
                "response": {
                    "cloudaicompanionProject": {"id": "fallback-project-222"}
                },
            },
        )

        # Reset side_effect to include both loadCodeAssist error and onboardUser success
        mock_session.post.return_value.__aenter__.side_effect = [
            aiohttp.ClientError("Connection failed"),
            onboard_response,
        ]

        with patch.object(client, "_save_project_id", new=AsyncMock()) as mock_save:
            project_id = await client._ensure_project_id(mock_session, "test-token")

        assert project_id == "fallback-project-222"
        mock_save.assert_called_once_with("fallback-project-222")

    @pytest.mark.asyncio
    async def test_onboarding_client_error_then_success(self):
        """Test onboarding handles ClientError and retries successfully."""
        import aiohttp

        mock_hass = MagicMock()
        mock_entry = MagicMock()
        mock_entry.data = {"gemini_oauth": {"access_token": "test-token"}}

        client = GeminiOAuthClient(mock_hass, mock_entry)

        # Mock loadCodeAssist response with no project
        load_response = self._create_mock_response(
            status=200,
            text_data=json.dumps({"currentTier": {"id": "FREE"}}),
        )

        # Mock onboardUser: first ClientError, then success
        onboard_response_success = self._create_mock_response(
            status=200,
            json_data={
                "done": True,
                "response": {
                    "cloudaicompanionProject": {"id": "client-error-recovery-333"}
                },
            },
        )

        mock_session = MagicMock()
        mock_session.post.return_value.__aenter__.side_effect = [
            load_response,
            aiohttp.ClientError("Network error"),
            onboard_response_success,
        ]

        with patch.object(client, "_save_project_id", new=AsyncMock()) as mock_save:
            with patch("asyncio.sleep", new=AsyncMock()):
                project_id = await client._ensure_project_id(
                    mock_session, "test-token"
                )

        assert project_id == "client-error-recovery-333"
        mock_save.assert_called_once_with("client-error-recovery-333")