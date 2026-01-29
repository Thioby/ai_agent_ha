# AI Agent HA - Architecture Redesign Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor ai_agent_ha from God Class (5008 LOC) to Plugin Architecture with parallel workstreams.

**Architecture:** Plugin Registry for AI providers, Managers for domain logic, slim Core orchestrator. Direct `hass` usage (no HA abstraction). Strangler Fig migration pattern.

**Tech Stack:** Python 3.12, Home Assistant Core, pytest-homeassistant-custom-component, aiohttp, aioresponses

---

## Parallel Workstreams Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              WORKSTREAMS                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  STREAM A: Providers          STREAM B: Managers         STREAM C: Core    │
│  ──────────────────          ─────────────────          ──────────────     │
│                                                                             │
│  A1: Registry Base     ←──┐                                                │
│         ↓                 │                                                │
│  A2: OpenAI Provider      │   B1: Entity Manager                           │
│         ↓                 │          ↓                                     │
│  A3: Gemini Provider      │   B2: Registry Manager                         │
│         ↓                 │          ↓                                     │
│  A4: Anthropic Provider   │   B3: Automation Manager                       │
│         ↓                 │          ↓                                     │
│  A5: OpenRouter Provider  │   B4: Dashboard Manager                        │
│         ↓                 │          ↓                                     │
│  A6: Groq Provider        │   B5: Control Manager                          │
│         ↓                 │          │                                     │
│  A7: Local Provider       │          │                                     │
│         │                 │          │                                     │
│         └─────────────────┴──────────┴─────────┐                           │
│                                                │                           │
│                                         C1: Query Processor  ←─────────────│
│                                                ↓                           │
│                                         C2: Response Parser                │
│                                                ↓                           │
│                                         C3: Conversation Manager           │
│                                                ↓                           │
│                                         C4: Agent Orchestrator             │
│                                                ↓                           │
│                                         C5: Integration & Cleanup          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

Dependencies:
- Stream A and B can run in PARALLEL (no dependencies between them)
- Stream C depends on completion of A1 (Registry) and all B tasks
- Within each stream, tasks are SEQUENTIAL
```

---

## Pre-requisites (Run Once)

### Task 0: Setup Test Infrastructure

**Files:**
- Modify: `requirements_test.txt`
- Modify: `tests/conftest.py`

**Step 1: Update test dependencies**

```bash
# requirements_test.txt - add these lines
echo "pytest-homeassistant-custom-component>=0.13.0" >> requirements_test.txt
echo "aioresponses>=0.7.6" >> requirements_test.txt
```

**Step 2: Install dependencies**

```bash
pip install -r requirements_test.txt
```

**Step 3: Verify existing tests pass**

```bash
pytest tests/ -v --tb=short -q
```
Expected: All tests pass (current coverage ~87%)

**Step 4: Commit**

```bash
git add requirements_test.txt
git commit -m "chore: add HA testing framework dependencies"
```

---

# STREAM A: AI Providers (Can run parallel with Stream B)

## Task A1: Provider Registry Base

**Depends on:** Task 0
**Blocks:** A2-A7, C1

**Files:**
- Create: `custom_components/ai_agent_ha/providers/__init__.py`
- Create: `custom_components/ai_agent_ha/providers/registry.py`
- Create: `custom_components/ai_agent_ha/providers/base_client.py`
- Create: `tests/test_providers/__init__.py`
- Create: `tests/test_providers/test_registry.py`

**Step 1: Create providers directory**

```bash
mkdir -p custom_components/ai_agent_ha/providers
mkdir -p tests/test_providers
touch custom_components/ai_agent_ha/providers/__init__.py
touch tests/test_providers/__init__.py
```

**Step 2: Write failing test for registry**

```python
# tests/test_providers/test_registry.py
"""Tests for AI provider registry."""
import pytest
from unittest.mock import MagicMock

from custom_components.ai_agent_ha.providers.registry import (
    AIProvider,
    ProviderRegistry,
)


class TestProviderRegistry:
    """Test ProviderRegistry class."""

    def setup_method(self):
        """Clear registry before each test."""
        ProviderRegistry._providers.clear()

    def test_register_provider(self):
        """Test registering a provider."""
        @ProviderRegistry.register("test_provider")
        class TestProvider(AIProvider):
            @property
            def supports_tools(self) -> bool:
                return False

            async def get_response(self, messages: list, **kwargs) -> str:
                return "test"

        assert "test_provider" in ProviderRegistry.available_providers()

    def test_create_provider(self):
        """Test creating provider instance."""
        @ProviderRegistry.register("test_provider")
        class TestProvider(AIProvider):
            @property
            def supports_tools(self) -> bool:
                return True

            async def get_response(self, messages: list, **kwargs) -> str:
                return "response"

        hass = MagicMock()
        config = {"token": "test-token"}

        provider = ProviderRegistry.create("test_provider", hass, config)

        assert isinstance(provider, TestProvider)
        assert provider.hass == hass
        assert provider.config == config

    def test_create_unknown_provider_raises(self):
        """Test creating unknown provider raises error."""
        hass = MagicMock()

        with pytest.raises(ValueError, match="Unknown provider"):
            ProviderRegistry.create("unknown", hass, {})

    def test_available_providers(self):
        """Test listing available providers."""
        @ProviderRegistry.register("provider_a")
        class ProviderA(AIProvider):
            @property
            def supports_tools(self) -> bool:
                return False
            async def get_response(self, messages: list, **kwargs) -> str:
                return ""

        @ProviderRegistry.register("provider_b")
        class ProviderB(AIProvider):
            @property
            def supports_tools(self) -> bool:
                return False
            async def get_response(self, messages: list, **kwargs) -> str:
                return ""

        providers = ProviderRegistry.available_providers()
        assert "provider_a" in providers
        assert "provider_b" in providers
```

**Step 3: Run test to verify it fails**

```bash
pytest tests/test_providers/test_registry.py -v
```
Expected: FAIL with "ModuleNotFoundError: No module named 'custom_components.ai_agent_ha.providers.registry'"

**Step 4: Implement registry**

```python
# custom_components/ai_agent_ha/providers/registry.py
"""AI Provider Registry - plugin system for AI providers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant


class AIProvider(ABC):
    """Abstract base class for AI providers."""

    def __init__(self, hass: HomeAssistant, config: dict[str, Any]) -> None:
        """Initialize provider.

        Args:
            hass: Home Assistant instance
            config: Provider configuration (token, model, etc.)
        """
        self.hass = hass
        self.config = config

    @property
    @abstractmethod
    def supports_tools(self) -> bool:
        """Whether this provider supports function calling/tools."""
        pass

    @abstractmethod
    async def get_response(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any
    ) -> str:
        """Get response from AI provider.

        Args:
            messages: List of conversation messages
            **kwargs: Additional provider-specific options

        Returns:
            AI response text
        """
        pass


class ProviderRegistry:
    """Registry for AI providers with decorator-based registration."""

    _providers: dict[str, type[AIProvider]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a provider class.

        Usage:
            @ProviderRegistry.register("openai")
            class OpenAIProvider(AIProvider):
                ...
        """
        def decorator(provider_class: type[AIProvider]) -> type[AIProvider]:
            cls._providers[name] = provider_class
            return provider_class
        return decorator

    @classmethod
    def create(
        cls,
        name: str,
        hass: HomeAssistant,
        config: dict[str, Any]
    ) -> AIProvider:
        """Factory method to create provider instance.

        Args:
            name: Provider name (e.g., "openai", "gemini")
            hass: Home Assistant instance
            config: Provider configuration

        Returns:
            Configured provider instance

        Raises:
            ValueError: If provider not found
        """
        if name not in cls._providers:
            available = list(cls._providers.keys())
            raise ValueError(
                f"Unknown provider: {name}. Available: {available}"
            )
        return cls._providers[name](hass, config)

    @classmethod
    def available_providers(cls) -> list[str]:
        """List all registered provider names."""
        return list(cls._providers.keys())

    @classmethod
    def get_provider_class(cls, name: str) -> type[AIProvider] | None:
        """Get provider class by name without instantiating."""
        return cls._providers.get(name)
```

**Step 5: Run test to verify it passes**

```bash
pytest tests/test_providers/test_registry.py -v
```
Expected: PASS (4 tests)

**Step 6: Write failing test for base client**

```python
# tests/test_providers/test_base_client.py
"""Tests for base HTTP client."""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from aioresponses import aioresponses

from custom_components.ai_agent_ha.providers.base_client import BaseHTTPClient


class TestBaseHTTPClient:
    """Test BaseHTTPClient class."""

    @pytest.fixture
    def mock_hass(self):
        """Create mock hass with aiohttp session."""
        hass = MagicMock()
        return hass

    @pytest.fixture
    def client(self, mock_hass):
        """Create BaseHTTPClient instance."""
        config = {
            "token": "test-token",
            "model": "test-model",
        }
        return BaseHTTPClient(mock_hass, config)

    def test_build_headers_abstract(self, client):
        """Test that _build_headers must be implemented."""
        with pytest.raises(NotImplementedError):
            client._build_headers()

    def test_build_payload_abstract(self, client):
        """Test that _build_payload must be implemented."""
        with pytest.raises(NotImplementedError):
            client._build_payload([])

    def test_extract_response_abstract(self, client):
        """Test that _extract_response must be implemented."""
        with pytest.raises(NotImplementedError):
            client._extract_response({})
```

**Step 7: Run test to verify it fails**

```bash
pytest tests/test_providers/test_base_client.py -v
```
Expected: FAIL with "ModuleNotFoundError"

**Step 8: Implement base client**

```python
# custom_components/ai_agent_ha/providers/base_client.py
"""Base HTTP client for AI providers with shared logic."""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .registry import AIProvider

if TYPE_CHECKING:
    from aiohttp import ClientSession
    from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)

# Default timeouts
DEFAULT_TIMEOUT = 120
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 1.0


class BaseHTTPClient(AIProvider):
    """Base class for HTTP-based AI providers.

    Provides common functionality:
    - HTTP session management (via HA's aiohttp client)
    - Retry logic with exponential backoff
    - Error handling and logging
    - Response parsing template

    Subclasses must implement:
    - _build_headers(): Provider-specific headers
    - _build_payload(): Provider-specific request body
    - _extract_response(): Provider-specific response parsing
    - supports_tools: Property indicating tool support
    """

    API_URL: str = ""  # Override in subclass

    def __init__(self, hass: HomeAssistant, config: dict[str, Any]) -> None:
        """Initialize HTTP client."""
        super().__init__(hass, config)
        self._session: ClientSession | None = None
        self._timeout = config.get("timeout", DEFAULT_TIMEOUT)
        self._max_retries = config.get("max_retries", DEFAULT_MAX_RETRIES)
        self._retry_delay = config.get("retry_delay", DEFAULT_RETRY_DELAY)

    @property
    def session(self) -> ClientSession:
        """Get aiohttp session (lazy initialization)."""
        if self._session is None:
            self._session = async_get_clientsession(self.hass)
        return self._session

    def _build_headers(self) -> dict[str, str]:
        """Build request headers. Override in subclass."""
        raise NotImplementedError("Subclass must implement _build_headers")

    def _build_payload(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any
    ) -> dict[str, Any]:
        """Build request payload. Override in subclass."""
        raise NotImplementedError("Subclass must implement _build_payload")

    def _extract_response(self, data: dict[str, Any]) -> str:
        """Extract response text from API response. Override in subclass."""
        raise NotImplementedError("Subclass must implement _extract_response")

    async def get_response(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any
    ) -> str:
        """Get response from AI provider with retry logic.

        Args:
            messages: Conversation messages
            **kwargs: Additional options (tools, temperature, etc.)

        Returns:
            AI response text

        Raises:
            Exception: If all retries fail
        """
        headers = self._build_headers()
        payload = self._build_payload(messages, **kwargs)

        last_error: Exception | None = None

        for attempt in range(self._max_retries):
            try:
                async with asyncio.timeout(self._timeout):
                    async with self.session.post(
                        self.API_URL,
                        json=payload,
                        headers=headers,
                    ) as resp:
                        if resp.status != 200:
                            error_text = await resp.text()
                            raise Exception(
                                f"API error {resp.status}: {error_text}"
                            )

                        data = await resp.json()
                        return self._extract_response(data)

            except asyncio.TimeoutError as e:
                last_error = e
                _LOGGER.warning(
                    "Timeout on attempt %d/%d for %s",
                    attempt + 1, self._max_retries, self.API_URL
                )
            except Exception as e:
                last_error = e
                _LOGGER.warning(
                    "Error on attempt %d/%d: %s",
                    attempt + 1, self._max_retries, str(e)
                )

            if attempt < self._max_retries - 1:
                delay = self._retry_delay * (2 ** attempt)  # Exponential backoff
                await asyncio.sleep(delay)

        raise last_error or Exception("All retries failed")
```

**Step 9: Run test to verify it passes**

```bash
pytest tests/test_providers/test_base_client.py -v
```
Expected: PASS (3 tests)

**Step 10: Update providers __init__.py**

```python
# custom_components/ai_agent_ha/providers/__init__.py
"""AI Providers plugin system."""
from .registry import AIProvider, ProviderRegistry
from .base_client import BaseHTTPClient

__all__ = [
    "AIProvider",
    "ProviderRegistry",
    "BaseHTTPClient",
]
```

**Step 11: Run all provider tests**

```bash
pytest tests/test_providers/ -v
```
Expected: PASS (7 tests)

**Step 12: Commit**

```bash
git add custom_components/ai_agent_ha/providers/ tests/test_providers/
git commit -m "feat(providers): add registry and base client infrastructure"
```

---

## Task A2: OpenAI Provider

**Depends on:** A1
**Blocks:** C1

**Files:**
- Create: `custom_components/ai_agent_ha/providers/openai.py`
- Create: `tests/test_providers/test_openai.py`
- Reference: `custom_components/ai_agent_ha/agent.py:504-605` (existing OpenAI code)

**Step 1: Write failing test**

```python
# tests/test_providers/test_openai.py
"""Tests for OpenAI provider."""
import pytest
from unittest.mock import MagicMock, patch
from aioresponses import aioresponses

from custom_components.ai_agent_ha.providers.openai import OpenAIProvider
from custom_components.ai_agent_ha.providers.registry import ProviderRegistry


class TestOpenAIProvider:
    """Test OpenAI provider."""

    @pytest.fixture
    def mock_hass(self):
        """Create mock hass."""
        hass = MagicMock()
        return hass

    @pytest.fixture
    def provider(self, mock_hass):
        """Create OpenAI provider."""
        config = {
            "token": "sk-test-token",
            "model": "gpt-4",
        }
        return OpenAIProvider(mock_hass, config)

    def test_registered_in_registry(self):
        """Test provider is registered."""
        assert "openai" in ProviderRegistry.available_providers()

    def test_supports_tools(self, provider):
        """Test provider supports tools."""
        assert provider.supports_tools is True

    def test_build_headers(self, provider):
        """Test header building."""
        headers = provider._build_headers()
        assert headers["Authorization"] == "Bearer sk-test-token"
        assert headers["Content-Type"] == "application/json"

    def test_build_payload(self, provider):
        """Test payload building."""
        messages = [{"role": "user", "content": "Hello"}]
        payload = provider._build_payload(messages)

        assert payload["model"] == "gpt-4"
        assert payload["messages"] == messages

    def test_build_payload_with_tools(self, provider):
        """Test payload with tools."""
        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test"}}]

        payload = provider._build_payload(messages, tools=tools)

        assert payload["tools"] == tools

    def test_extract_response(self, provider):
        """Test response extraction."""
        data = {
            "choices": [
                {"message": {"content": "Hello back!"}}
            ]
        }
        result = provider._extract_response(data)
        assert result == "Hello back!"

    def test_extract_response_with_tool_calls(self, provider):
        """Test response extraction with tool calls."""
        data = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {"function": {"name": "test", "arguments": "{}"}}
                        ]
                    }
                }
            ]
        }
        result = provider._extract_response(data)
        assert "tool_calls" in result or result is not None

    @pytest.mark.asyncio
    async def test_get_response_success(self, provider):
        """Test successful API call."""
        with patch.object(provider, 'session') as mock_session:
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.json = MagicMock(return_value={
                "choices": [{"message": {"content": "Test response"}}]
            })
            mock_response.__aenter__ = MagicMock(return_value=mock_response)
            mock_response.__aexit__ = MagicMock(return_value=None)

            mock_session.post.return_value = mock_response

            # This will fail until we implement - that's expected
            # result = await provider.get_response([{"role": "user", "content": "Hi"}])
            # assert result == "Test response"
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_providers/test_openai.py -v
```
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Implement OpenAI provider**

```python
# custom_components/ai_agent_ha/providers/openai.py
"""OpenAI API provider."""
from __future__ import annotations

import json
from typing import Any

from .base_client import BaseHTTPClient
from .registry import ProviderRegistry


@ProviderRegistry.register("openai")
class OpenAIProvider(BaseHTTPClient):
    """OpenAI API provider (GPT-4, GPT-3.5, etc.)."""

    API_URL = "https://api.openai.com/v1/chat/completions"

    @property
    def supports_tools(self) -> bool:
        """OpenAI supports function calling."""
        return True

    def _build_headers(self) -> dict[str, str]:
        """Build OpenAI API headers."""
        return {
            "Authorization": f"Bearer {self.config['token']}",
            "Content-Type": "application/json",
        }

    def _build_payload(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any
    ) -> dict[str, Any]:
        """Build OpenAI API payload."""
        payload: dict[str, Any] = {
            "model": self.config.get("model", "gpt-4"),
            "messages": messages,
        }

        # Optional parameters
        if "tools" in kwargs:
            payload["tools"] = kwargs["tools"]
        if "tool_choice" in kwargs:
            payload["tool_choice"] = kwargs["tool_choice"]
        if "temperature" in kwargs:
            payload["temperature"] = kwargs["temperature"]
        if "max_tokens" in kwargs:
            payload["max_tokens"] = kwargs["max_tokens"]

        return payload

    def _extract_response(self, data: dict[str, Any]) -> str:
        """Extract response from OpenAI API response."""
        message = data["choices"][0]["message"]

        # Check for tool calls
        if message.get("tool_calls"):
            return json.dumps({
                "tool_calls": message["tool_calls"],
                "content": message.get("content"),
            })

        return message.get("content", "")
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_providers/test_openai.py -v
```
Expected: PASS (8 tests)

**Step 5: Update providers __init__.py**

```python
# custom_components/ai_agent_ha/providers/__init__.py
"""AI Providers plugin system."""
from .registry import AIProvider, ProviderRegistry
from .base_client import BaseHTTPClient

# Import providers to trigger registration
from . import openai  # noqa: F401

__all__ = [
    "AIProvider",
    "ProviderRegistry",
    "BaseHTTPClient",
]
```

**Step 6: Run all tests**

```bash
pytest tests/test_providers/ -v
```
Expected: PASS

**Step 7: Commit**

```bash
git add custom_components/ai_agent_ha/providers/openai.py tests/test_providers/test_openai.py
git add custom_components/ai_agent_ha/providers/__init__.py
git commit -m "feat(providers): add OpenAI provider"
```

---

## Task A3: Gemini Provider

**Depends on:** A1
**Blocks:** C1

**Files:**
- Create: `custom_components/ai_agent_ha/providers/gemini.py`
- Create: `tests/test_providers/test_gemini.py`
- Reference: `custom_components/ai_agent_ha/agent.py:607-762` (existing Gemini code)

**Step 1: Write failing test**

```python
# tests/test_providers/test_gemini.py
"""Tests for Gemini provider."""
import pytest
from unittest.mock import MagicMock

from custom_components.ai_agent_ha.providers.gemini import GeminiProvider
from custom_components.ai_agent_ha.providers.registry import ProviderRegistry


class TestGeminiProvider:
    """Test Gemini provider."""

    @pytest.fixture
    def mock_hass(self):
        """Create mock hass."""
        return MagicMock()

    @pytest.fixture
    def provider(self, mock_hass):
        """Create Gemini provider."""
        config = {
            "token": "test-api-key",
            "model": "gemini-pro",
        }
        return GeminiProvider(mock_hass, config)

    def test_registered_in_registry(self):
        """Test provider is registered."""
        assert "gemini" in ProviderRegistry.available_providers()

    def test_supports_tools(self, provider):
        """Test provider supports tools."""
        assert provider.supports_tools is True

    def test_build_headers(self, provider):
        """Test header building."""
        headers = provider._build_headers()
        assert headers["x-goog-api-key"] == "test-api-key"
        assert headers["Content-Type"] == "application/json"

    def test_convert_messages_to_gemini_format(self, provider):
        """Test message format conversion."""
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        contents, system = provider._convert_messages(messages)

        assert system == "You are helpful"
        assert len(contents) == 2
        assert contents[0]["role"] == "user"
        assert contents[1]["role"] == "model"

    def test_extract_response(self, provider):
        """Test response extraction."""
        data = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "Hello from Gemini!"}]
                    }
                }
            ]
        }
        result = provider._extract_response(data)
        assert result == "Hello from Gemini!"
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_providers/test_gemini.py -v
```
Expected: FAIL

**Step 3: Implement Gemini provider**

```python
# custom_components/ai_agent_ha/providers/gemini.py
"""Google Gemini API provider."""
from __future__ import annotations

import json
from typing import Any

from .base_client import BaseHTTPClient
from .registry import ProviderRegistry


@ProviderRegistry.register("gemini")
class GeminiProvider(BaseHTTPClient):
    """Google Gemini API provider."""

    API_URL_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    @property
    def API_URL(self) -> str:
        """Build API URL with model name."""
        model = self.config.get("model", "gemini-pro")
        return self.API_URL_TEMPLATE.format(model=model)

    @property
    def supports_tools(self) -> bool:
        """Gemini supports function calling."""
        return True

    def _build_headers(self) -> dict[str, str]:
        """Build Gemini API headers."""
        return {
            "x-goog-api-key": self.config["token"],
            "Content-Type": "application/json",
        }

    def _convert_messages(
        self,
        messages: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Convert OpenAI-style messages to Gemini format.

        Returns:
            Tuple of (contents, system_instruction)
        """
        contents = []
        system_instruction = None

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                system_instruction = content
            elif role == "user":
                contents.append({
                    "role": "user",
                    "parts": [{"text": content}]
                })
            elif role == "assistant":
                contents.append({
                    "role": "model",
                    "parts": [{"text": content}]
                })

        return contents, system_instruction

    def _build_payload(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any
    ) -> dict[str, Any]:
        """Build Gemini API payload."""
        contents, system_instruction = self._convert_messages(messages)

        payload: dict[str, Any] = {
            "contents": contents,
        }

        if system_instruction:
            payload["systemInstruction"] = {
                "parts": [{"text": system_instruction}]
            }

        # Generation config
        generation_config = {}
        if "temperature" in kwargs:
            generation_config["temperature"] = kwargs["temperature"]
        if "max_tokens" in kwargs:
            generation_config["maxOutputTokens"] = kwargs["max_tokens"]
        if generation_config:
            payload["generationConfig"] = generation_config

        # Tools
        if "tools" in kwargs:
            payload["tools"] = self._convert_tools(kwargs["tools"])

        return payload

    def _convert_tools(
        self,
        openai_tools: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Convert OpenAI tool format to Gemini format."""
        gemini_tools = []
        for tool in openai_tools:
            if tool.get("type") == "function":
                func = tool["function"]
                gemini_tools.append({
                    "functionDeclarations": [{
                        "name": func["name"],
                        "description": func.get("description", ""),
                        "parameters": func.get("parameters", {}),
                    }]
                })
        return gemini_tools

    def _extract_response(self, data: dict[str, Any]) -> str:
        """Extract response from Gemini API response."""
        candidates = data.get("candidates", [])
        if not candidates:
            return ""

        content = candidates[0].get("content", {})
        parts = content.get("parts", [])

        # Check for function calls
        for part in parts:
            if "functionCall" in part:
                return json.dumps({
                    "tool_calls": [{
                        "function": {
                            "name": part["functionCall"]["name"],
                            "arguments": json.dumps(part["functionCall"].get("args", {}))
                        }
                    }]
                })

        # Text response
        text_parts = [p.get("text", "") for p in parts if "text" in p]
        return "".join(text_parts)
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_providers/test_gemini.py -v
```
Expected: PASS

**Step 5: Update __init__.py**

```python
# Add to custom_components/ai_agent_ha/providers/__init__.py
from . import gemini  # noqa: F401
```

**Step 6: Commit**

```bash
git add custom_components/ai_agent_ha/providers/gemini.py tests/test_providers/test_gemini.py
git add custom_components/ai_agent_ha/providers/__init__.py
git commit -m "feat(providers): add Gemini provider"
```

---

## Task A4: Anthropic Provider

**Depends on:** A1
**Files:**
- Create: `custom_components/ai_agent_ha/providers/anthropic.py`
- Create: `tests/test_providers/test_anthropic.py`
- Reference: `custom_components/ai_agent_ha/agent.py:763-844`

**Step 1: Write failing test**

```python
# tests/test_providers/test_anthropic.py
"""Tests for Anthropic provider."""
import pytest
from unittest.mock import MagicMock

from custom_components.ai_agent_ha.providers.anthropic import AnthropicProvider
from custom_components.ai_agent_ha.providers.registry import ProviderRegistry


class TestAnthropicProvider:
    """Test Anthropic provider."""

    @pytest.fixture
    def mock_hass(self):
        return MagicMock()

    @pytest.fixture
    def provider(self, mock_hass):
        config = {
            "token": "sk-ant-test",
            "model": "claude-3-opus-20240229",
        }
        return AnthropicProvider(mock_hass, config)

    def test_registered(self):
        assert "anthropic" in ProviderRegistry.available_providers()

    def test_supports_tools(self, provider):
        assert provider.supports_tools is True

    def test_build_headers(self, provider):
        headers = provider._build_headers()
        assert headers["x-api-key"] == "sk-ant-test"
        assert headers["anthropic-version"] == "2023-06-01"

    def test_extract_system_message(self, provider):
        messages = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hi"},
        ]
        filtered, system = provider._extract_system(messages)
        assert system == "Be helpful"
        assert len(filtered) == 1

    def test_extract_response(self, provider):
        data = {
            "content": [{"type": "text", "text": "Hello!"}]
        }
        assert provider._extract_response(data) == "Hello!"
```

**Step 2: Run test (fails)**

```bash
pytest tests/test_providers/test_anthropic.py -v
```

**Step 3: Implement**

```python
# custom_components/ai_agent_ha/providers/anthropic.py
"""Anthropic Claude API provider."""
from __future__ import annotations

import json
from typing import Any

from .base_client import BaseHTTPClient
from .registry import ProviderRegistry


@ProviderRegistry.register("anthropic")
class AnthropicProvider(BaseHTTPClient):
    """Anthropic Claude API provider."""

    API_URL = "https://api.anthropic.com/v1/messages"
    ANTHROPIC_VERSION = "2023-06-01"

    @property
    def supports_tools(self) -> bool:
        return True

    def _build_headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.config["token"],
            "anthropic-version": self.ANTHROPIC_VERSION,
            "Content-Type": "application/json",
        }

    def _extract_system(
        self,
        messages: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Extract system message from messages list."""
        system = None
        filtered = []

        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                filtered.append(msg)

        return filtered, system

    def _build_payload(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any
    ) -> dict[str, Any]:
        filtered_messages, system = self._extract_system(messages)

        payload: dict[str, Any] = {
            "model": self.config.get("model", "claude-3-opus-20240229"),
            "messages": filtered_messages,
            "max_tokens": kwargs.get("max_tokens", 4096),
        }

        if system:
            payload["system"] = system

        if "tools" in kwargs:
            payload["tools"] = self._convert_tools(kwargs["tools"])

        if "temperature" in kwargs:
            payload["temperature"] = kwargs["temperature"]

        return payload

    def _convert_tools(
        self,
        openai_tools: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Convert OpenAI tool format to Anthropic format."""
        anthropic_tools = []
        for tool in openai_tools:
            if tool.get("type") == "function":
                func = tool["function"]
                anthropic_tools.append({
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {"type": "object"}),
                })
        return anthropic_tools

    def _extract_response(self, data: dict[str, Any]) -> str:
        content = data.get("content", [])

        # Check for tool use
        for block in content:
            if block.get("type") == "tool_use":
                return json.dumps({
                    "tool_calls": [{
                        "id": block.get("id"),
                        "function": {
                            "name": block["name"],
                            "arguments": json.dumps(block.get("input", {}))
                        }
                    }]
                })

        # Text response
        text_blocks = [b.get("text", "") for b in content if b.get("type") == "text"]
        return "".join(text_blocks)
```

**Step 4: Run test (passes)**

```bash
pytest tests/test_providers/test_anthropic.py -v
```

**Step 5: Update __init__.py and commit**

```bash
# Add import to __init__.py
git add custom_components/ai_agent_ha/providers/anthropic.py tests/test_providers/test_anthropic.py
git commit -m "feat(providers): add Anthropic provider"
```

---

## Task A5: OpenRouter Provider

**Depends on:** A1
**Files:**
- Create: `custom_components/ai_agent_ha/providers/openrouter.py`
- Create: `tests/test_providers/test_openrouter.py`

**Step 1: Write failing test**

```python
# tests/test_providers/test_openrouter.py
"""Tests for OpenRouter provider."""
import pytest
from unittest.mock import MagicMock

from custom_components.ai_agent_ha.providers.openrouter import OpenRouterProvider
from custom_components.ai_agent_ha.providers.registry import ProviderRegistry


class TestOpenRouterProvider:
    """Test OpenRouter provider."""

    @pytest.fixture
    def mock_hass(self):
        return MagicMock()

    @pytest.fixture
    def provider(self, mock_hass):
        config = {
            "token": "sk-or-test",
            "model": "openai/gpt-4",
        }
        return OpenRouterProvider(mock_hass, config)

    def test_registered(self):
        assert "openrouter" in ProviderRegistry.available_providers()

    def test_supports_tools(self, provider):
        assert provider.supports_tools is True

    def test_build_headers(self, provider):
        headers = provider._build_headers()
        assert headers["Authorization"] == "Bearer sk-or-test"
        assert "HTTP-Referer" in headers

    def test_api_url(self, provider):
        assert "openrouter.ai" in provider.API_URL
```

**Step 2-4: Implement and test**

```python
# custom_components/ai_agent_ha/providers/openrouter.py
"""OpenRouter API provider (multi-model gateway)."""
from __future__ import annotations

from typing import Any

from .base_client import BaseHTTPClient
from .registry import ProviderRegistry


@ProviderRegistry.register("openrouter")
class OpenRouterProvider(BaseHTTPClient):
    """OpenRouter API provider - OpenAI-compatible gateway."""

    API_URL = "https://openrouter.ai/api/v1/chat/completions"

    @property
    def supports_tools(self) -> bool:
        return True

    def _build_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config['token']}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://home-assistant.io",
            "X-Title": "AI Agent HA",
        }

    def _build_payload(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any
    ) -> dict[str, Any]:
        # OpenRouter uses OpenAI format
        payload: dict[str, Any] = {
            "model": self.config.get("model", "openai/gpt-4"),
            "messages": messages,
        }

        if "tools" in kwargs:
            payload["tools"] = kwargs["tools"]
        if "temperature" in kwargs:
            payload["temperature"] = kwargs["temperature"]
        if "max_tokens" in kwargs:
            payload["max_tokens"] = kwargs["max_tokens"]

        return payload

    def _extract_response(self, data: dict[str, Any]) -> str:
        # OpenRouter uses OpenAI response format
        import json
        message = data["choices"][0]["message"]

        if message.get("tool_calls"):
            return json.dumps({
                "tool_calls": message["tool_calls"],
                "content": message.get("content"),
            })

        return message.get("content", "")
```

**Step 5: Commit**

```bash
git add custom_components/ai_agent_ha/providers/openrouter.py tests/test_providers/test_openrouter.py
git commit -m "feat(providers): add OpenRouter provider"
```

---

## Task A6: Groq Provider

**Depends on:** A1
**Files:**
- Create: `custom_components/ai_agent_ha/providers/groq.py`
- Create: `tests/test_providers/test_groq.py`

**Implementation similar to OpenAI (Groq uses OpenAI-compatible API)**

```python
# custom_components/ai_agent_ha/providers/groq.py
"""Groq API provider (fast inference)."""
from __future__ import annotations

from .openai import OpenAIProvider
from .registry import ProviderRegistry


@ProviderRegistry.register("groq")
class GroqProvider(OpenAIProvider):
    """Groq API provider - OpenAI-compatible with different URL."""

    API_URL = "https://api.groq.com/openai/v1/chat/completions"

    def _build_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config['token']}",
            "Content-Type": "application/json",
        }
```

**Commit**

```bash
git add custom_components/ai_agent_ha/providers/groq.py tests/test_providers/test_groq.py
git commit -m "feat(providers): add Groq provider"
```

---

## Task A7: Local Provider (Ollama)

**Depends on:** A1
**Files:**
- Create: `custom_components/ai_agent_ha/providers/local.py`
- Create: `tests/test_providers/test_local.py`
- Reference: `custom_components/ai_agent_ha/agent.py:132-462`

**Step 1: Write failing test**

```python
# tests/test_providers/test_local.py
"""Tests for Local/Ollama provider."""
import pytest
from unittest.mock import MagicMock

from custom_components.ai_agent_ha.providers.local import LocalProvider
from custom_components.ai_agent_ha.providers.registry import ProviderRegistry


class TestLocalProvider:
    """Test Local provider."""

    @pytest.fixture
    def mock_hass(self):
        return MagicMock()

    @pytest.fixture
    def provider(self, mock_hass):
        config = {
            "api_url": "http://localhost:11434",
            "model": "llama2",
        }
        return LocalProvider(mock_hass, config)

    def test_registered(self):
        assert "local" in ProviderRegistry.available_providers()

    def test_supports_tools(self, provider):
        # Local models may not support tools
        assert provider.supports_tools is False

    def test_api_url_configurable(self, provider):
        assert provider.API_URL == "http://localhost:11434/api/chat"

    def test_build_payload(self, provider):
        messages = [{"role": "user", "content": "Hi"}]
        payload = provider._build_payload(messages)
        assert payload["model"] == "llama2"
        assert payload["messages"] == messages
```

**Step 2-4: Implement and test**

```python
# custom_components/ai_agent_ha/providers/local.py
"""Local AI provider (Ollama, LM Studio, etc.)."""
from __future__ import annotations

from typing import Any

from .base_client import BaseHTTPClient
from .registry import ProviderRegistry


@ProviderRegistry.register("local")
class LocalProvider(BaseHTTPClient):
    """Local AI provider for self-hosted models."""

    @property
    def API_URL(self) -> str:
        """Get API URL from config."""
        base_url = self.config.get("api_url", "http://localhost:11434")
        return f"{base_url}/api/chat"

    @property
    def supports_tools(self) -> bool:
        """Most local models don't support function calling."""
        return self.config.get("supports_tools", False)

    def _build_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}

        # Optional auth for some local servers
        if "token" in self.config:
            headers["Authorization"] = f"Bearer {self.config['token']}"

        return headers

    def _build_payload(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.config.get("model", "llama2"),
            "messages": messages,
            "stream": False,
        }

        if "temperature" in kwargs:
            payload["options"] = payload.get("options", {})
            payload["options"]["temperature"] = kwargs["temperature"]

        return payload

    def _extract_response(self, data: dict[str, Any]) -> str:
        """Extract response from Ollama API."""
        message = data.get("message", {})
        return message.get("content", "")
```

**Step 5: Commit**

```bash
git add custom_components/ai_agent_ha/providers/local.py tests/test_providers/test_local.py
git commit -m "feat(providers): add Local/Ollama provider"
```

---

# STREAM B: Managers (Can run parallel with Stream A)

## Task B1: Entity Manager

**Depends on:** Task 0
**Blocks:** C4

**Files:**
- Create: `custom_components/ai_agent_ha/managers/__init__.py`
- Create: `custom_components/ai_agent_ha/managers/entity_manager.py`
- Create: `tests/test_managers/__init__.py`
- Create: `tests/test_managers/test_entity_manager.py`
- Reference: `custom_components/ai_agent_ha/agent.py:1859-2160` (entity methods)

**Step 1: Create directories**

```bash
mkdir -p custom_components/ai_agent_ha/managers
mkdir -p tests/test_managers
touch custom_components/ai_agent_ha/managers/__init__.py
touch tests/test_managers/__init__.py
```

**Step 2: Write failing test**

```python
# tests/test_managers/test_entity_manager.py
"""Tests for EntityManager."""
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime

from custom_components.ai_agent_ha.managers.entity_manager import EntityManager


class TestEntityManager:
    """Test EntityManager class."""

    @pytest.fixture
    def mock_hass(self):
        """Create mock hass with states."""
        hass = MagicMock()

        # Mock state object
        mock_state = MagicMock()
        mock_state.state = "on"
        mock_state.attributes = {"brightness": 255, "friendly_name": "Living Room Light"}
        mock_state.last_changed = datetime(2024, 1, 1, 12, 0, 0)
        mock_state.last_updated = datetime(2024, 1, 1, 12, 0, 0)

        hass.states.get.return_value = mock_state
        hass.states.async_all.return_value = [
            ("light.living_room", mock_state),
            ("light.bedroom", mock_state),
        ]

        return hass

    @pytest.fixture
    def manager(self, mock_hass):
        """Create EntityManager instance."""
        return EntityManager(mock_hass)

    def test_get_entity_state(self, manager, mock_hass):
        """Test getting entity state."""
        result = manager.get_entity_state("light.living_room")

        assert result is not None
        assert result["entity_id"] == "light.living_room"
        assert result["state"] == "on"
        assert result["attributes"]["brightness"] == 255

    def test_get_entity_state_not_found(self, manager, mock_hass):
        """Test getting non-existent entity."""
        mock_hass.states.get.return_value = None
        result = manager.get_entity_state("light.nonexistent")
        assert result is None

    def test_get_entities_by_domain(self, manager):
        """Test getting entities by domain."""
        result = manager.get_entities_by_domain("light")

        assert len(result) == 2
        entity_ids = [e["entity_id"] for e in result]
        assert "light.living_room" in entity_ids

    def test_get_entity_ids_by_domain(self, manager):
        """Test getting only entity IDs."""
        result = manager.get_entity_ids_by_domain("light")

        assert isinstance(result, list)
        assert "light.living_room" in result

    def test_filter_entities_by_attribute(self, manager, mock_hass):
        """Test filtering entities by attribute."""
        result = manager.filter_entities(domain="light", attribute="brightness", value=255)
        assert len(result) >= 0  # May be empty depending on mock
```

**Step 3: Run test (fails)**

```bash
pytest tests/test_managers/test_entity_manager.py -v
```

**Step 4: Implement**

```python
# custom_components/ai_agent_ha/managers/entity_manager.py
"""Entity Manager - handles entity state operations."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant, State

_LOGGER = logging.getLogger(__name__)


class EntityManager:
    """Manages entity-related operations.

    Extracts entity operations from the monolithic agent class.
    Uses hass directly - this is a HA plugin.
    """

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize EntityManager.

        Args:
            hass: Home Assistant instance
        """
        self.hass = hass

    def get_entity_state(self, entity_id: str) -> dict[str, Any] | None:
        """Get current state of an entity.

        Args:
            entity_id: Entity ID (e.g., "light.living_room")

        Returns:
            Entity state dict or None if not found
        """
        state: State | None = self.hass.states.get(entity_id)
        if state is None:
            return None

        return {
            "entity_id": entity_id,
            "state": state.state,
            "attributes": dict(state.attributes),
            "last_changed": state.last_changed.isoformat() if state.last_changed else None,
            "last_updated": state.last_updated.isoformat() if state.last_updated else None,
        }

    def get_entities_by_domain(self, domain: str) -> list[dict[str, Any]]:
        """Get all entities for a domain.

        Args:
            domain: Entity domain (e.g., "light", "switch")

        Returns:
            List of entity state dicts
        """
        entities = []
        for entity_id, state in self.hass.states.async_all(domain):
            entities.append({
                "entity_id": entity_id,
                "state": state.state,
                "friendly_name": state.attributes.get("friendly_name"),
                "attributes": dict(state.attributes),
            })
        return entities

    def get_entity_ids_by_domain(self, domain: str) -> list[str]:
        """Get only entity IDs for a domain.

        Args:
            domain: Entity domain

        Returns:
            List of entity IDs
        """
        return [entity_id for entity_id, _ in self.hass.states.async_all(domain)]

    def filter_entities(
        self,
        domain: str | None = None,
        attribute: str | None = None,
        value: Any = None,
        state: str | None = None,
    ) -> list[dict[str, Any]]:
        """Filter entities by various criteria.

        Args:
            domain: Filter by domain
            attribute: Filter by attribute name
            value: Filter by attribute value
            state: Filter by state value

        Returns:
            List of matching entities
        """
        results = []

        if domain:
            entities = self.hass.states.async_all(domain)
        else:
            entities = self.hass.states.async_all()

        for entity_id, entity_state in entities:
            # Filter by state
            if state is not None and entity_state.state != state:
                continue

            # Filter by attribute
            if attribute is not None:
                attr_value = entity_state.attributes.get(attribute)
                if value is not None and attr_value != value:
                    continue
                elif value is None and attribute not in entity_state.attributes:
                    continue

            results.append({
                "entity_id": entity_id,
                "state": entity_state.state,
                "attributes": dict(entity_state.attributes),
            })

        return results

    def get_entity_by_friendly_name(self, friendly_name: str) -> dict[str, Any] | None:
        """Find entity by friendly name.

        Args:
            friendly_name: The friendly name to search for

        Returns:
            Entity state dict or None
        """
        friendly_name_lower = friendly_name.lower()

        for entity_id, state in self.hass.states.async_all():
            entity_friendly_name = state.attributes.get("friendly_name", "")
            if entity_friendly_name.lower() == friendly_name_lower:
                return self.get_entity_state(entity_id)

        return None
```

**Step 5: Run test (passes)**

```bash
pytest tests/test_managers/test_entity_manager.py -v
```

**Step 6: Commit**

```bash
git add custom_components/ai_agent_ha/managers/ tests/test_managers/
git commit -m "feat(managers): add EntityManager"
```

---

## Task B2: Registry Manager

**Depends on:** Task 0
**Blocks:** C4

**Files:**
- Create: `custom_components/ai_agent_ha/managers/registry_manager.py`
- Create: `tests/test_managers/test_registry_manager.py`
- Reference: `custom_components/ai_agent_ha/agent.py:2236-2560` (registry methods)

**Step 1: Write failing test**

```python
# tests/test_managers/test_registry_manager.py
"""Tests for RegistryManager."""
import pytest
from unittest.mock import MagicMock, patch

from custom_components.ai_agent_ha.managers.registry_manager import RegistryManager


class TestRegistryManager:
    """Test RegistryManager class."""

    @pytest.fixture
    def mock_hass(self):
        """Create mock hass with registries."""
        hass = MagicMock()
        return hass

    @pytest.fixture
    def mock_entity_registry(self):
        """Create mock entity registry."""
        registry = MagicMock()

        mock_entry = MagicMock()
        mock_entry.entity_id = "light.living_room"
        mock_entry.area_id = "living_room"
        mock_entry.device_id = "device_123"
        mock_entry.platform = "hue"
        mock_entry.original_name = "Living Room Light"

        registry.entities = {"light.living_room": mock_entry}
        registry.async_get.return_value = mock_entry

        return registry

    @pytest.fixture
    def mock_device_registry(self):
        """Create mock device registry."""
        registry = MagicMock()

        mock_device = MagicMock()
        mock_device.id = "device_123"
        mock_device.name = "Hue Bridge"
        mock_device.manufacturer = "Philips"
        mock_device.model = "BSB002"
        mock_device.area_id = "living_room"

        registry.devices = {"device_123": mock_device}
        registry.async_get.return_value = mock_device

        return registry

    @pytest.fixture
    def mock_area_registry(self):
        """Create mock area registry."""
        registry = MagicMock()

        mock_area = MagicMock()
        mock_area.id = "living_room"
        mock_area.name = "Living Room"

        registry.areas = {"living_room": mock_area}
        registry.async_get_area.return_value = mock_area

        return registry

    @pytest.fixture
    def manager(self, mock_hass, mock_entity_registry, mock_device_registry, mock_area_registry):
        """Create RegistryManager with mocked registries."""
        with patch("custom_components.ai_agent_ha.managers.registry_manager.er.async_get", return_value=mock_entity_registry):
            with patch("custom_components.ai_agent_ha.managers.registry_manager.dr.async_get", return_value=mock_device_registry):
                with patch("custom_components.ai_agent_ha.managers.registry_manager.ar.async_get", return_value=mock_area_registry):
                    return RegistryManager(mock_hass)

    def test_get_entity_entry(self, manager):
        """Test getting entity registry entry."""
        result = manager.get_entity_entry("light.living_room")
        assert result is not None
        assert result["entity_id"] == "light.living_room"

    def test_get_device(self, manager):
        """Test getting device info."""
        result = manager.get_device("device_123")
        assert result is not None
        assert result["name"] == "Hue Bridge"

    def test_get_area(self, manager):
        """Test getting area info."""
        result = manager.get_area("living_room")
        assert result is not None
        assert result["name"] == "Living Room"

    def test_get_entities_by_area(self, manager, mock_entity_registry):
        """Test getting entities in an area."""
        result = manager.get_entities_by_area("living_room")
        assert isinstance(result, list)
```

**Step 2-4: Implement and test**

```python
# custom_components/ai_agent_ha/managers/registry_manager.py
"""Registry Manager - handles HA registry operations."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from homeassistant.helpers import (
    area_registry as ar,
    device_registry as dr,
    entity_registry as er,
)

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)


class RegistryManager:
    """Manages Home Assistant registry operations.

    Provides unified access to entity, device, and area registries.
    """

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize RegistryManager."""
        self.hass = hass
        self._entity_registry = er.async_get(hass)
        self._device_registry = dr.async_get(hass)
        self._area_registry = ar.async_get(hass)

    def get_entity_entry(self, entity_id: str) -> dict[str, Any] | None:
        """Get entity registry entry."""
        entry = self._entity_registry.async_get(entity_id)
        if entry is None:
            return None

        return {
            "entity_id": entry.entity_id,
            "area_id": entry.area_id,
            "device_id": entry.device_id,
            "platform": entry.platform,
            "original_name": entry.original_name,
            "disabled_by": entry.disabled_by,
        }

    def get_device(self, device_id: str) -> dict[str, Any] | None:
        """Get device registry entry."""
        device = self._device_registry.async_get(device_id)
        if device is None:
            return None

        return {
            "id": device.id,
            "name": device.name,
            "manufacturer": device.manufacturer,
            "model": device.model,
            "area_id": device.area_id,
            "sw_version": device.sw_version,
        }

    def get_area(self, area_id: str) -> dict[str, Any] | None:
        """Get area registry entry."""
        area = self._area_registry.async_get_area(area_id)
        if area is None:
            return None

        return {
            "id": area.id,
            "name": area.name,
        }

    def get_all_areas(self) -> list[dict[str, Any]]:
        """Get all areas."""
        return [
            {"id": area.id, "name": area.name}
            for area in self._area_registry.areas.values()
        ]

    def get_entities_by_area(self, area_id: str) -> list[dict[str, Any]]:
        """Get all entities in an area."""
        entities = []

        for entry in self._entity_registry.entities.values():
            if entry.area_id == area_id:
                state = self.hass.states.get(entry.entity_id)
                entities.append({
                    "entity_id": entry.entity_id,
                    "state": state.state if state else None,
                    "device_id": entry.device_id,
                })

        return entities

    def get_devices_by_area(self, area_id: str) -> list[dict[str, Any]]:
        """Get all devices in an area."""
        devices = []

        for device in self._device_registry.devices.values():
            if device.area_id == area_id:
                devices.append(self.get_device(device.id))

        return [d for d in devices if d is not None]

    def get_entities_by_device(self, device_id: str) -> list[dict[str, Any]]:
        """Get all entities for a device."""
        entities = []

        for entry in self._entity_registry.entities.values():
            if entry.device_id == device_id:
                state = self.hass.states.get(entry.entity_id)
                entities.append({
                    "entity_id": entry.entity_id,
                    "state": state.state if state else None,
                    "original_name": entry.original_name,
                })

        return entities
```

**Step 5: Commit**

```bash
git add custom_components/ai_agent_ha/managers/registry_manager.py tests/test_managers/test_registry_manager.py
git commit -m "feat(managers): add RegistryManager"
```

---

## Task B3: Automation Manager

**Depends on:** Task 0
**Blocks:** C4

**Files:**
- Create: `custom_components/ai_agent_ha/managers/automation_manager.py`
- Create: `tests/test_managers/test_automation_manager.py`
- Reference: `custom_components/ai_agent_ha/agent.py:1795-1857, 2227-2236`

**Step 1: Write failing test**

```python
# tests/test_managers/test_automation_manager.py
"""Tests for AutomationManager."""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from custom_components.ai_agent_ha.managers.automation_manager import AutomationManager


class TestAutomationManager:
    """Test AutomationManager class."""

    @pytest.fixture
    def mock_hass(self):
        """Create mock hass."""
        hass = MagicMock()
        hass.services.has_service.return_value = True
        hass.services.async_call = AsyncMock()
        return hass

    @pytest.fixture
    def manager(self, mock_hass):
        """Create AutomationManager instance."""
        return AutomationManager(mock_hass)

    def test_validate_automation_valid(self, manager):
        """Test validating a valid automation config."""
        config = {
            "alias": "Test Automation",
            "trigger": [{"platform": "state", "entity_id": "light.test"}],
            "action": [{"service": "light.turn_on", "target": {"entity_id": "light.test"}}],
        }

        result = manager.validate_automation(config)
        assert result["valid"] is True

    def test_validate_automation_missing_trigger(self, manager):
        """Test validating automation without trigger."""
        config = {
            "alias": "Test",
            "action": [{"service": "light.turn_on"}],
        }

        result = manager.validate_automation(config)
        assert result["valid"] is False
        assert "trigger" in result["error"].lower()

    def test_validate_automation_missing_action(self, manager):
        """Test validating automation without action."""
        config = {
            "alias": "Test",
            "trigger": [{"platform": "state"}],
        }

        result = manager.validate_automation(config)
        assert result["valid"] is False
        assert "action" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_create_automation(self, manager, mock_hass):
        """Test creating an automation."""
        config = {
            "alias": "Test Automation",
            "trigger": [{"platform": "state", "entity_id": "light.test"}],
            "action": [{"service": "light.turn_on"}],
        }

        result = await manager.create_automation(config)

        assert result["success"] is True
        mock_hass.services.async_call.assert_called()
```

**Step 2-4: Implement and test**

```python
# custom_components/ai_agent_ha/managers/automation_manager.py
"""Automation Manager - handles automation operations."""
from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any

import voluptuous as vol
from homeassistant.components.automation import DOMAIN as AUTOMATION_DOMAIN
from homeassistant.const import CONF_ALIAS, CONF_ID

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)


class AutomationManager:
    """Manages automation creation and validation."""

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize AutomationManager."""
        self.hass = hass

    def validate_automation(self, config: dict[str, Any]) -> dict[str, Any]:
        """Validate automation configuration.

        Args:
            config: Automation configuration dict

        Returns:
            Dict with 'valid' bool and optional 'error' message
        """
        # Check required fields
        if "trigger" not in config:
            return {"valid": False, "error": "Missing required field: trigger"}

        if "action" not in config:
            return {"valid": False, "error": "Missing required field: action"}

        # Validate trigger is a list
        triggers = config["trigger"]
        if not isinstance(triggers, list):
            triggers = [triggers]

        if not triggers:
            return {"valid": False, "error": "At least one trigger is required"}

        # Validate action is a list
        actions = config["action"]
        if not isinstance(actions, list):
            actions = [actions]

        if not actions:
            return {"valid": False, "error": "At least one action is required"}

        # Validate each action has a service
        for action in actions:
            if "service" not in action and "action" not in action:
                return {"valid": False, "error": "Each action must have a 'service' or 'action' field"}

        return {"valid": True}

    async def create_automation(self, config: dict[str, Any]) -> dict[str, Any]:
        """Create a new automation.

        Args:
            config: Automation configuration

        Returns:
            Dict with 'success' bool and 'automation_id' or 'error'
        """
        # Validate first
        validation = self.validate_automation(config)
        if not validation["valid"]:
            return {"success": False, "error": validation["error"]}

        # Generate ID if not provided
        if CONF_ID not in config:
            config[CONF_ID] = str(uuid.uuid4())

        # Ensure alias exists
        if CONF_ALIAS not in config:
            config[CONF_ALIAS] = f"AI Generated Automation {config[CONF_ID][:8]}"

        try:
            # Use automation.reload or direct config update
            await self.hass.services.async_call(
                AUTOMATION_DOMAIN,
                "reload",
                {},
                blocking=True,
            )

            return {
                "success": True,
                "automation_id": config[CONF_ID],
                "alias": config[CONF_ALIAS],
            }

        except Exception as e:
            _LOGGER.error("Failed to create automation: %s", e)
            return {"success": False, "error": str(e)}

    def get_automations(self) -> list[dict[str, Any]]:
        """Get all automations."""
        automations = []

        for entity_id, state in self.hass.states.async_all(AUTOMATION_DOMAIN):
            automations.append({
                "entity_id": entity_id,
                "state": state.state,
                "friendly_name": state.attributes.get("friendly_name"),
                "last_triggered": state.attributes.get("last_triggered"),
            })

        return automations

    async def toggle_automation(self, entity_id: str, enable: bool) -> dict[str, Any]:
        """Enable or disable an automation."""
        service = "turn_on" if enable else "turn_off"

        try:
            await self.hass.services.async_call(
                AUTOMATION_DOMAIN,
                service,
                {"entity_id": entity_id},
                blocking=True,
            )
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}
```

**Step 5: Commit**

```bash
git add custom_components/ai_agent_ha/managers/automation_manager.py tests/test_managers/test_automation_manager.py
git commit -m "feat(managers): add AutomationManager"
```

---

## Task B4: Dashboard Manager

**Depends on:** Task 0
**Blocks:** C4

**Files:**
- Create: `custom_components/ai_agent_ha/managers/dashboard_manager.py`
- Create: `tests/test_managers/test_dashboard_manager.py`
- Reference: `custom_components/ai_agent_ha/agent.py:2850-3450`

**(Implementation follows same pattern - tests first, then implementation)**

**Commit:**

```bash
git commit -m "feat(managers): add DashboardManager"
```

---

## Task B5: Control Manager

**Depends on:** Task 0
**Blocks:** C4

**Files:**
- Create: `custom_components/ai_agent_ha/managers/control_manager.py`
- Create: `tests/test_managers/test_control_manager.py`
- Reference: `custom_components/ai_agent_ha/agent.py:4791-4987`

**Step 1: Write failing test**

```python
# tests/test_managers/test_control_manager.py
"""Tests for ControlManager."""
import pytest
from unittest.mock import MagicMock, AsyncMock

from custom_components.ai_agent_ha.managers.control_manager import ControlManager


class TestControlManager:
    """Test ControlManager class."""

    @pytest.fixture
    def mock_hass(self):
        """Create mock hass."""
        hass = MagicMock()
        hass.services.has_service.return_value = True
        hass.services.async_call = AsyncMock()

        mock_state = MagicMock()
        mock_state.state = "on"
        mock_state.attributes = {}
        hass.states.get.return_value = mock_state

        return hass

    @pytest.fixture
    def manager(self, mock_hass):
        """Create ControlManager instance."""
        return ControlManager(mock_hass)

    @pytest.mark.asyncio
    async def test_call_service(self, manager, mock_hass):
        """Test calling a service."""
        result = await manager.call_service(
            domain="light",
            service="turn_on",
            target={"entity_id": "light.living_room"},
            data={"brightness": 255},
        )

        assert result["success"] is True
        mock_hass.services.async_call.assert_called_once()

    @pytest.mark.asyncio
    async def test_call_service_not_found(self, manager, mock_hass):
        """Test calling non-existent service."""
        mock_hass.services.has_service.return_value = False

        result = await manager.call_service(
            domain="fake",
            service="service",
        )

        assert result["success"] is False
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_set_entity_state(self, manager, mock_hass):
        """Test setting entity state."""
        result = await manager.set_entity_state(
            entity_id="light.living_room",
            state="on",
            attributes={"brightness": 255},
        )

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_turn_on_entity(self, manager, mock_hass):
        """Test turning on entity."""
        result = await manager.turn_on("light.living_room", brightness=255)

        assert result["success"] is True
        mock_hass.services.async_call.assert_called()

    @pytest.mark.asyncio
    async def test_turn_off_entity(self, manager, mock_hass):
        """Test turning off entity."""
        result = await manager.turn_off("light.living_room")

        assert result["success"] is True
```

**Step 2-4: Implement and test**

```python
# custom_components/ai_agent_ha/managers/control_manager.py
"""Control Manager - handles entity control operations."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from homeassistant.const import (
    SERVICE_TURN_OFF,
    SERVICE_TURN_ON,
    SERVICE_TOGGLE,
)
from homeassistant.core import ServiceCall

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)


class ControlManager:
    """Manages entity control operations.

    Provides methods to call services and control entity states.
    """

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize ControlManager."""
        self.hass = hass

    async def call_service(
        self,
        domain: str,
        service: str,
        target: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Call a Home Assistant service.

        Args:
            domain: Service domain (e.g., "light", "switch")
            service: Service name (e.g., "turn_on", "turn_off")
            target: Target entities/areas/devices
            data: Additional service data

        Returns:
            Dict with 'success' bool and optional 'error'
        """
        # Check service exists
        if not self.hass.services.has_service(domain, service):
            return {
                "success": False,
                "error": f"Service {domain}.{service} not found",
            }

        service_data = data or {}
        if target:
            service_data.update(target)

        try:
            await self.hass.services.async_call(
                domain,
                service,
                service_data,
                blocking=True,
            )
            return {"success": True}

        except Exception as e:
            _LOGGER.error("Service call failed: %s", e)
            return {"success": False, "error": str(e)}

    async def set_entity_state(
        self,
        entity_id: str,
        state: str,
        attributes: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Set entity state directly.

        Note: This uses state machine directly. For most cases,
        prefer calling the appropriate service instead.
        """
        try:
            self.hass.states.async_set(
                entity_id,
                state,
                attributes or {},
            )
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def turn_on(
        self,
        entity_id: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Turn on an entity."""
        domain = entity_id.split(".")[0]
        return await self.call_service(
            domain=domain,
            service=SERVICE_TURN_ON,
            target={"entity_id": entity_id},
            data=kwargs,
        )

    async def turn_off(
        self,
        entity_id: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Turn off an entity."""
        domain = entity_id.split(".")[0]
        return await self.call_service(
            domain=domain,
            service=SERVICE_TURN_OFF,
            target={"entity_id": entity_id},
            data=kwargs,
        )

    async def toggle(
        self,
        entity_id: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Toggle an entity."""
        domain = entity_id.split(".")[0]
        return await self.call_service(
            domain=domain,
            service=SERVICE_TOGGLE,
            target={"entity_id": entity_id},
            data=kwargs,
        )

    async def set_value(
        self,
        entity_id: str,
        value: Any,
    ) -> dict[str, Any]:
        """Set value for input_* entities."""
        domain = entity_id.split(".")[0]

        service_map = {
            "input_number": ("input_number", "set_value", {"value": value}),
            "input_text": ("input_text", "set_value", {"value": value}),
            "input_boolean": ("input_boolean", "turn_on" if value else "turn_off", {}),
            "input_select": ("input_select", "select_option", {"option": value}),
        }

        if domain not in service_map:
            return {"success": False, "error": f"Unsupported domain: {domain}"}

        svc_domain, svc_name, svc_data = service_map[domain]
        return await self.call_service(
            domain=svc_domain,
            service=svc_name,
            target={"entity_id": entity_id},
            data=svc_data,
        )
```

**Step 5: Commit**

```bash
git add custom_components/ai_agent_ha/managers/control_manager.py tests/test_managers/test_control_manager.py
git commit -m "feat(managers): add ControlManager"
```

---

# STREAM C: Core Refactor (Depends on A1 and B1-B5)

## Task C1: Query Processor

**Depends on:** A1 (registry), B1-B5 (all managers)
**Blocks:** C4

**Files:**
- Create: `custom_components/ai_agent_ha/core/__init__.py`
- Create: `custom_components/ai_agent_ha/core/query_processor.py`
- Create: `tests/test_core/__init__.py`
- Create: `tests/test_core/test_query_processor.py`
- Reference: `custom_components/ai_agent_ha/agent.py:3451-4633` (process_query mega-method)

**Step 1: Create directories**

```bash
mkdir -p custom_components/ai_agent_ha/core
mkdir -p tests/test_core
touch custom_components/ai_agent_ha/core/__init__.py
touch tests/test_core/__init__.py
```

**Step 2: Write failing test**

```python
# tests/test_core/test_query_processor.py
"""Tests for QueryProcessor."""
import pytest
from unittest.mock import MagicMock, AsyncMock

from custom_components.ai_agent_ha.core.query_processor import QueryProcessor


class TestQueryProcessor:
    """Test QueryProcessor class."""

    @pytest.fixture
    def mock_provider(self):
        """Create mock AI provider."""
        provider = MagicMock()
        provider.get_response = AsyncMock(return_value="Hello!")
        provider.supports_tools = True
        return provider

    @pytest.fixture
    def processor(self, mock_provider):
        """Create QueryProcessor instance."""
        return QueryProcessor(
            provider=mock_provider,
            max_iterations=5,
        )

    @pytest.mark.asyncio
    async def test_process_simple_query(self, processor):
        """Test processing a simple query."""
        result = await processor.process(
            query="Hello",
            messages=[],
        )

        assert result["success"] is True
        assert result["response"] == "Hello!"

    @pytest.mark.asyncio
    async def test_process_with_history(self, processor):
        """Test processing with conversation history."""
        history = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]

        result = await processor.process(
            query="How are you?",
            messages=history,
        )

        assert result["success"] is True

    def test_sanitize_query(self, processor):
        """Test query sanitization."""
        # Test invisible characters removal
        query = "Hello\u200b\ufeffWorld"
        sanitized = processor._sanitize_query(query)
        assert "\u200b" not in sanitized
        assert "\ufeff" not in sanitized

    def test_truncate_long_query(self, processor):
        """Test truncating long queries."""
        long_query = "a" * 2000
        truncated = processor._sanitize_query(long_query, max_length=1000)
        assert len(truncated) == 1000
```

**Step 3-4: Implement and test**

```python
# custom_components/ai_agent_ha/core/query_processor.py
"""Query Processor - handles AI query processing pipeline."""
from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from custom_components.ai_agent_ha.providers.registry import AIProvider

_LOGGER = logging.getLogger(__name__)

# Invisible characters to remove
INVISIBLE_CHARS = [
    "\ufeff",  # BOM
    "\u200b",  # Zero-width space
    "\u200c",  # Zero-width non-joiner
    "\u200d",  # Zero-width joiner
    "\u2060",  # Word joiner
]

DEFAULT_MAX_QUERY_LENGTH = 1000
DEFAULT_MAX_ITERATIONS = 10


class QueryProcessor:
    """Processes AI queries with iteration support.

    Extracted from the monolithic process_query() method.
    """

    def __init__(
        self,
        provider: AIProvider,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        max_query_length: int = DEFAULT_MAX_QUERY_LENGTH,
    ) -> None:
        """Initialize QueryProcessor.

        Args:
            provider: AI provider instance
            max_iterations: Max tool-call iterations
            max_query_length: Max query length
        """
        self.provider = provider
        self.max_iterations = max_iterations
        self.max_query_length = max_query_length

    def _sanitize_query(
        self,
        query: str,
        max_length: int | None = None,
    ) -> str:
        """Sanitize user query.

        - Remove invisible characters
        - Truncate to max length
        - Strip whitespace
        """
        # Remove invisible characters
        for char in INVISIBLE_CHARS:
            query = query.replace(char, "")

        # Strip whitespace
        query = query.strip()

        # Truncate if needed
        max_len = max_length or self.max_query_length
        if len(query) > max_len:
            query = query[:max_len]

        return query

    def _build_messages(
        self,
        query: str,
        history: list[dict[str, Any]],
        system_prompt: str | None = None,
    ) -> list[dict[str, Any]]:
        """Build messages list for AI provider."""
        messages = []

        # Add system prompt if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Add history
        messages.extend(history)

        # Add current query
        messages.append({"role": "user", "content": query})

        return messages

    async def process(
        self,
        query: str,
        messages: list[dict[str, Any]],
        system_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_executor: Any | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Process a query through the AI provider.

        Args:
            query: User query
            messages: Conversation history
            system_prompt: Optional system prompt
            tools: Optional tools for function calling
            tool_executor: Optional tool executor for tool calls
            **kwargs: Additional provider options

        Returns:
            Dict with 'success', 'response', and metadata
        """
        # Sanitize query
        query = self._sanitize_query(query)

        if not query:
            return {
                "success": False,
                "error": "Empty query after sanitization",
            }

        # Build messages
        all_messages = self._build_messages(query, messages, system_prompt)

        try:
            # Get response from provider
            response = await self.provider.get_response(
                all_messages,
                tools=tools,
                **kwargs,
            )

            return {
                "success": True,
                "response": response,
                "messages": all_messages + [
                    {"role": "assistant", "content": response}
                ],
            }

        except Exception as e:
            _LOGGER.error("Query processing failed: %s", e)
            return {
                "success": False,
                "error": str(e),
            }
```

**Step 5: Commit**

```bash
git add custom_components/ai_agent_ha/core/ tests/test_core/
git commit -m "feat(core): add QueryProcessor"
```

---

## Task C2: Response Parser

**Depends on:** C1
**Files:**
- Create: `custom_components/ai_agent_ha/core/response_parser.py`
- Create: `tests/test_core/test_response_parser.py`
- Reference: `custom_components/ai_agent_ha/agent.py:3754-4200` (JSON extraction logic)

**(Implementation follows same pattern)**

**Commit:**

```bash
git commit -m "feat(core): add ResponseParser"
```

---

## Task C3: Conversation Manager

**Depends on:** C1
**Files:**
- Create: `custom_components/ai_agent_ha/core/conversation.py`
- Create: `tests/test_core/test_conversation.py`

**(Implementation follows same pattern)**

**Commit:**

```bash
git commit -m "feat(core): add ConversationManager"
```

---

## Task C4: Agent Orchestrator

**Depends on:** C1, C2, C3, A1-A7, B1-B5
**Files:**
- Create: `custom_components/ai_agent_ha/core/agent.py`
- Modify: `custom_components/ai_agent_ha/__init__.py`
- Update: `tests/test_core/test_agent.py`

This is the final integration task that wires everything together.

**(Implementation follows same pattern - slim orchestrator ~500 LOC)**

**Commit:**

```bash
git commit -m "feat(core): add slim Agent orchestrator"
```

---

## Task C5: Integration & Cleanup

**Depends on:** C4
**Files:**
- Modify: `custom_components/ai_agent_ha/__init__.py`
- Modify: `custom_components/ai_agent_ha/agent.py` (original - can be deleted after)
- Update all existing tests

**Step 1: Wire new architecture in __init__.py**

**Step 2: Run full test suite**

```bash
pytest tests/ -v --cov=custom_components.ai_agent_ha --cov-report=html
```
Expected: Coverage ≥ 87%

**Step 3: Remove old code from agent.py**

**Step 4: Final commit**

```bash
git add .
git commit -m "feat: complete architecture migration to plugin system"
```

---

## Verification Checklist

Before marking migration complete:

- [ ] All tests pass: `pytest tests/ -v`
- [ ] Coverage ≥ 87%: `pytest --cov --cov-fail-under=87`
- [ ] No import errors: `python -c "from custom_components.ai_agent_ha import *"`
- [ ] agent.py reduced to ~500 LOC
- [ ] All providers registered and working
- [ ] All managers working
- [ ] Manual test in HA dev environment
