---
**STATUS: COMPLETED (2026-01-29)**

**Final Results:**
- Old agent.py: 5,008 LOC -> REMOVED
- New architecture: 3,083 LOC (modular)
- Test coverage: 92% on new code
- All 1,513 tests passing
- 6 AI providers available
---

# AI Agent HA - Architecture Redesign Proposal

**Date:** 2026-01-29
**Current Test Coverage:** ~87%
**Priority Order:** Extensibility > Maintainability > Performance
**Status:** Validated by Gemini (2026-01-29)

---

## Executive Summary

Projekt `ai_agent_ha` wymaga kompleksowego redesignu architektury. Główne problemy to God Class (5008 LOC), Mega Method (1182 LOC), oraz duplikacja kodu w klientach AI.

**Kluczowe założenie:** To jest plugin Home Assistant - coupling do `hass` jest naturalny i oczekiwany. Nie abstrakcujemy HA, tylko poprawiamy wewnętrzną organizację kodu.

---

## Current State Analysis

### Project Structure

```
ai_agent_ha/
├── custom_components/ai_agent_ha/
│   ├── __init__.py                   # Setup i entry point (533 LOC)
│   ├── agent.py                      # GŁÓWNY PROBLEM (5008 LOC) ⚠️
│   ├── config_flow.py                # Konfiguracja (797 LOC)
│   ├── websocket_api.py              # WebSocket API (370 LOC)
│   ├── storage.py                    # Sesje i chat storage (444 LOC)
│   ├── function_calling.py           # Konwertery narzędzi (452 LOC)
│   ├── error_handler.py              # Obsługa błędów (222 LOC)
│   ├── dashboard_templates.py        # Szablony dashboardów (390 LOC)
│   ├── prompts.py                    # System prompts (193 LOC)
│   ├── const.py                      # Konstante (51 LOC)
│   ├── oauth.py                      # OAuth2 (149 LOC)
│   ├── gemini_oauth.py               # Gemini OAuth (200 LOC)
│   ├── tools/                        # Narzędzia dla agenta (dobrze zaprojektowane ✓)
│   │   ├── base.py                   # Abstrakcja narzędzi (494 LOC)
│   │   ├── ha_native.py              # Home Assistant tools (696 LOC)
│   │   ├── webfetch.py               # Web fetch tool (450 LOC)
│   │   ├── websearch.py              # Web search tool (380 LOC)
│   │   └── context7.py               # Documentation tool (356 LOC)
│   └── rag/                          # RAG system (dobrze zaprojektowane ✓)
│       ├── __init__.py               # RAGManager facade
│       ├── sqlite_store.py           # SQLite storage
│       ├── embeddings.py             # Embeddings
│       ├── entity_indexer.py         # Indeksowanie
│       ├── query_engine.py           # Query processing
│       ├── semantic_learner.py       # Learning
│       └── event_handlers.py         # Event handling
├── tests/                            # 18,740 LOC testów
└── docs/
```

### Critical Issues Identified

| Problem | Severity | Location | Impact |
|---------|----------|----------|--------|
| **God Class** - AiAgentHaAgent | 🔴 CRITICAL | agent.py (5008 LOC, 77 methods) | Impossible to extend without touching core |
| **Mega Method** - process_query() | 🔴 CRITICAL | agent.py:3451-4633 (1182 LOC) | Hard to test, maintain, debug |
| **Code Duplication** in AI clients | 🟠 HIGH | 9 client classes with similar logic | Adding new provider requires copy-paste |
| **Hardcoded Values** | 🟠 HIGH | URLs, models, timeouts scattered | Configuration changes require code changes |
| **No Layer Separation** | 🟡 MEDIUM | All layers in one file | Business logic mixed with AI client code |

**NOT a problem:** Coupling do Home Assistant - to plugin HA, bezpośrednie użycie `hass` jest prawidłowe.

### SOLID Violations

#### Single Responsibility (VIOLATED)
`AiAgentHaAgent` handles: chat processing, entity queries, automation management, dashboard management, RAG integration, caching, rate limiting - all in one class.

#### Open/Closed (VIOLATED)
```python
# agent.py:3494-3539 - Adding new provider requires modifying this dict
provider_config = {
    "openai": {"token_key": "openai_token", "client_class": OpenAIClient},
    "gemini": {"token_key": "gemini_token", "client_class": GeminiClient},
    # ... hardcoded list
}
```

#### Dependency Inversion (VIOLATED)
```python
# agent.py:1664-1704 - Direct instantiation instead of factory
if provider == "openai":
    self.ai_client = OpenAIClient(config.get("openai_token"), model)
elif provider == "gemini":
    self.ai_client = GeminiClient(config.get("gemini_token"), model)
```

### Code Duplication Example (AI Clients)

Each AI client duplicates: HTTP request building, error handling, response parsing, timeout configuration.

```python
# OpenAI (agent.py:504-568)
async def get_response(self, messages, **kwargs):
    headers = {"Authorization": f"Bearer {self.token}"}
    payload = {"model": self.model, "messages": messages}
    async with session.post(...) as resp:
        return data["choices"][0]["message"]["content"]

# Gemini (agent.py:607-730) - Same pattern, different extraction path
async def get_response(self, messages, **kwargs):
    headers = {"x-goog-api-key": self.token}
    payload = {"contents": [...]}
    async with session.post(...) as resp:
        return data["candidates"][0]["content"]["parts"][0]["text"]

# Anthropic (agent.py:763-839) - Same pattern, different extraction path
async def get_response(self, messages, **kwargs):
    headers = {"x-api-key": self.token}
    payload = {"model": self.model, "messages": messages}
    async with session.post(...) as resp:
        return data["content"][0]["text"]
```

---

## Proposed Architecture: Pragmatic Plugin Architecture

### Design Principles

1. **HA-native:** Używamy `hass` bezpośrednio - to plugin HA, nie abstrakcujemy frameworka
2. **Plugin system dla AI providers:** Tu jest główna wartość extensibility
3. **Managers dla logiki domenowej:** Rozbicie God Class na mniejsze klasy
4. **Config Flow dla konfiguracji:** Brak plików YAML - wszystko przez UI HA
5. **HA testing framework:** `pytest-homeassistant-custom-component` dla pewności

### Target Structure

```
custom_components/ai_agent_ha/
├── __init__.py                   # Setup, async_setup_entry (używa hass bezpośrednio)
├── const.py                      # Stałe
├── config_flow.py                # UI configuration (rozbudowany o provider options)
│
├── core/
│   ├── __init__.py
│   ├── agent.py                  # Slim orchestrator (~500 LOC)
│   ├── query_processor.py        # Extracted from process_query()
│   ├── response_parser.py        # JSON/text extraction logic
│   └── conversation.py           # Conversation history management
│
├── providers/                    # AI providers - PLUGIN SYSTEM
│   ├── __init__.py
│   ├── registry.py               # Provider registration & factory
│   ├── base_client.py            # Abstract base + shared HTTP logic
│   ├── openai.py                 # OpenAI provider
│   ├── gemini.py                 # Gemini provider
│   ├── anthropic.py              # Anthropic provider
│   ├── openrouter.py             # OpenRouter provider
│   ├── groq.py                   # Groq provider
│   └── local.py                  # Local/Ollama provider
│
├── managers/                     # Domain logic (nie "services" - konflikt z HA)
│   ├── __init__.py
│   ├── entity_manager.py         # get_entity_state, get_entities_by_*
│   ├── registry_manager.py       # entity/device/area registry operations
│   ├── automation_manager.py     # create/validate automations
│   ├── dashboard_manager.py      # create/update dashboards
│   └── control_manager.py        # set_entity_state, call_service
│
├── tools/                        # Already well-designed ✓
│   ├── __init__.py
│   ├── base.py
│   ├── ha_native.py
│   ├── webfetch.py
│   ├── websearch.py
│   └── context7.py
│
├── rag/                          # Already well-designed ✓
│   └── (bez zmian)
│
├── websocket_api.py              # WebSocket API
├── storage.py                    # Session storage
├── function_calling.py           # Tool converters
├── error_handler.py              # Error handling
├── dashboard_templates.py        # Dashboard templates
├── prompts.py                    # System prompts
├── oauth.py                      # OAuth2
└── gemini_oauth.py               # Gemini OAuth
```

### Key Design Patterns

| Pattern | Where | Purpose |
|---------|-------|---------|
| **Registry Pattern** | `providers/registry.py` | Dynamic provider registration |
| **Strategy Pattern** | `providers/*.py` | Interchangeable AI clients |
| **Template Method** | `providers/base_client.py` | Shared HTTP logic with hooks |
| **Facade Pattern** | `managers/*.py` | Simplified interface to HA operations |
| **Factory Pattern** | `ProviderRegistry.create()` | Provider instantiation |

### Provider Registry Implementation

```python
# providers/registry.py
from typing import Dict, Type
from abc import ABC, abstractmethod
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession

class AIProvider(ABC):
    """Base interface for all AI providers."""

    def __init__(self, hass: HomeAssistant, config: dict):
        self.hass = hass
        self.config = config
        self._session = async_get_clientsession(hass)  # HA managed session

    @abstractmethod
    async def get_response(self, messages: list, **kwargs) -> str:
        """Get response from AI provider."""
        pass

    @property
    @abstractmethod
    def supports_tools(self) -> bool:
        """Whether provider supports function calling."""
        pass


class ProviderRegistry:
    """Registry for AI providers with automatic discovery."""

    _providers: Dict[str, Type[AIProvider]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a provider."""
        def decorator(provider_class: Type[AIProvider]):
            cls._providers[name] = provider_class
            return provider_class
        return decorator

    @classmethod
    def create(cls, name: str, hass: HomeAssistant, config: dict) -> AIProvider:
        """Factory method to create provider instance."""
        if name not in cls._providers:
            raise ValueError(f"Unknown provider: {name}. Available: {list(cls._providers.keys())}")
        return cls._providers[name](hass, config)

    @classmethod
    def available_providers(cls) -> list[str]:
        """List registered providers."""
        return list(cls._providers.keys())


# providers/openai.py
@ProviderRegistry.register("openai")
class OpenAIProvider(AIProvider):
    """OpenAI API provider."""

    API_URL = "https://api.openai.com/v1/chat/completions"

    @property
    def supports_tools(self) -> bool:
        return True

    async def get_response(self, messages: list, **kwargs) -> str:
        headers = {
            "Authorization": f"Bearer {self.config['token']}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.config.get("model", "gpt-4"),
            "messages": messages,
            **kwargs
        }

        async with self._session.post(self.API_URL, json=payload, headers=headers) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return data["choices"][0]["message"]["content"]
```

### Adding New Provider (Zero Core Changes)

```python
# providers/new_provider.py
from .registry import ProviderRegistry, AIProvider

@ProviderRegistry.register("new_provider")
class NewProvider(AIProvider):
    """New AI provider implementation."""

    API_URL = "https://api.newprovider.com/v1/chat"

    @property
    def supports_tools(self) -> bool:
        return False

    async def get_response(self, messages: list, **kwargs) -> str:
        # Implementation
        pass
```

No changes to `agent.py`, `config_flow.py`, or any other file. Just add new file and it's available.

### Manager Example (Entity Operations)

```python
# managers/entity_manager.py
from homeassistant.core import HomeAssistant, State
from homeassistant.helpers import entity_registry as er

class EntityManager:
    """Manages entity-related operations."""

    def __init__(self, hass: HomeAssistant):
        self.hass = hass  # Direct HA access - this is a HA plugin

    async def get_entity_state(self, entity_id: str) -> dict | None:
        """Get current state of an entity."""
        state: State | None = self.hass.states.get(entity_id)
        if state is None:
            return None
        return {
            "entity_id": entity_id,
            "state": state.state,
            "attributes": dict(state.attributes),
            "last_changed": state.last_changed.isoformat(),
        }

    async def get_entities_by_domain(self, domain: str) -> list[dict]:
        """Get all entities for a domain."""
        entities = []
        for entity_id, state in self.hass.states.async_all(domain):
            entities.append({
                "entity_id": entity_id,
                "state": state.state,
                "friendly_name": state.attributes.get("friendly_name"),
            })
        return entities

    async def get_entities_by_area(self, area_id: str) -> list[dict]:
        """Get all entities in an area."""
        registry = er.async_get(self.hass)
        entities = []
        for entry in registry.entities.values():
            if entry.area_id == area_id:
                state = self.hass.states.get(entry.entity_id)
                if state:
                    entities.append({
                        "entity_id": entry.entity_id,
                        "state": state.state,
                    })
        return entities
```

---

## Testing Strategy

### Home Assistant Testing Framework

Używamy oficjalnego pakietu `pytest-homeassistant-custom-component` który zapewnia:

- **Mock `hass` instance** - pełna symulacja Home Assistant
- **Config Entry fixtures** - testowanie config flow
- **Entity/Device/Area registry mocks** - testowanie operacji na rejestrach
- **Async test support** - prawidłowe testowanie async kodu

### Test Structure

```
tests/
├── conftest.py                   # HA fixtures, common mocks
├── test_providers/
│   ├── test_registry.py          # Provider registration tests
│   ├── test_openai.py            # OpenAI provider tests
│   ├── test_gemini.py            # Gemini provider tests
│   └── test_base_client.py       # Base client tests
├── test_managers/
│   ├── test_entity_manager.py    # Entity operations tests
│   ├── test_automation_manager.py
│   └── test_dashboard_manager.py
├── test_core/
│   ├── test_agent.py             # Orchestrator tests
│   ├── test_query_processor.py   # Query processing tests
│   └── test_response_parser.py   # Response parsing tests
├── test_integration/
│   ├── test_init.py              # Integration setup tests
│   ├── test_config_flow.py       # Config flow tests
│   └── test_websocket_api.py     # WebSocket tests
└── test_tools/                   # Existing tool tests
```

### Example Test with HA Fixtures

```python
# tests/test_managers/test_entity_manager.py
import pytest
from homeassistant.core import HomeAssistant
from homeassistant.const import STATE_ON, STATE_OFF

from custom_components.ai_agent_ha.managers.entity_manager import EntityManager


@pytest.fixture
def entity_manager(hass: HomeAssistant) -> EntityManager:
    """Create EntityManager with mocked hass."""
    return EntityManager(hass)


async def test_get_entity_state(hass: HomeAssistant, entity_manager: EntityManager):
    """Test getting entity state."""
    # Setup - use HA test helpers
    hass.states.async_set("light.living_room", STATE_ON, {"brightness": 255})

    # Execute
    result = await entity_manager.get_entity_state("light.living_room")

    # Assert
    assert result is not None
    assert result["state"] == STATE_ON
    assert result["attributes"]["brightness"] == 255


async def test_get_entity_state_not_found(entity_manager: EntityManager):
    """Test getting non-existent entity."""
    result = await entity_manager.get_entity_state("light.nonexistent")
    assert result is None


async def test_get_entities_by_domain(hass: HomeAssistant, entity_manager: EntityManager):
    """Test getting entities by domain."""
    # Setup
    hass.states.async_set("light.one", STATE_ON)
    hass.states.async_set("light.two", STATE_OFF)
    hass.states.async_set("switch.three", STATE_ON)  # Different domain

    # Execute
    result = await entity_manager.get_entities_by_domain("light")

    # Assert
    assert len(result) == 2
    entity_ids = [e["entity_id"] for e in result]
    assert "light.one" in entity_ids
    assert "light.two" in entity_ids
    assert "switch.three" not in entity_ids
```

### Testing Dependencies

```
# requirements_test.txt
pytest>=8.0.0
pytest-asyncio>=0.23.0
pytest-homeassistant-custom-component>=0.13.0
pytest-cov>=4.0.0
aioresponses>=0.7.6  # For mocking HTTP requests to AI providers
```

---

## Migration Strategy: Strangler Fig Pattern

### Principle

Zamiast "Big Bang" refactor, stopniowo owijamy stary kod nowym, przenosząc funkcjonalność kawałek po kawałku. Testy migrujemy równolegle z kodem.

### Phase 1: Foundation + First Provider (Week 1)

**Goal:** Infrastruktura + jeden działający provider (OpenAI)

**Tasks:**
1. Create `providers/` directory structure
2. Implement `ProviderRegistry` and `AIProvider` ABC
3. Extract `OpenAIProvider` from `agent.py`
4. Create tests for `OpenAIProvider` using `aioresponses`
5. Modify `agent.py` to use registry for OpenAI only (feature flag)

**Validation:**
```bash
pytest tests/test_providers/test_openai.py -v
pytest tests/test_agent.py -v  # Existing tests still pass
```

### Phase 2: Remaining Providers (Week 2)

**Goal:** All AI providers migrated to plugin system

**Tasks:**
1. Extract `GeminiProvider`
2. Extract `AnthropicProvider`
3. Extract `OpenRouterProvider`
4. Extract `GroqProvider`
5. Extract `LocalProvider`
6. Create shared `BaseHTTPClient` for common logic
7. Tests for each provider

**Validation:**
```bash
pytest tests/test_providers/ -v
pytest tests/ -v  # Full test suite passes
```

### Phase 3: Managers Extraction (Week 3)

**Goal:** Extract domain logic from God Class

**Tasks:**
1. Create `managers/entity_manager.py` - extract entity operations
2. Create `managers/registry_manager.py` - extract registry operations
3. Create `managers/automation_manager.py` - extract automation logic
4. Create `managers/dashboard_manager.py` - extract dashboard logic
5. Create `managers/control_manager.py` - extract control operations
6. Tests for each manager

**Validation:**
```bash
pytest tests/test_managers/ -v
pytest tests/ -v
```

### Phase 4: Core Refactor (Week 4)

**Goal:** Slim down agent.py to orchestrator

**Tasks:**
1. Create `core/query_processor.py` - extract from `process_query()`
2. Create `core/response_parser.py` - extract JSON/text parsing
3. Create `core/conversation.py` - extract conversation management
4. Refactor `agent.py` to ~500 LOC orchestrator
5. Update integration tests

**Validation:**
```bash
pytest tests/test_core/ -v
pytest tests/ -v
# Manual testing in HA dev environment
```

### Phase 5: Cleanup & Documentation (Week 5)

**Goal:** Remove old code, optimize, document

**Tasks:**
1. Remove dead code from `agent.py`
2. Performance optimization (lazy loading)
3. Update documentation
4. Final test coverage check (target: 90%+)

**Validation:**
```bash
pytest tests/ -v --cov=custom_components.ai_agent_ha --cov-report=html
# Coverage should be >= 87% (current) ideally 90%+
```

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Regression during migration | Tests migrated with code, CI runs on every PR |
| Breaking HA integration | Use `pytest-homeassistant-custom-component` fixtures |
| Performance degradation | Lazy loading, profile before/after |
| Incomplete migration | Strangler Fig allows partial states to work |

### CI Pipeline

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - name: Install dependencies
        run: |
          pip install -r requirements_test.txt
          pip install -e .
      - name: Run tests
        run: pytest tests/ -v --cov --cov-fail-under=87
```

---

## Success Criteria

| Metric | Current | Target |
|--------|---------|--------|
| `agent.py` LOC | 5008 | ~500 |
| `process_query()` LOC | 1182 | ~100 (delegating) |
| Test coverage | 87% | 90%+ |
| Time to add new provider | ~2h (copy-paste) | ~30min (implement interface) |
| AI client code duplication | High | Minimal (shared base) |

---

## Appendix: Code Metrics

| File | Current LOC | Target LOC |
|------|-------------|------------|
| agent.py | 5008 | ~500 |
| config_flow.py | 797 | ~800 (minor changes) |
| tools/base.py | 494 | ~500 (unchanged) |
| storage.py | 444 | ~450 (unchanged) |
| **New: providers/** | 0 | ~1500 |
| **New: managers/** | 0 | ~1200 |
| **New: core/** | 0 | ~800 |
| tests/ | 18,740 | ~20,000 |

---

## Gemini Validation Summary (2026-01-29)

**Verdict:** Approved with corrections

**Key feedback incorporated:**
- ✅ Removed `ha_adapter.py` - HA coupling is expected
- ✅ Renamed `services/` to `managers/` - avoid HA naming conflict
- ✅ Removed `providers.yaml` - use Config Flow instead
- ✅ Added HA testing framework (`pytest-homeassistant-custom-component`)
- ✅ Changed to Strangler Fig migration pattern
- ✅ Tests migrated in parallel with code, not at the end
