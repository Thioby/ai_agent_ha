# Integration Tests Rewrite Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rewrite mock-heavy tests to use real Home Assistant fixtures from `pytest-homeassistant-custom-component`

**Architecture:** Replace `mock_hass = MagicMock()` with real `hass` fixture, use HA test utilities for proper integration testing

**Tech Stack:** pytest, pytest-homeassistant-custom-component, pytest-asyncio

---

## Background

### Current Problem
Tests use `MagicMock()` for Home Assistant instance which:
1. Accepts any method call (e.g., `rag.query()` works even though method doesn't exist)
2. Doesn't validate actual HA behavior
3. Misses integration bugs between components

### Solution
Use `pytest-homeassistant-custom-component` which provides:
- Real `hass` fixture with working state machine
- `MockConfigEntry` for config entry testing
- `async_mock_service` for service testing
- Registry mocks (`mock_device_registry`, `mock_area_registry`)

### Reference Implementation
`tests/test_tools.py` already does this correctly:
```python
async def test_execute_success(self, tool, hass):
    hass.states.async_set("light.living_room", "on", {"brightness": 255})
    result = await tool.execute(entity_id="light.living_room")
    assert result.success is True
```

---

## Files to Rewrite (Priority Order)

### Priority 1: Core Agent Tests (Critical Path)
| File | mock_hass count | Impact |
|------|-----------------|--------|
| `tests/test_agent_compat.py` | 1 fixture | RAG integration bug missed |
| `tests/test_ai_agent_ha/test_init.py` | 20+ | Setup/teardown bugs |
| `tests/test_ai_agent_ha/test_integration.py` | 1 fixture | "Integration" tests that don't integrate |

### Priority 2: Manager Tests
| File | mock_hass count |
|------|-----------------|
| `tests/test_managers/test_entity_manager.py` | 1 |
| `tests/test_managers/test_registry_manager.py` | 2 |
| `tests/test_managers/test_control_manager.py` | 1 |
| `tests/test_managers/test_automation_manager.py` | 1 |
| `tests/test_managers/test_dashboard_manager.py` | 1 |

### Priority 3: Provider Tests
| File | mock_hass count |
|------|-----------------|
| `tests/test_providers/test_gemini.py` | 19 |
| `tests/test_providers/test_anthropic.py` | 16 |
| `tests/test_providers/test_openai.py` | 9 |
| `tests/test_providers/test_groq.py` | 9 |
| `tests/test_providers/test_openrouter.py` | 10 |
| `tests/test_providers/test_local.py` | 14 |
| `tests/test_providers/test_registry.py` | 5 |
| `tests/test_providers/test_base_client.py` | 7 |

### Priority 4: Other Tests
| File | mock_hass count |
|------|-----------------|
| `tests/test_storage.py` | 5 |
| `tests/test_chat_integration.py` | 1 |
| `tests/test_rag/test_event_handlers.py` | 1 |
| `tests/test_rag/test_rag_init.py` | 1 (already wraps hass) |

---

## Task 1: Create Shared Test Fixtures

**Files:**
- Create: `tests/fixtures/__init__.py`
- Create: `tests/fixtures/ha_fixtures.py`
- Modify: `tests/conftest.py`

**Step 1: Create fixtures module**

```python
# tests/fixtures/__init__.py
"""Shared test fixtures for AI Agent HA."""
from .ha_fixtures import *
```

**Step 2: Create HA fixtures**

```python
# tests/fixtures/ha_fixtures.py
"""Home Assistant integration test fixtures."""
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch

from homeassistant.core import HomeAssistant
from homeassistant.config_entries import ConfigEntry
from pytest_homeassistant_custom_component.common import (
    MockConfigEntry,
    async_mock_service,
    mock_area_registry,
    mock_device_registry,
)


@pytest.fixture
def ai_agent_config_entry() -> MockConfigEntry:
    """Create a mock config entry for AI Agent HA."""
    return MockConfigEntry(
        domain="ai_agent_ha",
        title="AI Agent HA",
        data={
            "ai_provider": "openai",
            "openai_token": "sk-test-token",
            "models": {"openai": "gpt-4"},
            "rag_enabled": False,
        },
        entry_id="test_entry_id",
        version=1,
    )


@pytest.fixture
def ai_agent_config_entry_with_rag() -> MockConfigEntry:
    """Create a mock config entry with RAG enabled."""
    return MockConfigEntry(
        domain="ai_agent_ha",
        title="AI Agent HA with RAG",
        data={
            "ai_provider": "gemini",
            "gemini_token": "test-gemini-token",
            "models": {"gemini": "gemini-2.5-flash"},
            "rag_enabled": True,
        },
        entry_id="test_entry_rag",
        version=1,
    )


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response."""
    return {
        "choices": [
            {
                "message": {
                    "content": "Test response from AI",
                    "role": "assistant",
                }
            }
        ]
    }


@pytest_asyncio.fixture
async def setup_ai_agent(hass: HomeAssistant, ai_agent_config_entry: MockConfigEntry):
    """Set up AI Agent HA integration for testing."""
    ai_agent_config_entry.add_to_hass(hass)

    with patch("custom_components.ai_agent_ha.agent_compat.ProviderRegistry") as mock_registry:
        mock_provider = AsyncMock()
        mock_provider.get_response = AsyncMock(return_value="Test response")
        mock_provider.supports_tools = True
        mock_registry.create.return_value = mock_provider

        await hass.config_entries.async_setup(ai_agent_config_entry.entry_id)
        await hass.async_block_till_done()

        yield {
            "entry": ai_agent_config_entry,
            "provider": mock_provider,
        }


@pytest.fixture
def mock_entities(hass: HomeAssistant):
    """Set up common test entities."""
    hass.states.async_set("light.living_room", "on", {
        "brightness": 255,
        "friendly_name": "Living Room Light",
    })
    hass.states.async_set("light.bedroom", "off", {
        "friendly_name": "Bedroom Light",
    })
    hass.states.async_set("sensor.temperature", "22.5", {
        "unit_of_measurement": "°C",
        "friendly_name": "Temperature Sensor",
        "device_class": "temperature",
    })
    hass.states.async_set("switch.fan", "off", {
        "friendly_name": "Fan Switch",
    })
    return hass
```

**Step 3: Update conftest.py**

```python
# Add to tests/conftest.py
from tests.fixtures import *
```

**Step 4: Run tests to verify fixtures work**

Run: `uv run pytest tests/fixtures/ -v`
Expected: No errors (module imports correctly)

**Step 5: Commit**

```bash
git add tests/fixtures/ tests/conftest.py
git commit -m "feat(tests): add shared HA integration test fixtures"
```

---

## Task 2: Rewrite test_agent_compat.py

**Files:**
- Modify: `tests/test_agent_compat.py`

**Step 1: Replace mock_hass fixture with real hass**

Replace:
```python
@pytest.fixture
def mock_hass(self):
    return MagicMock()
```

With:
```python
# Use hass fixture from conftest.py (provided by pytest-homeassistant-custom-component)
# No local fixture needed
```

**Step 2: Update test class to use real hass**

```python
class TestAiAgentHaAgentCompat:
    """Test compatibility wrapper with real HA fixtures."""

    @pytest.fixture
    def config(self):
        return {
            "ai_provider": "openai",
            "openai_token": "sk-test",
            "models": {"openai": "gpt-4"},
        }

    @pytest.fixture
    def patch_managers(self):
        """Patch managers - still needed as they have external deps."""
        with patch("custom_components.ai_agent_ha.managers.entity_manager.EntityManager"), \
             patch("custom_components.ai_agent_ha.managers.registry_manager.RegistryManager"), \
             patch("custom_components.ai_agent_ha.managers.automation_manager.AutomationManager"), \
             patch("custom_components.ai_agent_ha.managers.dashboard_manager.DashboardManager"), \
             patch("custom_components.ai_agent_ha.managers.control_manager.ControlManager"):
            yield

    @patch("custom_components.ai_agent_ha.agent_compat.ProviderRegistry")
    def test_init_creates_agent(self, mock_registry, patch_managers, hass, config):
        """Test initialization with real hass."""
        mock_provider = MagicMock()
        mock_registry.create.return_value = mock_provider

        agent = AiAgentHaAgent(hass, config)

        assert agent.hass == hass
        assert agent._provider == mock_provider
```

**Step 3: Add integration test for RAG context**

```python
@pytest.mark.asyncio
@patch("custom_components.ai_agent_ha.agent_compat.ProviderRegistry")
async def test_rag_context_integration(self, mock_registry, patch_managers, hass, config):
    """Test RAG context flows through to query processor."""
    mock_registry.create.return_value = MagicMock()
    agent = AiAgentHaAgent(hass, config)

    # Create mock RAG manager that behaves like real one
    mock_rag = MagicMock()
    mock_rag.get_relevant_context = AsyncMock(return_value="Light: bedroom is OFF")
    agent.set_rag_manager(mock_rag)

    # Set up real entity state
    hass.states.async_set("light.bedroom", "off")

    # Verify RAG context is retrieved
    context = await agent._get_rag_context("what is the bedroom light status?")

    assert context == "Light: bedroom is OFF"
    mock_rag.get_relevant_context.assert_called_once()
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_agent_compat.py -v`
Expected: All tests pass

**Step 5: Commit**

```bash
git add tests/test_agent_compat.py
git commit -m "refactor(tests): use real hass fixture in test_agent_compat"
```

---

## Task 3: Rewrite test_ai_agent_ha/test_init.py

**Files:**
- Modify: `tests/test_ai_agent_ha/test_init.py`

**Step 1: Remove mock_hass fixtures from test classes**

Find all `def mock_hass` fixtures and remove them.

**Step 2: Update TestAsyncSetupEntry to use real hass**

```python
class TestAsyncSetupEntry:
    """Test async_setup_entry with real HA fixtures."""

    @pytest.mark.asyncio
    async def test_setup_entry_success(self, hass, ai_agent_config_entry):
        """Test successful setup with real hass."""
        ai_agent_config_entry.add_to_hass(hass)

        with patch("custom_components.ai_agent_ha.agent_compat.AiAgentHaAgent") as mock_agent:
            mock_agent.return_value = MagicMock()

            result = await async_setup_entry(hass, ai_agent_config_entry)

            assert result is True
            assert DOMAIN in hass.data
            assert "agents" in hass.data[DOMAIN]
```

**Step 3: Update service handler tests**

```python
@pytest.mark.asyncio
async def test_query_service(self, hass, setup_ai_agent):
    """Test query service with real service registry."""
    # Service should be registered
    assert hass.services.has_service(DOMAIN, "query")

    # Call service
    await hass.services.async_call(
        DOMAIN,
        "query",
        {"query": "What is the status?"},
        blocking=True,
    )
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_ai_agent_ha/test_init.py -v`
Expected: All tests pass

**Step 5: Commit**

```bash
git add tests/test_ai_agent_ha/test_init.py
git commit -m "refactor(tests): use real hass fixture in test_init"
```

---

## Task 4: Rewrite test_ai_agent_ha/test_integration.py

**Files:**
- Modify: `tests/test_ai_agent_ha/test_integration.py`

**Step 1: Remove mock_hass and use real fixtures**

**Step 2: Write actual integration tests**

```python
class TestIntegration:
    """Real integration tests for AI Agent HA."""

    @pytest.mark.asyncio
    async def test_full_setup_and_query(self, hass, ai_agent_config_entry):
        """Test complete flow: setup -> query -> response."""
        ai_agent_config_entry.add_to_hass(hass)

        with patch("aiohttp.ClientSession") as mock_session:
            # Mock external API call
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={
                "choices": [{"message": {"content": "The light is on"}}]
            })
            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response

            # Setup integration
            await hass.config_entries.async_setup(ai_agent_config_entry.entry_id)
            await hass.async_block_till_done()

            # Set up test entity
            hass.states.async_set("light.test", "on")

            # Call query service
            await hass.services.async_call(
                DOMAIN, "query",
                {"query": "Is the light on?"},
                blocking=True,
            )

            # Verify response event was fired
            # (capture with async_capture_events)
```

**Step 3: Run tests**

Run: `uv run pytest tests/test_ai_agent_ha/test_integration.py -v`
Expected: All tests pass

**Step 4: Commit**

```bash
git add tests/test_ai_agent_ha/test_integration.py
git commit -m "refactor(tests): rewrite integration tests with real HA"
```

---

## Task 5: Rewrite Manager Tests

**Files:**
- Modify: `tests/test_managers/test_entity_manager.py`
- Modify: `tests/test_managers/test_registry_manager.py`
- Modify: `tests/test_managers/test_control_manager.py`

**Step 1: Update entity_manager tests**

```python
class TestEntityManager:
    """Test EntityManager with real hass."""

    @pytest.mark.asyncio
    async def test_get_entity_state(self, hass):
        """Test getting entity state from real state machine."""
        hass.states.async_set("light.test", "on", {"brightness": 200})

        manager = EntityManager(hass)
        state = manager.get_entity_state("light.test")

        assert state is not None
        assert state["state"] == "on"
        assert state["attributes"]["brightness"] == 200
```

**Step 2: Update registry_manager tests with mock_device_registry**

```python
from pytest_homeassistant_custom_component.common import mock_device_registry

async def test_get_devices(self, hass):
    """Test getting devices from registry."""
    device_registry = mock_device_registry(hass)
    device_registry.async_get_or_create(
        config_entry_id="test",
        identifiers={("test", "device1")},
        name="Test Device",
    )

    manager = RegistryManager(hass)
    devices = manager.get_devices()

    assert len(devices) >= 1
```

**Step 3: Update control_manager tests with async_mock_service**

```python
from pytest_homeassistant_custom_component.common import async_mock_service

async def test_call_service(self, hass):
    """Test calling service."""
    calls = async_mock_service(hass, "light", "turn_on")

    manager = ControlManager(hass)
    result = await manager.call_service("light", "turn_on", entity_id="light.test")

    assert len(calls) == 1
    assert calls[0].data["entity_id"] == "light.test"
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_managers/ -v`
Expected: All tests pass

**Step 5: Commit**

```bash
git add tests/test_managers/
git commit -m "refactor(tests): use real hass in manager tests"
```

---

## Task 6: Rewrite Provider Tests

**Files:**
- Modify: `tests/test_providers/test_*.py`

**Note:** Provider tests mostly test HTTP calls to external APIs. The `hass` fixture is needed for:
- Config entry access
- OAuth token refresh
- Storing state

**Step 1: Create provider test base class**

```python
# tests/test_providers/conftest.py
import pytest
from unittest.mock import AsyncMock, patch

@pytest.fixture
def mock_aiohttp_response():
    """Create mock aiohttp response."""
    def _create(status=200, json_data=None, text_data=None):
        response = AsyncMock()
        response.status = status
        if json_data:
            response.json = AsyncMock(return_value=json_data)
        if text_data:
            response.text = AsyncMock(return_value=text_data)
        return response
    return _create
```

**Step 2: Update each provider test file**

Replace `mock_hass = MagicMock()` with `hass` fixture parameter.

**Step 3: Run tests**

Run: `uv run pytest tests/test_providers/ -v`
Expected: All tests pass

**Step 4: Commit**

```bash
git add tests/test_providers/
git commit -m "refactor(tests): use real hass in provider tests"
```

---

## Task 7: Final Cleanup and Verification

**Step 1: Search for remaining mock_hass**

Run: `grep -r "mock_hass.*MagicMock\|def mock_hass" tests/`
Expected: No matches (or only intentional ones with comments)

**Step 2: Run full test suite**

Run: `uv run pytest tests/ -v --tb=short`
Expected: All tests pass

**Step 3: Run with coverage**

Run: `uv run pytest tests/ --cov=custom_components/ai_agent_ha --cov-report=term-missing`
Expected: Coverage remains similar or improves

**Step 4: Commit**

```bash
git add .
git commit -m "refactor(tests): complete migration to real HA fixtures"
```

---

## Summary

### What Changes
| Before | After |
|--------|-------|
| `mock_hass = MagicMock()` | `hass` fixture from pytest-homeassistant-custom-component |
| Manual service mocking | `async_mock_service()` |
| No state validation | Real `hass.states.async_set/get` |
| MagicMock accepts anything | Real interfaces catch mismatches |

### Benefits
1. **Catches interface bugs** - Like `rag.query()` vs `rag.get_relevant_context()`
2. **Tests real behavior** - State machine, service registry, event bus
3. **Better confidence** - Tests prove integration actually works
4. **Matches production** - Uses same HA core as production

### Risks
1. **Slower tests** - Real fixtures have overhead (mitigate with pytest-xdist)
2. **More complex setup** - Need proper fixtures (mitigate with shared fixtures)
3. **Breaking changes** - HA updates may break tests (but that's good - catches issues)
