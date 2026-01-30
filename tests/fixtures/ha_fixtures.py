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
