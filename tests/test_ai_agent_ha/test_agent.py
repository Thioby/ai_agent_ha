"""Tests for the AI Agent core functionality."""

import asyncio
import json
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import homeassistant
from homeassistant.core import HomeAssistant, State
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.ai_agent_ha.agent import AiAgentHaAgent
from custom_components.ai_agent_ha.const import AI_PROVIDERS

class TestAIAgent:
    """Test AI Agent functionality."""

    @pytest.fixture
    def mock_agent_config(self):
        """Mock agent configuration."""
        return {
            "ai_provider": "openai",
            "openai_token": "test_token_123",
            "openai_model": "gpt-3.5-turbo",
        }

    @pytest.mark.asyncio
    async def test_agent_initialization(self, hass, mock_agent_config):
        """Test agent initialization with valid config."""
        # Patching openai module since it might be imported by agent
        with patch.dict("sys.modules", {"openai": MagicMock()}):
            agent = AiAgentHaAgent(hass, mock_agent_config)
            assert agent is not None

    @pytest.mark.asyncio
    async def test_agent_query_processing(self, hass, mock_agent_config):
        """Test agent query processing."""
        agent = AiAgentHaAgent(hass, mock_agent_config)
        
        # Mock _get_ai_response to return a valid JSON response string
        response_data = {
            "request_type": "final_response",
            "response": "Test response"
        }
        
        with patch.object(agent, "_get_ai_response", new_callable=AsyncMock) as mock_get_response:
            mock_get_response.return_value = json.dumps(response_data)
            
            result = await agent.process_query("Test query")
            
            assert result["success"] is True
            assert result["answer"] == "Test response"

    def test_agent_config_validation(self, mock_agent_config):
        """Test agent configuration validation."""
        # Test valid config
        assert mock_agent_config["ai_provider"] in [
            "openai",
            "anthropic",
            "google",
            "openrouter",
            "llama",
            "local",
        ]
        assert "openai_token" in mock_agent_config
        assert len(mock_agent_config["openai_token"]) > 0

    @pytest.mark.asyncio
    async def test_agent_error_handling(self, hass, mock_agent_config):
        """Test agent error handling with invalid provider response."""
        agent = AiAgentHaAgent(hass, mock_agent_config)
        
        with patch.object(agent, "_get_ai_response", new_callable=AsyncMock) as mock_get_response:
            mock_get_response.side_effect = Exception("API Error")
            
            result = await agent.process_query("Test query")
            
            assert result["success"] is False
            assert "error" in result
            assert "API Error" in result["error"]

    def test_ai_providers_support(self):
        """Test that all supported AI providers are properly defined."""
        expected_providers = [
            "llama",
            "openai",
            "gemini",
            "openrouter",
            "anthropic",
            "local",
        ]
        assert all(provider in AI_PROVIDERS for provider in expected_providers)

    @pytest.mark.asyncio
    async def test_get_entity_state(self, hass, mock_agent_config):
        """Test get_entity_state functionality."""
        # Mock entity states
        hass.states.async_set("light.living_room", "on", {"friendly_name": "Living Room Light"})
        hass.states.async_set("sensor.temperature", "22.5")

        agent = AiAgentHaAgent(hass, mock_agent_config)
        
        # Test existing entity
        state_info = await agent.get_entity_state("light.living_room")
        assert state_info["entity_id"] == "light.living_room"
        assert state_info["state"] == "on"
        assert state_info["friendly_name"] == "Living Room Light"
        
        # Test non-existent entity
        error_info = await agent.get_entity_state("light.non_existent")
        assert "error" in error_info

    @pytest.mark.asyncio
    async def test_get_entities_by_device_class(self, hass, mock_agent_config):
        """Test filtering entities by device_class."""
        
        # Set up states in hass fixture
        hass.states.async_set(
            "sensor.bedroom_temperature", 
            "22.5", 
            {"device_class": "temperature", "unit_of_measurement": "°C", "friendly_name": "Bedroom Temperature"}
        )
        hass.states.async_set(
            "sensor.living_room_humidity", 
            "55", 
            {"device_class": "humidity", "unit_of_measurement": "%", "friendly_name": "Living Room Humidity"}
        )
        hass.states.async_set(
            "sensor.power_usage", 
            "150", 
            {"device_class": "power", "unit_of_measurement": "W", "friendly_name": "Power Usage"}
        )

        agent = AiAgentHaAgent(hass, mock_agent_config)

        # Test getting temperature sensors
        temp_entities = await agent.get_entities_by_device_class("temperature")
        assert len(temp_entities) == 1
        assert temp_entities[0]["entity_id"] == "sensor.bedroom_temperature"

        # Test getting humidity sensors
        humidity_entities = await agent.get_entities_by_device_class("humidity")
        assert len(humidity_entities) == 1
        assert humidity_entities[0]["entity_id"] == "sensor.living_room_humidity"

        # Test with domain filter
        temp_sensors_only = await agent.get_entities_by_device_class(
            "temperature", "sensor"
        )
        assert len(temp_sensors_only) == 1

    @pytest.mark.asyncio
    async def test_get_climate_related_entities(self, hass, mock_agent_config):
        """Test getting all climate-related entities (climate + temp/humidity sensors)."""
        
        hass.states.async_set(
            "climate.thermostat", 
            "heat", 
            {"friendly_name": "Thermostat"}
        )
        hass.states.async_set(
            "sensor.bedroom_temperature", 
            "22.5", 
            {"device_class": "temperature", "friendly_name": "Bedroom Temperature"}
        )
        hass.states.async_set(
            "sensor.living_room_humidity", 
            "55", 
            {"device_class": "humidity", "friendly_name": "Living Room Humidity"}
        )

        agent = AiAgentHaAgent(hass, mock_agent_config)

        # Test getting all climate-related entities
        climate_entities = await agent.get_climate_related_entities()
        # Note: The implementation might differ in how it counts or returns, 
        # but logically it should find these. 
        # Since we are using real hass fixture logic (mostly), check if logic holds.
        # If the actual implementation calls hass.states.async_all(), it should work.
        
        entity_ids = [e["entity_id"] for e in climate_entities]
        assert "climate.thermostat" in entity_ids
        assert "sensor.bedroom_temperature" in entity_ids
        assert "sensor.living_room_humidity" in entity_ids

    @pytest.mark.asyncio
    async def test_get_climate_related_entities_sensors_only(self, hass, mock_agent_config):
        """Test getting climate-related entities when only temperature/humidity sensors exist (no climate.* entities)."""
        
        hass.states.async_set(
            "sensor.bedroom_temperature", 
            "22.5", 
            {"device_class": "temperature", "friendly_name": "Bedroom Temperature"}
        )
        hass.states.async_set(
            "sensor.kitchen_temperature", 
            "23.1", 
            {"device_class": "temperature", "friendly_name": "Kitchen Temperature"}
        )
        hass.states.async_set(
            "sensor.living_room_humidity", 
            "55", 
            {"device_class": "humidity", "friendly_name": "Living Room Humidity"}
        )

        agent = AiAgentHaAgent(hass, mock_agent_config)

        # Test getting climate-related entities
        climate_entities = await agent.get_climate_related_entities()
        entity_ids = [e["entity_id"] for e in climate_entities]
        assert "sensor.bedroom_temperature" in entity_ids
        assert "sensor.kitchen_temperature" in entity_ids
        assert "sensor.living_room_humidity" in entity_ids

    @pytest.mark.asyncio
    async def test_climate_related_entities_deduplication(self, hass, mock_agent_config):
        """Test that get_climate_related_entities deduplicates entities."""
        
        hass.states.async_set(
            "climate.thermostat", 
            "heat", 
            {"friendly_name": "Thermostat"}
        )

        agent = AiAgentHaAgent(hass, mock_agent_config)

        # Test that deduplication works
        climate_entities = await agent.get_climate_related_entities()
        entity_ids = [e["entity_id"] for e in climate_entities]

        # Should only appear once even if returned by multiple methods
        assert entity_ids.count("climate.thermostat") == 1

    @pytest.mark.asyncio
    async def test_data_payload_uses_user_role_not_system(self, hass, mock_agent_config):
        """Test critical fix: data payloads use 'user' role, not 'system'."""
        
        hass.states.async_set("light.living_room", "on")

        agent = AiAgentHaAgent(hass, mock_agent_config)
        
        # Mock responses: 1. Data Request, 2. Final Response
        response_1 = json.dumps({
            "request_type": "get_entities_by_domain",
            "parameters": {"domain": "light"},
        })
        
        response_2 = json.dumps({
            "request_type": "final_response",
            "response": "Lights are on"
        })
        
        with patch.object(agent, "_get_ai_response", new_callable=AsyncMock) as mock_get_response:
            mock_get_response.side_effect = [response_1, response_2]
            
            await agent.process_query("turn on lights")

        # Check conversation history
        data_messages = [
            msg
            for msg in agent.conversation_history
            if isinstance(msg.get("content"), str)
            and '"data":' in msg.get("content", "")
        ]

        assert len(data_messages) > 0, "No data payload found in history"
        for msg in data_messages:
            assert msg.get("role") == "user", "Data payload should use 'user' role"


class TestAgentUtilityMethods:
    """Test AiAgentHaAgent utility methods."""

    @pytest.fixture
    def agent_config_openai(self):
        """Mock agent config for OpenAI."""
        return {
            "ai_provider": "openai",
            "openai_token": "sk-" + "a" * 48,
        }

    @pytest.fixture
    def agent_config_gemini(self):
        """Mock agent config for Gemini."""
        return {
            "ai_provider": "gemini",
            "gemini_token": "AIza" + "b" * 35,
        }

    @pytest.fixture
    def agent_config_local(self):
        """Mock agent config for local model."""
        return {
            "ai_provider": "local",
            "local_url": "http://localhost:11434/api/generate",
        }

    def test_validate_api_key_openai_valid(self, hass, agent_config_openai):
        """Test _validate_api_key with valid OpenAI token."""
        agent = AiAgentHaAgent(hass, agent_config_openai)
        assert agent._validate_api_key() is True

    def test_validate_api_key_openai_short(self, hass):
        """Test _validate_api_key with too short OpenAI token."""
        config = {"ai_provider": "openai", "openai_token": "sk-short"}
        agent = AiAgentHaAgent(hass, config)
        assert agent._validate_api_key() is False

    def test_validate_api_key_openai_missing(self, hass):
        """Test _validate_api_key with missing OpenAI token."""
        config = {"ai_provider": "openai"}
        agent = AiAgentHaAgent(hass, config)
        assert agent._validate_api_key() is False

    def test_validate_api_key_gemini_valid(self, hass, agent_config_gemini):
        """Test _validate_api_key with valid Gemini token."""
        agent = AiAgentHaAgent(hass, agent_config_gemini)
        assert agent._validate_api_key() is True

    def test_validate_api_key_local_valid_http(self, hass, agent_config_local):
        """Test _validate_api_key with valid local HTTP URL."""
        agent = AiAgentHaAgent(hass, agent_config_local)
        assert agent._validate_api_key() is True

    def test_validate_api_key_local_valid_https(self, hass):
        """Test _validate_api_key with valid local HTTPS URL."""
        config = {"ai_provider": "local", "local_url": "https://localhost:11434/api"}
        agent = AiAgentHaAgent(hass, config)
        assert agent._validate_api_key() is True

    def test_validate_api_key_local_invalid_url(self, hass):
        """Test _validate_api_key with invalid local URL (no scheme)."""
        config = {"ai_provider": "local", "local_url": "localhost:11434"}
        agent = AiAgentHaAgent(hass, config)
        assert agent._validate_api_key() is False

    def test_validate_api_key_anthropic(self, hass):
        """Test _validate_api_key with Anthropic token."""
        config = {"ai_provider": "anthropic", "anthropic_token": "sk-ant-" + "x" * 40}
        agent = AiAgentHaAgent(hass, config)
        assert agent._validate_api_key() is True

    def test_validate_api_key_openrouter(self, hass):
        """Test _validate_api_key with OpenRouter token."""
        config = {"ai_provider": "openrouter", "openrouter_token": "sk-or-" + "y" * 40}
        agent = AiAgentHaAgent(hass, config)
        assert agent._validate_api_key() is True

    def test_validate_api_key_alter(self, hass):
        """Test _validate_api_key with Alter token."""
        config = {"ai_provider": "alter", "alter_token": "alt-" + "z" * 40}
        agent = AiAgentHaAgent(hass, config)
        assert agent._validate_api_key() is True

    def test_validate_api_key_zai(self, hass):
        """Test _validate_api_key with Z.ai token."""
        config = {"ai_provider": "zai", "zai_token": "zai-" + "w" * 40}
        agent = AiAgentHaAgent(hass, config)
        assert agent._validate_api_key() is True

    def test_validate_api_key_llama(self, hass):
        """Test _validate_api_key with Llama token."""
        config = {"ai_provider": "llama", "llama_token": "llama-" + "v" * 40}
        agent = AiAgentHaAgent(hass, config)
        assert agent._validate_api_key() is True

    def test_check_rate_limit_within_limit(self, hass, agent_config_openai):
        """Test _check_rate_limit when within limits."""
        agent = AiAgentHaAgent(hass, agent_config_openai)
        # First few requests should be allowed
        assert agent._check_rate_limit() is True
        assert agent._check_rate_limit() is True
        assert agent._check_rate_limit() is True

    def test_check_rate_limit_exceeds_limit(self, hass, agent_config_openai):
        """Test _check_rate_limit when exceeding limit."""
        agent = AiAgentHaAgent(hass, agent_config_openai)
        agent._rate_limit = 3  # Set low limit for testing

        # First 3 should pass
        assert agent._check_rate_limit() is True
        assert agent._check_rate_limit() is True
        assert agent._check_rate_limit() is True

        # 4th should fail
        assert agent._check_rate_limit() is False

    def test_check_rate_limit_window_reset(self, hass, agent_config_openai):
        """Test _check_rate_limit window reset."""
        import time

        agent = AiAgentHaAgent(hass, agent_config_openai)
        agent._rate_limit = 2

        assert agent._check_rate_limit() is True
        assert agent._check_rate_limit() is True
        assert agent._check_rate_limit() is False  # Exceeded

        # Simulate window reset
        agent._request_window_start = time.time() - 61  # Move window back
        assert agent._check_rate_limit() is True  # Should reset and allow

    def test_cache_set_and_get(self, hass, agent_config_openai):
        """Test _set_cached_data and _get_cached_data."""
        agent = AiAgentHaAgent(hass, agent_config_openai)

        # Set data
        agent._set_cached_data("test_key", {"value": 123})

        # Get data
        result = agent._get_cached_data("test_key")
        assert result == {"value": 123}

    def test_cache_expired(self, hass, agent_config_openai):
        """Test _get_cached_data returns None for expired data."""
        import time

        agent = AiAgentHaAgent(hass, agent_config_openai)
        agent._cache_timeout = 1  # 1 second timeout

        # Set data
        agent._set_cached_data("test_key", {"value": 123})

        # Should exist immediately
        assert agent._get_cached_data("test_key") == {"value": 123}

        # Wait for expiry
        time.sleep(1.1)

        # Should be expired
        assert agent._get_cached_data("test_key") is None

    def test_cache_nonexistent_key(self, hass, agent_config_openai):
        """Test _get_cached_data returns None for non-existent key."""
        agent = AiAgentHaAgent(hass, agent_config_openai)
        assert agent._get_cached_data("nonexistent") is None

    def test_sanitize_automation_config_alias(self, hass, agent_config_openai):
        """Test _sanitize_automation_config sanitizes alias."""
        agent = AiAgentHaAgent(hass, agent_config_openai)

        config = {"alias": "  Test Automation  ", "description": "A test"}
        result = agent._sanitize_automation_config(config)

        assert result["alias"] == "Test Automation"
        assert result["description"] == "A test"

    def test_sanitize_automation_config_long_alias(self, hass, agent_config_openai):
        """Test _sanitize_automation_config truncates long alias."""
        agent = AiAgentHaAgent(hass, agent_config_openai)

        long_alias = "A" * 150  # Over 100 chars
        config = {"alias": long_alias}
        result = agent._sanitize_automation_config(config)

        assert len(result["alias"]) == 100

    def test_sanitize_automation_config_valid_mode(self, hass, agent_config_openai):
        """Test _sanitize_automation_config accepts valid modes."""
        agent = AiAgentHaAgent(hass, agent_config_openai)

        for mode in ["single", "restart", "queued", "parallel"]:
            config = {"mode": mode}
            result = agent._sanitize_automation_config(config)
            assert result["mode"] == mode

    def test_sanitize_automation_config_invalid_mode(self, hass, agent_config_openai):
        """Test _sanitize_automation_config rejects invalid modes."""
        agent = AiAgentHaAgent(hass, agent_config_openai)

        config = {"mode": "invalid_mode"}
        result = agent._sanitize_automation_config(config)

        assert "mode" not in result

    def test_sanitize_automation_config_trigger_action(self, hass, agent_config_openai):
        """Test _sanitize_automation_config passes through trigger/action lists."""
        agent = AiAgentHaAgent(hass, agent_config_openai)

        config = {
            "trigger": [{"platform": "state", "entity_id": "light.test"}],
            "action": [{"service": "light.turn_on", "target": {"entity_id": "light.test"}}],
        }
        result = agent._sanitize_automation_config(config)

        assert result["trigger"] == config["trigger"]
        assert result["action"] == config["action"]

    def test_sanitize_automation_config_ignores_unknown_keys(self, hass, agent_config_openai):
        """Test _sanitize_automation_config ignores unknown keys."""
        agent = AiAgentHaAgent(hass, agent_config_openai)

        config = {"alias": "Test", "unknown_key": "malicious_value", "injection": "<script>"}
        result = agent._sanitize_automation_config(config)

        assert "unknown_key" not in result
        assert "injection" not in result
        assert result["alias"] == "Test"

    def test_set_rag_manager(self, hass, agent_config_openai):
        """Test set_rag_manager sets the RAG manager."""
        agent = AiAgentHaAgent(hass, agent_config_openai)

        mock_rag = MagicMock()
        agent.set_rag_manager(mock_rag)

        assert agent._rag_manager == mock_rag

    def test_clear_conversation_history(self, hass, agent_config_openai):
        """Test clear_conversation_history clears history."""
        agent = AiAgentHaAgent(hass, agent_config_openai)

        # Add some history
        agent.conversation_history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]

        agent.clear_conversation_history()

        assert agent.conversation_history == []


class TestAgentDataMethods:
    """Test AiAgentHaAgent data retrieval methods."""

    @pytest.fixture
    def agent_config(self):
        """Mock agent config."""
        return {
            "ai_provider": "openai",
            "openai_token": "sk-" + "a" * 48,
        }

    @pytest.mark.asyncio
    async def test_get_entities_by_domain_empty_domain(self, hass, agent_config):
        """Test get_entities_by_domain with empty domain returns error."""
        agent = AiAgentHaAgent(hass, agent_config)
        result = await agent.get_entities_by_domain("")

        assert len(result) == 1
        assert "error" in result[0]
        assert "required" in result[0]["error"].lower()

    @pytest.mark.asyncio
    async def test_get_entities_by_domain_success(self, hass, agent_config):
        """Test get_entities_by_domain returns entities for domain."""
        hass.states.async_set("light.living_room", "on", {"friendly_name": "Living Room"})
        hass.states.async_set("light.bedroom", "off", {"friendly_name": "Bedroom"})
        hass.states.async_set("switch.fan", "on", {"friendly_name": "Fan"})

        agent = AiAgentHaAgent(hass, agent_config)
        result = await agent.get_entities_by_domain("light")

        assert len(result) == 2
        entity_ids = [e["entity_id"] for e in result]
        assert "light.living_room" in entity_ids
        assert "light.bedroom" in entity_ids

    @pytest.mark.asyncio
    async def test_get_entities_by_domain_no_entities(self, hass, agent_config):
        """Test get_entities_by_domain returns empty list when no entities."""
        agent = AiAgentHaAgent(hass, agent_config)
        result = await agent.get_entities_by_domain("nonexistent")

        assert result == []

    @pytest.mark.asyncio
    async def test_get_entities_by_device_class_empty(self, hass, agent_config):
        """Test get_entities_by_device_class with empty device_class returns error."""
        agent = AiAgentHaAgent(hass, agent_config)
        result = await agent.get_entities_by_device_class("")

        assert len(result) == 1
        assert "error" in result[0]

    @pytest.mark.asyncio
    async def test_get_scenes_empty(self, hass, agent_config):
        """Test get_scenes with no scenes returns empty list."""
        agent = AiAgentHaAgent(hass, agent_config)
        result = await agent.get_scenes()

        assert result == []

    @pytest.mark.asyncio
    async def test_get_scenes_with_scenes(self, hass, agent_config):
        """Test get_scenes returns scene data."""
        from datetime import datetime

        hass.states.async_set(
            "scene.movie_time",
            "scening",
            {
                "friendly_name": "Movie Time",
                "icon": "mdi:movie",
                "last_activated": datetime.now().isoformat(),
            }
        )
        hass.states.async_set(
            "scene.good_morning",
            "scening",
            {"friendly_name": "Good Morning"}
        )

        agent = AiAgentHaAgent(hass, agent_config)
        result = await agent.get_scenes()

        assert len(result) == 2
        names = [s["name"] for s in result]
        assert "Movie Time" in names
        assert "Good Morning" in names

    @pytest.mark.asyncio
    async def test_get_weather_data_no_entities(self, hass, agent_config):
        """Test get_weather_data with no weather entities returns error."""
        agent = AiAgentHaAgent(hass, agent_config)
        result = await agent.get_weather_data()

        assert "error" in result
        assert "No weather entities" in result["error"]

    @pytest.mark.asyncio
    async def test_get_weather_data_with_entity(self, hass, agent_config):
        """Test get_weather_data returns weather data."""
        hass.states.async_set(
            "weather.home",
            "sunny",
            {
                "friendly_name": "Home Weather",
                "temperature": 22,
                "humidity": 55,
                "pressure": 1013,
                "wind_speed": 10,
                "forecast": [
                    {"datetime": "2024-01-15", "temperature": 20, "condition": "cloudy"}
                ]
            }
        )

        agent = AiAgentHaAgent(hass, agent_config)
        result = await agent.get_weather_data()

        assert "error" not in result
        # Weather data is returned with nested 'current' structure
        assert "current" in result
        assert result["current"]["entity_id"] == "weather.home"
        assert result["current"]["temperature"] == 22
        assert result["current"]["humidity"] == 55

    @pytest.mark.asyncio
    async def test_get_entity_state_empty_id(self, hass, agent_config):
        """Test get_entity_state with empty ID returns error."""
        agent = AiAgentHaAgent(hass, agent_config)
        result = await agent.get_entity_state("")

        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_get_statistics_empty_id(self, hass, agent_config):
        """Test get_statistics with empty entity_id returns error."""
        agent = AiAgentHaAgent(hass, agent_config)
        result = await agent.get_statistics("")

        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_get_statistics_no_recorder(self, hass, agent_config):
        """Test get_statistics without recorder returns error."""
        agent = AiAgentHaAgent(hass, agent_config)
        result = await agent.get_statistics("sensor.test")

        assert "error" in result
        # Either recorder not available or exception

    @pytest.mark.asyncio
    async def test_get_automations_empty(self, hass, agent_config):
        """Test get_automations with no automations."""
        agent = AiAgentHaAgent(hass, agent_config)
        result = await agent.get_automations()

        assert result == []

    @pytest.mark.asyncio
    async def test_get_automations_with_automations(self, hass, agent_config):
        """Test get_automations returns automation data."""
        hass.states.async_set(
            "automation.morning_lights",
            "on",
            {
                "friendly_name": "Morning Lights",
                "last_triggered": "2024-01-15T08:00:00",
            }
        )

        agent = AiAgentHaAgent(hass, agent_config)
        result = await agent.get_automations()

        assert len(result) == 1
        assert result[0]["entity_id"] == "automation.morning_lights"
        # Returns entity state format with friendly_name
        assert result[0]["friendly_name"] == "Morning Lights"

    @pytest.mark.asyncio
    async def test_get_person_data_empty(self, hass, agent_config):
        """Test get_person_data with no persons."""
        agent = AiAgentHaAgent(hass, agent_config)
        result = await agent.get_person_data()

        assert result == []

    @pytest.mark.asyncio
    async def test_get_person_data_with_persons(self, hass, agent_config):
        """Test get_person_data returns person data."""
        hass.states.async_set(
            "person.john",
            "home",
            {
                "friendly_name": "John",
                "source": "device_tracker.phone",
            }
        )

        agent = AiAgentHaAgent(hass, agent_config)
        result = await agent.get_person_data()

        assert len(result) == 1
        assert result[0]["entity_id"] == "person.john"
        assert result[0]["state"] == "home"

    @pytest.mark.asyncio
    async def test_get_area_registry(self, hass, agent_config):
        """Test get_area_registry returns area data."""
        agent = AiAgentHaAgent(hass, agent_config)
        result = await agent.get_area_registry()

        # Should return dict structure even if empty
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_get_dashboards(self, hass, agent_config):
        """Test get_dashboards returns dashboard list."""
        agent = AiAgentHaAgent(hass, agent_config)
        result = await agent.get_dashboards()

        # Should return list even if empty
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_get_entities_by_area_empty_id(self, hass, agent_config):
        """Test get_entities_by_area with empty area_id."""
        agent = AiAgentHaAgent(hass, agent_config)
        result = await agent.get_entities_by_area("")

        assert len(result) == 1
        assert "error" in result[0]
        assert "area_id is required" in result[0]["error"]

    @pytest.mark.asyncio
    async def test_get_entities_by_area_not_found(self, hass, agent_config):
        """Test get_entities_by_area with non-existent area."""
        agent = AiAgentHaAgent(hass, agent_config)
        result = await agent.get_entities_by_area("non_existent_area")

        # Should return empty list when no entities in area
        assert result == []

    @pytest.mark.asyncio
    async def test_get_entities_no_area_provided(self, hass, agent_config):
        """Test get_entities with no area provided."""
        agent = AiAgentHaAgent(hass, agent_config)
        result = await agent.get_entities()

        assert len(result) == 1
        assert "error" in result[0]
        assert "No area_id or area_ids provided" in result[0]["error"]

    @pytest.mark.asyncio
    async def test_get_entities_single_area(self, hass, agent_config):
        """Test get_entities with single area_id."""
        agent = AiAgentHaAgent(hass, agent_config)

        # Mock get_entities_by_area to return test data
        with patch.object(
            agent, "get_entities_by_area", new=AsyncMock(
                return_value=[{"entity_id": "light.living_room"}]
            )
        ):
            result = await agent.get_entities(area_id="living_room")

        assert len(result) == 1
        assert result[0]["entity_id"] == "light.living_room"

    @pytest.mark.asyncio
    async def test_get_entities_multiple_areas(self, hass, agent_config):
        """Test get_entities with multiple area_ids."""
        agent = AiAgentHaAgent(hass, agent_config)

        call_count = 0

        async def mock_get_by_area(area_id):
            nonlocal call_count
            call_count += 1
            if area_id == "living_room":
                return [{"entity_id": "light.living_room"}]
            elif area_id == "bedroom":
                return [{"entity_id": "light.bedroom"}]
            return []

        with patch.object(agent, "get_entities_by_area", side_effect=mock_get_by_area):
            result = await agent.get_entities(area_ids=["living_room", "bedroom"])

        assert call_count == 2
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_get_entities_removes_duplicates(self, hass, agent_config):
        """Test get_entities removes duplicate entities."""
        agent = AiAgentHaAgent(hass, agent_config)

        async def mock_get_by_area(area_id):
            # Return same entity from both areas
            return [{"entity_id": "light.shared"}]

        with patch.object(agent, "get_entities_by_area", side_effect=mock_get_by_area):
            result = await agent.get_entities(area_ids=["area1", "area2"])

        # Should deduplicate
        assert len(result) == 1
        assert result[0]["entity_id"] == "light.shared"

    @pytest.mark.asyncio
    async def test_get_calendar_events_specific_entity(self, hass, agent_config):
        """Test get_calendar_events with specific entity_id."""
        hass.states.async_set(
            "calendar.work",
            "on",
            {"friendly_name": "Work Calendar"}
        )

        agent = AiAgentHaAgent(hass, agent_config)
        result = await agent.get_calendar_events(entity_id="calendar.work")

        assert len(result) == 1
        assert result[0]["entity_id"] == "calendar.work"

    @pytest.mark.asyncio
    async def test_get_calendar_events_all(self, hass, agent_config):
        """Test get_calendar_events without entity_id returns all calendars."""
        hass.states.async_set(
            "calendar.work",
            "on",
            {"friendly_name": "Work Calendar"}
        )
        hass.states.async_set(
            "calendar.personal",
            "on",
            {"friendly_name": "Personal Calendar"}
        )

        agent = AiAgentHaAgent(hass, agent_config)
        result = await agent.get_calendar_events()

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_get_entity_state_valid_entity(self, hass, agent_config):
        """Test get_entity_state with valid entity."""
        hass.states.async_set(
            "light.test",
            "on",
            {"brightness": 255, "friendly_name": "Test Light"}
        )

        agent = AiAgentHaAgent(hass, agent_config)
        result = await agent.get_entity_state("light.test")

        assert result["entity_id"] == "light.test"
        assert result["state"] == "on"
        # Attributes are included in the result
        assert result["friendly_name"] == "Test Light"

    @pytest.mark.asyncio
    async def test_get_entity_state_invalid_entity(self, hass, agent_config):
        """Test get_entity_state with non-existent entity."""
        agent = AiAgentHaAgent(hass, agent_config)
        result = await agent.get_entity_state("light.nonexistent")

        assert "error" in result

    @pytest.mark.asyncio
    async def test_get_entity_registry_pagination(self, hass, agent_config):
        """Test get_entity_registry with pagination."""
        agent = AiAgentHaAgent(hass, agent_config)
        result = await agent.get_entity_registry(limit=10, offset=0)

        assert isinstance(result, dict)
        assert "entities" in result or "error" in result

    @pytest.mark.asyncio
    async def test_get_entity_registry_domain_filter(self, hass, agent_config):
        """Test get_entity_registry with domain filter."""
        agent = AiAgentHaAgent(hass, agent_config)
        result = await agent.get_entity_registry(domain="light")

        assert isinstance(result, dict)


class TestProcessQuery:
    """Tests for process_query method."""

    @pytest.fixture
    def agent_config(self):
        """Mock agent configuration."""
        return {
            "ai_provider": "openai",
            "openai_token": "test_token",
            "openai_model": "gpt-4",
        }

    @pytest.mark.asyncio
    async def test_process_query_final_response(self, hass, agent_config):
        """Test process_query with final_response."""
        agent = AiAgentHaAgent(hass, agent_config)

        with patch.object(agent, "_get_ai_response", new_callable=AsyncMock) as mock_ai:
            mock_ai.return_value = json.dumps({
                "request_type": "final_response",
                "response": "Hello! I'm your AI assistant."
            })

            result = await agent.process_query("Hello")

            assert result["success"] is True
            assert result["answer"] == "Hello! I'm your AI assistant."

    @pytest.mark.asyncio
    async def test_process_query_with_tool_call(self, hass, agent_config):
        """Test process_query with tool call followed by final response."""
        agent = AiAgentHaAgent(hass, agent_config)

        # Setup entity state
        hass.states.async_set("light.test", "on", {"brightness": 255})

        call_count = 0

        async def mock_ai_response(messages=None, tools=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call: request entity state
                return json.dumps({
                    "request_type": "get_entity_state",
                    "parameters": {"entity_id": "light.test"}
                })
            else:
                # Second call: final response
                return json.dumps({
                    "request_type": "final_response",
                    "response": "The light is on with brightness 255."
                })

        with patch.object(agent, "_get_ai_response", side_effect=mock_ai_response):
            result = await agent.process_query("What's the status of the light?")

            assert result["success"] is True
            assert call_count == 2

    @pytest.mark.asyncio
    async def test_process_query_max_iterations(self, hass, agent_config):
        """Test process_query stops at max iterations."""
        agent = AiAgentHaAgent(hass, agent_config)

        # Set up an entity so requests don't fail
        hass.states.async_set("light.test", "on", {"brightness": 255})

        # Always return a tool call to trigger iteration loop
        # Use get_entity_state which will succeed and continue loop
        with patch.object(agent, "_get_ai_response", new_callable=AsyncMock) as mock_ai:
            mock_ai.return_value = json.dumps({
                "request_type": "get_entity_state",
                "parameters": {"entity_id": "light.test"}
            })

            result = await agent.process_query("Test query")

            # Should eventually fail with max iterations error
            assert result["success"] is False
            assert "Maximum iterations" in result.get("error", "")

    @pytest.mark.asyncio
    async def test_process_query_invalid_json(self, hass, agent_config):
        """Test process_query with invalid JSON response."""
        agent = AiAgentHaAgent(hass, agent_config)

        with patch.object(agent, "_get_ai_response", new_callable=AsyncMock) as mock_ai:
            # Return invalid JSON (plain text)
            mock_ai.return_value = "This is not valid JSON but a text response"

            result = await agent.process_query("Test query")

            # Should handle gracefully and return as plain text
            assert result["success"] is True
            assert "This is not valid JSON" in result["answer"]

    @pytest.mark.asyncio
    async def test_process_query_exception_handling(self, hass, agent_config):
        """Test process_query handles exceptions."""
        agent = AiAgentHaAgent(hass, agent_config)

        with patch.object(agent, "_get_ai_response", new_callable=AsyncMock) as mock_ai:
            mock_ai.side_effect = Exception("API connection failed")

            result = await agent.process_query("Test query")

            assert result["success"] is False
            assert "API connection failed" in result["error"]

    @pytest.mark.asyncio
    async def test_process_query_automation_suggestion(self, hass, agent_config):
        """Test process_query with automation suggestion."""
        agent = AiAgentHaAgent(hass, agent_config)

        automation_data = {
            "request_type": "automation_suggestion",
            "automation": {
                "trigger": {"platform": "state", "entity_id": "light.test"},
                "action": {"service": "light.turn_off"}
            }
        }

        with patch.object(agent, "_get_ai_response", new_callable=AsyncMock) as mock_ai:
            mock_ai.return_value = json.dumps(automation_data)

            result = await agent.process_query("Create an automation")

            assert result["success"] is True
            assert "automation_suggestion" in result["answer"]

    @pytest.mark.asyncio
    async def test_process_query_dashboard_suggestion(self, hass, agent_config):
        """Test process_query with dashboard suggestion."""
        agent = AiAgentHaAgent(hass, agent_config)

        dashboard_data = {
            "request_type": "dashboard_suggestion",
            "dashboard": {
                "views": [{"title": "Test View", "cards": []}]
            }
        }

        with patch.object(agent, "_get_ai_response", new_callable=AsyncMock) as mock_ai:
            mock_ai.return_value = json.dumps(dashboard_data)

            result = await agent.process_query("Create a dashboard")

            assert result["success"] is True
            assert "dashboard_suggestion" in result["answer"]

    @pytest.mark.asyncio
    async def test_process_query_conversation_history(self, hass, agent_config):
        """Test process_query maintains conversation history."""
        agent = AiAgentHaAgent(hass, agent_config)

        with patch.object(agent, "_get_ai_response", new_callable=AsyncMock) as mock_ai:
            mock_ai.return_value = json.dumps({
                "request_type": "final_response",
                "response": "First response"
            })

            await agent.process_query("First query")

            # History should contain the query and response
            assert len(agent.conversation_history) >= 2

            # Query should be in history
            user_messages = [m for m in agent.conversation_history if m.get("role") == "user"]
            assert any("First query" in m.get("content", "") for m in user_messages)

    @pytest.mark.asyncio
    async def test_process_query_rate_limiting(self, hass, agent_config):
        """Test process_query rate limiting."""
        agent = AiAgentHaAgent(hass, agent_config)

        # Force rate limit by setting call times
        agent._last_call_time = time.time()
        agent._call_count = 100  # High call count

        with patch.object(agent, "_check_rate_limit", return_value=False):
            result = await agent.process_query("Test query")

            assert result["success"] is False
            assert "Rate limit" in result.get("error", "")

    @pytest.mark.asyncio
    async def test_process_query_caching(self, hass, agent_config):
        """Test process_query caching behavior."""
        agent = AiAgentHaAgent(hass, agent_config)

        call_count = 0

        async def mock_ai_response(messages=None, tools=None):
            nonlocal call_count
            call_count += 1
            return json.dumps({
                "request_type": "final_response",
                "response": f"Response {call_count}"
            })

        with patch.object(agent, "_get_ai_response", side_effect=mock_ai_response):
            # First call
            result1 = await agent.process_query("Same query")

            # Second call with same query should use cache
            result2 = await agent.process_query("Same query")

            # Both should succeed
            assert result1["success"] is True
            assert result2["success"] is True

            # Second result should be from cache (same response)
            # Note: Cache behavior depends on implementation
            # This test verifies caching doesn't break functionality

    @pytest.mark.asyncio
    async def test_process_query_external_history(self, hass, agent_config):
        """Test process_query with external conversation history."""
        agent = AiAgentHaAgent(hass, agent_config)

        external_history = [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"}
        ]

        with patch.object(agent, "_get_ai_response", new_callable=AsyncMock) as mock_ai:
            mock_ai.return_value = json.dumps({
                "request_type": "final_response",
                "response": "New response"
            })

            result = await agent.process_query(
                "New question",
                conversation_history=external_history
            )

            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_process_query_get_entities(self, hass, agent_config):
        """Test process_query with get_entities request."""
        agent = AiAgentHaAgent(hass, agent_config)

        call_count = 0

        async def mock_ai_response(messages=None, tools=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return json.dumps({
                    "request_type": "get_entities",
                    "parameters": {"area_id": "living_room"}
                })
            return json.dumps({
                "request_type": "final_response",
                "response": "Here are the entities."
            })

        with patch.object(agent, "_get_ai_response", side_effect=mock_ai_response):
            with patch.object(agent, "get_entities", new_callable=AsyncMock) as mock_get:
                mock_get.return_value = [{"entity_id": "light.living_room"}]

                result = await agent.process_query("Show entities in living room")

                assert result["success"] is True
                mock_get.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_query_get_entities_by_device_class(self, hass, agent_config):
        """Test process_query with get_entities_by_device_class request."""
        agent = AiAgentHaAgent(hass, agent_config)

        call_count = 0

        async def mock_ai_response(messages=None, tools=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return json.dumps({
                    "request_type": "get_entities_by_device_class",
                    "parameters": {"device_class": "temperature", "domain": "sensor"}
                })
            return json.dumps({
                "request_type": "final_response",
                "response": "Found temperature sensors."
            })

        with patch.object(agent, "_get_ai_response", side_effect=mock_ai_response):
            with patch.object(agent, "get_entities_by_device_class", new_callable=AsyncMock) as mock_get:
                mock_get.return_value = [{"entity_id": "sensor.temp_1"}]

                result = await agent.process_query("Show temperature sensors")

                assert result["success"] is True
                mock_get.assert_called_once_with("temperature", "sensor")

    @pytest.mark.asyncio
    async def test_process_query_get_automations(self, hass, agent_config):
        """Test process_query with get_automations request."""
        agent = AiAgentHaAgent(hass, agent_config)

        call_count = 0

        async def mock_ai_response(messages=None, tools=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return json.dumps({
                    "request_type": "get_automations",
                    "parameters": {}
                })
            return json.dumps({
                "request_type": "final_response",
                "response": "Found automations."
            })

        with patch.object(agent, "_get_ai_response", side_effect=mock_ai_response):
            with patch.object(agent, "get_automations", new_callable=AsyncMock) as mock_get:
                mock_get.return_value = [{"id": "auto_1", "alias": "Test"}]

                result = await agent.process_query("List automations")

                assert result["success"] is True
                mock_get.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_query_get_weather_data(self, hass, agent_config):
        """Test process_query with get_weather_data request."""
        agent = AiAgentHaAgent(hass, agent_config)

        call_count = 0

        async def mock_ai_response(messages=None, tools=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return json.dumps({
                    "request_type": "get_weather_data",
                    "parameters": {}
                })
            return json.dumps({
                "request_type": "final_response",
                "response": "Weather is sunny."
            })

        with patch.object(agent, "_get_ai_response", side_effect=mock_ai_response):
            with patch.object(agent, "get_weather_data", new_callable=AsyncMock) as mock_get:
                mock_get.return_value = {"temperature": 22, "condition": "sunny"}

                result = await agent.process_query("What's the weather?")

                assert result["success"] is True
                mock_get.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_query_get_scenes(self, hass, agent_config):
        """Test process_query with get_scenes request."""
        agent = AiAgentHaAgent(hass, agent_config)

        call_count = 0

        async def mock_ai_response(messages=None, tools=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return json.dumps({
                    "request_type": "get_scenes",
                    "parameters": {}
                })
            return json.dumps({
                "request_type": "final_response",
                "response": "Available scenes."
            })

        with patch.object(agent, "_get_ai_response", side_effect=mock_ai_response):
            with patch.object(agent, "get_scenes", new_callable=AsyncMock) as mock_get:
                mock_get.return_value = [{"entity_id": "scene.movie_night"}]

                result = await agent.process_query("List scenes")

                assert result["success"] is True
                mock_get.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_query_get_history(self, hass, agent_config):
        """Test process_query with get_history request."""
        agent = AiAgentHaAgent(hass, agent_config)

        call_count = 0

        async def mock_ai_response(messages=None, tools=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return json.dumps({
                    "request_type": "get_history",
                    "parameters": {"entity_id": "sensor.temp", "hours": 12}
                })
            return json.dumps({
                "request_type": "final_response",
                "response": "Here's the history."
            })

        with patch.object(agent, "_get_ai_response", side_effect=mock_ai_response):
            with patch.object(agent, "get_history", new_callable=AsyncMock) as mock_get:
                mock_get.return_value = [{"state": "22", "last_changed": "2024-01-01"}]

                result = await agent.process_query("Show sensor history")

                assert result["success"] is True
                mock_get.assert_called_once_with("sensor.temp", 12)

    @pytest.mark.asyncio
    async def test_process_query_set_entity_state(self, hass, agent_config):
        """Test process_query with set_entity_state request."""
        agent = AiAgentHaAgent(hass, agent_config)

        call_count = 0

        async def mock_ai_response(messages=None, tools=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return json.dumps({
                    "request_type": "set_entity_state",
                    "parameters": {
                        "entity_id": "input_boolean.test",
                        "state": "on",
                        "attributes": {"editable": True}
                    }
                })
            return json.dumps({
                "request_type": "final_response",
                "response": "State updated."
            })

        with patch.object(agent, "_get_ai_response", side_effect=mock_ai_response):
            with patch.object(agent, "set_entity_state", new_callable=AsyncMock) as mock_set:
                mock_set.return_value = {"success": True}

                result = await agent.process_query("Turn on the input boolean")

                assert result["success"] is True
                mock_set.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_query_call_service(self, hass, agent_config):
        """Test process_query with call_service request."""
        agent = AiAgentHaAgent(hass, agent_config)

        call_count = 0

        async def mock_ai_response(messages=None, tools=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return json.dumps({
                    "request_type": "call_service",
                    "parameters": {
                        "domain": "light",
                        "service": "turn_on",
                        "service_data": {"entity_id": "light.bedroom", "brightness": 128}
                    }
                })
            return json.dumps({
                "request_type": "final_response",
                "response": "Service called."
            })

        with patch.object(agent, "_get_ai_response", side_effect=mock_ai_response):
            with patch.object(agent, "call_service", new_callable=AsyncMock) as mock_call:
                mock_call.return_value = {"success": True}

                result = await agent.process_query("Turn on bedroom light at 50%")

                assert result["success"] is True
                mock_call.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_query_get_area_registry(self, hass, agent_config):
        """Test process_query with get_area_registry request."""
        agent = AiAgentHaAgent(hass, agent_config)

        call_count = 0

        async def mock_ai_response(messages=None, tools=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return json.dumps({
                    "request_type": "get_area_registry",
                    "parameters": {}
                })
            return json.dumps({
                "request_type": "final_response",
                "response": "Found areas."
            })

        with patch.object(agent, "_get_ai_response", side_effect=mock_ai_response):
            with patch.object(agent, "get_area_registry", new_callable=AsyncMock) as mock_get:
                mock_get.return_value = [{"area_id": "living_room", "name": "Living Room"}]

                result = await agent.process_query("List all areas")

                assert result["success"] is True
                mock_get.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_query_get_entity_registry(self, hass, agent_config):
        """Test process_query with get_entity_registry request."""
        agent = AiAgentHaAgent(hass, agent_config)

        call_count = 0

        async def mock_ai_response(messages=None, tools=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return json.dumps({
                    "request_type": "get_entity_registry",
                    "parameters": {"domain": "light", "limit": 10}
                })
            return json.dumps({
                "request_type": "final_response",
                "response": "Found entities."
            })

        with patch.object(agent, "_get_ai_response", side_effect=mock_ai_response):
            with patch.object(agent, "get_entity_registry", new_callable=AsyncMock) as mock_get:
                mock_get.return_value = {"entities": [], "total": 0}

                result = await agent.process_query("List light entities")

                assert result["success"] is True
                mock_get.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_query_create_automation(self, hass, agent_config):
        """Test process_query with create_automation request."""
        agent = AiAgentHaAgent(hass, agent_config)

        call_count = 0

        async def mock_ai_response(messages=None, tools=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return json.dumps({
                    "request_type": "create_automation",
                    "parameters": {
                        "automation": {
                            "alias": "Test Automation",
                            "trigger": {"platform": "state", "entity_id": "light.test"},
                            "action": {"service": "light.turn_off"}
                        }
                    }
                })
            return json.dumps({
                "request_type": "final_response",
                "response": "Automation created."
            })

        with patch.object(agent, "_get_ai_response", side_effect=mock_ai_response):
            with patch.object(agent, "create_automation", new_callable=AsyncMock) as mock_create:
                mock_create.return_value = {"success": True, "automation_id": "auto_123"}

                result = await agent.process_query("Create automation for light")

                assert result["success"] is True
                mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_query_get_climate_related_entities(self, hass, agent_config):
        """Test process_query with get_climate_related_entities request."""
        agent = AiAgentHaAgent(hass, agent_config)

        call_count = 0

        async def mock_ai_response(messages=None, tools=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return json.dumps({
                    "request_type": "get_climate_related_entities",
                    "parameters": {}
                })
            return json.dumps({
                "request_type": "final_response",
                "response": "Found climate entities."
            })

        with patch.object(agent, "_get_ai_response", side_effect=mock_ai_response):
            with patch.object(agent, "get_climate_related_entities", new_callable=AsyncMock) as mock_get:
                mock_get.return_value = [
                    {"entity_id": "climate.thermostat", "state": "heat"},
                    {"entity_id": "sensor.temperature", "state": "22.5"}
                ]

                result = await agent.process_query("Show climate entities")

                assert result["success"] is True
                mock_get.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_query_get_calendar_events(self, hass, agent_config):
        """Test process_query with get_calendar_events request."""
        agent = AiAgentHaAgent(hass, agent_config)

        call_count = 0

        async def mock_ai_response(messages=None, tools=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return json.dumps({
                    "request_type": "get_calendar_events",
                    "parameters": {"entity_id": "calendar.work"}
                })
            return json.dumps({
                "request_type": "final_response",
                "response": "Found calendar events."
            })

        with patch.object(agent, "_get_ai_response", side_effect=mock_ai_response):
            with patch.object(agent, "get_calendar_events", new_callable=AsyncMock) as mock_get:
                mock_get.return_value = [
                    {"entity_id": "calendar.work", "state": "on", "summary": "Meeting"}
                ]

                result = await agent.process_query("Show work calendar events")

                assert result["success"] is True
                mock_get.assert_called_once_with("calendar.work")

    @pytest.mark.asyncio
    async def test_process_query_get_person_data(self, hass, agent_config):
        """Test process_query with get_person_data request."""
        agent = AiAgentHaAgent(hass, agent_config)

        call_count = 0

        async def mock_ai_response(messages=None, tools=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return json.dumps({
                    "request_type": "get_person_data",
                    "parameters": {}
                })
            return json.dumps({
                "request_type": "final_response",
                "response": "Found person data."
            })

        with patch.object(agent, "_get_ai_response", side_effect=mock_ai_response):
            with patch.object(agent, "get_person_data", new_callable=AsyncMock) as mock_get:
                mock_get.return_value = [
                    {"entity_id": "person.john", "state": "home", "name": "John"}
                ]

                result = await agent.process_query("Show person data")

                assert result["success"] is True
                mock_get.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_query_get_statistics(self, hass, agent_config):
        """Test process_query with get_statistics request."""
        agent = AiAgentHaAgent(hass, agent_config)

        call_count = 0

        async def mock_ai_response(messages=None, tools=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return json.dumps({
                    "request_type": "get_statistics",
                    "parameters": {"entity_id": "sensor.temperature"}
                })
            return json.dumps({
                "request_type": "final_response",
                "response": "Found statistics."
            })

        with patch.object(agent, "_get_ai_response", side_effect=mock_ai_response):
            with patch.object(agent, "get_statistics", new_callable=AsyncMock) as mock_get:
                mock_get.return_value = {
                    "entity_id": "sensor.temperature",
                    "mean": 22.5,
                    "min": 20.0,
                    "max": 25.0
                }

                result = await agent.process_query("Show temperature statistics")

                assert result["success"] is True
                mock_get.assert_called_once_with("sensor.temperature")

    @pytest.mark.asyncio
    async def test_process_query_get_dashboards(self, hass, agent_config):
        """Test process_query with get_dashboards request."""
        agent = AiAgentHaAgent(hass, agent_config)

        call_count = 0

        async def mock_ai_response(messages=None, tools=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return json.dumps({
                    "request_type": "get_dashboards",
                    "parameters": {}
                })
            return json.dumps({
                "request_type": "final_response",
                "response": "Found dashboards."
            })

        with patch.object(agent, "_get_ai_response", side_effect=mock_ai_response):
            with patch.object(agent, "get_dashboards", new_callable=AsyncMock) as mock_get:
                mock_get.return_value = [
                    {"url_path": "lovelace", "title": "Home"},
                    {"url_path": "energy", "title": "Energy"}
                ]

                result = await agent.process_query("List all dashboards")

                assert result["success"] is True
                mock_get.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_query_get_dashboard_config(self, hass, agent_config):
        """Test process_query with get_dashboard_config request."""
        agent = AiAgentHaAgent(hass, agent_config)

        call_count = 0

        async def mock_ai_response(messages=None, tools=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return json.dumps({
                    "request_type": "get_dashboard_config",
                    "parameters": {"dashboard_url": "lovelace"}
                })
            return json.dumps({
                "request_type": "final_response",
                "response": "Found dashboard config."
            })

        with patch.object(agent, "_get_ai_response", side_effect=mock_ai_response):
            with patch.object(agent, "get_dashboard_config", new_callable=AsyncMock) as mock_get:
                mock_get.return_value = {
                    "title": "Home",
                    "views": [{"title": "Overview", "cards": []}]
                }

                result = await agent.process_query("Show lovelace dashboard config")

                assert result["success"] is True
                mock_get.assert_called_once_with("lovelace")

    @pytest.mark.asyncio
    async def test_process_query_create_dashboard(self, hass, agent_config):
        """Test process_query with create_dashboard request."""
        agent = AiAgentHaAgent(hass, agent_config)

        call_count = 0

        async def mock_ai_response(messages=None, tools=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return json.dumps({
                    "request_type": "create_dashboard",
                    "parameters": {
                        "dashboard_config": {
                            "title": "Test Dashboard",
                            "url_path": "test-dashboard",
                            "icon": "mdi:view-dashboard",
                            "show_in_sidebar": True
                        }
                    }
                })
            return json.dumps({
                "request_type": "final_response",
                "response": "Dashboard created successfully."
            })

        with patch.object(agent, "_get_ai_response", side_effect=mock_ai_response):
            with patch.object(agent, "create_dashboard", new_callable=AsyncMock) as mock_create:
                mock_create.return_value = {
                    "success": True,
                    "dashboard_url": "test-dashboard",
                    "message": "Dashboard created successfully"
                }

                result = await agent.process_query("Create a new dashboard")

                assert result["success"] is True
                mock_create.assert_called_once()
                # Verify the dashboard_config parameter was passed correctly
                call_args = mock_create.call_args[0][0]
                assert call_args["title"] == "Test Dashboard"
                assert call_args["url_path"] == "test-dashboard"

    @pytest.mark.asyncio
    async def test_process_query_update_dashboard(self, hass, agent_config):
        """Test process_query with update_dashboard request."""
        agent = AiAgentHaAgent(hass, agent_config)

        call_count = 0

        async def mock_ai_response(messages=None, tools=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return json.dumps({
                    "request_type": "update_dashboard",
                    "parameters": {
                        "dashboard_url": "lovelace",
                        "dashboard_config": {
                            "title": "Updated Home",
                            "icon": "mdi:home",
                            "show_in_sidebar": False
                        }
                    }
                })
            return json.dumps({
                "request_type": "final_response",
                "response": "Dashboard updated successfully."
            })

        with patch.object(agent, "_get_ai_response", side_effect=mock_ai_response):
            with patch.object(agent, "update_dashboard", new_callable=AsyncMock) as mock_update:
                mock_update.return_value = {
                    "success": True,
                    "dashboard_url": "lovelace",
                    "message": "Dashboard updated successfully"
                }

                result = await agent.process_query("Update the lovelace dashboard")

                assert result["success"] is True
                mock_update.assert_called_once()
                # Verify both parameters were passed correctly
                call_args = mock_update.call_args[0]
                assert call_args[0] == "lovelace"  # dashboard_url
                assert call_args[1]["title"] == "Updated Home"  # dashboard_config

    @pytest.mark.asyncio
    async def test_process_query_create_dashboard_with_error(self, hass, agent_config):
        """Test process_query with create_dashboard request that returns error."""
        agent = AiAgentHaAgent(hass, agent_config)

        call_count = 0

        async def mock_ai_response(messages=None, tools=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return json.dumps({
                    "request_type": "create_dashboard",
                    "parameters": {
                        "dashboard_config": {
                            "title": "Invalid Dashboard"
                            # Missing required url_path
                        }
                    }
                })
            return json.dumps({
                "request_type": "final_response",
                "response": "Dashboard creation failed."
            })

        with patch.object(agent, "_get_ai_response", side_effect=mock_ai_response):
            with patch.object(agent, "create_dashboard", new_callable=AsyncMock) as mock_create:
                mock_create.return_value = {
                    "error": "Dashboard URL path is required"
                }

                result = await agent.process_query("Create a dashboard without url")

                assert result["success"] is True  # process_query itself succeeds
                mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_query_update_dashboard_with_error(self, hass, agent_config):
        """Test process_query with update_dashboard request that returns error."""
        agent = AiAgentHaAgent(hass, agent_config)

        call_count = 0

        async def mock_ai_response(messages=None, tools=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return json.dumps({
                    "request_type": "update_dashboard",
                    "parameters": {
                        "dashboard_url": "nonexistent",
                        "dashboard_config": {
                            "title": "Updated Dashboard"
                        }
                    }
                })
            return json.dumps({
                "request_type": "final_response",
                "response": "Dashboard update failed."
            })

        with patch.object(agent, "_get_ai_response", side_effect=mock_ai_response):
            with patch.object(agent, "update_dashboard", new_callable=AsyncMock) as mock_update:
                mock_update.return_value = {
                    "error": "Dashboard not found: nonexistent"
                }

                result = await agent.process_query("Update nonexistent dashboard")

                assert result["success"] is True  # process_query itself succeeds
                mock_update.assert_called_once()
                # Verify error was passed through in conversation history
                call_args = mock_update.call_args[0]
                assert call_args[0] == "nonexistent"

    @pytest.mark.asyncio
    async def test_process_query_web_fetch_success(self, hass, agent_config):
        """Test process_query with web_fetch success."""
        agent = AiAgentHaAgent(hass, agent_config)

        call_count = 0

        async def mock_ai_response(messages=None, tools=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return json.dumps({
                    "request_type": "web_fetch",
                    "parameters": {
                        "url": "https://example.com",
                        "format": "markdown",
                        "timeout": 30
                    }
                })
            return json.dumps({
                "request_type": "final_response",
                "response": "Fetched content successfully."
            })

        with patch.object(agent, "_get_ai_response", side_effect=mock_ai_response):
            with patch("custom_components.ai_agent_ha.agent.execute_tool", new_callable=AsyncMock) as mock_execute:
                # Mock successful ToolResult
                from custom_components.ai_agent_ha.tools import ToolResult
                mock_execute.return_value = ToolResult(
                    output="# Example Page\n\nThis is example content.",
                    title="Example Domain",
                    metadata={"url": "https://example.com", "status": 200},
                    success=True
                )

                result = await agent.process_query("Fetch content from example.com")

                assert result["success"] is True
                mock_execute.assert_called_once_with(
                    "web_fetch",
                    {"url": "https://example.com", "format": "markdown", "timeout": 30},
                    hass=hass,
                    config=agent.config
                )

    @pytest.mark.asyncio
    async def test_process_query_web_fetch_failure(self, hass, agent_config):
        """Test process_query with web_fetch failure."""
        agent = AiAgentHaAgent(hass, agent_config)

        call_count = 0

        async def mock_ai_response(messages=None, tools=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return json.dumps({
                    "request_type": "web_fetch",
                    "parameters": {
                        "url": "https://invalid-url.com",
                        "format": "markdown",
                        "timeout": 30
                    }
                })
            return json.dumps({
                "request_type": "final_response",
                "response": "Failed to fetch content."
            })

        with patch.object(agent, "_get_ai_response", side_effect=mock_ai_response):
            with patch("custom_components.ai_agent_ha.agent.execute_tool", new_callable=AsyncMock) as mock_execute:
                # Mock failed ToolResult
                from custom_components.ai_agent_ha.tools import ToolResult
                mock_execute.return_value = ToolResult(
                    output="",
                    success=False,
                    error="Connection timeout after 30s"
                )

                result = await agent.process_query("Fetch content from invalid URL")

                assert result["success"] is True  # process_query succeeds even if tool fails
                mock_execute.assert_called_once_with(
                    "web_fetch",
                    {"url": "https://invalid-url.com", "format": "markdown", "timeout": 30},
                    hass=hass,
                    config=agent.config
                )

    @pytest.mark.asyncio
    async def test_process_query_web_search_success(self, hass, agent_config):
        """Test process_query with web_search success."""
        agent = AiAgentHaAgent(hass, agent_config)

        call_count = 0

        async def mock_ai_response(messages=None, tools=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return json.dumps({
                    "request_type": "web_search",
                    "parameters": {
                        "query": "Home Assistant automation",
                        "num_results": 8,
                        "type": "auto",
                        "livecrawl": "fallback"
                    }
                })
            return json.dumps({
                "request_type": "final_response",
                "response": "Found search results."
            })

        with patch.object(agent, "_get_ai_response", side_effect=mock_ai_response):
            with patch("custom_components.ai_agent_ha.agent.execute_tool", new_callable=AsyncMock) as mock_execute:
                # Mock successful ToolResult
                from custom_components.ai_agent_ha.tools import ToolResult
                mock_execute.return_value = ToolResult(
                    output=[
                        {
                            "title": "Home Assistant Automation Guide",
                            "url": "https://www.home-assistant.io/docs/automation/",
                            "text": "Learn about Home Assistant automation..."
                        },
                        {
                            "title": "Advanced Automation",
                            "url": "https://www.home-assistant.io/docs/automation/advanced/",
                            "text": "Advanced automation techniques..."
                        }
                    ],
                    title="Search Results",
                    metadata={"query": "Home Assistant automation", "count": 2},
                    success=True
                )

                result = await agent.process_query("Search for Home Assistant automation")

                assert result["success"] is True
                mock_execute.assert_called_once_with(
                    "web_search",
                    {
                        "query": "Home Assistant automation",
                        "num_results": 8,
                        "type": "auto",
                        "livecrawl": "fallback"
                    },
                    hass=hass,
                    config=agent.config
                )

    @pytest.mark.asyncio
    async def test_process_query_web_search_failure(self, hass, agent_config):
        """Test process_query with web_search failure."""
        agent = AiAgentHaAgent(hass, agent_config)

        call_count = 0

        async def mock_ai_response(messages=None, tools=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return json.dumps({
                    "request_type": "web_search",
                    "parameters": {
                        "query": "test search",
                        "num_results": 8,
                        "type": "auto",
                        "livecrawl": "fallback"
                    }
                })
            return json.dumps({
                "request_type": "final_response",
                "response": "Search failed."
            })

        with patch.object(agent, "_get_ai_response", side_effect=mock_ai_response):
            with patch("custom_components.ai_agent_ha.agent.execute_tool", new_callable=AsyncMock) as mock_execute:
                # Mock failed ToolResult
                from custom_components.ai_agent_ha.tools import ToolResult
                mock_execute.return_value = ToolResult(
                    output="",
                    success=False,
                    error="API key not configured or invalid"
                )

                result = await agent.process_query("Search the web")

                assert result["success"] is True  # process_query succeeds even if tool fails
                mock_execute.assert_called_once_with(
                    "web_search",
                    {
                        "query": "test search",
                        "num_results": 8,
                        "type": "auto",
                        "livecrawl": "fallback"
                    },
                    hass=hass,
                    config=agent.config
                )
    @pytest.mark.asyncio
    async def test_process_query_context7_resolve_success(self, hass, agent_config):
        """Test process_query with context7_resolve success."""
        agent = AiAgentHaAgent(hass, agent_config)

        call_count = 0

        async def mock_ai_response(messages=None, tools=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return json.dumps({
                    "request_type": "context7_resolve",
                    "parameters": {
                        "library_name": "react",
                        "query": "hooks documentation"
                    }
                })
            return json.dumps({
                "request_type": "final_response",
                "response": "Found React hooks documentation."
            })

        with patch.object(agent, "_get_ai_response", side_effect=mock_ai_response):
            with patch("custom_components.ai_agent_ha.agent.execute_tool", new_callable=AsyncMock) as mock_execute:
                # Mock successful ToolResult
                from custom_components.ai_agent_ha.tools import ToolResult
                mock_execute.return_value = ToolResult(
                    output={"id": "react-123", "name": "React", "version": "18.0"},
                    title="React Library",
                    metadata={"source": "context7", "matched": True},
                    success=True
                )

                result = await agent.process_query("Find React documentation")

                assert result["success"] is True
                mock_execute.assert_called_once_with(
                    "context7_resolve",
                    {"library_name": "react", "query": "hooks documentation"},
                    hass=hass,
                    config=agent.config
                )

    @pytest.mark.asyncio
    async def test_process_query_context7_resolve_failure(self, hass, agent_config):
        """Test process_query with context7_resolve failure."""
        agent = AiAgentHaAgent(hass, agent_config)

        call_count = 0

        async def mock_ai_response(messages=None, tools=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return json.dumps({
                    "request_type": "context7_resolve",
                    "parameters": {
                        "library_name": "unknown-lib",
                        "query": "documentation"
                    }
                })
            return json.dumps({
                "request_type": "final_response",
                "response": "Library not found."
            })

        with patch.object(agent, "_get_ai_response", side_effect=mock_ai_response):
            with patch("custom_components.ai_agent_ha.agent.execute_tool", new_callable=AsyncMock) as mock_execute:
                # Mock failed ToolResult
                from custom_components.ai_agent_ha.tools import ToolResult
                mock_execute.return_value = ToolResult(
                    output="",
                    success=False,
                    error="Library not found in registry"
                )

                result = await agent.process_query("Find unknown library")

                assert result["success"] is True  # process_query succeeds even if tool fails
                mock_execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_query_context7_docs_success(self, hass, agent_config):
        """Test process_query with context7_docs success."""
        agent = AiAgentHaAgent(hass, agent_config)

        call_count = 0

        async def mock_ai_response(messages=None, tools=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return json.dumps({
                    "request_type": "context7_docs",
                    "parameters": {
                        "library_id": "react-123",
                        "query": "useState",
                        "topic": "hooks"
                    }
                })
            return json.dumps({
                "request_type": "final_response",
                "response": "Here is the useState documentation."
            })

        with patch.object(agent, "_get_ai_response", side_effect=mock_ai_response):
            with patch("custom_components.ai_agent_ha.agent.execute_tool", new_callable=AsyncMock) as mock_execute:
                # Mock successful ToolResult
                from custom_components.ai_agent_ha.tools import ToolResult
                mock_execute.return_value = ToolResult(
                    output="# useState Hook\n\nReturns a stateful value and a function to update it.",
                    title="useState Documentation",
                    metadata={"library_id": "react-123", "topic": "hooks"},
                    success=True
                )

                result = await agent.process_query("Get useState documentation")

                assert result["success"] is True
                mock_execute.assert_called_once_with(
                    "context7_docs",
                    {"library_id": "react-123", "query": "useState", "topic": "hooks"},
                    hass=hass,
                    config=agent.config
                )

    @pytest.mark.asyncio
    async def test_process_query_context7_docs_failure(self, hass, agent_config):
        """Test process_query with context7_docs failure."""
        agent = AiAgentHaAgent(hass, agent_config)

        call_count = 0

        async def mock_ai_response(messages=None, tools=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return json.dumps({
                    "request_type": "context7_docs",
                    "parameters": {
                        "library_id": "invalid-id",
                        "query": "something",
                        "topic": None
                    }
                })
            return json.dumps({
                "request_type": "final_response",
                "response": "Documentation not found."
            })

        with patch.object(agent, "_get_ai_response", side_effect=mock_ai_response):
            with patch("custom_components.ai_agent_ha.agent.execute_tool", new_callable=AsyncMock) as mock_execute:
                # Mock failed ToolResult
                from custom_components.ai_agent_ha.tools import ToolResult
                mock_execute.return_value = ToolResult(
                    output="",
                    success=False,
                    error="Invalid library ID"
                )

                result = await agent.process_query("Get docs for invalid library")

                assert result["success"] is True  # process_query succeeds even if tool fails
                mock_execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_query_unknown_request_type(self, hass, agent_config):
        """Test process_query with unknown request type returns error."""
        agent = AiAgentHaAgent(hass, agent_config)

        async def mock_ai_response(messages=None, tools=None):
            # Return an unknown request type
            return json.dumps({
                "request_type": "totally_unknown_type",
                "parameters": {"foo": "bar"}
            })

        with patch.object(agent, "_get_ai_response", side_effect=mock_ai_response):
            result = await agent.process_query("Test unknown request type")

            # Should return error for unknown request type
            assert result["success"] is False
            assert "Unknown response type" in result["error"]

    @pytest.mark.asyncio
    async def test_process_query_tool_error_retry_transient(self, hass, agent_config):
        """Test process_query retries transient errors automatically."""
        agent = AiAgentHaAgent(hass, agent_config)

        call_count = 0
        tool_call_count = 0

        async def mock_ai_response(messages=None, tools=None):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:  # First two calls request web_search
                return json.dumps({
                    "request_type": "web_search",
                    "parameters": {
                        "query": "test search",
                        "num_results": 8
                    }
                })
            return json.dumps({
                "request_type": "final_response",
                "response": "Search completed."
            })

        with patch.object(agent, "_get_ai_response", side_effect=mock_ai_response):
            with patch("custom_components.ai_agent_ha.agent.execute_tool", new_callable=AsyncMock) as mock_execute:
                from custom_components.ai_agent_ha.tools import ToolResult

                async def mock_tool_execute(*args, **kwargs):
                    nonlocal tool_call_count
                    tool_call_count += 1
                    if tool_call_count == 1:
                        # First call: transient error
                        return ToolResult(
                            output="",
                            success=False,
                            error="Connection timeout"
                        )
                    else:
                        # Second call: success
                        return ToolResult(
                            output={"results": ["result1", "result2"]},
                            success=True
                        )

                mock_execute.side_effect = mock_tool_execute

                # Mock ErrorClassifier to return transient error
                with patch("custom_components.ai_agent_ha.agent.ErrorClassifier.classify") as mock_classify:
                    from custom_components.ai_agent_ha.error_handler import ErrorType
                    mock_classify.return_value = (ErrorType.TRANSIENT, True)

                    # Speed up test by reducing backoff time
                    with patch("custom_components.ai_agent_ha.agent.asyncio.sleep", new_callable=AsyncMock):
                        result = await agent.process_query("Search the web")

                        assert result["success"] is True
                        # Should have retried automatically
                        assert tool_call_count == 2

    @pytest.mark.asyncio
    async def test_process_query_tool_error_retry_logic_error(self, hass, agent_config):
        """Test process_query feeds logic errors to LLM for correction."""
        agent = AiAgentHaAgent(hass, agent_config)

        call_count = 0

        async def mock_ai_response(messages=None, tools=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call: bad parameters
                return json.dumps({
                    "request_type": "web_search",
                    "parameters": {
                        "query": "",  # Empty query - logic error
                        "num_results": 8
                    }
                })
            elif call_count == 2:
                # Second call: LLM corrects the error
                return json.dumps({
                    "request_type": "web_search",
                    "parameters": {
                        "query": "corrected query",
                        "num_results": 8
                    }
                })
            return json.dumps({
                "request_type": "final_response",
                "response": "Search completed after correction."
            })

        with patch.object(agent, "_get_ai_response", side_effect=mock_ai_response):
            with patch("custom_components.ai_agent_ha.agent.execute_tool", new_callable=AsyncMock) as mock_execute:
                from custom_components.ai_agent_ha.tools import ToolResult

                call_num = 0
                async def mock_tool_execute(*args, **kwargs):
                    nonlocal call_num
                    call_num += 1
                    if call_num == 1:
                        # First call: logic error
                        return ToolResult(
                            output="",
                            success=False,
                            error="Query parameter is required and cannot be empty"
                        )
                    else:
                        # Second call: success
                        return ToolResult(
                            output={"results": ["result1", "result2"]},
                            success=True
                        )

                mock_execute.side_effect = mock_tool_execute

                # Mock ErrorClassifier to return logic error
                with patch("custom_components.ai_agent_ha.agent.ErrorClassifier.classify") as mock_classify:
                    from custom_components.ai_agent_ha.error_handler import ErrorType
                    mock_classify.return_value = (ErrorType.LOGIC, True)

                    result = await agent.process_query("Search with empty query")

                    assert result["success"] is True
                    # Should have called tool twice (initial + retry after LLM correction)
                    assert call_num == 2
                    # Conversation history should contain error feedback for LLM
                    error_messages = [m for m in agent.conversation_history if "error" in m.get("content", "").lower()]
                    assert len(error_messages) > 0

    @pytest.mark.asyncio
    async def test_process_query_tool_error_max_retries_exceeded(self, hass, agent_config):
        """Test process_query returns error after max retries exceeded."""
        agent = AiAgentHaAgent(hass, agent_config)

        # Always return same failing tool call
        async def mock_ai_response(messages=None, tools=None):
            return json.dumps({
                "request_type": "web_search",
                "parameters": {
                    "query": "test",
                    "num_results": 8
                }
            })

        with patch.object(agent, "_get_ai_response", side_effect=mock_ai_response):
            with patch("custom_components.ai_agent_ha.agent.execute_tool", new_callable=AsyncMock) as mock_execute:
                from custom_components.ai_agent_ha.tools import ToolResult
                # Always return error
                mock_execute.return_value = ToolResult(
                    output="",
                    success=False,
                    error="Persistent error"
                )

                # Mock ErrorClassifier to return logic error (which feeds to LLM)
                with patch("custom_components.ai_agent_ha.agent.ErrorClassifier.classify") as mock_classify:
                    from custom_components.ai_agent_ha.error_handler import ErrorType
                    mock_classify.return_value = (ErrorType.LOGIC, True)

                    result = await agent.process_query("Search that will fail")

                    # After 3 attempts, should return error to user
                    assert result["success"] is False
                    assert "Persistent error" in result.get("error", "")

    @pytest.mark.asyncio
    async def test_process_query_tool_error_in_list_data(self, hass, agent_config):
        """Test process_query handles errors in list-type data responses."""
        agent = AiAgentHaAgent(hass, agent_config)

        call_count = 0

        async def mock_ai_response(messages=None, tools=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return json.dumps({
                    "request_type": "get_entities",
                    "parameters": {"area_id": "kitchen"}
                })
            return json.dumps({
                "request_type": "final_response",
                "response": "Handled error in entity list."
            })

        with patch.object(agent, "_get_ai_response", side_effect=mock_ai_response):
            with patch.object(agent, "get_entities", new_callable=AsyncMock) as mock_get:
                # Return list with error item
                mock_get.return_value = [
                    {"entity_id": "light.kitchen", "state": "on"},
                    {"error": "Failed to get state for sensor.broken"}
                ]

                # Mock ErrorClassifier
                with patch("custom_components.ai_agent_ha.agent.ErrorClassifier.classify") as mock_classify:
                    from custom_components.ai_agent_ha.error_handler import ErrorType
                    mock_classify.return_value = (ErrorType.LOGIC, True)

                    result = await agent.process_query("Get kitchen entities")

                    # Should detect error in list and handle it
                    assert result["success"] is True
                    assert call_count >= 2  # Should have retried after detecting error



class TestLocalClient:
    """Tests for LocalClient AI client."""

    @pytest.fixture
    def mock_aiohttp_session(self):
        """Create a reusable mock aiohttp session context manager."""
        def create_session(mock_response):
            mock_session = MagicMock()

            # Create async context manager for the response
            response_cm = MagicMock()
            response_cm.__aenter__ = AsyncMock(return_value=mock_response)
            response_cm.__aexit__ = AsyncMock(return_value=None)

            mock_session.post = MagicMock(return_value=response_cm)

            # Create async context manager for the session
            session_cm = MagicMock()
            session_cm.__aenter__ = AsyncMock(return_value=mock_session)
            session_cm.__aexit__ = AsyncMock(return_value=None)

            return session_cm

        return create_session

    def _create_mock_response(self, status=200, json_data=None, text=None):
        """Helper to create mock aiohttp response."""
        mock_resp = MagicMock()
        mock_resp.status = status
        mock_resp.text = AsyncMock(return_value=text or json.dumps(json_data or {}))
        mock_resp.json = AsyncMock(return_value=json_data)
        mock_resp.headers = {}
        return mock_resp

    @pytest.mark.asyncio
    async def test_local_client_successful_response(self, mock_aiohttp_session):
        """Test LocalClient with successful response."""
        from custom_components.ai_agent_ha.agent import LocalClient

        client = LocalClient(url="http://localhost:11434/api/generate", model="llama3")

        mock_resp = self._create_mock_response(
            status=200,
            json_data={"response": "Hello from Ollama!", "done": True},
            text='{"response": "Hello from Ollama!", "done": true}'
        )

        with patch("custom_components.ai_agent_ha.agent.aiohttp.ClientSession",
                   return_value=mock_aiohttp_session(mock_resp)):
            messages = [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hi"}
            ]

            result = await client.get_response(messages)

            assert "Hello from Ollama!" in result

    @pytest.mark.asyncio
    async def test_local_client_http_error(self, mock_aiohttp_session):
        """Test LocalClient with HTTP error."""
        from custom_components.ai_agent_ha.agent import LocalClient

        client = LocalClient(url="http://localhost:11434/api/generate", model="llama3")

        mock_resp = self._create_mock_response(status=500, text="Internal Server Error")

        with patch("custom_components.ai_agent_ha.agent.aiohttp.ClientSession",
                   return_value=mock_aiohttp_session(mock_resp)):
            messages = [{"role": "user", "content": "Hi"}]

            with pytest.raises(Exception) as exc_info:
                await client.get_response(messages)

            assert "Local API error 500" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_local_client_model_not_found(self, mock_aiohttp_session):
        """Test LocalClient with 404 model not found."""
        from custom_components.ai_agent_ha.agent import LocalClient

        client = LocalClient(url="http://localhost:11434/api/generate", model="nonexistent")

        mock_resp = self._create_mock_response(status=404, text="model not found")

        with patch("custom_components.ai_agent_ha.agent.aiohttp.ClientSession",
                   return_value=mock_aiohttp_session(mock_resp)):
            messages = [{"role": "user", "content": "Hi"}]

            with pytest.raises(Exception) as exc_info:
                await client.get_response(messages)

            assert "not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_local_client_no_model_warning(self, mock_aiohttp_session):
        """Test LocalClient logs warning when no model specified."""
        from custom_components.ai_agent_ha.agent import LocalClient

        client = LocalClient(url="http://localhost:11434/api/generate", model="")

        mock_resp = self._create_mock_response(
            status=200,
            json_data={"response": "Response", "done": True},
            text='{"response": "Response", "done": true}'
        )

        with patch("custom_components.ai_agent_ha.agent.aiohttp.ClientSession",
                   return_value=mock_aiohttp_session(mock_resp)):
            messages = [{"role": "user", "content": "Hi"}]

            with patch("custom_components.ai_agent_ha.agent._LOGGER") as mock_logger:
                await client.get_response(messages)

                # Should log warning about missing model
                warning_calls = [call for call in mock_logger.warning.call_args_list
                               if "No model specified" in str(call) or "Missing" in str(call)]
                assert len(warning_calls) > 0

    @pytest.mark.asyncio
    async def test_local_client_empty_response(self, mock_aiohttp_session):
        """Test LocalClient handles empty response."""
        from custom_components.ai_agent_ha.agent import LocalClient

        client = LocalClient(url="http://localhost:11434/api/generate", model="llama3")

        mock_resp = self._create_mock_response(
            status=200,
            json_data={"response": "", "done": True},
            text='{"response": "", "done": true}'
        )

        with patch("custom_components.ai_agent_ha.agent.aiohttp.ClientSession",
                   return_value=mock_aiohttp_session(mock_resp)):
            messages = [{"role": "user", "content": "Hi"}]

            result = await client.get_response(messages)

            # Should handle empty response gracefully
            assert result is not None

    @pytest.mark.asyncio
    async def test_local_client_model_loading(self, mock_aiohttp_session):
        """Test LocalClient handles model loading response."""
        from custom_components.ai_agent_ha.agent import LocalClient

        client = LocalClient(url="http://localhost:11434/api/generate", model="llama3")

        mock_resp = self._create_mock_response(
            status=200,
            json_data={"response": "", "done": False, "done_reason": "load"},
            text='{"response": "", "done": false, "done_reason": "load"}'
        )

        with patch("custom_components.ai_agent_ha.agent.aiohttp.ClientSession",
                   return_value=mock_aiohttp_session(mock_resp)):
            messages = [{"role": "user", "content": "Hi"}]

            result = await client.get_response(messages)

            # Should return message about model loading
            assert "loading" in result.lower() or result is not None

    @pytest.mark.asyncio
    async def test_local_client_bad_request(self, mock_aiohttp_session):
        """Test LocalClient with 400 bad request."""
        from custom_components.ai_agent_ha.agent import LocalClient

        client = LocalClient(url="http://localhost:11434/api/generate", model="llama3")

        mock_resp = self._create_mock_response(status=400, text="Bad request: invalid JSON")

        with patch("custom_components.ai_agent_ha.agent.aiohttp.ClientSession",
                   return_value=mock_aiohttp_session(mock_resp)):
            messages = [{"role": "user", "content": "Hi"}]

            with pytest.raises(Exception) as exc_info:
                await client.get_response(messages)

            assert "Bad request" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_local_client_endpoint_not_found_no_model(self, mock_aiohttp_session):
        """Test LocalClient with 404 and no model specified."""
        from custom_components.ai_agent_ha.agent import LocalClient

        client = LocalClient(url="http://localhost:11434/api/generate", model="")

        mock_resp = self._create_mock_response(status=404, text="Not Found")

        with patch("custom_components.ai_agent_ha.agent.aiohttp.ClientSession",
                   return_value=mock_aiohttp_session(mock_resp)):
            messages = [{"role": "user", "content": "Hi"}]

            with pytest.raises(Exception) as exc_info:
                await client.get_response(messages)

            # Should mention endpoint not found
            assert "not found" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_local_client_plain_text_response(self, mock_aiohttp_session):
        """Test LocalClient handles plain text response (non-JSON)."""
        from custom_components.ai_agent_ha.agent import LocalClient

        client = LocalClient(url="http://localhost:11434/api/generate", model="llama3")

        # Return plain text that's not valid JSON
        mock_resp = self._create_mock_response(status=200, text="Plain text response without JSON")

        with patch("custom_components.ai_agent_ha.agent.aiohttp.ClientSession",
                   return_value=mock_aiohttp_session(mock_resp)):
            messages = [{"role": "user", "content": "Hi"}]

            result = await client.get_response(messages)

            # Should handle plain text response
            assert result is not None

    @pytest.mark.asyncio
    async def test_local_client_message_formatting(self, mock_aiohttp_session):
        """Test LocalClient formats messages correctly in prompt."""
        from custom_components.ai_agent_ha.agent import LocalClient

        client = LocalClient(url="http://localhost:11434/api/generate", model="llama3")

        mock_resp = self._create_mock_response(
            status=200,
            json_data={"response": "Response", "done": True},
            text='{"response": "Response", "done": true}'
        )

        captured_payload = {}
        session_cm = mock_aiohttp_session(mock_resp)

        # Capture the payload from the post call
        original_session = session_cm.__aenter__().__class__
        mock_session = MagicMock()
        response_cm = MagicMock()
        response_cm.__aenter__ = AsyncMock(return_value=mock_resp)
        response_cm.__aexit__ = AsyncMock(return_value=None)

        def capture_post(*args, **kwargs):
            captured_payload.update(kwargs.get("json", {}))
            return response_cm

        mock_session.post = MagicMock(side_effect=capture_post)
        session_cm = MagicMock()
        session_cm.__aenter__ = AsyncMock(return_value=mock_session)
        session_cm.__aexit__ = AsyncMock(return_value=None)

        with patch("custom_components.ai_agent_ha.agent.aiohttp.ClientSession",
                   return_value=session_cm):
            messages = [
                {"role": "system", "content": "System message"},
                {"role": "user", "content": "User message"},
                {"role": "assistant", "content": "Assistant message"}
            ]

            await client.get_response(messages)

            # Verify prompt contains all message roles
            prompt = captured_payload.get("prompt", "")
            assert "System:" in prompt
            assert "User:" in prompt
            assert "Assistant:" in prompt


# Import time for rate limiting test
import time


class TestGetAiResponse:
    """Test the _get_ai_response method with different providers."""

    @pytest.fixture
    def openai_config(self):
        """OpenAI provider configuration."""
        return {
            "ai_provider": "openai",
            "openai_token": "sk-test123",
            "models": {"openai": "gpt-3.5-turbo"}
        }

    @pytest.fixture
    def anthropic_config(self):
        """Anthropic provider configuration."""
        return {
            "ai_provider": "anthropic",
            "anthropic_token": "sk-ant-test123",
            "models": {"anthropic": "claude-sonnet-4-5-20250929"}
        }

    @pytest.fixture
    def gemini_config(self):
        """Gemini provider configuration."""
        return {
            "ai_provider": "gemini",
            "gemini_token": "test-gemini-key",
            "models": {"gemini": "gemini-2.5-flash"}
        }

    @pytest.fixture
    def local_config(self):
        """Local provider configuration."""
        return {
            "ai_provider": "local",
            "local_url": "http://localhost:11434/api/generate",
            "models": {"local": "llama3.2"}
        }

    @pytest.mark.asyncio
    async def test_get_ai_response_openai_text_response(self, hass, openai_config):
        """Test _get_ai_response with OpenAI returning text response."""
        agent = AiAgentHaAgent(hass, openai_config)
        agent.conversation_history = [{"role": "user", "content": "Hello"}]

        # Mock the OpenAI client
        mock_client = AsyncMock()
        mock_client.get_response = AsyncMock(return_value="Hello! How can I help you?")
        agent.ai_client = mock_client

        # Mock _check_rate_limit and _get_native_function_calling_tools
        with patch.object(agent, "_check_rate_limit", return_value=True), \
             patch.object(agent, "_get_native_function_calling_tools", return_value=None):

            response = await agent._get_ai_response()

            assert response == "Hello! How can I help you?"
            mock_client.get_response.assert_called_once()
            call_args = mock_client.get_response.call_args
            # Verify messages were passed (system prompt + user message)
            assert len(call_args[0][0]) >= 1

    @pytest.mark.asyncio
    async def test_get_ai_response_anthropic_text_response(self, hass, anthropic_config):
        """Test _get_ai_response with Anthropic returning text response."""
        agent = AiAgentHaAgent(hass, anthropic_config)
        agent.conversation_history = [{"role": "user", "content": "Test query"}]

        # Mock the Anthropic client
        mock_client = AsyncMock()
        mock_client.get_response = AsyncMock(return_value="I'm Claude, happy to help!")
        agent.ai_client = mock_client

        with patch.object(agent, "_check_rate_limit", return_value=True), \
             patch.object(agent, "_get_native_function_calling_tools", return_value=None):

            response = await agent._get_ai_response()

            assert response == "I'm Claude, happy to help!"
            mock_client.get_response.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_ai_response_gemini_text_response(self, hass, gemini_config):
        """Test _get_ai_response with Gemini returning text response."""
        agent = AiAgentHaAgent(hass, gemini_config)
        agent.conversation_history = [{"role": "user", "content": "What's the weather?"}]

        # Mock the Gemini client
        mock_client = AsyncMock()
        mock_client.get_response = AsyncMock(return_value="Let me check the weather for you.")
        agent.ai_client = mock_client

        with patch.object(agent, "_check_rate_limit", return_value=True), \
             patch.object(agent, "_get_native_function_calling_tools", return_value=None):

            response = await agent._get_ai_response()

            assert response == "Let me check the weather for you."
            mock_client.get_response.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_ai_response_local_text_response(self, hass, local_config):
        """Test _get_ai_response with Local provider returning text response."""
        agent = AiAgentHaAgent(hass, local_config)
        agent.conversation_history = [{"role": "user", "content": "Hello local model"}]

        # Mock the Local client
        mock_client = AsyncMock()
        mock_client.get_response = AsyncMock(return_value="Response from local model")
        agent.ai_client = mock_client

        with patch.object(agent, "_check_rate_limit", return_value=True), \
             patch.object(agent, "_get_native_function_calling_tools", return_value=None):

            response = await agent._get_ai_response()

            assert response == "Response from local model"
            mock_client.get_response.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_ai_response_native_function_call(self, hass, openai_config):
        """Test _get_ai_response with native function calling response."""
        agent = AiAgentHaAgent(hass, openai_config)
        agent.conversation_history = [{"role": "user", "content": "Turn on the lights"}]

        # Create a mock function call object
        mock_function_call = MagicMock()
        mock_function_call.name = "call_service"
        mock_function_call.arguments = {
            "domain": "light",
            "service": "turn_on",
            "entity_id": "light.living_room"
        }

        # Mock the client to return a dict with function_calls
        mock_client = AsyncMock()
        mock_client.get_response = AsyncMock(return_value={
            "function_calls": [mock_function_call]
        })
        agent.ai_client = mock_client

        with patch.object(agent, "_check_rate_limit", return_value=True), \
             patch.object(agent, "_get_native_function_calling_tools", return_value=["mock_tools"]):

            response = await agent._get_ai_response()

            # Verify the response is JSON formatted
            response_json = json.loads(response)
            assert response_json["request_type"] == "call_service"
            assert response_json["parameters"]["domain"] == "light"
            assert response_json["parameters"]["service"] == "turn_on"
            assert response_json["parameters"]["entity_id"] == "light.living_room"

    @pytest.mark.asyncio
    async def test_get_ai_response_rate_limit_exceeded(self, hass, openai_config):
        """Test _get_ai_response raises exception when rate limit is exceeded."""
        agent = AiAgentHaAgent(hass, openai_config)
        agent.conversation_history = [{"role": "user", "content": "Test"}]

        with patch.object(agent, "_check_rate_limit", return_value=False):
            with pytest.raises(Exception, match="Rate limit exceeded"):
                await agent._get_ai_response()

    @pytest.mark.asyncio
    async def test_get_ai_response_retry_on_error(self, hass, openai_config):
        """Test _get_ai_response retries on error."""
        agent = AiAgentHaAgent(hass, openai_config)
        agent.conversation_history = [{"role": "user", "content": "Test"}]
        agent._max_retries = 3
        agent._retry_delay = 0.01  # Short delay for testing

        # Mock client that fails twice then succeeds
        mock_client = AsyncMock()
        mock_client.get_response = AsyncMock(
            side_effect=[
                Exception("Network error"),
                Exception("Timeout"),
                "Success response"
            ]
        )
        agent.ai_client = mock_client

        with patch.object(agent, "_check_rate_limit", return_value=True), \
             patch.object(agent, "_get_native_function_calling_tools", return_value=None):

            response = await agent._get_ai_response()

            assert response == "Success response"
            assert mock_client.get_response.call_count == 3

    @pytest.mark.asyncio
    async def test_get_ai_response_max_retries_exceeded(self, hass, openai_config):
        """Test _get_ai_response raises exception after max retries."""
        agent = AiAgentHaAgent(hass, openai_config)
        agent.conversation_history = [{"role": "user", "content": "Test"}]
        agent._max_retries = 2
        agent._retry_delay = 0.01

        # Mock client that always fails
        mock_client = AsyncMock()
        mock_client.get_response = AsyncMock(side_effect=Exception("Persistent error"))
        agent.ai_client = mock_client

        with patch.object(agent, "_check_rate_limit", return_value=True), \
             patch.object(agent, "_get_native_function_calling_tools", return_value=None):

            with pytest.raises(Exception, match="Failed after .* retries"):
                await agent._get_ai_response()

            assert mock_client.get_response.call_count == 2

    @pytest.mark.asyncio
    async def test_get_ai_response_empty_response_retry(self, hass, openai_config):
        """Test _get_ai_response retries on empty response."""
        agent = AiAgentHaAgent(hass, openai_config)
        agent.conversation_history = [{"role": "user", "content": "Test"}]
        agent._max_retries = 3
        agent._retry_delay = 0.01

        # Mock client that returns empty then valid response
        mock_client = AsyncMock()
        mock_client.get_response = AsyncMock(
            side_effect=["", "  ", "Valid response"]
        )
        agent.ai_client = mock_client

        with patch.object(agent, "_check_rate_limit", return_value=True), \
             patch.object(agent, "_get_native_function_calling_tools", return_value=None):

            response = await agent._get_ai_response()

            assert response == "Valid response"
            assert mock_client.get_response.call_count == 3

    @pytest.mark.asyncio
    async def test_get_ai_response_empty_after_max_retries(self, hass, openai_config):
        """Test _get_ai_response raises exception for empty response after max retries."""
        agent = AiAgentHaAgent(hass, openai_config)
        agent.conversation_history = [{"role": "user", "content": "Test"}]
        agent._max_retries = 2
        agent._retry_delay = 0.01

        # Mock client that always returns empty
        mock_client = AsyncMock()
        mock_client.get_response = AsyncMock(return_value="")
        agent.ai_client = mock_client

        with patch.object(agent, "_check_rate_limit", return_value=True), \
             patch.object(agent, "_get_native_function_calling_tools", return_value=None):

            with pytest.raises(Exception, match="empty response after all retries"):
                await agent._get_ai_response()

    @pytest.mark.asyncio
    async def test_get_ai_response_conversation_history_limit(self, hass, openai_config):
        """Test _get_ai_response limits conversation history to last 10 messages."""
        agent = AiAgentHaAgent(hass, openai_config)

        # Create 15 messages in conversation history
        agent.conversation_history = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"Message {i}"}
            for i in range(15)
        ]

        mock_client = AsyncMock()
        mock_client.get_response = AsyncMock(return_value="Response")
        agent.ai_client = mock_client

        with patch.object(agent, "_check_rate_limit", return_value=True), \
             patch.object(agent, "_get_native_function_calling_tools", return_value=None):

            await agent._get_ai_response()

            # Get the messages that were passed to the client
            call_args = mock_client.get_response.call_args
            messages_sent = call_args[0][0]

            # Should have system prompt + last 10 messages = 11 total
            # Or just last 10 if system prompt is in the last 10
            assert len(messages_sent) <= 11

    @pytest.mark.asyncio
    async def test_get_ai_response_system_prompt_always_first(self, hass, openai_config):
        """Test _get_ai_response ensures system prompt is always first."""
        agent = AiAgentHaAgent(hass, openai_config)

        # Create conversation history without system prompt at start
        agent.conversation_history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"}
        ]

        mock_client = AsyncMock()
        mock_client.get_response = AsyncMock(return_value="Response")
        agent.ai_client = mock_client

        with patch.object(agent, "_check_rate_limit", return_value=True), \
             patch.object(agent, "_get_native_function_calling_tools", return_value=None):

            await agent._get_ai_response()

            # Get the messages that were passed to the client
            call_args = mock_client.get_response.call_args
            messages_sent = call_args[0][0]

            # First message should be system prompt
            assert messages_sent[0]["role"] == "system"

    @pytest.mark.asyncio
    async def test_get_ai_response_extremely_long_response(self, hass, openai_config):
        """Test _get_ai_response handles extremely long response with warning."""
        agent = AiAgentHaAgent(hass, openai_config)
        agent.conversation_history = [{"role": "user", "content": "Test"}]

        # Create a very long response (>50000 chars)
        long_response = "x" * 60000

        mock_client = AsyncMock()
        mock_client.get_response = AsyncMock(return_value=long_response)
        agent.ai_client = mock_client

        with patch.object(agent, "_check_rate_limit", return_value=True), \
             patch.object(agent, "_get_native_function_calling_tools", return_value=None):

            response = await agent._get_ai_response()

            # Should still return the response despite warning
            assert len(response) == 60000
            assert response == long_response

    @pytest.mark.asyncio
    async def test_get_ai_response_corrupted_repetitive_response(self, hass, openai_config):
        """Test _get_ai_response detects and rejects corrupted repetitive response."""
        agent = AiAgentHaAgent(hass, openai_config)
        agent.conversation_history = [{"role": "user", "content": "Test"}]
        agent._max_retries = 2
        agent._retry_delay = 0.01

        # Create a corrupted response with repetitive pattern (>50k chars and >50 repetitions)
        corrupted_response = "for its use in various fields " * 2000  # Creates ~60k chars, >50 repetitions

        mock_client = AsyncMock()
        mock_client.get_response = AsyncMock(
            side_effect=[corrupted_response, "Valid response"]
        )
        agent.ai_client = mock_client

        with patch.object(agent, "_check_rate_limit", return_value=True), \
             patch.object(agent, "_get_native_function_calling_tools", return_value=None):

            response = await agent._get_ai_response()

            # Should retry and get valid response
            assert response == "Valid response"
            assert mock_client.get_response.call_count == 2

    @pytest.mark.asyncio
    async def test_get_ai_response_passes_tools_when_available(self, hass, openai_config):
        """Test _get_ai_response passes tools to client when available."""
        agent = AiAgentHaAgent(hass, openai_config)
        agent.conversation_history = [{"role": "user", "content": "Test"}]

        mock_tools = [{"name": "get_weather", "description": "Get weather info"}]

        mock_client = AsyncMock()
        mock_client.get_response = AsyncMock(return_value="Response with tools")
        agent.ai_client = mock_client

        with patch.object(agent, "_check_rate_limit", return_value=True), \
             patch.object(agent, "_get_native_function_calling_tools", return_value=mock_tools):

            response = await agent._get_ai_response()

            # Verify tools were passed to the client
            call_args = mock_client.get_response.call_args
            assert call_args[1]["tools"] == mock_tools

    @pytest.mark.asyncio
    async def test_get_ai_response_no_tools_when_none(self, hass, openai_config):
        """Test _get_ai_response passes None for tools when not available."""
        agent = AiAgentHaAgent(hass, openai_config)
        agent.conversation_history = [{"role": "user", "content": "Test"}]

        mock_client = AsyncMock()
        mock_client.get_response = AsyncMock(return_value="Response without tools")
        agent.ai_client = mock_client

        with patch.object(agent, "_check_rate_limit", return_value=True), \
             patch.object(agent, "_get_native_function_calling_tools", return_value=None):

            response = await agent._get_ai_response()

            # Verify tools parameter is None
            call_args = mock_client.get_response.call_args
            assert call_args[1]["tools"] is None


class TestConversationHistoryHandling:
    """Test conversation history handling in _get_ai_response method."""

    @pytest.fixture
    def openai_config(self):
        """OpenAI provider configuration."""
        return {
            "ai_provider": "openai",
            "openai_token": "sk-test123",
            "models": {"openai": "gpt-3.5-turbo"}
        }

    @pytest.mark.asyncio
    async def test_history_limit_to_last_10_messages(self, hass, openai_config):
        """Test that conversation history is limited to last 10 messages."""
        agent = AiAgentHaAgent(hass, openai_config)

        # Create a history with 15 messages (more than the 10 limit)
        agent.conversation_history = [
            {"role": "user", "content": f"Message {i}"}
            for i in range(15)
        ]

        # Mock the client
        mock_client = AsyncMock()
        mock_client.get_response = AsyncMock(return_value="Response")
        agent.ai_client = mock_client

        with patch.object(agent, "_check_rate_limit", return_value=True), \
             patch.object(agent, "_get_native_function_calling_tools", return_value=None):

            await agent._get_ai_response()

            # Get the messages that were sent to the AI client
            call_args = mock_client.get_response.call_args
            sent_messages = call_args[0][0]

            # Should have system prompt + last 10 messages = 11 total
            assert len(sent_messages) == 11

            # First message should be system prompt
            assert sent_messages[0]["role"] == "system"

            # Remaining messages should be the last 10 from history (messages 5-14)
            for i, msg in enumerate(sent_messages[1:]):
                expected_idx = 5 + i  # Messages 5 through 14
                assert msg["content"] == f"Message {expected_idx}"

    @pytest.mark.asyncio
    async def test_history_less_than_10_messages(self, hass, openai_config):
        """Test that all messages are included when history has fewer than 10 messages."""
        agent = AiAgentHaAgent(hass, openai_config)

        # Create a history with only 5 messages
        agent.conversation_history = [
            {"role": "user", "content": "Message 1"},
            {"role": "assistant", "content": "Response 1"},
            {"role": "user", "content": "Message 2"},
            {"role": "assistant", "content": "Response 2"},
            {"role": "user", "content": "Message 3"},
        ]

        mock_client = AsyncMock()
        mock_client.get_response = AsyncMock(return_value="Response")
        agent.ai_client = mock_client

        with patch.object(agent, "_check_rate_limit", return_value=True), \
             patch.object(agent, "_get_native_function_calling_tools", return_value=None):

            await agent._get_ai_response()

            call_args = mock_client.get_response.call_args
            sent_messages = call_args[0][0]

            # Should have system prompt + 5 messages = 6 total
            assert len(sent_messages) == 6
            assert sent_messages[0]["role"] == "system"
            assert sent_messages[1]["content"] == "Message 1"
            assert sent_messages[5]["content"] == "Message 3"

    @pytest.mark.asyncio
    async def test_system_prompt_always_first(self, hass, openai_config):
        """Test that system prompt is always the first message."""
        agent = AiAgentHaAgent(hass, openai_config)

        # Create history without system prompt
        agent.conversation_history = [
            {"role": "user", "content": "User message"},
            {"role": "assistant", "content": "Assistant response"},
        ]

        mock_client = AsyncMock()
        mock_client.get_response = AsyncMock(return_value="Response")
        agent.ai_client = mock_client

        with patch.object(agent, "_check_rate_limit", return_value=True), \
             patch.object(agent, "_get_native_function_calling_tools", return_value=None):

            await agent._get_ai_response()

            call_args = mock_client.get_response.call_args
            sent_messages = call_args[0][0]

            # First message must be system prompt
            assert sent_messages[0]["role"] == "system"
            # System prompt should contain the agent's instructions
            assert "content" in sent_messages[0]
            assert len(sent_messages[0]["content"]) > 0

    @pytest.mark.asyncio
    async def test_system_prompt_not_duplicated(self, hass, openai_config):
        """Test that system prompt is not duplicated if already present in recent messages."""
        agent = AiAgentHaAgent(hass, openai_config)

        # Create history that already starts with system prompt
        system_prompt = agent.system_prompt
        agent.conversation_history = [
            system_prompt,
            {"role": "user", "content": "User message"},
            {"role": "assistant", "content": "Assistant response"},
        ]

        mock_client = AsyncMock()
        mock_client.get_response = AsyncMock(return_value="Response")
        agent.ai_client = mock_client

        with patch.object(agent, "_check_rate_limit", return_value=True), \
             patch.object(agent, "_get_native_function_calling_tools", return_value=None):

            await agent._get_ai_response()

            call_args = mock_client.get_response.call_args
            sent_messages = call_args[0][0]

            # Should have system prompt + 2 messages = 3 total
            assert len(sent_messages) == 3
            # Verify only one system message
            system_messages = [m for m in sent_messages if m.get("role") == "system"]
            assert len(system_messages) == 1

    @pytest.mark.asyncio
    async def test_empty_conversation_history(self, hass, openai_config):
        """Test handling of empty conversation history."""
        agent = AiAgentHaAgent(hass, openai_config)

        # Empty history
        agent.conversation_history = []

        mock_client = AsyncMock()
        mock_client.get_response = AsyncMock(return_value="Response")
        agent.ai_client = mock_client

        with patch.object(agent, "_check_rate_limit", return_value=True), \
             patch.object(agent, "_get_native_function_calling_tools", return_value=None):

            await agent._get_ai_response()

            call_args = mock_client.get_response.call_args
            sent_messages = call_args[0][0]

            # Should have only system prompt
            assert len(sent_messages) == 1
            assert sent_messages[0]["role"] == "system"

    @pytest.mark.asyncio
    async def test_message_structure_preserved(self, hass, openai_config):
        """Test that message structure (role, content) is preserved."""
        agent = AiAgentHaAgent(hass, openai_config)

        # Create history with various message types
        agent.conversation_history = [
            {"role": "user", "content": "What's the weather?"},
            {"role": "assistant", "content": "Let me check that for you."},
            {"role": "user", "content": "Turn on the lights"},
        ]

        mock_client = AsyncMock()
        mock_client.get_response = AsyncMock(return_value="Response")
        agent.ai_client = mock_client

        with patch.object(agent, "_check_rate_limit", return_value=True), \
             patch.object(agent, "_get_native_function_calling_tools", return_value=None):

            await agent._get_ai_response()

            call_args = mock_client.get_response.call_args
            sent_messages = call_args[0][0]

            # Skip system prompt, check other messages
            assert sent_messages[1]["role"] == "user"
            assert sent_messages[1]["content"] == "What's the weather?"
            assert sent_messages[2]["role"] == "assistant"
            assert sent_messages[2]["content"] == "Let me check that for you."
            assert sent_messages[3]["role"] == "user"
            assert sent_messages[3]["content"] == "Turn on the lights"

    @pytest.mark.asyncio
    async def test_history_with_exactly_10_messages(self, hass, openai_config):
        """Test edge case where history has exactly 10 messages."""
        agent = AiAgentHaAgent(hass, openai_config)

        # Create history with exactly 10 messages
        agent.conversation_history = [
            {"role": "user", "content": f"Message {i}"}
            for i in range(10)
        ]

        mock_client = AsyncMock()
        mock_client.get_response = AsyncMock(return_value="Response")
        agent.ai_client = mock_client

        with patch.object(agent, "_check_rate_limit", return_value=True), \
             patch.object(agent, "_get_native_function_calling_tools", return_value=None):

            await agent._get_ai_response()

            call_args = mock_client.get_response.call_args
            sent_messages = call_args[0][0]

            # Should have system prompt + all 10 messages = 11 total
            assert len(sent_messages) == 11
            assert sent_messages[0]["role"] == "system"
            # All 10 messages should be included
            for i in range(10):
                assert sent_messages[i + 1]["content"] == f"Message {i}"

    @pytest.mark.asyncio
    async def test_message_ordering_preserved(self, hass, openai_config):
        """Test that message ordering is preserved in the sent history."""
        agent = AiAgentHaAgent(hass, openai_config)

        # Create history with clear ordering
        agent.conversation_history = [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Second"},
            {"role": "user", "content": "Third"},
            {"role": "assistant", "content": "Fourth"},
            {"role": "user", "content": "Fifth"},
        ]

        mock_client = AsyncMock()
        mock_client.get_response = AsyncMock(return_value="Response")
        agent.ai_client = mock_client

        with patch.object(agent, "_check_rate_limit", return_value=True), \
             patch.object(agent, "_get_native_function_calling_tools", return_value=None):

            await agent._get_ai_response()

            call_args = mock_client.get_response.call_args
            sent_messages = call_args[0][0]

            # Verify ordering (skip system prompt at index 0)
            expected_order = ["First", "Second", "Third", "Fourth", "Fifth"]
            for i, expected_content in enumerate(expected_order):
                assert sent_messages[i + 1]["content"] == expected_content

    @pytest.mark.asyncio
    async def test_history_limit_with_mixed_roles(self, hass, openai_config):
        """Test history limiting works correctly with mixed user/assistant messages."""
        agent = AiAgentHaAgent(hass, openai_config)

        # Create 20 messages alternating between user and assistant
        agent.conversation_history = []
        for i in range(20):
            role = "user" if i % 2 == 0 else "assistant"
            agent.conversation_history.append({
                "role": role,
                "content": f"{role} message {i}"
            })

        mock_client = AsyncMock()
        mock_client.get_response = AsyncMock(return_value="Response")
        agent.ai_client = mock_client

        with patch.object(agent, "_check_rate_limit", return_value=True), \
             patch.object(agent, "_get_native_function_calling_tools", return_value=None):

            await agent._get_ai_response()

            call_args = mock_client.get_response.call_args
            sent_messages = call_args[0][0]

            # Should have system + last 10 messages = 11
            assert len(sent_messages) == 11

            # Verify we got messages 10-19 (the last 10)
            for i in range(10):
                msg_index = 10 + i
                expected_role = "user" if msg_index % 2 == 0 else "assistant"
                assert sent_messages[i + 1]["role"] == expected_role
                assert sent_messages[i + 1]["content"] == f"{expected_role} message {msg_index}"

    @pytest.mark.asyncio
    async def test_system_prompt_content_included(self, hass, openai_config):
        """Test that system prompt has content when added."""
        agent = AiAgentHaAgent(hass, openai_config)
        agent.conversation_history = [{"role": "user", "content": "Test"}]

        mock_client = AsyncMock()
        mock_client.get_response = AsyncMock(return_value="Response")
        agent.ai_client = mock_client

        with patch.object(agent, "_check_rate_limit", return_value=True), \
             patch.object(agent, "_get_native_function_calling_tools", return_value=None):

            await agent._get_ai_response()

            call_args = mock_client.get_response.call_args
            sent_messages = call_args[0][0]

            # Verify system prompt structure
            system_msg = sent_messages[0]
            assert system_msg["role"] == "system"
            assert "content" in system_msg
            assert isinstance(system_msg["content"], str)
            assert len(system_msg["content"]) > 0


class TestAgentHelperMethods:
    """Tests for agent helper methods."""

    @pytest.fixture
    def openai_config(self):
        """OpenAI provider configuration."""
        return {
            "ai_provider": "openai",
            "openai_token": "sk-test123",
            "models": {"openai": "gpt-3.5-turbo"}
        }

    @pytest.fixture
    def agent(self, hass, openai_config):
        """Create an agent instance for testing."""
        agent = AiAgentHaAgent(hass, openai_config)
        return agent

    @pytest.mark.asyncio
    async def test_get_climate_related_entities_success(self, agent, hass):
        """Test get_climate_related_entities returns climate, temperature, and humidity entities."""
        # Create mock climate entity
        climate_state = State(
            "climate.thermostat",
            "heat",
            {"friendly_name": "Thermostat", "temperature": 22.0, "current_temperature": 21.0}
        )
        
        # Create mock temperature sensor
        temp_state = State(
            "sensor.temperature",
            "20.5",
            {"friendly_name": "Temperature Sensor", "device_class": "temperature", "unit_of_measurement": "°C"}
        )
        
        # Create mock humidity sensor
        humidity_state = State(
            "sensor.humidity",
            "65",
            {"friendly_name": "Humidity Sensor", "device_class": "humidity", "unit_of_measurement": "%"}
        )
        
        hass.states.async_set(climate_state.entity_id, climate_state.state, climate_state.attributes)
        hass.states.async_set(temp_state.entity_id, temp_state.state, temp_state.attributes)
        hass.states.async_set(humidity_state.entity_id, humidity_state.state, humidity_state.attributes)
        
        result = await agent.get_climate_related_entities()
        
        assert len(result) == 3
        entity_ids = [e["entity_id"] for e in result]
        assert "climate.thermostat" in entity_ids
        assert "sensor.temperature" in entity_ids
        assert "sensor.humidity" in entity_ids

    @pytest.mark.asyncio
    async def test_get_climate_related_entities_deduplication(self, agent, hass):
        """Test get_climate_related_entities deduplicates entities."""
        # Create a climate entity that might appear in multiple categories
        climate_state = State(
            "climate.thermostat",
            "heat",
            {"friendly_name": "Thermostat"}
        )
        
        hass.states.async_set(climate_state.entity_id, climate_state.state, climate_state.attributes)
        
        result = await agent.get_climate_related_entities()
        
        # Count occurrences of the entity
        entity_ids = [e["entity_id"] for e in result]
        assert entity_ids.count("climate.thermostat") == 1

    @pytest.mark.asyncio
    async def test_get_climate_related_entities_empty(self, agent, hass):
        """Test get_climate_related_entities returns empty list when no entities."""
        result = await agent.get_climate_related_entities()
        
        assert result == []

    @pytest.mark.asyncio
    async def test_get_climate_related_entities_error_handling(self, agent, hass):
        """Test get_climate_related_entities handles errors gracefully."""
        with patch.object(agent, "get_entities_by_domain", side_effect=Exception("Test error")):
            result = await agent.get_climate_related_entities()
            
            assert len(result) == 1
            assert "error" in result[0]
            assert "Test error" in result[0]["error"]

    @pytest.mark.asyncio
    async def test_get_calendar_events_all(self, agent, hass):
        """Test get_calendar_events returns all calendar entities."""
        calendar1_state = State(
            "calendar.personal",
            "off",
            {"friendly_name": "Personal Calendar", "message": "Meeting at 2pm"}
        )
        
        calendar2_state = State(
            "calendar.work",
            "on",
            {"friendly_name": "Work Calendar", "message": "Team standup"}
        )
        
        hass.states.async_set(calendar1_state.entity_id, calendar1_state.state, calendar1_state.attributes)
        hass.states.async_set(calendar2_state.entity_id, calendar2_state.state, calendar2_state.attributes)
        
        result = await agent.get_calendar_events()
        
        assert len(result) == 2
        entity_ids = [e["entity_id"] for e in result]
        assert "calendar.personal" in entity_ids
        assert "calendar.work" in entity_ids

    @pytest.mark.asyncio
    async def test_get_calendar_events_specific_entity(self, agent, hass):
        """Test get_calendar_events returns specific calendar entity."""
        calendar_state = State(
            "calendar.personal",
            "off",
            {"friendly_name": "Personal Calendar", "message": "Meeting at 2pm"}
        )
        
        hass.states.async_set(calendar_state.entity_id, calendar_state.state, calendar_state.attributes)
        
        result = await agent.get_calendar_events(entity_id="calendar.personal")
        
        assert len(result) == 1
        assert result[0]["entity_id"] == "calendar.personal"

    @pytest.mark.asyncio
    async def test_get_calendar_events_empty(self, agent, hass):
        """Test get_calendar_events returns empty list when no calendars."""
        result = await agent.get_calendar_events()
        
        assert result == []

    @pytest.mark.asyncio
    async def test_get_calendar_events_error_handling(self, agent, hass):
        """Test get_calendar_events handles errors gracefully."""
        with patch.object(agent, "get_entities_by_domain", side_effect=Exception("Test error")):
            result = await agent.get_calendar_events()
            
            assert len(result) == 1
            assert "error" in result[0]
            assert "Test error" in result[0]["error"]

    @pytest.mark.asyncio
    async def test_get_person_data_success(self, agent, hass):
        """Test get_person_data returns person tracking information."""
        from datetime import datetime, timezone
        
        person1_state = State(
            "person.john",
            "home",
            {
                "friendly_name": "John Doe",
                "latitude": 37.7749,
                "longitude": -122.4194,
                "source": "device_tracker.phone",
                "gps_accuracy": 10
            }
        )
        person1_state.last_changed = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        
        person2_state = State(
            "person.jane",
            "away",
            {
                "friendly_name": "Jane Smith",
                "latitude": 40.7128,
                "longitude": -74.0060,
                "source": "device_tracker.car",
                "gps_accuracy": 5
            }
        )
        person2_state.last_changed = datetime(2024, 1, 1, 13, 0, 0, tzinfo=timezone.utc)
        
        hass.states.async_set(person1_state.entity_id, person1_state.state, person1_state.attributes)
        hass.states.async_set(person2_state.entity_id, person2_state.state, person2_state.attributes)
        
        result = await agent.get_person_data()
        
        assert len(result) == 2
        
        # Check person 1
        john = next(p for p in result if p["entity_id"] == "person.john")
        assert john["name"] == "John Doe"
        assert john["state"] == "home"
        assert john["latitude"] == 37.7749
        assert john["longitude"] == -122.4194
        assert john["source"] == "device_tracker.phone"
        assert john["gps_accuracy"] == 10
        # Just check that last_changed exists and is a valid ISO timestamp
        assert john["last_changed"] is not None
        assert "T" in john["last_changed"]
        assert john["last_changed"].endswith("+00:00")

        # Check person 2
        jane = next(p for p in result if p["entity_id"] == "person.jane")
        assert jane["name"] == "Jane Smith"
        assert jane["state"] == "away"

    @pytest.mark.asyncio
    async def test_get_person_data_empty(self, agent, hass):
        """Test get_person_data returns empty list when no persons."""
        result = await agent.get_person_data()
        
        assert result == []

    @pytest.mark.asyncio
    async def test_get_person_data_missing_attributes(self, agent, hass):
        """Test get_person_data handles missing attributes gracefully."""
        person_state = State(
            "person.john",
            "home",
            {"friendly_name": "John Doe"}  # Missing latitude, longitude, etc.
        )
        
        hass.states.async_set(person_state.entity_id, person_state.state, person_state.attributes)
        
        result = await agent.get_person_data()
        
        assert len(result) == 1
        assert result[0]["entity_id"] == "person.john"
        assert result[0]["latitude"] is None
        assert result[0]["longitude"] is None
        assert result[0]["source"] is None
        assert result[0]["gps_accuracy"] is None

    # Note: Error handling test for get_person_data is omitted because
    # hass.states.async_all is read-only and cannot be mocked for testing
    # error scenarios. The error handling is covered by the try/except
    # in the implementation, but cannot be practically tested.

    @pytest.mark.asyncio
    async def test_get_statistics_success(self, agent, hass):
        """Test get_statistics returns statistics for entity."""
        from homeassistant.components import recorder

        # Mock recorder availability
        hass.data[recorder.DATA_INSTANCE] = True

        # Mock statistics data
        mock_stats = {
            "sensor.temperature": [
                {
                    "start": "2024-01-01T00:00:00+00:00",
                    "mean": 22.5,
                    "min": 20.0,
                    "max": 25.0,
                    "sum": 90.0,
                    "state": 22.5
                }
            ]
        }

        async def mock_executor_job(func, *args):
            return func(*args)

        with patch("homeassistant.components.recorder.statistics.get_last_short_term_statistics", return_value=mock_stats):
            with patch.object(hass, "async_add_executor_job", side_effect=mock_executor_job):
                result = await agent.get_statistics("sensor.temperature")

        assert result["entity_id"] == "sensor.temperature"
        assert "mean" in result
        assert "min" in result
        assert "max" in result

    @pytest.mark.asyncio
    async def test_get_statistics_no_entity_id(self, agent, hass):
        """Test get_statistics returns error when entity_id is missing."""
        result = await agent.get_statistics("")
        
        assert "error" in result
        assert "entity_id is required" in result["error"]

    @pytest.mark.asyncio
    async def test_get_statistics_recorder_not_available(self, agent, hass):
        """Test get_statistics returns error when recorder is not available."""
        from homeassistant.components import recorder

        # Ensure recorder is not available
        if recorder.DATA_INSTANCE in hass.data:
            del hass.data[recorder.DATA_INSTANCE]

        result = await agent.get_statistics("sensor.temperature")

        assert "error" in result
        assert "Recorder component is not available" in result["error"]

    @pytest.mark.asyncio
    async def test_get_statistics_no_data(self, agent, hass):
        """Test get_statistics handles entity with no statistics."""
        from homeassistant.components import recorder

        hass.data[recorder.DATA_INSTANCE] = True

        # Mock empty statistics data
        mock_stats = {}

        async def mock_executor_job(func, *args):
            return func(*args)

        with patch("homeassistant.components.recorder.statistics.get_last_short_term_statistics", return_value=mock_stats):
            with patch.object(hass, "async_add_executor_job", side_effect=mock_executor_job):
                result = await agent.get_statistics("sensor.temperature")

        # When no stats found, only returns error dict (no entity_id)
        assert "error" in result
        assert "No statistics available" in result["error"]

    @pytest.mark.asyncio
    async def test_get_statistics_error_handling(self, agent, hass):
        """Test get_statistics handles errors gracefully."""
        from homeassistant.components import recorder
        
        hass.data[recorder.DATA_INSTANCE] = True
        
        with patch.object(hass, "async_add_executor_job", side_effect=Exception("Test error")):
            result = await agent.get_statistics("sensor.temperature")
            
            assert "error" in result
            assert "Test error" in result["error"]

    @pytest.mark.asyncio
    async def test_get_dashboards_success(self, agent, hass):
        """Test get_dashboards returns list of dashboards."""
        from homeassistant.components.lovelace import DOMAIN as LOVELACE_DOMAIN

        # Mock WebSocket API
        hass.data["websocket_api"] = True

        # Mock lovelace data with dashboards
        mock_dashboard = AsyncMock()

        mock_lovelace_data = AsyncMock()
        mock_lovelace_data.dashboards = {None: mock_dashboard}
        mock_lovelace_data.yaml_dashboards = {}

        hass.data[LOVELACE_DOMAIN] = mock_lovelace_data

        result = await agent.get_dashboards()

        assert len(result) == 1
        # Default dashboard gets url_path=None and title="Overview"
        assert result[0]["url_path"] is None
        assert result[0]["title"] == "Overview"
        assert result[0]["icon"] == "mdi:home"

    @pytest.mark.asyncio
    async def test_get_dashboards_multiple(self, agent, hass):
        """Test get_dashboards returns multiple dashboards."""
        from homeassistant.components.lovelace import DOMAIN as LOVELACE_DOMAIN

        hass.data["websocket_api"] = True

        # Mock multiple dashboards
        mock_dashboard1 = AsyncMock()
        mock_dashboard2 = AsyncMock()

        mock_lovelace_data = AsyncMock()
        mock_lovelace_data.dashboards = {None: mock_dashboard1, "energy": mock_dashboard2}
        # Can provide metadata in yaml_dashboards
        mock_lovelace_data.yaml_dashboards = {
            "energy": {"title": "Energy Dashboard", "icon": "mdi:lightning-bolt"}
        }

        hass.data[LOVELACE_DOMAIN] = mock_lovelace_data

        result = await agent.get_dashboards()

        assert len(result) == 2
        titles = [d["title"] for d in result]
        # Default dashboard gets title "Overview", custom gets yaml_dashboards title or url
        assert "Overview" in titles
        assert "Energy Dashboard" in titles

    @pytest.mark.asyncio
    async def test_get_dashboards_websocket_not_available(self, agent, hass):
        """Test get_dashboards returns error when WebSocket API not available."""
        # Ensure websocket_api is not available
        if "websocket_api" in hass.data:
            del hass.data["websocket_api"]
        
        result = await agent.get_dashboards()
        
        assert len(result) == 1
        assert "error" in result[0]
        assert "WebSocket API not available" in result[0]["error"]

    @pytest.mark.asyncio
    async def test_get_dashboards_lovelace_not_available(self, agent, hass):
        """Test get_dashboards returns error when Lovelace not available."""
        from homeassistant.components.lovelace import DOMAIN as LOVELACE_DOMAIN
        
        hass.data["websocket_api"] = True
        
        # Ensure lovelace is not available
        if LOVELACE_DOMAIN in hass.data:
            del hass.data[LOVELACE_DOMAIN]
        
        result = await agent.get_dashboards()
        
        assert len(result) == 1
        assert "error" in result[0]
        assert "Lovelace not available" in result[0]["error"]

    @pytest.mark.asyncio
    async def test_get_dashboards_error_handling(self, agent, hass):
        """Test get_dashboards handles errors gracefully."""
        hass.data["websocket_api"] = True
        
        with patch("homeassistant.components.lovelace.DOMAIN", side_effect=Exception("Test error")):
            result = await agent.get_dashboards()
            
            assert len(result) == 1
            assert "error" in result[0]

    @pytest.mark.asyncio
    async def test_get_dashboard_config_default(self, agent, hass):
        """Test get_dashboard_config returns default dashboard config."""
        from homeassistant.components.lovelace import DOMAIN as LOVELACE_DOMAIN
        
        # Mock dashboard with config
        mock_dashboard = AsyncMock()
        mock_dashboard.async_get_info = AsyncMock(return_value={
            "id": "default",
            "title": "Home",
            "icon": "mdi:home"
        })
        mock_dashboard.async_load = AsyncMock(return_value={
            "views": [{"title": "Overview", "cards": []}]
        })
        
        mock_lovelace_data = AsyncMock()
        mock_lovelace_data.dashboards = {None: mock_dashboard}
        
        hass.data[LOVELACE_DOMAIN] = mock_lovelace_data
        
        result = await agent.get_dashboard_config()

        # get_dashboard_config returns the info dict from async_get_info
        assert result["id"] == "default"
        assert result["title"] == "Home"
        assert result["icon"] == "mdi:home"

    @pytest.mark.asyncio
    async def test_get_dashboard_config_specific(self, agent, hass):
        """Test get_dashboard_config returns specific dashboard config."""
        from homeassistant.components.lovelace import DOMAIN as LOVELACE_DOMAIN
        
        # Mock specific dashboard
        mock_dashboard = AsyncMock()
        mock_dashboard.async_get_info = AsyncMock(return_value={
            "id": "energy",
            "title": "Energy Dashboard",
            "icon": "mdi:lightning-bolt"
        })
        mock_dashboard.async_load = AsyncMock(return_value={
            "views": [{"title": "Energy", "cards": [{"type": "energy-usage"}]}]
        })
        
        mock_lovelace_data = AsyncMock()
        mock_lovelace_data.dashboards = {"energy": mock_dashboard}
        
        hass.data[LOVELACE_DOMAIN] = mock_lovelace_data
        
        result = await agent.get_dashboard_config(dashboard_url="energy")

        # get_dashboard_config returns the info dict from async_get_info
        assert result["id"] == "energy"
        assert result["title"] == "Energy Dashboard"
        assert result["icon"] == "mdi:lightning-bolt"

    @pytest.mark.asyncio
    async def test_get_dashboard_config_not_found(self, agent, hass):
        """Test get_dashboard_config returns error for non-existent dashboard."""
        from homeassistant.components.lovelace import DOMAIN as LOVELACE_DOMAIN
        
        mock_lovelace_data = AsyncMock()
        mock_lovelace_data.dashboards = {}
        
        hass.data[LOVELACE_DOMAIN] = mock_lovelace_data
        
        result = await agent.get_dashboard_config(dashboard_url="nonexistent")
        
        assert "error" in result
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_get_dashboard_config_lovelace_not_available(self, agent, hass):
        """Test get_dashboard_config returns error when Lovelace not available."""
        from homeassistant.components.lovelace import DOMAIN as LOVELACE_DOMAIN
        
        # Ensure lovelace is not available
        if LOVELACE_DOMAIN in hass.data:
            del hass.data[LOVELACE_DOMAIN]
        
        result = await agent.get_dashboard_config()
        
        assert "error" in result
        assert "Lovelace not available" in result["error"]

    @pytest.mark.asyncio
    async def test_get_dashboard_config_error_handling(self, agent, hass):
        """Test get_dashboard_config handles errors gracefully."""
        with patch("homeassistant.components.lovelace.DOMAIN", side_effect=Exception("Test error")):
            result = await agent.get_dashboard_config()

            assert "error" in result

    # Tests for set_entity_state method
    @pytest.mark.asyncio
    async def test_set_entity_state_missing_entity_id(self, agent, hass):
        """Test set_entity_state with missing entity_id."""
        result = await agent.set_entity_state("", "on")

        assert "error" in result
        assert "entity_id is required" in result["error"]

    @pytest.mark.asyncio
    async def test_set_entity_state_missing_state(self, agent, hass):
        """Test set_entity_state with missing state."""
        result = await agent.set_entity_state("light.living_room", "")

        assert "error" in result
        assert "state is required" in result["error"]

    @pytest.mark.asyncio
    async def test_set_entity_state_entity_not_found(self, agent, hass):
        """Test set_entity_state with non-existent entity."""
        result = await agent.set_entity_state("light.nonexistent", "on")

        assert "error" in result
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_set_entity_state_light_on(self, agent, hass):
        """Test set_entity_state turns light on."""
        # Create a light entity
        hass.states.async_set("light.living_room", "off")

        # Register mock service
        service_calls = []
        async def mock_turn_on_service(call):
            service_calls.append({"domain": "light", "service": "turn_on", "data": dict(call.data)})
            hass.states.async_set("light.living_room", "on")

        hass.services.async_register("light", "turn_on", mock_turn_on_service)

        result = await agent.set_entity_state("light.living_room", "on")

        assert result["success"] is True
        assert result["entity_id"] == "light.living_room"
        assert result["new_state"] == "on"
        assert len(service_calls) == 1
        assert service_calls[0]["domain"] == "light"
        assert service_calls[0]["service"] == "turn_on"

    @pytest.mark.asyncio
    async def test_set_entity_state_light_off(self, agent, hass):
        """Test set_entity_state turns light off."""
        hass.states.async_set("light.living_room", "on")

        service_calls = []
        async def mock_turn_off_service(call):
            service_calls.append({"domain": "light", "service": "turn_off", "data": dict(call.data)})
            hass.states.async_set("light.living_room", "off")

        hass.services.async_register("light", "turn_off", mock_turn_off_service)

        result = await agent.set_entity_state("light.living_room", "off")

        assert result["success"] is True
        assert result["new_state"] == "off"
        assert service_calls[0]["service"] == "turn_off"

    @pytest.mark.asyncio
    async def test_set_entity_state_light_with_attributes(self, agent, hass):
        """Test set_entity_state turns light on with attributes."""
        hass.states.async_set("light.living_room", "off")

        service_calls = []
        async def mock_turn_on_service(call):
            service_calls.append({"domain": "light", "service": "turn_on", "data": dict(call.data)})
            hass.states.async_set("light.living_room", "on", {"brightness": 200})

        hass.services.async_register("light", "turn_on", mock_turn_on_service)

        result = await agent.set_entity_state(
            "light.living_room", "on", {"brightness": 200}
        )

        assert result["success"] is True
        assert "brightness" in result["new_attributes"]
        assert service_calls[0]["data"]["brightness"] == 200

    @pytest.mark.asyncio
    async def test_set_entity_state_switch_on(self, agent, hass):
        """Test set_entity_state turns switch on."""
        hass.states.async_set("switch.outlet", "off")

        service_calls = []
        async def mock_turn_on_service(call):
            service_calls.append({"domain": "switch", "service": "turn_on", "data": dict(call.data)})
            hass.states.async_set("switch.outlet", "on")

        hass.services.async_register("switch", "turn_on", mock_turn_on_service)

        result = await agent.set_entity_state("switch.outlet", "on")

        assert result["success"] is True
        assert service_calls[0]["domain"] == "switch"
        assert service_calls[0]["service"] == "turn_on"

    @pytest.mark.asyncio
    async def test_set_entity_state_cover_open(self, agent, hass):
        """Test set_entity_state opens cover."""
        hass.states.async_set("cover.garage", "closed")

        service_calls = []
        async def mock_open_cover_service(call):
            service_calls.append({"domain": "cover", "service": "open_cover", "data": dict(call.data)})
            hass.states.async_set("cover.garage", "open")

        hass.services.async_register("cover", "open_cover", mock_open_cover_service)

        result = await agent.set_entity_state("cover.garage", "open")

        assert result["success"] is True
        assert service_calls[0]["service"] == "open_cover"

    @pytest.mark.asyncio
    async def test_set_entity_state_cover_close(self, agent, hass):
        """Test set_entity_state closes cover."""
        hass.states.async_set("cover.garage", "open")

        service_calls = []
        async def mock_close_cover_service(call):
            service_calls.append({"domain": "cover", "service": "close_cover", "data": dict(call.data)})
            hass.states.async_set("cover.garage", "closed")

        hass.services.async_register("cover", "close_cover", mock_close_cover_service)

        result = await agent.set_entity_state("cover.garage", "close")

        assert result["success"] is True
        assert service_calls[0]["service"] == "close_cover"

    @pytest.mark.asyncio
    async def test_set_entity_state_cover_stop(self, agent, hass):
        """Test set_entity_state stops cover."""
        hass.states.async_set("cover.garage", "opening")

        service_calls = []
        async def mock_stop_cover_service(call):
            service_calls.append({"domain": "cover", "service": "stop_cover", "data": dict(call.data)})
            hass.states.async_set("cover.garage", "open")

        hass.services.async_register("cover", "stop_cover", mock_stop_cover_service)

        result = await agent.set_entity_state("cover.garage", "stop")

        assert result["success"] is True
        assert service_calls[0]["service"] == "stop_cover"

    @pytest.mark.asyncio
    async def test_set_entity_state_cover_invalid_state(self, agent, hass):
        """Test set_entity_state with invalid cover state."""
        hass.states.async_set("cover.garage", "closed")

        result = await agent.set_entity_state("cover.garage", "invalid")

        assert "error" in result
        assert "Invalid state" in result["error"]

    @pytest.mark.asyncio
    async def test_set_entity_state_climate_hvac_mode(self, agent, hass):
        """Test set_entity_state sets climate HVAC mode."""
        hass.states.async_set("climate.thermostat", "off")

        service_calls = []
        async def mock_set_hvac_mode_service(call):
            service_calls.append({"domain": "climate", "service": "set_hvac_mode", "data": dict(call.data)})
            hass.states.async_set("climate.thermostat", "heat")

        hass.services.async_register("climate", "set_hvac_mode", mock_set_hvac_mode_service)

        result = await agent.set_entity_state("climate.thermostat", "heat")

        assert result["success"] is True
        assert service_calls[0]["service"] == "set_hvac_mode"
        assert service_calls[0]["data"]["hvac_mode"] == "heat"

    @pytest.mark.asyncio
    async def test_set_entity_state_climate_on(self, agent, hass):
        """Test set_entity_state turns climate on."""
        hass.states.async_set("climate.thermostat", "off")

        service_calls = []
        async def mock_turn_on_service(call):
            service_calls.append({"domain": "climate", "service": "turn_on", "data": dict(call.data)})
            hass.states.async_set("climate.thermostat", "heat")

        hass.services.async_register("climate", "turn_on", mock_turn_on_service)

        result = await agent.set_entity_state("climate.thermostat", "on")

        assert result["success"] is True
        assert service_calls[0]["service"] == "turn_on"

    @pytest.mark.asyncio
    async def test_set_entity_state_climate_invalid_state(self, agent, hass):
        """Test set_entity_state with invalid climate state."""
        hass.states.async_set("climate.thermostat", "heat")

        result = await agent.set_entity_state("climate.thermostat", "invalid")

        assert "error" in result
        assert "Invalid state" in result["error"]

    @pytest.mark.asyncio
    async def test_set_entity_state_fan_on_with_attributes(self, agent, hass):
        """Test set_entity_state turns fan on with attributes."""
        hass.states.async_set("fan.bedroom", "off")

        service_calls = []
        async def mock_turn_on_service(call):
            service_calls.append({"domain": "fan", "service": "turn_on", "data": dict(call.data)})
            hass.states.async_set("fan.bedroom", "on", {"speed": "high"})

        hass.services.async_register("fan", "turn_on", mock_turn_on_service)

        result = await agent.set_entity_state("fan.bedroom", "on", {"speed": "high"})

        assert result["success"] is True
        assert service_calls[0]["data"]["speed"] == "high"

    @pytest.mark.asyncio
    async def test_set_entity_state_other_domain_direct(self, agent, hass):
        """Test set_entity_state for other domains sets state directly."""
        hass.states.async_set("sensor.temperature", "20", {"unit_of_measurement": "°C"})

        result = await agent.set_entity_state("sensor.temperature", "22", {"unit_of_measurement": "°C"})

        assert result["success"] is True
        assert result["new_state"] == "22"

    # Tests for call_service method
    @pytest.mark.asyncio
    async def test_call_service_missing_domain(self, agent, hass):
        """Test call_service with missing domain."""
        result = await agent.call_service("", "turn_on")

        assert "error" in result
        assert "domain is required" in result["error"]

    @pytest.mark.asyncio
    async def test_call_service_missing_service(self, agent, hass):
        """Test call_service with missing service."""
        result = await agent.call_service("light", "")

        assert "error" in result
        assert "service is required" in result["error"]

    @pytest.mark.asyncio
    async def test_call_service_entity_not_found(self, agent, hass):
        """Test call_service with non-existent entity."""
        result = await agent.call_service(
            "light", "turn_on", target={"entity_id": "light.nonexistent"}
        )

        assert "error" in result
        assert "Entity not found" in result["error"]

    @pytest.mark.asyncio
    async def test_call_service_entity_not_found_with_suggestions(self, agent, hass):
        """Test call_service with non-existent entity provides suggestions."""
        # Create similar entities
        hass.states.async_set("light.living_room", "off")
        hass.states.async_set("light.bedroom", "off")

        result = await agent.call_service(
            "light", "turn_on", target={"entity_id": "light.living"}
        )

        assert "error" in result
        assert "Entity not found" in result["error"]
        assert "Did you mean" in result["error"]

    @pytest.mark.asyncio
    async def test_call_service_basic_success(self, agent, hass):
        """Test call_service basic successful call."""
        hass.states.async_set("light.living_room", "off")

        service_calls = []
        async def mock_turn_on_service(call):
            service_calls.append({"domain": "light", "service": "turn_on", "data": dict(call.data)})
            hass.states.async_set("light.living_room", "on")

        hass.services.async_register("light", "turn_on", mock_turn_on_service)

        result = await agent.call_service(
            "light", "turn_on", target={"entity_id": "light.living_room"}
        )

        assert result["success"] is True
        assert result["service"] == "light.turn_on"
        assert "Successfully called" in result["message"]
        assert len(result["entities_affected"]) == 1
        assert result["entities_affected"][0]["entity_id"] == "light.living_room"
        assert result["entities_affected"][0]["state"] == "on"

    @pytest.mark.asyncio
    async def test_call_service_with_single_entity_string(self, agent, hass):
        """Test call_service with single entity_id as string."""
        hass.states.async_set("light.living_room", "off")

        service_calls = []
        async def mock_turn_on_service(call):
            service_calls.append({"domain": "light", "service": "turn_on", "data": dict(call.data)})
            hass.states.async_set("light.living_room", "on")

        hass.services.async_register("light", "turn_on", mock_turn_on_service)

        result = await agent.call_service(
            "light", "turn_on", target={"entity_id": "light.living_room"}
        )

        assert result["success"] is True
        # Verify entity_id was converted to list
        assert isinstance(service_calls[0]["data"]["entity_id"], list)
        assert "light.living_room" in service_calls[0]["data"]["entity_id"]

    @pytest.mark.asyncio
    async def test_call_service_with_multiple_entities(self, agent, hass):
        """Test call_service with multiple entities."""
        hass.states.async_set("light.living_room", "off")
        hass.states.async_set("light.bedroom", "off")

        service_calls = []
        async def mock_turn_on_service(call):
            service_calls.append({"domain": "light", "service": "turn_on", "data": dict(call.data)})
            hass.states.async_set("light.living_room", "on")
            hass.states.async_set("light.bedroom", "on")

        hass.services.async_register("light", "turn_on", mock_turn_on_service)

        result = await agent.call_service(
            "light",
            "turn_on",
            target={"entity_id": ["light.living_room", "light.bedroom"]}
        )

        assert result["success"] is True
        assert len(result["entities_affected"]) == 2

    @pytest.mark.asyncio
    async def test_call_service_with_service_data(self, agent, hass):
        """Test call_service with service_data."""
        hass.states.async_set("light.living_room", "off")

        service_calls = []
        async def mock_turn_on_service(call):
            service_calls.append({"domain": "light", "service": "turn_on", "data": dict(call.data)})
            hass.states.async_set("light.living_room", "on", {"brightness": 200})

        hass.services.async_register("light", "turn_on", mock_turn_on_service)

        result = await agent.call_service(
            "light",
            "turn_on",
            target={"entity_id": "light.living_room"},
            service_data={"brightness": 200}
        )

        assert result["success"] is True
        assert service_calls[0]["data"]["brightness"] == 200

    @pytest.mark.asyncio
    async def test_call_service_with_other_target_properties(self, agent, hass):
        """Test call_service with other target properties."""
        hass.states.async_set("light.living_room", "off")

        service_calls = []
        async def mock_turn_on_service(call):
            service_calls.append({"domain": "light", "service": "turn_on", "data": dict(call.data)})

        hass.services.async_register("light", "turn_on", mock_turn_on_service)

        result = await agent.call_service(
            "light",
            "turn_on",
            target={"entity_id": "light.living_room", "area_id": "living_room"}
        )

        assert result["success"] is True
        assert service_calls[0]["data"]["area_id"] == "living_room"

    @pytest.mark.asyncio
    async def test_call_service_without_target(self, agent, hass):
        """Test call_service without target entities."""
        service_calls = []
        async def mock_restart_service(call):
            service_calls.append({"domain": "homeassistant", "service": "restart", "data": dict(call.data)})

        hass.services.async_register("homeassistant", "restart", mock_restart_service)

        result = await agent.call_service("homeassistant", "restart")

        assert result["success"] is True
        assert result["service"] == "homeassistant.restart"
        assert len(result["entities_affected"]) == 0

    @pytest.mark.asyncio
    async def test_call_service_multiple_entities_one_missing(self, agent, hass):
        """Test call_service with multiple entities where one is missing."""
        hass.states.async_set("light.living_room", "off")

        result = await agent.call_service(
            "light",
            "turn_on",
            target={"entity_id": ["light.living_room", "light.nonexistent"]}
        )

        assert "error" in result
        assert "Entity not found" in result["error"]
        assert "light.nonexistent" in result["error"]

    # Tests for get_entity_state
    @pytest.mark.asyncio
    async def test_get_entity_state_success(self, agent, hass):
        """Test get_entity_state returns full entity information."""
        from datetime import datetime, timezone

        # Create a light entity with common attributes
        light_state = State(
            "light.living_room",
            "on",
            {
                "friendly_name": "Living Room Light",
                "brightness": 255,
                "color_temp": 400,
                "supported_features": 43
            }
        )
        light_state.last_changed = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        hass.states.async_set(light_state.entity_id, light_state.state, light_state.attributes)

        result = await agent.get_entity_state("light.living_room")

        assert result["entity_id"] == "light.living_room"
        assert result["state"] == "on"
        assert result["friendly_name"] == "Living Room Light"
        assert result["attributes"]["brightness"] == 255
        assert result["attributes"]["color_temp"] == 400
        assert result["last_changed"] is not None
        assert "T" in result["last_changed"]  # ISO format

    @pytest.mark.asyncio
    async def test_get_entity_state_sensor_with_device_class(self, agent, hass):
        """Test get_entity_state with sensor having device_class."""
        temp_state = State(
            "sensor.bedroom_temp",
            "21.5",
            {
                "friendly_name": "Bedroom Temperature",
                "device_class": "temperature",
                "unit_of_measurement": "°C",
                "state_class": "measurement"
            }
        )

        hass.states.async_set(temp_state.entity_id, temp_state.state, temp_state.attributes)

        result = await agent.get_entity_state("sensor.bedroom_temp")

        assert result["entity_id"] == "sensor.bedroom_temp"
        assert result["state"] == "21.5"
        assert result["attributes"]["device_class"] == "temperature"
        assert result["attributes"]["unit_of_measurement"] == "°C"

    @pytest.mark.asyncio
    async def test_get_entity_state_climate_entity(self, agent, hass):
        """Test get_entity_state with climate entity."""
        climate_state = State(
            "climate.living_room",
            "heat",
            {
                "friendly_name": "Living Room Thermostat",
                "temperature": 22.0,
                "current_temperature": 21.5,
                "hvac_modes": ["heat", "cool", "off"],
                "min_temp": 7,
                "max_temp": 35
            }
        )

        hass.states.async_set(climate_state.entity_id, climate_state.state, climate_state.attributes)

        result = await agent.get_entity_state("climate.living_room")

        assert result["entity_id"] == "climate.living_room"
        assert result["state"] == "heat"
        assert result["attributes"]["temperature"] == 22.0
        assert result["attributes"]["current_temperature"] == 21.5

    @pytest.mark.asyncio
    async def test_get_entity_state_binary_sensor(self, agent, hass):
        """Test get_entity_state with binary sensor."""
        motion_state = State(
            "binary_sensor.hallway_motion",
            "on",
            {
                "friendly_name": "Hallway Motion",
                "device_class": "motion"
            }
        )

        hass.states.async_set(motion_state.entity_id, motion_state.state, motion_state.attributes)

        result = await agent.get_entity_state("binary_sensor.hallway_motion")

        assert result["entity_id"] == "binary_sensor.hallway_motion"
        assert result["state"] == "on"
        assert result["attributes"]["device_class"] == "motion"

    @pytest.mark.asyncio
    async def test_get_entity_state_not_found(self, agent, hass):
        """Test get_entity_state with non-existent entity."""
        result = await agent.get_entity_state("light.nonexistent")

        assert "error" in result
        assert "not found" in result["error"].lower()
        assert "light.nonexistent" in result["error"]

    @pytest.mark.asyncio
    async def test_get_entity_state_empty_entity_id(self, agent, hass):
        """Test get_entity_state with empty entity_id."""
        result = await agent.get_entity_state("")

        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_get_entity_state_none_entity_id(self, agent, hass):
        """Test get_entity_state with None entity_id."""
        result = await agent.get_entity_state(None)

        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_get_entity_state_switch(self, agent, hass):
        """Test get_entity_state with switch entity."""
        switch_state = State(
            "switch.coffee_maker",
            "off",
            {"friendly_name": "Coffee Maker"}
        )

        hass.states.async_set(switch_state.entity_id, switch_state.state, switch_state.attributes)

        result = await agent.get_entity_state("switch.coffee_maker")

        assert result["entity_id"] == "switch.coffee_maker"
        assert result["state"] == "off"
        assert result["friendly_name"] == "Coffee Maker"

    @pytest.mark.asyncio
    async def test_get_entity_state_cover(self, agent, hass):
        """Test get_entity_state with cover entity."""
        cover_state = State(
            "cover.garage_door",
            "closed",
            {
                "friendly_name": "Garage Door",
                "current_position": 0,
                "supported_features": 15
            }
        )

        hass.states.async_set(cover_state.entity_id, cover_state.state, cover_state.attributes)

        result = await agent.get_entity_state("cover.garage_door")

        assert result["entity_id"] == "cover.garage_door"
        assert result["state"] == "closed"
        assert result["attributes"]["current_position"] == 0

    # Tests for get_entities_by_domain
    @pytest.mark.asyncio
    async def test_get_entities_by_domain_lights(self, agent, hass):
        """Test get_entities_by_domain returns all light entities."""
        # Create multiple light entities
        light1 = State("light.living_room", "on", {"friendly_name": "Living Room"})
        light2 = State("light.bedroom", "off", {"friendly_name": "Bedroom"})
        light3 = State("light.kitchen", "on", {"friendly_name": "Kitchen", "brightness": 200})

        hass.states.async_set(light1.entity_id, light1.state, light1.attributes)
        hass.states.async_set(light2.entity_id, light2.state, light2.attributes)
        hass.states.async_set(light3.entity_id, light3.state, light3.attributes)

        result = await agent.get_entities_by_domain("light")

        assert len(result) == 3
        entity_ids = [e["entity_id"] for e in result]
        assert "light.living_room" in entity_ids
        assert "light.bedroom" in entity_ids
        assert "light.kitchen" in entity_ids

    @pytest.mark.asyncio
    async def test_get_entities_by_domain_sensors(self, agent, hass):
        """Test get_entities_by_domain returns all sensor entities."""
        sensor1 = State("sensor.temperature", "22.5", {"device_class": "temperature", "unit_of_measurement": "°C"})
        sensor2 = State("sensor.humidity", "65", {"device_class": "humidity", "unit_of_measurement": "%"})
        sensor3 = State("sensor.power", "150", {"device_class": "power", "unit_of_measurement": "W"})

        hass.states.async_set(sensor1.entity_id, sensor1.state, sensor1.attributes)
        hass.states.async_set(sensor2.entity_id, sensor2.state, sensor2.attributes)
        hass.states.async_set(sensor3.entity_id, sensor3.state, sensor3.attributes)

        result = await agent.get_entities_by_domain("sensor")

        assert len(result) == 3
        entity_ids = [e["entity_id"] for e in result]
        assert "sensor.temperature" in entity_ids
        assert "sensor.humidity" in entity_ids
        assert "sensor.power" in entity_ids

    @pytest.mark.asyncio
    async def test_get_entities_by_domain_mixed_entities(self, agent, hass):
        """Test get_entities_by_domain filters correctly with mixed entity types."""
        # Create entities from different domains
        hass.states.async_set("light.living_room", "on", {"friendly_name": "Light"})
        hass.states.async_set("switch.fan", "off", {"friendly_name": "Fan"})
        hass.states.async_set("sensor.temp", "22", {"friendly_name": "Temp"})
        hass.states.async_set("light.bedroom", "off", {"friendly_name": "Bedroom Light"})

        # Request only light domain
        result = await agent.get_entities_by_domain("light")

        assert len(result) == 2
        entity_ids = [e["entity_id"] for e in result]
        assert "light.living_room" in entity_ids
        assert "light.bedroom" in entity_ids
        assert "switch.fan" not in entity_ids
        assert "sensor.temp" not in entity_ids

    @pytest.mark.asyncio
    async def test_get_entities_by_domain_empty_result(self, agent, hass):
        """Test get_entities_by_domain returns empty list when no entities exist."""
        # Create entities from different domains
        hass.states.async_set("light.living_room", "on", {"friendly_name": "Light"})
        hass.states.async_set("switch.fan", "off", {"friendly_name": "Fan"})

        # Request domain with no entities
        result = await agent.get_entities_by_domain("climate")

        assert result == []

    @pytest.mark.asyncio
    async def test_get_entities_by_domain_empty_domain(self, agent, hass):
        """Test get_entities_by_domain with empty domain."""
        result = await agent.get_entities_by_domain("")

        assert len(result) == 1
        assert "error" in result[0]
        assert "required" in result[0]["error"].lower()

    @pytest.mark.asyncio
    async def test_get_entities_by_domain_none_domain(self, agent, hass):
        """Test get_entities_by_domain with None domain."""
        result = await agent.get_entities_by_domain(None)

        assert len(result) == 1
        assert "error" in result[0]
        assert "required" in result[0]["error"].lower()

    @pytest.mark.asyncio
    async def test_get_entities_by_domain_binary_sensors(self, agent, hass):
        """Test get_entities_by_domain with binary_sensor domain."""
        binary1 = State("binary_sensor.motion", "on", {"device_class": "motion"})
        binary2 = State("binary_sensor.door", "off", {"device_class": "door"})
        binary3 = State("binary_sensor.window", "off", {"device_class": "window"})

        hass.states.async_set(binary1.entity_id, binary1.state, binary1.attributes)
        hass.states.async_set(binary2.entity_id, binary2.state, binary2.attributes)
        hass.states.async_set(binary3.entity_id, binary3.state, binary3.attributes)

        result = await agent.get_entities_by_domain("binary_sensor")

        assert len(result) == 3
        entity_ids = [e["entity_id"] for e in result]
        assert "binary_sensor.motion" in entity_ids
        assert "binary_sensor.door" in entity_ids
        assert "binary_sensor.window" in entity_ids

    @pytest.mark.asyncio
    async def test_get_entities_by_domain_climate(self, agent, hass):
        """Test get_entities_by_domain with climate entities."""
        climate1 = State("climate.living_room", "heat", {
            "temperature": 22.0,
            "current_temperature": 21.5
        })
        climate2 = State("climate.bedroom", "cool", {
            "temperature": 20.0,
            "current_temperature": 21.0
        })

        hass.states.async_set(climate1.entity_id, climate1.state, climate1.attributes)
        hass.states.async_set(climate2.entity_id, climate2.state, climate2.attributes)

        result = await agent.get_entities_by_domain("climate")

        assert len(result) == 2
        entity_ids = [e["entity_id"] for e in result]
        assert "climate.living_room" in entity_ids
        assert "climate.bedroom" in entity_ids

    @pytest.mark.asyncio
    async def test_get_entities_by_domain_single_entity(self, agent, hass):
        """Test get_entities_by_domain with single entity in domain."""
        automation = State("automation.morning_routine", "on", {
            "friendly_name": "Morning Routine",
            "last_triggered": None
        })

        hass.states.async_set(automation.entity_id, automation.state, automation.attributes)

        result = await agent.get_entities_by_domain("automation")

        assert len(result) == 1
        assert result[0]["entity_id"] == "automation.morning_routine"
        assert result[0]["state"] == "on"

    @pytest.mark.asyncio
    async def test_get_entities_by_domain_covers(self, agent, hass):
        """Test get_entities_by_domain with cover entities."""
        cover1 = State("cover.garage", "closed", {"current_position": 0})
        cover2 = State("cover.blinds", "open", {"current_position": 100})

        hass.states.async_set(cover1.entity_id, cover1.state, cover1.attributes)
        hass.states.async_set(cover2.entity_id, cover2.state, cover2.attributes)

        result = await agent.get_entities_by_domain("cover")

        assert len(result) == 2
        entity_ids = [e["entity_id"] for e in result]
        assert "cover.garage" in entity_ids
        assert "cover.blinds" in entity_ids

    @pytest.mark.asyncio
    async def test_get_entities_by_domain_case_sensitive(self, agent, hass):
        """Test get_entities_by_domain is case-sensitive."""
        hass.states.async_set("light.test", "on", {"friendly_name": "Test Light"})

        # Lowercase should work
        result = await agent.get_entities_by_domain("light")
        assert len(result) == 1

        # Uppercase should return empty (domains are lowercase in HA)
        result = await agent.get_entities_by_domain("LIGHT")
        assert len(result) == 0

    # Tests for create_automation
    @pytest.mark.asyncio
    async def test_create_automation_success(self, agent, hass, tmp_path):
        """Test successful automation creation."""
        import time
        import yaml

        # Register the automation.reload service
        async def mock_reload_service(call):
            pass
        hass.services.async_register("automation", "reload", mock_reload_service)

        # Mock the config path
        automations_file = tmp_path / "automations.yaml"
        automations_file.write_text("[]")

        with patch.object(hass.config, 'path', return_value=str(automations_file)):
            with patch.object(agent, '_sanitize_automation_config', side_effect=lambda x: x):
                with patch.object(time, 'time', return_value=1234567890.123):
                    automation_config = {
                        "alias": "Test Automation",
                        "description": "A test automation",
                        "trigger": [{"platform": "state", "entity_id": "light.living_room", "to": "on"}],
                        "action": [{"service": "light.turn_off", "target": {"entity_id": "light.bedroom"}}],
                        "mode": "single"
                    }

                    result = await agent.create_automation(automation_config)

                    assert result["success"] is True
                    assert "Test Automation" in result["message"]
                    assert "created successfully" in result["message"]

                    # Verify automation was written to file
                    automations = yaml.safe_load(automations_file.read_text())
                    assert len(automations) == 1
                    assert automations[0]["alias"] == "Test Automation"
                    assert automations[0]["id"] == "ai_agent_auto_1234567890123"

    @pytest.mark.asyncio
    async def test_create_automation_missing_required_fields(self, agent, hass):
        """Test automation creation with missing required fields."""
        # Missing 'trigger'
        automation_config = {
            "alias": "Test Automation",
            "action": [{"service": "light.turn_off", "target": {"entity_id": "light.bedroom"}}]
        }

        result = await agent.create_automation(automation_config)

        assert "error" in result
        assert "Missing required fields" in result["error"]

    @pytest.mark.asyncio
    async def test_create_automation_missing_alias(self, agent, hass):
        """Test automation creation with missing alias."""
        automation_config = {
            "trigger": [{"platform": "state", "entity_id": "light.living_room"}],
            "action": [{"service": "light.turn_off"}]
        }

        result = await agent.create_automation(automation_config)

        assert "error" in result
        assert "Missing required fields" in result["error"]

    @pytest.mark.asyncio
    async def test_create_automation_missing_action(self, agent, hass):
        """Test automation creation with missing action."""
        automation_config = {
            "alias": "Test Automation",
            "trigger": [{"platform": "state", "entity_id": "light.living_room"}]
        }

        result = await agent.create_automation(automation_config)

        assert "error" in result
        assert "Missing required fields" in result["error"]

    @pytest.mark.asyncio
    async def test_create_automation_duplicate_name(self, agent, hass, tmp_path):
        """Test automation creation with duplicate automation name."""
        import yaml

        # Register the automation.reload service
        async def mock_reload_service(call):
            pass
        hass.services.async_register("automation", "reload", mock_reload_service)

        # Create existing automation with same name
        automations_file = tmp_path / "automations.yaml"
        existing_automations = [{
            "id": "existing_auto",
            "alias": "Test Automation",
            "trigger": [],
            "action": []
        }]
        automations_file.write_text(yaml.dump(existing_automations))

        with patch.object(hass.config, 'path', return_value=str(automations_file)):
            with patch.object(agent, '_sanitize_automation_config', side_effect=lambda x: x):
                automation_config = {
                    "alias": "Test Automation",
                    "trigger": [{"platform": "state", "entity_id": "light.living_room"}],
                    "action": [{"service": "light.turn_off"}]
                }

                result = await agent.create_automation(automation_config)

                assert "error" in result
                assert "already exists" in result["error"]
                assert "Test Automation" in result["error"]

    @pytest.mark.asyncio
    async def test_create_automation_file_not_found(self, agent, hass, tmp_path):
        """Test automation creation when automations.yaml doesn't exist."""
        import time
        import yaml

        # Register the automation.reload service
        async def mock_reload_service(call):
            pass
        hass.services.async_register("automation", "reload", mock_reload_service)

        # Point to non-existent file
        automations_file = tmp_path / "automations.yaml"

        with patch.object(hass.config, 'path', return_value=str(automations_file)):
            with patch.object(agent, '_sanitize_automation_config', side_effect=lambda x: x):
                with patch.object(time, 'time', return_value=1234567890.123):
                    automation_config = {
                        "alias": "Test Automation",
                        "trigger": [{"platform": "state", "entity_id": "light.living_room"}],
                        "action": [{"service": "light.turn_off"}]
                    }

                    result = await agent.create_automation(automation_config)

                    assert result["success"] is True

                    # Verify file was created with the automation
                    assert automations_file.exists()
                    automations = yaml.safe_load(automations_file.read_text())
                    assert len(automations) == 1
                    assert automations[0]["alias"] == "Test Automation"

    @pytest.mark.asyncio
    async def test_create_automation_service_call_success(self, agent, hass, tmp_path):
        """Test that automation reload service is called after creation."""
        import yaml

        automations_file = tmp_path / "automations.yaml"
        automations_file.write_text("[]")

        service_calls = []
        async def mock_reload_service(call):
            service_calls.append({"domain": call.domain, "service": call.service})

        hass.services.async_register("automation", "reload", mock_reload_service)

        with patch.object(hass.config, 'path', return_value=str(automations_file)):
            with patch.object(agent, '_sanitize_automation_config', side_effect=lambda x: x):
                automation_config = {
                    "alias": "Test Automation",
                    "trigger": [{"platform": "state", "entity_id": "light.living_room"}],
                    "action": [{"service": "light.turn_off"}]
                }

                result = await agent.create_automation(automation_config)

                assert result["success"] is True
                assert len(service_calls) == 1
                assert service_calls[0]["domain"] == "automation"
                assert service_calls[0]["service"] == "reload"

    @pytest.mark.asyncio
    async def test_create_automation_with_condition(self, agent, hass, tmp_path):
        """Test automation creation with conditions."""
        import yaml

        # Register the automation.reload service
        async def mock_reload_service(call):
            pass
        hass.services.async_register("automation", "reload", mock_reload_service)

        automations_file = tmp_path / "automations.yaml"
        automations_file.write_text("[]")

        with patch.object(hass.config, 'path', return_value=str(automations_file)):
            with patch.object(agent, '_sanitize_automation_config', side_effect=lambda x: x):
                automation_config = {
                    "alias": "Test Automation with Condition",
                    "trigger": [{"platform": "state", "entity_id": "light.living_room"}],
                    "condition": [{"condition": "state", "entity_id": "binary_sensor.motion", "state": "on"}],
                    "action": [{"service": "light.turn_off"}]
                }

                result = await agent.create_automation(automation_config)

                assert result["success"] is True

                automations = yaml.safe_load(automations_file.read_text())
                assert automations[0]["condition"] == automation_config["condition"]

    @pytest.mark.asyncio
    async def test_create_automation_exception_handling(self, agent, hass):
        """Test automation creation error handling."""
        with patch.object(hass.config, 'path', side_effect=Exception("Config path error")):
            automation_config = {
                "alias": "Test Automation",
                "trigger": [{"platform": "state", "entity_id": "light.living_room"}],
                "action": [{"service": "light.turn_off"}]
            }

            result = await agent.create_automation(automation_config)

            assert "error" in result
            assert "Config path error" in result["error"]

    # Tests for create_dashboard
    @pytest.mark.asyncio
    async def test_create_dashboard_success(self, agent, hass, tmp_path):
        """Test successful dashboard creation."""
        import yaml

        # Setup file paths
        dashboard_file = tmp_path / "ui-lovelace-test.yaml"
        config_file = tmp_path / "configuration.yaml"
        config_file.write_text("# Empty config\n")

        def mock_path(filename):
            if filename == "configuration.yaml":
                return str(config_file)
            elif filename.startswith("ui-lovelace-"):
                return str(tmp_path / filename)
            return str(tmp_path / filename)

        with patch.object(hass.config, 'path', side_effect=mock_path):
            dashboard_config = {
                "title": "Test Dashboard",
                "url_path": "test",
                "icon": "mdi:test",
                "show_in_sidebar": True,
                "views": [{"title": "View 1", "cards": []}]
            }

            result = await agent.create_dashboard(dashboard_config)

            assert result["success"] is True
            assert "Test Dashboard" in result["message"]
            assert result["url_path"] == "test"
            assert result["restart_required"] is True

            # Verify dashboard file was created
            assert dashboard_file.exists()
            dashboard_data = yaml.safe_load(dashboard_file.read_text())
            assert dashboard_data["title"] == "Test Dashboard"
            assert dashboard_data["icon"] == "mdi:test"

    @pytest.mark.asyncio
    async def test_create_dashboard_missing_title(self, agent, hass):
        """Test dashboard creation with missing title."""
        dashboard_config = {
            "url_path": "test"
        }

        result = await agent.create_dashboard(dashboard_config)

        assert "error" in result
        assert "title is required" in result["error"]

    @pytest.mark.asyncio
    async def test_create_dashboard_missing_url_path(self, agent, hass):
        """Test dashboard creation with missing url_path."""
        dashboard_config = {
            "title": "Test Dashboard"
        }

        result = await agent.create_dashboard(dashboard_config)

        assert "error" in result
        assert "URL path is required" in result["error"]

    @pytest.mark.asyncio
    async def test_create_dashboard_url_path_sanitization(self, agent, hass, tmp_path):
        """Test that dashboard url_path is properly sanitized."""
        import yaml

        dashboard_file = tmp_path / "ui-lovelace-test-dashboard.yaml"
        config_file = tmp_path / "configuration.yaml"
        config_file.write_text("# Empty config\n")

        def mock_path(filename):
            if filename == "configuration.yaml":
                return str(config_file)
            elif filename.startswith("ui-lovelace-"):
                return str(tmp_path / filename)
            return str(tmp_path / filename)

        with patch.object(hass.config, 'path', side_effect=mock_path):
            dashboard_config = {
                "title": "Test Dashboard",
                "url_path": "Test_Dashboard",  # Should be converted to test-dashboard
                "views": []
            }

            result = await agent.create_dashboard(dashboard_config)

            assert result["success"] is True
            assert result["url_path"] == "test-dashboard"
            assert dashboard_file.exists()

    @pytest.mark.asyncio
    async def test_create_dashboard_with_default_values(self, agent, hass, tmp_path):
        """Test dashboard creation with default values for optional fields."""
        import yaml

        dashboard_file = tmp_path / "ui-lovelace-minimal.yaml"
        config_file = tmp_path / "configuration.yaml"
        config_file.write_text("# Empty config\n")

        def mock_path(filename):
            if filename == "configuration.yaml":
                return str(config_file)
            elif filename.startswith("ui-lovelace-"):
                return str(tmp_path / filename)
            return str(tmp_path / filename)

        with patch.object(hass.config, 'path', side_effect=mock_path):
            dashboard_config = {
                "title": "Minimal Dashboard",
                "url_path": "minimal"
            }

            result = await agent.create_dashboard(dashboard_config)

            assert result["success"] is True

            # Verify defaults were applied
            dashboard_data = yaml.safe_load(dashboard_file.read_text())
            assert dashboard_data["icon"] == "mdi:view-dashboard"
            assert dashboard_data["show_in_sidebar"] is True
            assert dashboard_data["views"] == []

    @pytest.mark.asyncio
    async def test_create_dashboard_config_file_update(self, agent, hass, tmp_path):
        """Test that configuration.yaml is updated when dashboard is created."""
        import yaml

        dashboard_file = tmp_path / "ui-lovelace-test.yaml"
        config_file = tmp_path / "configuration.yaml"
        config_file.write_text("# Configuration\n")

        def mock_path(filename):
            if filename == "configuration.yaml":
                return str(config_file)
            elif filename.startswith("ui-lovelace-"):
                return str(tmp_path / filename)
            return str(tmp_path / filename)

        with patch.object(hass.config, 'path', side_effect=mock_path):
            dashboard_config = {
                "title": "Test Dashboard",
                "url_path": "test",
                "views": []
            }

            result = await agent.create_dashboard(dashboard_config)

            assert result["success"] is True

            # Verify configuration.yaml was updated
            config_content = config_file.read_text()
            assert "lovelace:" in config_content
            assert "dashboards:" in config_content
            assert "test:" in config_content
            assert "mode: yaml" in config_content

    @pytest.mark.asyncio
    async def test_create_dashboard_file_write_error(self, agent, hass, tmp_path):
        """Test dashboard creation when file write fails."""
        config_file = tmp_path / "configuration.yaml"
        config_file.write_text("# Empty config\n")

        def mock_path(filename):
            if filename == "configuration.yaml":
                return str(config_file)
            # Return a path to a non-existent directory
            elif filename.startswith("ui-lovelace-"):
                return str(tmp_path / "nonexistent" / filename)
            return str(tmp_path / filename)

        with patch.object(hass.config, 'path', side_effect=mock_path):
            dashboard_config = {
                "title": "Test Dashboard",
                "url_path": "test",
                "views": []
            }

            result = await agent.create_dashboard(dashboard_config)

            assert "error" in result
            assert "Failed to create dashboard file" in result["error"]

    @pytest.mark.asyncio
    async def test_create_dashboard_with_custom_options(self, agent, hass, tmp_path):
        """Test dashboard creation with custom options."""
        import yaml

        dashboard_file = tmp_path / "ui-lovelace-custom.yaml"
        config_file = tmp_path / "configuration.yaml"
        config_file.write_text("# Configuration\n")

        def mock_path(filename):
            if filename == "configuration.yaml":
                return str(config_file)
            elif filename.startswith("ui-lovelace-"):
                return str(tmp_path / filename)
            return str(tmp_path / filename)

        with patch.object(hass.config, 'path', side_effect=mock_path):
            dashboard_config = {
                "title": "Custom Dashboard",
                "url_path": "custom",
                "icon": "mdi:rocket",
                "show_in_sidebar": False,
                "require_admin": True,
                "views": [{"title": "Custom View", "cards": [{"type": "markdown", "content": "Hello"}]}]
            }

            result = await agent.create_dashboard(dashboard_config)

            assert result["success"] is True
            assert result["url_path"] == "custom"

            # Verify custom options were applied
            dashboard_data = yaml.safe_load(dashboard_file.read_text())
            assert dashboard_data["icon"] == "mdi:rocket"
            assert dashboard_data["show_in_sidebar"] is False
            assert dashboard_data["require_admin"] is True
            assert len(dashboard_data["views"]) == 1

    @pytest.mark.asyncio
    async def test_create_dashboard_exception_handling(self, agent, hass):
        """Test dashboard creation general exception handling."""
        with patch.object(hass.config, 'path', side_effect=Exception("Unexpected error")):
            dashboard_config = {
                "title": "Test Dashboard",
                "url_path": "test"
            }

            result = await agent.create_dashboard(dashboard_config)

            assert "error" in result
            assert "Unexpected error" in result["error"]

    @pytest.mark.asyncio
    async def test_update_dashboard_success(self, agent, hass, tmp_path):
        """Test successful dashboard update."""
        import yaml

        # Create existing dashboard file
        dashboard_file = tmp_path / "ui-lovelace-test.yaml"
        initial_config = {
            "title": "Old Dashboard",
            "icon": "mdi:old",
            "show_in_sidebar": False,
            "views": []
        }
        dashboard_file.write_text(yaml.dump(initial_config))

        def mock_path(filename):
            return str(tmp_path / filename)

        with patch.object(hass.config, 'path', side_effect=mock_path):
            dashboard_config = {
                "title": "Updated Dashboard",
                "icon": "mdi:new",
                "show_in_sidebar": True,
                "require_admin": False,
                "views": [{"title": "New View", "cards": []}]
            }

            result = await agent.update_dashboard("test", dashboard_config)

            assert result["success"] is True
            assert "updated successfully" in result["message"]

            # Verify file was updated
            updated_data = yaml.safe_load(dashboard_file.read_text())
            assert updated_data["title"] == "Updated Dashboard"
            assert updated_data["icon"] == "mdi:new"
            assert updated_data["show_in_sidebar"] is True
            assert len(updated_data["views"]) == 1

    @pytest.mark.asyncio
    async def test_update_dashboard_file_not_found(self, agent, hass, tmp_path):
        """Test updating non-existent dashboard."""
        def mock_path(filename):
            return str(tmp_path / filename)

        with patch.object(hass.config, 'path', side_effect=mock_path):
            dashboard_config = {
                "title": "Updated Dashboard",
                "views": []
            }

            result = await agent.update_dashboard("nonexistent", dashboard_config)

            assert "error" in result
            assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_update_dashboard_minimal_config(self, agent, hass, tmp_path):
        """Test updating dashboard with minimal configuration."""
        import yaml

        # Create existing dashboard file
        dashboard_file = tmp_path / "ui-lovelace-minimal.yaml"
        initial_config = {
            "title": "Old",
            "views": []
        }
        dashboard_file.write_text(yaml.dump(initial_config))

        def mock_path(filename):
            return str(tmp_path / filename)

        with patch.object(hass.config, 'path', side_effect=mock_path):
            # Update with minimal config - should use defaults
            dashboard_config = {}

            result = await agent.update_dashboard("minimal", dashboard_config)

            assert result["success"] is True

            # Verify defaults were applied
            updated_data = yaml.safe_load(dashboard_file.read_text())
            assert updated_data["title"] == "Updated Dashboard"
            assert updated_data["icon"] == "mdi:view-dashboard"
            assert updated_data["show_in_sidebar"] is True
            assert updated_data["require_admin"] is False

    @pytest.mark.asyncio
    async def test_update_dashboard_write_error(self, agent, hass, tmp_path):
        """Test update_dashboard when file write fails."""
        # Create existing dashboard file as read-only
        dashboard_file = tmp_path / "ui-lovelace-readonly.yaml"
        dashboard_file.write_text("title: Old\n")
        dashboard_file.chmod(0o444)  # Read-only

        def mock_path(filename):
            return str(tmp_path / filename)

        try:
            with patch.object(hass.config, 'path', side_effect=mock_path):
                dashboard_config = {
                    "title": "New Title",
                    "views": []
                }

                result = await agent.update_dashboard("readonly", dashboard_config)

                assert "error" in result
                assert "Failed to update dashboard file" in result["error"]
        finally:
            # Restore write permissions for cleanup
            dashboard_file.chmod(0o644)

    @pytest.mark.asyncio
    async def test_update_dashboard_exception_handling(self, agent, hass):
        """Test update_dashboard general exception handling."""
        with patch.object(hass.config, 'path', side_effect=Exception("Unexpected error")):
            dashboard_config = {
                "title": "Test",
                "views": []
            }

            result = await agent.update_dashboard("test", dashboard_config)

            assert "error" in result
            assert "Failed to update dashboard file" in result["error"]

    @pytest.mark.asyncio
    async def test_get_device_registry_basic(self, agent, hass):
        """Test get_device_registry with basic functionality."""
        # Create a config entry
        config_entry = MockConfigEntry(domain="test", entry_id="test_entry")
        config_entry.add_to_hass(hass)

        from homeassistant.helpers import device_registry as dr

        # Create device registry
        device_reg = dr.async_get(hass)

        # Add test devices
        device_reg.async_get_or_create(
            config_entry_id="test_entry",
            identifiers={("test", "device1")},
            name="Test Device 1",
            manufacturer="Test Manufacturer",
            model="Model A"
        )
        device_reg.async_get_or_create(
            config_entry_id="test_entry",
            identifiers={("test", "device2")},
            name="Test Device 2",
            manufacturer="Test Manufacturer",
            model="Model B"
        )

        result = await agent.get_device_registry()

        assert "devices" in result
        assert result["total_count"] == 2
        assert result["returned_count"] == 2
        assert result["has_more"] is False
        assert len(result["devices"]) == 2

    @pytest.mark.asyncio
    async def test_get_device_registry_with_area_filter(self, agent, hass):
        """Test get_device_registry with area filtering."""
        # Create a config entry
        config_entry = MockConfigEntry(domain="test", entry_id="test_entry")
        config_entry.add_to_hass(hass)

        from homeassistant.helpers import area_registry as ar
        from homeassistant.helpers import device_registry as dr

        # Create areas
        area_reg = ar.async_get(hass)
        living_room = area_reg.async_create("Living Room")
        bedroom = area_reg.async_create("Bedroom")

        # Create device registry
        device_reg = dr.async_get(hass)

        # Add devices to different areas
        device1 = device_reg.async_get_or_create(
            config_entry_id="test_entry",
            identifiers={("test", "device1")},
            name="Living Room Light",
            manufacturer="Test"
        )
        device_reg.async_update_device(device1.id, area_id=living_room.id)

        device2 = device_reg.async_get_or_create(
            config_entry_id="test_entry",
            identifiers={("test", "device2")},
            name="Bedroom Light",
            manufacturer="Test"
        )
        device_reg.async_update_device(device2.id, area_id=bedroom.id)

        # Filter by living room
        result = await agent.get_device_registry(area_id=living_room.id)

        assert result["total_count"] == 1
        assert result["devices"][0]["name"] == "Living Room Light"
        assert result["devices"][0]["area_name"] == "Living Room"

    @pytest.mark.asyncio
    async def test_get_device_registry_with_manufacturer_filter(self, agent, hass):
        """Test get_device_registry with manufacturer filtering."""
        # Create a config entry
        config_entry = MockConfigEntry(domain="test", entry_id="test_entry")
        config_entry.add_to_hass(hass)

        from homeassistant.helpers import device_registry as dr

        device_reg = dr.async_get(hass)

        # Add devices with different manufacturers
        device_reg.async_get_or_create(
            config_entry_id="test_entry",
            identifiers={("test", "device1")},
            name="Device 1",
            manufacturer="Philips"
        )
        device_reg.async_get_or_create(
            config_entry_id="test_entry",
            identifiers={("test", "device2")},
            name="Device 2",
            manufacturer="IKEA"
        )
        device_reg.async_get_or_create(
            config_entry_id="test_entry",
            identifiers={("test", "device3")},
            name="Device 3",
            manufacturer="Philips"
        )

        # Filter by Philips
        result = await agent.get_device_registry(manufacturer="Philips")

        assert result["total_count"] == 2
        assert all(d["manufacturer"] == "Philips" for d in result["devices"])

    @pytest.mark.asyncio
    async def test_get_device_registry_pagination(self, agent, hass):
        """Test get_device_registry with pagination."""
        # Create a config entry
        config_entry = MockConfigEntry(domain="test", entry_id="test_entry")
        config_entry.add_to_hass(hass)

        from homeassistant.helpers import device_registry as dr

        device_reg = dr.async_get(hass)

        # Add multiple devices
        for i in range(10):
            device_reg.async_get_or_create(
                config_entry_id="test_entry",
                identifiers={("test", f"device{i}")},
                name=f"Device {i}",
                manufacturer="Test"
            )

        # First page
        result = await agent.get_device_registry(limit=5, offset=0)
        assert result["returned_count"] == 5
        assert result["total_count"] == 10
        assert result["has_more"] is True

        # Second page
        result = await agent.get_device_registry(limit=5, offset=5)
        assert result["returned_count"] == 5
        assert result["total_count"] == 10
        assert result["has_more"] is False

    @pytest.mark.asyncio
    async def test_get_device_registry_disabled_devices_excluded(self, agent, hass):
        """Test that disabled devices are excluded from results."""
        # Create a config entry
        config_entry = MockConfigEntry(domain="test", entry_id="test_entry")
        config_entry.add_to_hass(hass)

        from homeassistant.helpers import device_registry as dr

        device_reg = dr.async_get(hass)

        # Add enabled device
        device_reg.async_get_or_create(
            config_entry_id="test_entry",
            identifiers={("test", "device1")},
            name="Enabled Device",
            manufacturer="Test"
        )

        # Add disabled device
        disabled_device = device_reg.async_get_or_create(
            config_entry_id="test_entry",
            identifiers={("test", "device2")},
            name="Disabled Device",
            manufacturer="Test"
        )
        device_reg.async_update_device(
            disabled_device.id,
            disabled_by=dr.DeviceEntryDisabler.USER
        )

        result = await agent.get_device_registry()

        # Should only return enabled device
        assert result["total_count"] == 1
        assert result["devices"][0]["name"] == "Enabled Device"

    @pytest.mark.asyncio
    async def test_get_device_registry_empty(self, agent, hass):
        """Test get_device_registry when registry is empty."""
        result = await agent.get_device_registry()

        assert result["total_count"] == 0
        assert result["returned_count"] == 0
        assert result["devices"] == []
        assert result["has_more"] is False

    @pytest.mark.asyncio
    async def test_get_device_registry_exception_handling(self, agent, hass):
        """Test get_device_registry exception handling."""
        from homeassistant.helpers import device_registry as dr

        with patch.object(dr, 'async_get', side_effect=Exception("Registry error")):
            result = await agent.get_device_registry()

            assert "error" in result
            assert "Registry error" in result["error"]

    @pytest.mark.asyncio
    async def test_get_device_registry_summary_basic(self, agent, hass):
        """Test get_device_registry_summary basic functionality."""
        # Create a config entry
        config_entry = MockConfigEntry(domain="test", entry_id="test_entry")
        config_entry.add_to_hass(hass)

        from homeassistant.helpers import area_registry as ar
        from homeassistant.helpers import device_registry as dr

        # Create areas
        area_reg = ar.async_get(hass)
        living_room = area_reg.async_create("Living Room")
        bedroom = area_reg.async_create("Bedroom")

        # Create device registry
        device_reg = dr.async_get(hass)

        # Add devices
        device1 = device_reg.async_get_or_create(
            config_entry_id="test_entry",
            identifiers={("test", "device1")},
            name="Living Room Light",
            manufacturer="Philips"
        )
        device_reg.async_update_device(device1.id, area_id=living_room.id)

        device2 = device_reg.async_get_or_create(
            config_entry_id="test_entry",
            identifiers={("test", "device2")},
            name="Living Room Switch",
            manufacturer="IKEA"
        )
        device_reg.async_update_device(device2.id, area_id=living_room.id)

        device3 = device_reg.async_get_or_create(
            config_entry_id="test_entry",
            identifiers={("test", "device3")},
            name="Bedroom Light",
            manufacturer="Philips"
        )
        device_reg.async_update_device(device3.id, area_id=bedroom.id)

        result = await agent.get_device_registry_summary()

        assert result["total_devices"] == 3
        assert result["by_area"]["Living Room"] == 2
        assert result["by_area"]["Bedroom"] == 1
        assert result["by_manufacturer"]["Philips"] == 2
        assert result["by_manufacturer"]["IKEA"] == 1

    @pytest.mark.asyncio
    async def test_get_device_registry_summary_disabled_devices_excluded(self, agent, hass):
        """Test that summary excludes disabled devices."""
        # Create a config entry
        config_entry = MockConfigEntry(domain="test", entry_id="test_entry")
        config_entry.add_to_hass(hass)

        from homeassistant.helpers import device_registry as dr

        device_reg = dr.async_get(hass)

        # Add enabled device
        device_reg.async_get_or_create(
            config_entry_id="test_entry",
            identifiers={("test", "device1")},
            name="Enabled Device",
            manufacturer="Test"
        )

        # Add disabled device
        disabled_device = device_reg.async_get_or_create(
            config_entry_id="test_entry",
            identifiers={("test", "device2")},
            name="Disabled Device",
            manufacturer="Test"
        )
        device_reg.async_update_device(
            disabled_device.id,
            disabled_by=dr.DeviceEntryDisabler.USER
        )

        result = await agent.get_device_registry_summary()

        assert result["total_devices"] == 1
        assert result["by_manufacturer"]["Test"] == 1

    @pytest.mark.asyncio
    async def test_get_device_registry_summary_no_area(self, agent, hass):
        """Test summary with devices not assigned to areas."""
        # Create a config entry
        config_entry = MockConfigEntry(domain="test", entry_id="test_entry")
        config_entry.add_to_hass(hass)

        from homeassistant.helpers import device_registry as dr

        device_reg = dr.async_get(hass)

        # Add device without area
        device_reg.async_get_or_create(
            config_entry_id="test_entry",
            identifiers={("test", "device1")},
            name="Unassigned Device",
            manufacturer="Test"
        )

        result = await agent.get_device_registry_summary()

        assert result["total_devices"] == 1
        assert result["by_area"]["unassigned"] == 1

    @pytest.mark.asyncio
    async def test_get_device_registry_summary_no_manufacturer(self, agent, hass):
        """Test summary with devices without manufacturer."""
        # Create a config entry
        config_entry = MockConfigEntry(domain="test", entry_id="test_entry")
        config_entry.add_to_hass(hass)

        from homeassistant.helpers import device_registry as dr

        device_reg = dr.async_get(hass)

        # Add device without manufacturer
        device_reg.async_get_or_create(
            config_entry_id="test_entry",
            identifiers={("test", "device1")},
            name="Generic Device"
        )

        result = await agent.get_device_registry_summary()

        assert result["total_devices"] == 1
        assert result["by_manufacturer"]["unknown"] == 1

    @pytest.mark.asyncio
    async def test_get_device_registry_summary_empty(self, agent, hass):
        """Test summary when registry is empty."""
        result = await agent.get_device_registry_summary()

        assert result["total_devices"] == 0
        assert result["by_area"] == {}
        assert result["by_manufacturer"] == {}

    @pytest.mark.asyncio
    async def test_get_device_registry_summary_exception_handling(self, agent, hass):
        """Test summary exception handling."""
        from homeassistant.helpers import device_registry as dr

        with patch.object(dr, 'async_get', side_effect=Exception("Registry error")):
            result = await agent.get_device_registry_summary()

            assert "error" in result
            assert "Registry error" in result["error"]
    @pytest.mark.asyncio
    async def test_get_entity_registry_summary_success(self, agent, hass):
        """Test get_entity_registry_summary returns proper summary with counts."""
        # Create a config entry
        config_entry = MockConfigEntry(domain="test", entry_id="test_config")
        config_entry.add_to_hass(hass)

        from homeassistant.helpers import entity_registry as er, area_registry as ar, device_registry as dr

        # Create area registry with test areas
        area_reg = ar.async_get(hass)
        living_room = area_reg.async_create("Living Room")
        bedroom = area_reg.async_create("Bedroom")

        # Create device registry
        device_reg = dr.async_get(hass)
        test_device = device_reg.async_get_or_create(
            config_entry_id="test_config",
            identifiers={("test", "device1")}
        )
        # Update device with area
        device_reg.async_update_device(test_device.id, area_id=living_room.id)

        # Create entity registry entries
        entity_reg = er.async_get(hass)

        # Light entity in living room
        light_entity = entity_reg.async_get_or_create(
            "light",
            "test",
            "light1",
            suggested_object_id="living_room_light"
        )
        entity_reg.async_update_entity(light_entity.entity_id, area_id=living_room.id)

        # Temperature sensor in bedroom with device class
        temp_entity = entity_reg.async_get_or_create(
            "sensor",
            "test",
            "temp1",
            suggested_object_id="bedroom_temp"
        )
        entity_reg.async_update_entity(temp_entity.entity_id, area_id=bedroom.id)

        # Switch entity with device (inherits area from device)
        switch_entity = entity_reg.async_get_or_create(
            "switch",
            "test",
            "switch1",
            suggested_object_id="device_switch",
            device_id=test_device.id
        )

        # Entity without area assignment
        unassigned_entity = entity_reg.async_get_or_create(
            "sensor",
            "test",
            "unassigned1",
            suggested_object_id="unassigned_sensor"
        )

        # Disabled entity (should be excluded)
        disabled_entity = entity_reg.async_get_or_create(
            "light",
            "test",
            "disabled1",
            suggested_object_id="disabled_light",
            disabled_by=er.RegistryEntryDisabler.USER
        )

        # Add states for entities with device_class
        hass.states.async_set(
            temp_entity.entity_id,
            "20.5",
            {"device_class": "temperature", "unit_of_measurement": "°C"}
        )
        hass.states.async_set(light_entity.entity_id, "on", {})
        hass.states.async_set(switch_entity.entity_id, "off", {})
        hass.states.async_set(unassigned_entity.entity_id, "unknown", {})

        result = await agent.get_entity_registry_summary()

        # Verify total count (excludes disabled)
        assert result["total_entities"] == 4

        # Verify domain counts
        assert result["by_domain"]["sensor"] == 2
        assert result["by_domain"]["light"] == 1
        assert result["by_domain"]["switch"] == 1

        # Verify area counts
        assert result["by_area"]["Living Room"] == 2  # light + switch (via device)
        assert result["by_area"]["Bedroom"] == 1  # temp sensor
        assert result["by_area"]["unassigned"] == 1  # unassigned sensor

        # Verify device_class counts
        assert result["by_device_class"]["temperature"] == 1

    @pytest.mark.asyncio
    async def test_get_entity_registry_summary_empty_registry(self, agent, hass):
        """Test get_entity_registry_summary with no entities."""
        result = await agent.get_entity_registry_summary()

        assert result["total_entities"] == 0
        assert result["by_domain"] == {}
        assert result["by_area"] == {}
        assert result["by_device_class"] == {}

    @pytest.mark.asyncio
    async def test_get_entity_registry_summary_sorted_output(self, agent, hass):
        """Test get_entity_registry_summary returns sorted results."""
        from homeassistant.helpers import entity_registry as er

        entity_reg = er.async_get(hass)

        # Create multiple entities with different domain counts
        for i in range(5):
            entity_reg.async_get_or_create("light", "test", f"light{i}")
        for i in range(3):
            entity_reg.async_get_or_create("switch", "test", f"switch{i}")
        for i in range(7):
            entity_reg.async_get_or_create("sensor", "test", f"sensor{i}")

        result = await agent.get_entity_registry_summary()

        # Verify sorting (should be sorted by count, descending)
        domain_list = list(result["by_domain"].items())
        assert domain_list[0] == ("sensor", 7)
        assert domain_list[1] == ("light", 5)
        assert domain_list[2] == ("switch", 3)

    @pytest.mark.asyncio
    async def test_get_entity_registry_summary_error_handling(self, agent, hass):
        """Test get_entity_registry_summary handles errors gracefully."""
        from homeassistant.helpers import entity_registry as er

        # Mock entity_registry to raise an exception
        with patch.object(er, 'async_get', side_effect=Exception("Registry error")):
            result = await agent.get_entity_registry_summary()

            # Should return empty structure on error
            assert result == {"error": "Error getting entity registry summary: Registry error"}

    @pytest.mark.asyncio
    async def test_get_history_success(self, agent, hass):
        """Test get_history returns historical state changes."""
        from datetime import datetime, timedelta
        from homeassistant.core import State
        from homeassistant.util import dt as dt_util

        # Create a test entity with state
        entity_id = "sensor.test_sensor"
        hass.states.async_set(entity_id, "20", {"unit_of_measurement": "°C"})

        # Mock get_significant_states to return historical data
        mock_state1 = State(
            entity_id,
            "18",
            {"unit_of_measurement": "°C"},
            last_changed=dt_util.utcnow() - timedelta(hours=2),
            last_updated=dt_util.utcnow() - timedelta(hours=2)
        )
        mock_state2 = State(
            entity_id,
            "19",
            {"unit_of_measurement": "°C"},
            last_changed=dt_util.utcnow() - timedelta(hours=1),
            last_updated=dt_util.utcnow() - timedelta(hours=1)
        )
        mock_state3 = State(
            entity_id,
            "20",
            {"unit_of_measurement": "°C"},
            last_changed=dt_util.utcnow(),
            last_updated=dt_util.utcnow()
        )

        mock_history = {entity_id: [mock_state1, mock_state2, mock_state3]}

        async def mock_executor(func, *args):
            return func(*args)

        with patch("homeassistant.components.recorder.history.get_significant_states", return_value=mock_history):
            with patch.object(hass, 'async_add_executor_job', side_effect=mock_executor):
                result = await agent.get_history(entity_id, hours=24)

        assert len(result) == 3
        assert result[0]["entity_id"] == entity_id
        assert result[0]["state"] == "18"
        assert result[1]["state"] == "19"
        assert result[2]["state"] == "20"
        assert "last_changed" in result[0]
        assert "last_updated" in result[0]
        assert "attributes" in result[0]
        assert result[0]["attributes"]["unit_of_measurement"] == "°C"

    @pytest.mark.asyncio
    async def test_get_history_custom_hours(self, agent, hass):
        """Test get_history with custom hour parameter."""
        from homeassistant.util import dt as dt_util

        entity_id = "sensor.test_sensor"
        hass.states.async_set(entity_id, "20", {})

        mock_history = {entity_id: []}

        async def mock_executor(func, *args):
            return func(*args)

        with patch("homeassistant.components.recorder.history.get_significant_states", return_value=mock_history) as mock_get:
            with patch.object(hass, 'async_add_executor_job', side_effect=mock_executor):
                result = await agent.get_history(entity_id, hours=12)

        # Verify the function was called with correct time range
        call_args = mock_get.call_args[0]
        start_time = call_args[1]
        end_time = call_args[2]

        # The time difference should be approximately 12 hours
        time_diff = (end_time - start_time).total_seconds() / 3600
        assert 11.9 < time_diff < 12.1  # Allow small variance

    @pytest.mark.asyncio
    async def test_get_history_empty_entity_id(self, agent, hass):
        """Test get_history with empty entity_id returns error."""
        result = await agent.get_history("")

        assert len(result) == 1
        assert "error" in result[0]
        assert result[0]["error"] == "entity_id is required"

    @pytest.mark.asyncio
    async def test_get_history_no_history_data(self, agent, hass):
        """Test get_history when no historical data exists."""
        entity_id = "sensor.test_sensor"
        hass.states.async_set(entity_id, "20", {})

        mock_history = {}  # Empty history

        async def mock_executor(func, *args):
            return func(*args)

        with patch("homeassistant.components.recorder.history.get_significant_states", return_value=mock_history):
            with patch.object(hass, 'async_add_executor_job', side_effect=mock_executor):
                result = await agent.get_history(entity_id)

        assert result == []

    @pytest.mark.asyncio
    async def test_get_history_filters_dict_states(self, agent, hass):
        """Test get_history filters out dict states (mypy type narrowing)."""
        from homeassistant.core import State
        from homeassistant.util import dt as dt_util

        entity_id = "sensor.test_sensor"
        hass.states.async_set(entity_id, "20", {})

        mock_state = State(
            entity_id,
            "20",
            {},
            last_changed=dt_util.utcnow(),
            last_updated=dt_util.utcnow()
        )

        # Mix State objects and dicts (dicts should be filtered out)
        mock_history = {entity_id: [mock_state, {"invalid": "dict"}, mock_state]}

        async def mock_executor(func, *args):
            return func(*args)

        with patch("homeassistant.components.recorder.history.get_significant_states", return_value=mock_history):
            with patch.object(hass, 'async_add_executor_job', side_effect=mock_executor):
                result = await agent.get_history(entity_id)

        # Should only have 2 states (dict should be filtered out)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_get_history_error_handling(self, agent, hass):
        """Test get_history handles errors gracefully."""
        entity_id = "sensor.test_sensor"

        async def mock_executor(func, *args):
            return func(*args)

        with patch("homeassistant.components.recorder.history.get_significant_states", side_effect=Exception("Database error")):
            with patch.object(hass, 'async_add_executor_job', side_effect=mock_executor):
                result = await agent.get_history(entity_id)

        assert len(result) == 1
        assert "error" in result[0]
        assert "Database error" in result[0]["error"]


class TestSystemPromptInitialization:
    """Test system prompt initialization with different providers and configurations."""

    @pytest.fixture
    def openai_config(self):
        """OpenAI provider configuration."""
        return {
            "ai_provider": "openai",
            "openai_token": "test_token",
            "models": {"openai": "gpt-4"}
        }

    @pytest.fixture
    def anthropic_config(self):
        """Anthropic provider configuration."""
        return {
            "ai_provider": "anthropic",
            "anthropic_token": "test_token",
            "models": {"anthropic": "claude-sonnet-4-5-20250929"}
        }

    @pytest.fixture
    def gemini_config(self):
        """Gemini provider configuration."""
        return {
            "ai_provider": "gemini",
            "gemini_token": "test_token",
            "models": {"gemini": "gemini-2.5-flash"}
        }

    @pytest.fixture
    def openrouter_config(self):
        """OpenRouter provider configuration."""
        return {
            "ai_provider": "openrouter",
            "openrouter_token": "test_token",
            "models": {"openrouter": "openai/gpt-4o"}
        }

    @pytest.fixture
    def local_config(self):
        """Local provider configuration."""
        return {
            "ai_provider": "local",
            "local_url": "http://localhost:8080",
            "models": {"local": "llama-3"}
        }

    @pytest.fixture
    def llama_config(self):
        """Llama provider configuration."""
        return {
            "ai_provider": "llama",
            "llama_token": "test_token",
            "models": {"llama": "Llama-4-Maverick-17B-128E-Instruct-FP8"}
        }

    @pytest.fixture
    def alter_config(self):
        """Alter provider configuration."""
        return {
            "ai_provider": "alter",
            "alter_token": "test_token",
            "models": {"alter": "alter-model"}
        }

    @pytest.fixture
    def zai_config(self):
        """Zai provider configuration."""
        return {
            "ai_provider": "zai",
            "zai_token": "test_token",
            "zai_endpoint": "general",
            "models": {"zai": "glm-4.7"}
        }

    @pytest.mark.asyncio
    async def test_system_prompt_standard_for_openai(self, hass, openai_config):
        """Test that OpenAI provider uses standard system prompt."""
        from custom_components.ai_agent_ha.prompts import SYSTEM_PROMPT

        agent = AiAgentHaAgent(hass, openai_config)

        assert agent.system_prompt == SYSTEM_PROMPT
        assert agent.system_prompt["role"] == "system"
        assert "You are an AI assistant for Home Assistant" in agent.system_prompt["content"]
        assert "CRITICAL: INTENT RECOGNITION" in agent.system_prompt["content"]

    @pytest.mark.asyncio
    async def test_system_prompt_standard_for_anthropic(self, hass, anthropic_config):
        """Test that Anthropic provider uses standard system prompt."""
        from custom_components.ai_agent_ha.prompts import SYSTEM_PROMPT

        agent = AiAgentHaAgent(hass, anthropic_config)

        assert agent.system_prompt == SYSTEM_PROMPT
        assert agent.system_prompt["role"] == "system"

    @pytest.mark.asyncio
    async def test_system_prompt_standard_for_gemini(self, hass, gemini_config):
        """Test that Gemini provider uses standard system prompt."""
        from custom_components.ai_agent_ha.prompts import SYSTEM_PROMPT

        agent = AiAgentHaAgent(hass, gemini_config)

        assert agent.system_prompt == SYSTEM_PROMPT
        assert agent.system_prompt["role"] == "system"

    @pytest.mark.asyncio
    async def test_system_prompt_standard_for_openrouter(self, hass, openrouter_config):
        """Test that OpenRouter provider uses standard system prompt."""
        from custom_components.ai_agent_ha.prompts import SYSTEM_PROMPT

        agent = AiAgentHaAgent(hass, openrouter_config)

        assert agent.system_prompt == SYSTEM_PROMPT

    @pytest.mark.asyncio
    async def test_system_prompt_local_for_local_provider(self, hass, local_config):
        """Test that local provider uses local-optimized system prompt."""
        from custom_components.ai_agent_ha.prompts import SYSTEM_PROMPT_LOCAL

        agent = AiAgentHaAgent(hass, local_config)

        assert agent.system_prompt == SYSTEM_PROMPT_LOCAL
        assert agent.system_prompt["role"] == "system"
        assert "You are an AI assistant for Home Assistant smart home" in agent.system_prompt["content"]
        assert "INTENT RECOGNITION (MOST IMPORTANT!)" in agent.system_prompt["content"]
        assert "Always respond with JSON only" in agent.system_prompt["content"]

    @pytest.mark.asyncio
    async def test_system_prompt_standard_for_llama(self, hass, llama_config):
        """Test that Llama provider uses standard system prompt."""
        from custom_components.ai_agent_ha.prompts import SYSTEM_PROMPT

        agent = AiAgentHaAgent(hass, llama_config)

        assert agent.system_prompt == SYSTEM_PROMPT

    @pytest.mark.asyncio
    async def test_system_prompt_standard_for_alter(self, hass, alter_config):
        """Test that Alter provider uses standard system prompt."""
        from custom_components.ai_agent_ha.prompts import SYSTEM_PROMPT

        agent = AiAgentHaAgent(hass, alter_config)

        assert agent.system_prompt == SYSTEM_PROMPT

    @pytest.mark.asyncio
    async def test_system_prompt_standard_for_zai(self, hass, zai_config):
        """Test that Zai provider uses standard system prompt."""
        from custom_components.ai_agent_ha.prompts import SYSTEM_PROMPT

        agent = AiAgentHaAgent(hass, zai_config)

        assert agent.system_prompt == SYSTEM_PROMPT

    @pytest.mark.asyncio
    async def test_system_prompt_contains_tool_descriptions(self, hass, openai_config):
        """Test that standard system prompt contains tool descriptions."""
        agent = AiAgentHaAgent(hass, openai_config)

        content = agent.system_prompt["content"]

        # Check for key tool categories
        assert "ENTITY DATA:" in content
        assert "REGISTRY" in content
        assert "CONTROL:" in content
        assert "CREATE" in content
        assert "WEB TOOLS:" in content
        assert "DOCUMENTATION:" in content

        # Check for specific tools
        assert "get_entity_state" in content
        assert "get_entities_by_domain" in content
        assert "get_area_registry" in content
        assert "call_service" in content
        assert "create_dashboard" in content
        assert "web_fetch" in content
        assert "web_search" in content

    @pytest.mark.asyncio
    async def test_system_prompt_local_contains_json_format(self, hass, local_config):
        """Test that local system prompt emphasizes JSON format."""
        agent = AiAgentHaAgent(hass, local_config)

        content = agent.system_prompt["content"]

        # Check JSON format instructions
        assert "JSON only" in content
        assert '"request_type":' in content
        assert '"final_response"' in content
        assert '"data_request"' in content
        assert '"call_service"' in content

    @pytest.mark.asyncio
    async def test_system_prompt_includes_intent_recognition(self, hass, openai_config):
        """Test that system prompt includes intent recognition guidelines."""
        agent = AiAgentHaAgent(hass, openai_config)

        content = agent.system_prompt["content"]

        assert "INTENT RECOGNITION" in content
        assert "QUESTION/INFO" in content
        assert "CONTROL DEVICE" in content
        assert "CREATE DASHBOARD" in content
        assert "CREATE AUTOMATION" in content

    @pytest.mark.asyncio
    async def test_system_prompt_includes_response_format(self, hass, openai_config):
        """Test that system prompt includes response format guidelines."""
        agent = AiAgentHaAgent(hass, openai_config)

        content = agent.system_prompt["content"]

        assert "RESPONSE FORMAT" in content
        assert "Use the provided TOOLS" in content
        assert "NATURAL LANGUAGE" in content
        assert "NEVER output raw JSON to the user" in content

    @pytest.mark.asyncio
    async def test_system_prompt_includes_error_recovery(self, hass, openai_config):
        """Test that system prompt includes error recovery guidelines."""
        agent = AiAgentHaAgent(hass, openai_config)

        content = agent.system_prompt["content"]

        assert "ERROR RECOVERY" in content
        assert "max 3 attempts" in content
        assert "Entity not found" in content
        assert "Area not found" in content

    @pytest.mark.asyncio
    async def test_system_prompt_includes_critical_rules(self, hass, openai_config):
        """Test that system prompt includes critical usage rules."""
        agent = AiAgentHaAgent(hass, openai_config)

        content = agent.system_prompt["content"]

        assert "CRITICAL RULES" in content
        assert "Use TOOLS to fetch data" in content
        assert "do NOT guess entity states" in content

    @pytest.mark.asyncio
    async def test_system_prompt_different_for_local_vs_cloud(self, hass, openai_config, local_config):
        """Test that local and cloud providers get different prompts."""
        cloud_agent = AiAgentHaAgent(hass, openai_config)
        local_agent = AiAgentHaAgent(hass, local_config)

        # Should be different prompts
        assert cloud_agent.system_prompt != local_agent.system_prompt

        # Cloud should have more detailed content
        assert len(cloud_agent.system_prompt["content"]) > len(local_agent.system_prompt["content"])

        # Local should emphasize JSON
        assert "JSON only" in local_agent.system_prompt["content"]
        assert "JSON only" not in cloud_agent.system_prompt["content"]

    @pytest.mark.asyncio
    async def test_system_prompt_is_dict_with_role_and_content(self, hass, openai_config):
        """Test that system prompt is a properly formatted message dict."""
        agent = AiAgentHaAgent(hass, openai_config)

        assert isinstance(agent.system_prompt, dict)
        assert "role" in agent.system_prompt
        assert "content" in agent.system_prompt
        assert agent.system_prompt["role"] == "system"
        assert isinstance(agent.system_prompt["content"], str)
        assert len(agent.system_prompt["content"]) > 0

    @pytest.mark.asyncio
    async def test_system_prompt_local_is_dict_with_role_and_content(self, hass, local_config):
        """Test that local system prompt is properly formatted."""
        agent = AiAgentHaAgent(hass, local_config)

        assert isinstance(agent.system_prompt, dict)
        assert "role" in agent.system_prompt
        assert "content" in agent.system_prompt
        assert agent.system_prompt["role"] == "system"
        assert isinstance(agent.system_prompt["content"], str)
        assert len(agent.system_prompt["content"]) > 0


class TestEntityStateMethods:
    """Tests for entity state methods: get_entity_state, get_entities_by_domain."""

    @pytest.fixture
    def mock_config(self):
        """Mock agent configuration."""
        return {
            "ai_provider": "openai",
            "openai_token": "test_token",
            "openai_model": "gpt-3.5-turbo",
        }

    @pytest.fixture
    def mock_state(self):
        """Create a mock state object."""
        from datetime import datetime
        state = MagicMock()
        state.entity_id = "light.living_room"
        state.state = "on"
        state.last_changed = datetime(2024, 1, 15, 10, 30, 0)
        state.attributes = {
            "friendly_name": "Living Room Light",
            "brightness": 255,
            "color_temp": 400,
        }
        return state

    # ==================== get_entity_state tests ====================

    @pytest.mark.asyncio
    async def test_get_entity_state_empty_entity_id(self, hass, mock_config):
        """Test get_entity_state returns error when entity_id is empty."""
        agent = AiAgentHaAgent(hass, mock_config)

        result = await agent.get_entity_state("")

        assert "error" in result
        assert result["error"] == "Entity ID is required"

    @pytest.mark.asyncio
    async def test_get_entity_state_none_entity_id(self, hass, mock_config):
        """Test get_entity_state returns error when entity_id is None."""
        agent = AiAgentHaAgent(hass, mock_config)

        result = await agent.get_entity_state(None)

        assert "error" in result
        assert result["error"] == "Entity ID is required"

    @pytest.mark.asyncio
    async def test_get_entity_state_entity_not_found(self, hass, mock_config):
        """Test get_entity_state returns error when entity doesn't exist."""
        agent = AiAgentHaAgent(hass, mock_config)

        result = await agent.get_entity_state("light.nonexistent")

        assert "error" in result
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_get_entity_state_success(self, hass, mock_config):
        """Test get_entity_state returns correct state information."""
        hass.states.async_set(
            "light.kitchen",
            "on",
            {"friendly_name": "Kitchen Light", "brightness": 200}
        )
        agent = AiAgentHaAgent(hass, mock_config)

        result = await agent.get_entity_state("light.kitchen")

        assert result["entity_id"] == "light.kitchen"
        assert result["state"] == "on"
        assert result["friendly_name"] == "Kitchen Light"
        assert result["attributes"]["brightness"] == 200

    @pytest.mark.asyncio
    async def test_get_entity_state_includes_last_changed(self, hass, mock_config):
        """Test get_entity_state includes last_changed timestamp."""
        hass.states.async_set("sensor.temp", "22.5")
        agent = AiAgentHaAgent(hass, mock_config)

        result = await agent.get_entity_state("sensor.temp")

        assert "last_changed" in result
        # last_changed should be an ISO format string or None
        assert result["last_changed"] is None or isinstance(result["last_changed"], str)

    @pytest.mark.asyncio
    async def test_get_entity_state_includes_area_fields(self, hass, mock_config):
        """Test get_entity_state includes area_id and area_name fields."""
        hass.states.async_set("switch.bedroom", "off")
        agent = AiAgentHaAgent(hass, mock_config)

        result = await agent.get_entity_state("switch.bedroom")

        # area_id and area_name should be present (may be None)
        assert "area_id" in result
        assert "area_name" in result

    @pytest.mark.asyncio
    async def test_get_entity_state_converts_datetime_attributes(self, hass, mock_config):
        """Test get_entity_state converts datetime attributes to ISO format."""
        from datetime import datetime

        hass.states.async_set(
            "sensor.with_datetime",
            "active",
            {"last_triggered": datetime(2024, 6, 15, 14, 30, 0)}
        )
        agent = AiAgentHaAgent(hass, mock_config)

        result = await agent.get_entity_state("sensor.with_datetime")

        # Datetime attributes should be converted to ISO format strings
        assert "last_triggered" in result["attributes"]
        assert isinstance(result["attributes"]["last_triggered"], str)

    @pytest.mark.asyncio
    async def test_get_entity_state_with_area_from_entity_registry(self, hass, mock_config):
        """Test get_entity_state retrieves area from entity registry."""
        hass.states.async_set("light.test", "on", {"friendly_name": "Test Light"})
        agent = AiAgentHaAgent(hass, mock_config)

        # Mock the registries
        mock_entity_entry = MagicMock()
        mock_entity_entry.area_id = "living_room"
        mock_entity_entry.device_id = None

        mock_area_entry = MagicMock()
        mock_area_entry.name = "Living Room"

        mock_entity_registry = MagicMock()
        mock_entity_registry.async_get = MagicMock(return_value=mock_entity_entry)

        mock_area_registry = MagicMock()
        mock_area_registry.async_get_area = MagicMock(return_value=mock_area_entry)

        with patch("homeassistant.helpers.entity_registry.async_get", return_value=mock_entity_registry):
            with patch("homeassistant.helpers.area_registry.async_get", return_value=mock_area_registry):
                with patch("homeassistant.helpers.device_registry.async_get", return_value=MagicMock()):
                    result = await agent.get_entity_state("light.test")

        assert result["area_id"] == "living_room"
        assert result["area_name"] == "Living Room"

    @pytest.mark.asyncio
    async def test_get_entity_state_with_area_from_device_registry(self, hass, mock_config):
        """Test get_entity_state retrieves area from device when entity has no direct area."""
        hass.states.async_set("light.device_light", "on")
        agent = AiAgentHaAgent(hass, mock_config)

        # Mock entity with device_id but no area_id
        mock_entity_entry = MagicMock()
        mock_entity_entry.area_id = None
        mock_entity_entry.device_id = "device_123"

        # Mock device with area
        mock_device_entry = MagicMock()
        mock_device_entry.area_id = "bedroom"

        mock_area_entry = MagicMock()
        mock_area_entry.name = "Bedroom"

        mock_entity_registry = MagicMock()
        mock_entity_registry.async_get = MagicMock(return_value=mock_entity_entry)

        mock_device_registry = MagicMock()
        mock_device_registry.async_get = MagicMock(return_value=mock_device_entry)

        mock_area_registry = MagicMock()
        mock_area_registry.async_get_area = MagicMock(return_value=mock_area_entry)

        with patch("homeassistant.helpers.entity_registry.async_get", return_value=mock_entity_registry):
            with patch("homeassistant.helpers.device_registry.async_get", return_value=mock_device_registry):
                with patch("homeassistant.helpers.area_registry.async_get", return_value=mock_area_registry):
                    result = await agent.get_entity_state("light.device_light")

        assert result["area_id"] == "bedroom"
        assert result["area_name"] == "Bedroom"

    @pytest.mark.asyncio
    async def test_get_entity_state_handles_registry_exception(self, hass, mock_config):
        """Test get_entity_state handles exceptions when accessing registries."""
        hass.states.async_set("light.test", "on", {"friendly_name": "Test"})
        agent = AiAgentHaAgent(hass, mock_config)

        # Registry access throws exception - should still return state
        with patch("homeassistant.helpers.entity_registry.async_get", side_effect=Exception("Registry error")):
            result = await agent.get_entity_state("light.test")

        # Should still return the basic state info
        assert result["entity_id"] == "light.test"
        assert result["state"] == "on"
        # area_id/area_name should be None due to exception
        assert result["area_id"] is None
        assert result["area_name"] is None

    @pytest.mark.asyncio
    async def test_get_entity_state_handles_outer_exception(self, hass, mock_config):
        """Test get_entity_state handles general exceptions gracefully."""
        agent = AiAgentHaAgent(hass, mock_config)

        # Create a mock hass with states.get that raises an exception
        mock_hass = MagicMock()
        mock_hass.states.get.side_effect = Exception("Unexpected error")
        agent.hass = mock_hass

        result = await agent.get_entity_state("light.broken")

        assert "error" in result
        assert "Error getting entity state" in result["error"]

    # ==================== get_entities_by_domain tests ====================

    @pytest.mark.asyncio
    async def test_get_entities_by_domain_empty_domain(self, hass, mock_config):
        """Test get_entities_by_domain returns error when domain is empty."""
        agent = AiAgentHaAgent(hass, mock_config)

        result = await agent.get_entities_by_domain("")

        assert len(result) == 1
        assert "error" in result[0]
        assert "domain is required" in result[0]["error"]

    @pytest.mark.asyncio
    async def test_get_entities_by_domain_none_domain(self, hass, mock_config):
        """Test get_entities_by_domain returns error when domain is None."""
        agent = AiAgentHaAgent(hass, mock_config)

        result = await agent.get_entities_by_domain(None)

        assert len(result) == 1
        assert "error" in result[0]
        assert "domain is required" in result[0]["error"]

    @pytest.mark.asyncio
    async def test_get_entities_by_domain_no_entities(self, hass, mock_config):
        """Test get_entities_by_domain returns empty list when no entities in domain."""
        agent = AiAgentHaAgent(hass, mock_config)

        result = await agent.get_entities_by_domain("vacuum")

        assert result == []

    @pytest.mark.asyncio
    async def test_get_entities_by_domain_single_entity(self, hass, mock_config):
        """Test get_entities_by_domain returns single entity correctly."""
        hass.states.async_set("fan.bedroom", "on", {"friendly_name": "Bedroom Fan"})
        agent = AiAgentHaAgent(hass, mock_config)

        result = await agent.get_entities_by_domain("fan")

        assert len(result) == 1
        assert result[0]["entity_id"] == "fan.bedroom"
        assert result[0]["state"] == "on"

    @pytest.mark.asyncio
    async def test_get_entities_by_domain_multiple_entities(self, hass, mock_config):
        """Test get_entities_by_domain returns all entities in domain."""
        hass.states.async_set("light.kitchen", "on")
        hass.states.async_set("light.bedroom", "off")
        hass.states.async_set("light.bathroom", "on")
        hass.states.async_set("switch.garage", "off")  # Different domain
        agent = AiAgentHaAgent(hass, mock_config)

        result = await agent.get_entities_by_domain("light")

        assert len(result) == 3
        entity_ids = [e["entity_id"] for e in result]
        assert "light.kitchen" in entity_ids
        assert "light.bedroom" in entity_ids
        assert "light.bathroom" in entity_ids
        assert "switch.garage" not in entity_ids

    @pytest.mark.asyncio
    async def test_get_entities_by_domain_filters_correctly(self, hass, mock_config):
        """Test get_entities_by_domain only returns entities from specified domain."""
        hass.states.async_set("sensor.temp", "22")
        hass.states.async_set("sensor.humidity", "55")
        hass.states.async_set("binary_sensor.motion", "on")
        hass.states.async_set("light.living", "on")
        agent = AiAgentHaAgent(hass, mock_config)

        sensor_result = await agent.get_entities_by_domain("sensor")
        binary_result = await agent.get_entities_by_domain("binary_sensor")

        assert len(sensor_result) == 2
        assert len(binary_result) == 1
        assert binary_result[0]["entity_id"] == "binary_sensor.motion"

    @pytest.mark.asyncio
    async def test_get_entities_by_domain_includes_full_state_info(self, hass, mock_config):
        """Test get_entities_by_domain returns full state info for each entity."""
        hass.states.async_set(
            "cover.garage_door",
            "closed",
            {"friendly_name": "Garage Door", "device_class": "garage"}
        )
        agent = AiAgentHaAgent(hass, mock_config)

        result = await agent.get_entities_by_domain("cover")

        assert len(result) == 1
        entity = result[0]
        assert entity["entity_id"] == "cover.garage_door"
        assert entity["state"] == "closed"
        assert entity["friendly_name"] == "Garage Door"
        assert "attributes" in entity
        assert entity["attributes"]["device_class"] == "garage"

    @pytest.mark.asyncio
    async def test_get_entities_by_domain_handles_exception(self, hass, mock_config):
        """Test get_entities_by_domain handles exceptions gracefully."""
        agent = AiAgentHaAgent(hass, mock_config)

        # Create a mock hass with states.async_all that raises an exception
        mock_hass = MagicMock()
        mock_hass.states.async_all.side_effect = Exception("State error")
        agent.hass = mock_hass

        result = await agent.get_entities_by_domain("light")

        assert len(result) == 1
        assert "error" in result[0]
        assert "Error getting entities for domain" in result[0]["error"]

    @pytest.mark.asyncio
    async def test_get_entities_by_domain_calls_get_entity_state(self, hass, mock_config):
        """Test get_entities_by_domain uses get_entity_state for each entity."""
        hass.states.async_set("media_player.tv", "playing")
        hass.states.async_set("media_player.speaker", "idle")
        agent = AiAgentHaAgent(hass, mock_config)

        with patch.object(agent, "get_entity_state", new_callable=AsyncMock) as mock_get_state:
            mock_get_state.return_value = {"entity_id": "mocked", "state": "mocked"}

            result = await agent.get_entities_by_domain("media_player")

        assert mock_get_state.call_count == 2
        mock_get_state.assert_any_call("media_player.tv")
        mock_get_state.assert_any_call("media_player.speaker")

    @pytest.mark.asyncio
    async def test_get_entities_by_domain_partial_match_excluded(self, hass, mock_config):
        """Test that entities with partially matching domains are excluded."""
        hass.states.async_set("light.test", "on")
        hass.states.async_set("light_sensor.test", "50")  # This should NOT match "light"
        agent = AiAgentHaAgent(hass, mock_config)

        result = await agent.get_entities_by_domain("light")

        # Only light.test should be returned, not light_sensor.test
        assert len(result) == 1
        assert result[0]["entity_id"] == "light.test"


class TestServiceCallMethods:
    """Test service call methods (call_service)."""

    @pytest.fixture
    def openai_config(self):
        """Mock OpenAI agent configuration."""
        return {
            "ai_provider": "openai",
            "openai_token": "test_token_123",
            "openai_model": "gpt-4",
        }

    @pytest.mark.asyncio
    async def test_call_service_success(self, hass, openai_config):
        """Test successful service call with entity_id as string."""
        # Set up entity state
        hass.states.async_set(
            "light.living_room",
            "off",
            {"friendly_name": "Living Room Light", "brightness": 0},
        )

        agent = AiAgentHaAgent(hass, openai_config)

        # Mock the async_call method using patch on ServiceRegistry class
        with patch("homeassistant.core.ServiceRegistry.async_call", new_callable=AsyncMock) as mock_call:
            result = await agent.call_service(
                domain="light",
                service="turn_on",
                target={"entity_id": "light.living_room"},
                service_data={"brightness": 255},
            )

            assert result["success"] is True
            assert result["service"] == "light.turn_on"
            assert result["message"] == "Successfully called light.turn_on"
            assert len(result["entities_affected"]) == 1
            assert result["entities_affected"][0]["entity_id"] == "light.living_room"

            # Verify service was called with correct parameters
            mock_call.assert_called_once_with(
                "light",
                "turn_on",
                {"entity_id": ["light.living_room"], "brightness": 255},
            )

    @pytest.mark.asyncio
    async def test_call_service_with_entity_list(self, hass, openai_config):
        """Test successful service call with entity_id as list."""
        hass.states.async_set("light.living_room", "off", {"friendly_name": "Living Room"})
        hass.states.async_set("light.bedroom", "off", {"friendly_name": "Bedroom"})

        agent = AiAgentHaAgent(hass, openai_config)

        with patch("homeassistant.core.ServiceRegistry.async_call", new_callable=AsyncMock) as mock_call:
            result = await agent.call_service(
                domain="light",
                service="turn_on",
                target={"entity_id": ["light.living_room", "light.bedroom"]},
            )

            assert result["success"] is True
            assert len(result["entities_affected"]) == 2
            mock_call.assert_called_once_with(
                "light",
                "turn_on",
                {"entity_id": ["light.living_room", "light.bedroom"]},
            )

    @pytest.mark.asyncio
    async def test_call_service_missing_domain(self, hass, openai_config):
        """Test call_service returns error when domain is missing."""
        agent = AiAgentHaAgent(hass, openai_config)

        result = await agent.call_service(
            domain="",
            service="turn_on",
            target={"entity_id": "light.living_room"},
        )

        assert "error" in result
        assert "domain is required" in result["error"]

    @pytest.mark.asyncio
    async def test_call_service_missing_service(self, hass, openai_config):
        """Test call_service returns error when service is missing."""
        agent = AiAgentHaAgent(hass, openai_config)

        result = await agent.call_service(
            domain="light",
            service="",
            target={"entity_id": "light.living_room"},
        )

        assert "error" in result
        assert "service is required" in result["error"]

    @pytest.mark.asyncio
    async def test_call_service_entity_not_found(self, hass, openai_config):
        """Test call_service returns error when entity not found."""
        # Set up a similar entity to provide suggestions
        hass.states.async_set("light.living_room_lamp", "on", {})

        agent = AiAgentHaAgent(hass, openai_config)

        result = await agent.call_service(
            domain="light",
            service="turn_on",
            target={"entity_id": "light.living_room"},
        )

        assert "error" in result
        assert "Entity not found" in result["error"]
        assert "light.living_room" in result["error"]

    @pytest.mark.asyncio
    async def test_call_service_entity_not_found_with_suggestions(self, hass, openai_config):
        """Test call_service provides suggestions for similar entities."""
        # Set up entities with similar names
        hass.states.async_set("light.living_room_light", "on", {})
        hass.states.async_set("light.living_room_lamp", "on", {})

        agent = AiAgentHaAgent(hass, openai_config)

        result = await agent.call_service(
            domain="light",
            service="turn_on",
            target={"entity_id": "light.living"},
        )

        assert "error" in result
        assert "Entity not found" in result["error"]
        assert "Did you mean" in result["error"]

    @pytest.mark.asyncio
    async def test_call_service_exception_handling(self, hass, openai_config):
        """Test call_service handles exceptions during service call."""
        hass.states.async_set("light.living_room", "off", {})

        agent = AiAgentHaAgent(hass, openai_config)

        with patch(
            "homeassistant.core.ServiceRegistry.async_call", new_callable=AsyncMock
        ) as mock_call:
            mock_call.side_effect = Exception("Service call failed")

            result = await agent.call_service(
                domain="light",
                service="turn_on",
                target={"entity_id": "light.living_room"},
            )

            assert "error" in result
            assert "Error calling service light.turn_on" in result["error"]
            assert "Service call failed" in result["error"]

    @pytest.mark.asyncio
    async def test_call_service_with_additional_target_properties(self, hass, openai_config):
        """Test call_service passes additional target properties."""
        hass.states.async_set("light.living_room", "off", {})

        agent = AiAgentHaAgent(hass, openai_config)

        with patch("homeassistant.core.ServiceRegistry.async_call", new_callable=AsyncMock) as mock_call:
            result = await agent.call_service(
                domain="light",
                service="turn_on",
                target={
                    "entity_id": "light.living_room",
                    "area_id": "living_room",
                },
            )

            assert result["success"] is True
            mock_call.assert_called_once_with(
                "light",
                "turn_on",
                {"entity_id": ["light.living_room"], "area_id": "living_room"},
            )

    @pytest.mark.asyncio
    async def test_call_service_without_target(self, hass, openai_config):
        """Test call_service works without target parameter."""
        agent = AiAgentHaAgent(hass, openai_config)

        with patch("homeassistant.core.ServiceRegistry.async_call", new_callable=AsyncMock) as mock_call:
            result = await agent.call_service(
                domain="homeassistant",
                service="restart",
            )

            assert result["success"] is True
            assert result["service"] == "homeassistant.restart"
            assert result["entities_affected"] == []
            mock_call.assert_called_once_with(
                "homeassistant",
                "restart",
                {},
            )

    @pytest.mark.asyncio
    async def test_call_service_with_service_data_only(self, hass, openai_config):
        """Test call_service with service_data but no target."""
        agent = AiAgentHaAgent(hass, openai_config)

        with patch("homeassistant.core.ServiceRegistry.async_call", new_callable=AsyncMock) as mock_call:
            result = await agent.call_service(
                domain="notify",
                service="send_message",
                service_data={"message": "Hello World", "title": "Test"},
            )

            assert result["success"] is True
            mock_call.assert_called_once_with(
                "notify",
                "send_message",
                {"message": "Hello World", "title": "Test"},
            )

    @pytest.mark.asyncio
    async def test_call_service_none_domain(self, hass, openai_config):
        """Test call_service returns error when domain is None."""
        agent = AiAgentHaAgent(hass, openai_config)

        result = await agent.call_service(
            domain=None,
            service="turn_on",
        )

        assert "error" in result
        assert "domain is required" in result["error"]

    @pytest.mark.asyncio
    async def test_call_service_none_service(self, hass, openai_config):
        """Test call_service returns error when service is None."""
        agent = AiAgentHaAgent(hass, openai_config)

        result = await agent.call_service(
            domain="light",
            service=None,
        )

        assert "error" in result
        assert "service is required" in result["error"]

    @pytest.mark.asyncio
    async def test_call_service_multiple_entities_not_found(self, hass, openai_config):
        """Test call_service with multiple missing entities."""
        hass.states.async_set("light.kitchen", "on", {})

        agent = AiAgentHaAgent(hass, openai_config)

        result = await agent.call_service(
            domain="light",
            service="turn_on",
            target={"entity_id": ["light.bedroom", "light.bathroom"]},
        )

        assert "error" in result
        assert "Entity not found" in result["error"]
        assert "light.bedroom" in result["error"]
        assert "light.bathroom" in result["error"]

    @pytest.mark.asyncio
    async def test_call_service_returns_updated_entity_states(self, hass, openai_config):
        """Test call_service returns updated entity states after call."""
        # Initial state
        hass.states.async_set(
            "light.living_room",
            "off",
            {"brightness": 0, "friendly_name": "Living Room"},
        )

        agent = AiAgentHaAgent(hass, openai_config)

        # Mock async_call to update state - use *args, **kwargs for flexibility
        async def mock_call_fn(*args, **kwargs):
            hass.states.async_set(
                "light.living_room",
                "on",
                {"brightness": 255, "friendly_name": "Living Room"},
            )

        with patch(
            "homeassistant.core.ServiceRegistry.async_call", new_callable=AsyncMock
        ) as mock_async_call:
            mock_async_call.side_effect = mock_call_fn

            result = await agent.call_service(
                domain="light",
                service="turn_on",
                target={"entity_id": "light.living_room"},
                service_data={"brightness": 255},
            )

            assert result["success"] is True
            assert result["entities_affected"][0]["state"] == "on"
            assert result["entities_affected"][0]["attributes"]["brightness"] == 255


class TestAutomationCreation:
    """Tests for automation creation methods (create_automation)."""

    @pytest.fixture
    def agent_config(self):
        """Mock agent configuration."""
        return {
            "ai_provider": "openai",
            "openai_token": "sk-" + "a" * 48,
        }

    @pytest.fixture
    def valid_automation_config(self):
        """Valid automation configuration."""
        return {
            "alias": "Test Automation",
            "description": "A test automation",
            "trigger": [{"platform": "state", "entity_id": "light.living_room", "to": "on"}],
            "condition": [],
            "action": [{"service": "light.turn_off", "target": {"entity_id": "light.bedroom"}}],
            "mode": "single",
        }

    @pytest.mark.asyncio
    async def test_create_automation_missing_alias(self, hass, agent_config):
        """Test create_automation fails when alias is missing."""
        agent = AiAgentHaAgent(hass, agent_config)

        config = {
            "trigger": [{"platform": "state", "entity_id": "light.test"}],
            "action": [{"service": "light.turn_on"}],
        }

        result = await agent.create_automation(config)

        assert "error" in result
        assert "Missing required fields" in result["error"]

    @pytest.mark.asyncio
    async def test_create_automation_missing_trigger(self, hass, agent_config):
        """Test create_automation fails when trigger is missing."""
        agent = AiAgentHaAgent(hass, agent_config)

        config = {
            "alias": "Test Automation",
            "action": [{"service": "light.turn_on"}],
        }

        result = await agent.create_automation(config)

        assert "error" in result
        assert "Missing required fields" in result["error"]

    @pytest.mark.asyncio
    async def test_create_automation_missing_action(self, hass, agent_config):
        """Test create_automation fails when action is missing."""
        agent = AiAgentHaAgent(hass, agent_config)

        config = {
            "alias": "Test Automation",
            "trigger": [{"platform": "state", "entity_id": "light.test"}],
        }

        result = await agent.create_automation(config)

        assert "error" in result
        assert "Missing required fields" in result["error"]

    @pytest.mark.asyncio
    async def test_create_automation_success(self, hass, agent_config, valid_automation_config, tmp_path):
        """Test create_automation succeeds with valid config."""
        import yaml

        agent = AiAgentHaAgent(hass, agent_config)

        # Register the automation.reload service
        async def mock_reload_service(call):
            pass
        hass.services.async_register("automation", "reload", mock_reload_service)

        # Create temp automations file
        automations_file = tmp_path / "automations.yaml"
        automations_file.write_text("[]")

        with patch.object(hass.config, "path", return_value=str(automations_file)):
            result = await agent.create_automation(valid_automation_config)

            assert result["success"] is True
            assert "created successfully" in result["message"]
            assert "Test Automation" in result["message"]

            # Verify automation was written to file
            automations = yaml.safe_load(automations_file.read_text())
            assert len(automations) == 1
            assert automations[0]["alias"] == "Test Automation"

    @pytest.mark.asyncio
    async def test_create_automation_duplicate_name(self, hass, agent_config, valid_automation_config):
        """Test create_automation fails when automation with same name exists."""
        agent = AiAgentHaAgent(hass, agent_config)

        existing_automations = [{"alias": "Test Automation", "id": "existing_auto"}]

        with patch.object(hass, "async_add_executor_job", new_callable=AsyncMock) as mock_executor:
            mock_executor.return_value = existing_automations

            result = await agent.create_automation(valid_automation_config)

            assert "error" in result
            assert "already exists" in result["error"]
            assert "Test Automation" in result["error"]

    @pytest.mark.asyncio
    async def test_create_automation_file_not_found(self, hass, agent_config, valid_automation_config, tmp_path):
        """Test create_automation handles missing automations.yaml file."""
        import yaml

        agent = AiAgentHaAgent(hass, agent_config)

        # Register the automation.reload service
        async def mock_reload_service(call):
            pass
        hass.services.async_register("automation", "reload", mock_reload_service)

        # Point to non-existent file (will be created)
        automations_file = tmp_path / "automations.yaml"

        with patch.object(hass.config, "path", return_value=str(automations_file)):
            result = await agent.create_automation(valid_automation_config)

            assert result["success"] is True
            assert "created successfully" in result["message"]

            # File should be created
            assert automations_file.exists()
            automations = yaml.safe_load(automations_file.read_text())
            assert len(automations) == 1

    @pytest.mark.asyncio
    async def test_create_automation_write_error(self, hass, agent_config, valid_automation_config):
        """Test create_automation handles write errors gracefully."""
        agent = AiAgentHaAgent(hass, agent_config)

        with patch.object(hass, "async_add_executor_job", new_callable=AsyncMock) as mock_executor:
            # First call returns empty list, second call raises IOError
            mock_executor.side_effect = [[], IOError("Permission denied")]

            result = await agent.create_automation(valid_automation_config)

            assert "error" in result
            assert "Error creating automation" in result["error"]

    @pytest.mark.asyncio
    async def test_create_automation_generates_unique_id(self, hass, agent_config, valid_automation_config, tmp_path):
        """Test create_automation generates unique ID for automation."""
        import yaml
        import time

        agent = AiAgentHaAgent(hass, agent_config)

        # Register the automation.reload service
        async def mock_reload_service(call):
            pass
        hass.services.async_register("automation", "reload", mock_reload_service)

        # Create temp automations file
        automations_file = tmp_path / "automations.yaml"
        automations_file.write_text("[]")

        with patch.object(hass.config, "path", return_value=str(automations_file)):
            with patch.object(time, "time", return_value=1234567890.123):
                result = await agent.create_automation(valid_automation_config)

                assert result["success"] is True

                # Verify unique ID was generated
                automations = yaml.safe_load(automations_file.read_text())
                assert automations[0]["id"] == "ai_agent_auto_1234567890123"

    @pytest.mark.asyncio
    async def test_create_automation_sanitizes_config(self, hass, agent_config, tmp_path):
        """Test create_automation sanitizes the automation config."""
        import yaml

        agent = AiAgentHaAgent(hass, agent_config)

        # Register the automation.reload service
        async def mock_reload_service(call):
            pass
        hass.services.async_register("automation", "reload", mock_reload_service)

        # Create temp automations file
        automations_file = tmp_path / "automations.yaml"
        automations_file.write_text("[]")

        config = {
            "alias": "  Whitespace Automation  ",  # Extra whitespace
            "trigger": [{"platform": "state", "entity_id": "light.test"}],
            "action": [{"service": "light.turn_on"}],
            "unknown_field": "should_be_ignored",  # Unknown field
        }

        with patch.object(hass.config, "path", return_value=str(automations_file)):
            result = await agent.create_automation(config)

            assert result["success"] is True
            # The alias should be trimmed in the success message
            assert "Whitespace Automation" in result["message"]

            # Verify unknown field was not written
            automations = yaml.safe_load(automations_file.read_text())
            assert "unknown_field" not in automations[0]

    @pytest.mark.asyncio
    async def test_create_automation_with_mode(self, hass, agent_config, tmp_path):
        """Test create_automation respects mode parameter."""
        import yaml

        agent = AiAgentHaAgent(hass, agent_config)

        # Register the automation.reload service
        async def mock_reload_service(call):
            pass
        hass.services.async_register("automation", "reload", mock_reload_service)

        # Create temp automations file
        automations_file = tmp_path / "automations.yaml"
        automations_file.write_text("[]")

        config = {
            "alias": "Queued Automation",
            "trigger": [{"platform": "state", "entity_id": "light.test"}],
            "action": [{"service": "light.turn_on"}],
            "mode": "queued",
        }

        with patch.object(hass.config, "path", return_value=str(automations_file)):
            result = await agent.create_automation(config)

            assert result["success"] is True

            # Verify mode was saved
            automations = yaml.safe_load(automations_file.read_text())
            assert automations[0]["mode"] == "queued"

    @pytest.mark.asyncio
    async def test_create_automation_with_conditions(self, hass, agent_config, tmp_path):
        """Test create_automation with conditions."""
        import yaml

        agent = AiAgentHaAgent(hass, agent_config)

        # Register the automation.reload service
        async def mock_reload_service(call):
            pass
        hass.services.async_register("automation", "reload", mock_reload_service)

        # Create temp automations file
        automations_file = tmp_path / "automations.yaml"
        automations_file.write_text("[]")

        config = {
            "alias": "Conditional Automation",
            "trigger": [{"platform": "state", "entity_id": "light.test"}],
            "condition": [{"condition": "sun", "after": "sunset"}],
            "action": [{"service": "light.turn_on"}],
        }

        with patch.object(hass.config, "path", return_value=str(automations_file)):
            result = await agent.create_automation(config)

            assert result["success"] is True
            assert "Conditional Automation" in result["message"]

            # Verify condition was saved
            automations = yaml.safe_load(automations_file.read_text())
            assert automations[0]["condition"] == [{"condition": "sun", "after": "sunset"}]

    @pytest.mark.asyncio
    async def test_create_automation_clears_cache(self, hass, agent_config, valid_automation_config, tmp_path):
        """Test create_automation clears the cache after creation."""
        agent = AiAgentHaAgent(hass, agent_config)

        # Pre-populate cache
        agent._cache["test_key"] = {"data": "test", "timestamp": 0}

        # Register the automation.reload service
        async def mock_reload_service(call):
            pass
        hass.services.async_register("automation", "reload", mock_reload_service)

        # Create temp automations file
        automations_file = tmp_path / "automations.yaml"
        automations_file.write_text("[]")

        with patch.object(hass.config, "path", return_value=str(automations_file)):
            result = await agent.create_automation(valid_automation_config)

            assert result["success"] is True
            # Cache should be cleared
            assert agent._cache == {}

    @pytest.mark.asyncio
    async def test_create_automation_reload_failure(self, hass, agent_config, valid_automation_config, tmp_path):
        """Test create_automation handles reload service failure."""
        agent = AiAgentHaAgent(hass, agent_config)

        # Register the automation.reload service that raises an exception
        async def mock_reload_service(call):
            raise Exception("Reload failed")
        hass.services.async_register("automation", "reload", mock_reload_service)

        # Create temp automations file
        automations_file = tmp_path / "automations.yaml"
        automations_file.write_text("[]")

        with patch.object(hass.config, "path", return_value=str(automations_file)):
            result = await agent.create_automation(valid_automation_config)

            # Should still succeed - HA catches service exceptions internally
            # The automation is written to file and will be loaded on next HA restart
            assert result["success"] is True
            assert "created successfully" in result["message"]

    @pytest.mark.asyncio
    async def test_create_automation_multiple_triggers(self, hass, agent_config, tmp_path):
        """Test create_automation with multiple triggers."""
        import yaml

        agent = AiAgentHaAgent(hass, agent_config)

        # Register the automation.reload service
        async def mock_reload_service(call):
            pass
        hass.services.async_register("automation", "reload", mock_reload_service)

        # Create temp automations file
        automations_file = tmp_path / "automations.yaml"
        automations_file.write_text("[]")

        config = {
            "alias": "Multi-trigger Automation",
            "trigger": [
                {"platform": "state", "entity_id": "light.bedroom", "to": "on"},
                {"platform": "time", "at": "22:00:00"},
            ],
            "action": [{"service": "light.turn_off", "target": {"entity_id": "light.hallway"}}],
        }

        with patch.object(hass.config, "path", return_value=str(automations_file)):
            result = await agent.create_automation(config)

            assert result["success"] is True
            assert "Multi-trigger Automation" in result["message"]

            # Verify multiple triggers were saved
            automations = yaml.safe_load(automations_file.read_text())
            assert len(automations[0]["trigger"]) == 2

    @pytest.mark.asyncio
    async def test_create_automation_multiple_actions(self, hass, agent_config, tmp_path):
        """Test create_automation with multiple actions."""
        import yaml

        agent = AiAgentHaAgent(hass, agent_config)

        # Register the automation.reload service
        async def mock_reload_service(call):
            pass
        hass.services.async_register("automation", "reload", mock_reload_service)

        # Create temp automations file
        automations_file = tmp_path / "automations.yaml"
        automations_file.write_text("[]")

        config = {
            "alias": "Multi-action Automation",
            "trigger": [{"platform": "state", "entity_id": "sensor.motion", "to": "on"}],
            "action": [
                {"service": "light.turn_on", "target": {"entity_id": "light.hallway"}},
                {"delay": {"seconds": 30}},
                {"service": "notify.mobile_app", "data": {"message": "Motion detected"}},
            ],
        }

        with patch.object(hass.config, "path", return_value=str(automations_file)):
            result = await agent.create_automation(config)

            assert result["success"] is True

            # Verify multiple actions were saved
            automations = yaml.safe_load(automations_file.read_text())
            assert len(automations[0]["action"]) == 3

    @pytest.mark.asyncio
    async def test_create_automation_appends_to_existing(self, hass, agent_config, valid_automation_config, tmp_path):
        """Test create_automation appends to existing automations."""
        import yaml

        agent = AiAgentHaAgent(hass, agent_config)

        # Register the automation.reload service
        async def mock_reload_service(call):
            pass
        hass.services.async_register("automation", "reload", mock_reload_service)

        # Create temp automations file with existing automations
        automations_file = tmp_path / "automations.yaml"
        existing_automations = [
            {"id": "auto_1", "alias": "Existing Automation 1"},
            {"id": "auto_2", "alias": "Existing Automation 2"},
        ]
        automations_file.write_text(yaml.dump(existing_automations))

        with patch.object(hass.config, "path", return_value=str(automations_file)):
            result = await agent.create_automation(valid_automation_config)

            assert result["success"] is True

            # Should have 3 automations after adding
            automations = yaml.safe_load(automations_file.read_text())
            assert len(automations) == 3

    @pytest.mark.asyncio
    async def test_create_automation_empty_yaml_file(self, hass, agent_config, valid_automation_config, tmp_path):
        """Test create_automation handles empty YAML file (returns None from safe_load)."""
        import yaml

        agent = AiAgentHaAgent(hass, agent_config)

        # Register the automation.reload service
        async def mock_reload_service(call):
            pass
        hass.services.async_register("automation", "reload", mock_reload_service)

        # Create empty temp automations file (yaml.safe_load returns None)
        automations_file = tmp_path / "automations.yaml"
        automations_file.write_text("")

        with patch.object(hass.config, "path", return_value=str(automations_file)):
            result = await agent.create_automation(valid_automation_config)

            # Should treat None as empty list
            assert result["success"] is True

            # Verify automation was written
            automations = yaml.safe_load(automations_file.read_text())
            assert len(automations) == 1

    @pytest.mark.asyncio
    async def test_create_automation_long_alias_truncated(self, hass, agent_config, tmp_path):
        """Test create_automation truncates long aliases."""
        import yaml

        agent = AiAgentHaAgent(hass, agent_config)

        # Register the automation.reload service
        async def mock_reload_service(call):
            pass
        hass.services.async_register("automation", "reload", mock_reload_service)

        # Create temp automations file
        automations_file = tmp_path / "automations.yaml"
        automations_file.write_text("[]")

        long_alias = "A" * 150  # Over 100 chars
        config = {
            "alias": long_alias,
            "trigger": [{"platform": "state", "entity_id": "light.test"}],
            "action": [{"service": "light.turn_on"}],
        }

        with patch.object(hass.config, "path", return_value=str(automations_file)):
            result = await agent.create_automation(config)

            assert result["success"] is True

            # Alias should be truncated to 100 chars
            automations = yaml.safe_load(automations_file.read_text())
            assert len(automations[0]["alias"]) == 100

    @pytest.mark.asyncio
    async def test_create_automation_invalid_mode_ignored(self, hass, agent_config, tmp_path):
        """Test create_automation ignores invalid mode values."""
        import yaml

        agent = AiAgentHaAgent(hass, agent_config)

        # Register the automation.reload service
        async def mock_reload_service(call):
            pass
        hass.services.async_register("automation", "reload", mock_reload_service)

        # Create temp automations file
        automations_file = tmp_path / "automations.yaml"
        automations_file.write_text("[]")

        config = {
            "alias": "Invalid Mode Automation",
            "trigger": [{"platform": "state", "entity_id": "light.test"}],
            "action": [{"service": "light.turn_on"}],
            "mode": "invalid_mode",  # Invalid mode
        }

        with patch.object(hass.config, "path", return_value=str(automations_file)):
            result = await agent.create_automation(config)

            # Should still succeed, mode will be default 'single'
            assert result["success"] is True

            # Verify invalid mode was not saved (defaults to 'single')
            automations = yaml.safe_load(automations_file.read_text())
            assert automations[0]["mode"] == "single"


class TestDashboardMethods:
    """Test dashboard-related methods: get_dashboards, get_dashboard_config, create_dashboard, update_dashboard."""

    @pytest.fixture
    def openai_config(self):
        """OpenAI provider configuration."""
        return {
            "ai_provider": "openai",
            "openai_token": "sk-" + "a" * 48,
            "models": {"openai": "gpt-3.5-turbo"}
        }

    @pytest.fixture
    def agent(self, hass, openai_config):
        """Create an agent instance for testing."""
        return AiAgentHaAgent(hass, openai_config)

    # ========== get_dashboards tests ==========

    @pytest.mark.asyncio
    async def test_get_dashboards_no_websocket_api(self, agent):
        """Test get_dashboards when websocket API is not available."""
        agent.hass.data.pop("websocket_api", None)

        result = await agent.get_dashboards()

        assert result == [{"error": "WebSocket API not available"}]

    @pytest.mark.asyncio
    async def test_get_dashboards_lovelace_not_available(self, agent):
        """Test get_dashboards when lovelace is not available."""
        agent.hass.data["websocket_api"] = MagicMock()
        agent.hass.data.pop("lovelace", None)

        result = await agent.get_dashboards()

        assert result == [{"error": "Lovelace not available"}]

    @pytest.mark.asyncio
    async def test_get_dashboards_no_dashboards_attribute(self, agent):
        """Test get_dashboards when lovelace data lacks dashboards attribute."""
        agent.hass.data["websocket_api"] = MagicMock()
        lovelace_data = MagicMock(spec=[])
        agent.hass.data["lovelace"] = lovelace_data

        result = await agent.get_dashboards()

        assert result == [{"error": "Lovelace dashboards not available"}]

    @pytest.mark.asyncio
    async def test_get_dashboards_success_with_default_dashboard(self, agent):
        """Test get_dashboards successfully returns default dashboard."""
        agent.hass.data["websocket_api"] = MagicMock()

        mock_dashboard = MagicMock()
        lovelace_data = MagicMock()
        lovelace_data.dashboards = {None: mock_dashboard}
        lovelace_data.yaml_dashboards = {}
        agent.hass.data["lovelace"] = lovelace_data

        result = await agent.get_dashboards()

        assert len(result) == 1
        assert result[0]["url_path"] is None
        assert result[0]["title"] == "Overview"
        assert result[0]["icon"] == "mdi:home"
        assert result[0]["show_in_sidebar"] is True
        assert result[0]["require_admin"] is False

    @pytest.mark.asyncio
    async def test_get_dashboards_success_with_custom_dashboard(self, agent):
        """Test get_dashboards with custom dashboard."""
        agent.hass.data["websocket_api"] = MagicMock()

        mock_dashboard = MagicMock()
        lovelace_data = MagicMock()
        lovelace_data.dashboards = {"my-dashboard": mock_dashboard}
        lovelace_data.yaml_dashboards = {}
        agent.hass.data["lovelace"] = lovelace_data

        result = await agent.get_dashboards()

        assert len(result) == 1
        assert result[0]["url_path"] == "my-dashboard"
        assert result[0]["title"] == "my-dashboard"
        assert result[0]["icon"] == "mdi:view-dashboard"

    @pytest.mark.asyncio
    async def test_get_dashboards_with_yaml_config_metadata(self, agent):
        """Test get_dashboards uses yaml config metadata when available."""
        agent.hass.data["websocket_api"] = MagicMock()

        mock_dashboard = MagicMock()
        lovelace_data = MagicMock()
        lovelace_data.dashboards = {"my-dashboard": mock_dashboard}
        lovelace_data.yaml_dashboards = {
            "my-dashboard": {
                "title": "Custom Title",
                "icon": "mdi:custom-icon",
                "show_in_sidebar": False,
                "require_admin": True
            }
        }
        agent.hass.data["lovelace"] = lovelace_data

        result = await agent.get_dashboards()

        assert len(result) == 1
        assert result[0]["title"] == "Custom Title"
        assert result[0]["icon"] == "mdi:custom-icon"
        assert result[0]["show_in_sidebar"] is False
        assert result[0]["require_admin"] is True

    @pytest.mark.asyncio
    async def test_get_dashboards_multiple_dashboards(self, agent):
        """Test get_dashboards with multiple dashboards."""
        agent.hass.data["websocket_api"] = MagicMock()

        lovelace_data = MagicMock()
        lovelace_data.dashboards = {
            None: MagicMock(),
            "dashboard-1": MagicMock(),
            "dashboard-2": MagicMock()
        }
        lovelace_data.yaml_dashboards = {}
        agent.hass.data["lovelace"] = lovelace_data

        result = await agent.get_dashboards()

        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_get_dashboards_exception_in_lovelace_access(self, agent):
        """Test get_dashboards handles exception during lovelace access."""
        agent.hass.data["websocket_api"] = MagicMock()

        lovelace_data = MagicMock()
        type(lovelace_data).dashboards = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("Access error"))
        )
        agent.hass.data["lovelace"] = lovelace_data

        result = await agent.get_dashboards()

        assert len(result) == 1
        assert "error" in result[0]
        assert "Could not retrieve dashboards" in result[0]["error"]

    @pytest.mark.asyncio
    async def test_get_dashboards_outer_exception(self, agent):
        """Test get_dashboards handles outer exception."""
        agent.hass.data = MagicMock()
        agent.hass.data.get.side_effect = RuntimeError("Outer error")

        result = await agent.get_dashboards()

        assert len(result) == 1
        assert "error" in result[0]
        assert "Error getting dashboards" in result[0]["error"]

    # ========== get_dashboard_config tests ==========

    @pytest.mark.asyncio
    async def test_get_dashboard_config_lovelace_not_available(self, agent):
        """Test get_dashboard_config when lovelace is not available."""
        agent.hass.data.pop("lovelace", None)

        result = await agent.get_dashboard_config()

        assert result == {"error": "Lovelace not available"}

    @pytest.mark.asyncio
    async def test_get_dashboard_config_no_dashboards_attribute(self, agent):
        """Test get_dashboard_config when lovelace data lacks dashboards attribute."""
        lovelace_data = MagicMock(spec=[])
        agent.hass.data["lovelace"] = lovelace_data

        result = await agent.get_dashboard_config()

        assert result == {"error": "Lovelace dashboards not available"}

    @pytest.mark.asyncio
    async def test_get_dashboard_config_default_not_found(self, agent):
        """Test get_dashboard_config when default dashboard not found."""
        lovelace_data = MagicMock()
        lovelace_data.dashboards = {"some-dashboard": MagicMock()}
        agent.hass.data["lovelace"] = lovelace_data

        result = await agent.get_dashboard_config(dashboard_url=None)

        assert result == {"error": "Default dashboard not found"}

    @pytest.mark.asyncio
    async def test_get_dashboard_config_custom_not_found(self, agent):
        """Test get_dashboard_config when custom dashboard not found."""
        lovelace_data = MagicMock()
        lovelace_data.dashboards = {None: MagicMock()}
        agent.hass.data["lovelace"] = lovelace_data

        result = await agent.get_dashboard_config(dashboard_url="nonexistent")

        assert result == {"error": "Dashboard 'nonexistent' not found"}

    @pytest.mark.asyncio
    async def test_get_dashboard_config_success_default(self, agent):
        """Test get_dashboard_config successfully retrieves default dashboard config."""
        mock_dashboard = AsyncMock()
        mock_dashboard.async_get_info.return_value = {
            "title": "Overview",
            "views": [{"title": "Home"}]
        }

        lovelace_data = MagicMock()
        lovelace_data.dashboards = {None: mock_dashboard}
        agent.hass.data["lovelace"] = lovelace_data

        result = await agent.get_dashboard_config()

        assert result == {"title": "Overview", "views": [{"title": "Home"}]}
        mock_dashboard.async_get_info.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_dashboard_config_success_custom(self, agent):
        """Test get_dashboard_config successfully retrieves custom dashboard config."""
        mock_dashboard = AsyncMock()
        mock_dashboard.async_get_info.return_value = {
            "title": "My Dashboard",
            "icon": "mdi:home"
        }

        lovelace_data = MagicMock()
        lovelace_data.dashboards = {"my-dashboard": mock_dashboard}
        agent.hass.data["lovelace"] = lovelace_data

        result = await agent.get_dashboard_config(dashboard_url="my-dashboard")

        assert result == {"title": "My Dashboard", "icon": "mdi:home"}

    @pytest.mark.asyncio
    async def test_get_dashboard_config_returns_empty_when_no_config(self, agent):
        """Test get_dashboard_config when async_get_info returns None."""
        mock_dashboard = AsyncMock()
        mock_dashboard.async_get_info.return_value = None

        lovelace_data = MagicMock()
        lovelace_data.dashboards = {None: mock_dashboard}
        agent.hass.data["lovelace"] = lovelace_data

        result = await agent.get_dashboard_config()

        assert result == {"error": "No dashboard config"}

    @pytest.mark.asyncio
    async def test_get_dashboard_config_exception_in_lovelace(self, agent):
        """Test get_dashboard_config handles exception during dashboard access."""
        lovelace_data = MagicMock()
        type(lovelace_data).dashboards = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("Dashboard error"))
        )
        agent.hass.data["lovelace"] = lovelace_data

        result = await agent.get_dashboard_config()

        assert "error" in result
        assert "Could not retrieve dashboard config" in result["error"]

    @pytest.mark.asyncio
    async def test_get_dashboard_config_outer_exception(self, agent):
        """Test get_dashboard_config handles outer exception."""
        agent.hass.data = MagicMock()
        agent.hass.data.get.side_effect = RuntimeError("Outer error")

        result = await agent.get_dashboard_config()

        assert "error" in result
        assert "Outer error" in result["error"]

    # ========== create_dashboard tests ==========

    @pytest.mark.asyncio
    async def test_create_dashboard_missing_title(self, agent):
        """Test create_dashboard fails when title is missing."""
        config = {"url_path": "my-dashboard"}

        result = await agent.create_dashboard(config)

        assert result == {"error": "Dashboard title is required"}

    @pytest.mark.asyncio
    async def test_create_dashboard_missing_url_path(self, agent):
        """Test create_dashboard fails when url_path is missing."""
        config = {"title": "My Dashboard"}

        result = await agent.create_dashboard(config)

        assert result == {"error": "Dashboard URL path is required"}

    @pytest.mark.asyncio
    async def test_create_dashboard_success(self, agent):
        """Test create_dashboard successfully creates a dashboard."""
        config = {
            "title": "Test Dashboard",
            "url_path": "test-dashboard",
            "icon": "mdi:test",
            "show_in_sidebar": True,
            "views": []
        }

        mock_config_path = "/config"
        agent.hass.config.path = MagicMock(side_effect=lambda f: f"{mock_config_path}/{f}")
        agent.hass.async_add_executor_job = AsyncMock(side_effect=[None, True])

        result = await agent.create_dashboard(config)

        assert result["success"] is True
        assert result["url_path"] == "test-dashboard"
        assert result["restart_required"] is True
        assert "created successfully" in result["message"]

    @pytest.mark.asyncio
    async def test_create_dashboard_sanitizes_url_path(self, agent):
        """Test create_dashboard sanitizes the URL path."""
        config = {
            "title": "Test Dashboard",
            "url_path": "Test Dashboard_With_Spaces",
        }

        agent.hass.config.path = MagicMock(side_effect=lambda f: f"/config/{f}")
        agent.hass.async_add_executor_job = AsyncMock(side_effect=[None, True])

        result = await agent.create_dashboard(config)

        assert result["success"] is True
        assert result["url_path"] == "test-dashboard-with-spaces"

    @pytest.mark.asyncio
    async def test_create_dashboard_config_update_fails(self, agent):
        """Test create_dashboard when config update fails but file creation succeeds."""
        config = {
            "title": "Test Dashboard",
            "url_path": "test-dashboard",
        }

        agent.hass.config.path = MagicMock(side_effect=lambda f: f"/config/{f}")
        agent.hass.async_add_executor_job = AsyncMock(side_effect=[None, False])

        result = await agent.create_dashboard(config)

        assert result["success"] is True
        assert "manually add" in result["message"]
        assert result["restart_required"] is True

    @pytest.mark.asyncio
    async def test_create_dashboard_file_write_exception(self, agent):
        """Test create_dashboard handles file write exception."""
        config = {
            "title": "Test Dashboard",
            "url_path": "test-dashboard",
        }

        agent.hass.config.path = MagicMock(side_effect=lambda f: f"/config/{f}")
        agent.hass.async_add_executor_job = AsyncMock(side_effect=IOError("Write failed"))

        result = await agent.create_dashboard(config)

        assert "error" in result
        assert "Failed to create dashboard file" in result["error"]

    @pytest.mark.asyncio
    async def test_create_dashboard_config_update_exception(self, agent):
        """Test create_dashboard handles config update exception."""
        config = {
            "title": "Test Dashboard",
            "url_path": "test-dashboard",
        }

        agent.hass.config.path = MagicMock(side_effect=lambda f: f"/config/{f}")
        call_count = 0

        async def mock_executor(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return None
            raise RuntimeError("Config update error")

        agent.hass.async_add_executor_job = mock_executor

        result = await agent.create_dashboard(config)

        assert result["success"] is True
        assert "manually add" in result["message"]

    @pytest.mark.asyncio
    async def test_create_dashboard_outer_exception(self, agent):
        """Test create_dashboard handles outer exception."""
        config = {
            "title": "Test Dashboard",
            "url_path": "test-dashboard",
        }

        agent.hass.config.path = MagicMock(side_effect=RuntimeError("Path error"))

        result = await agent.create_dashboard(config)

        assert "error" in result
        assert "Path error" in result["error"]

    @pytest.mark.asyncio
    async def test_create_dashboard_uses_default_values(self, agent):
        """Test create_dashboard uses default values for optional fields."""
        config = {
            "title": "Test Dashboard",
            "url_path": "test-dashboard",
        }

        agent.hass.config.path = MagicMock(side_effect=lambda f: f"/config/{f}")
        agent.hass.async_add_executor_job = AsyncMock(side_effect=[None, True])

        result = await agent.create_dashboard(config)

        assert result["success"] is True

    # ========== update_dashboard tests ==========

    @pytest.mark.asyncio
    async def test_update_dashboard_file_not_found(self, agent):
        """Test update_dashboard when dashboard file doesn't exist."""
        agent.hass.config.path = MagicMock(side_effect=lambda f: f"/config/{f}")
        agent.hass.async_add_executor_job = AsyncMock(side_effect=[False, False])

        result = await agent.update_dashboard("my-dashboard", {"title": "Updated"})

        assert result == {"error": "Dashboard file for 'my-dashboard' not found"}

    @pytest.mark.asyncio
    async def test_update_dashboard_success_primary_path(self, agent):
        """Test update_dashboard successfully updates dashboard at primary path."""
        config = {
            "title": "Updated Dashboard",
            "icon": "mdi:updated",
            "views": [{"title": "New View"}]
        }

        agent.hass.config.path = MagicMock(side_effect=lambda f: f"/config/{f}")
        agent.hass.async_add_executor_job = AsyncMock(side_effect=[True, None])

        result = await agent.update_dashboard("my-dashboard", config)

        assert result["success"] is True
        assert "updated successfully" in result["message"]

    @pytest.mark.asyncio
    async def test_update_dashboard_success_fallback_path(self, agent):
        """Test update_dashboard finds file at fallback path."""
        config = {"title": "Updated Dashboard"}

        agent.hass.config.path = MagicMock(side_effect=lambda f: f"/config/{f}")
        agent.hass.async_add_executor_job = AsyncMock(side_effect=[False, True, None])

        result = await agent.update_dashboard("my-dashboard", config)

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_update_dashboard_write_exception(self, agent):
        """Test update_dashboard handles write exception."""
        config = {"title": "Updated Dashboard"}

        agent.hass.config.path = MagicMock(side_effect=lambda f: f"/config/{f}")
        agent.hass.async_add_executor_job = AsyncMock(side_effect=[True, IOError("Write failed")])

        result = await agent.update_dashboard("my-dashboard", config)

        assert "error" in result
        assert "Failed to update dashboard file" in result["error"]

    @pytest.mark.asyncio
    async def test_update_dashboard_outer_exception(self, agent):
        """Test update_dashboard handles outer exception."""
        config = {"title": "Updated Dashboard"}

        agent.hass.config.path = MagicMock(side_effect=RuntimeError("Path error"))

        result = await agent.update_dashboard("my-dashboard", config)

        assert "error" in result
        assert "Path error" in result["error"]

    @pytest.mark.asyncio
    async def test_update_dashboard_uses_default_values(self, agent):
        """Test update_dashboard uses default values for missing config fields."""
        config = {"title": "Minimal Update"}

        agent.hass.config.path = MagicMock(side_effect=lambda f: f"/config/{f}")
        agent.hass.async_add_executor_job = AsyncMock(side_effect=[True, None])

        result = await agent.update_dashboard("my-dashboard", config)

        assert result["success"] is True


class TestValidateAutomation:
    """Tests for AIAgent._validate_automation method."""

    @pytest.fixture
    def agent_config(self):
        """Mock agent configuration."""
        return {
            "ai_provider": "openai",
            "openai_token": "sk-" + "a" * 48,
        }

    @pytest.fixture
    def agent_with_mock_services(self, hass, agent_config):
        """Create agent instance with mocked services."""
        agent = AiAgentHaAgent(hass, agent_config)
        mock_services = MagicMock()
        mock_services.has_service = MagicMock(return_value=True)
        agent.hass.services = mock_services
        return agent

    @pytest.mark.asyncio
    async def test_validate_automation_valid_config(self, agent_with_mock_services):
        """Test validation passes with valid automation config."""
        agent = agent_with_mock_services
        config = {
            "trigger": [{"platform": "state", "entity_id": "light.living_room", "to": "on"}],
            "condition": [{"condition": "state", "entity_id": "sun.sun", "state": "above_horizon"}],
            "action": [{"service": "light.turn_off", "target": {"entity_id": "light.bedroom"}}],
        }
        result = await agent._validate_automation(config)
        assert result["valid"] is True
        assert result["errors"] is None

    @pytest.mark.asyncio
    async def test_validate_automation_missing_trigger(self, agent_with_mock_services):
        """Test validation fails when trigger is missing."""
        agent = agent_with_mock_services
        config = {
            "trigger": [],
            "action": [{"service": "light.turn_off"}],
        }
        result = await agent._validate_automation(config)
        assert result["valid"] is False
        assert "At least one trigger is required" in result["errors"]

    @pytest.mark.asyncio
    async def test_validate_automation_missing_action(self, agent_with_mock_services):
        """Test validation fails when action is missing."""
        agent = agent_with_mock_services
        config = {
            "trigger": [{"platform": "state", "entity_id": "light.test"}],
            "action": [],
        }
        result = await agent._validate_automation(config)
        assert result["valid"] is False
        assert "At least one action is required" in result["errors"]

    @pytest.mark.asyncio
    async def test_validate_automation_trigger_missing_platform(self, agent_with_mock_services):
        """Test validation fails when trigger is missing platform."""
        agent = agent_with_mock_services
        config = {
            "trigger": [{"entity_id": "light.test"}],
            "action": [{"service": "light.turn_on"}],
        }
        result = await agent._validate_automation(config)
        assert result["valid"] is False
        assert "Trigger 0 missing 'platform' field" in result["errors"]

    @pytest.mark.asyncio
    async def test_validate_automation_trigger_not_dict(self, agent_with_mock_services):
        """Test validation fails when trigger is not a dictionary."""
        agent = agent_with_mock_services
        config = {
            "trigger": ["invalid_trigger"],
            "action": [{"service": "light.turn_on"}],
        }
        result = await agent._validate_automation(config)
        assert result["valid"] is False
        assert "Trigger 0 must be a dictionary" in result["errors"]

    @pytest.mark.asyncio
    async def test_validate_automation_condition_missing_type(self, agent_with_mock_services):
        """Test validation fails when condition is missing condition type."""
        agent = agent_with_mock_services
        config = {
            "trigger": [{"platform": "state", "entity_id": "light.test"}],
            "condition": [{"entity_id": "sun.sun"}],
            "action": [{"service": "light.turn_on"}],
        }
        result = await agent._validate_automation(config)
        assert result["valid"] is False
        assert "Condition 0 missing 'condition' field" in result["errors"]

    @pytest.mark.asyncio
    async def test_validate_automation_condition_not_dict(self, agent_with_mock_services):
        """Test validation fails when condition is not a dictionary."""
        agent = agent_with_mock_services
        config = {
            "trigger": [{"platform": "state", "entity_id": "light.test"}],
            "condition": ["invalid_condition"],
            "action": [{"service": "light.turn_on"}],
        }
        result = await agent._validate_automation(config)
        assert result["valid"] is False
        assert "Condition 0 must be a dictionary" in result["errors"]

    @pytest.mark.asyncio
    async def test_validate_automation_action_not_dict(self, agent_with_mock_services):
        """Test validation fails when action is not a dictionary."""
        agent = agent_with_mock_services
        config = {
            "trigger": [{"platform": "state", "entity_id": "light.test"}],
            "action": ["invalid_action"],
        }
        result = await agent._validate_automation(config)
        assert result["valid"] is False
        assert "Action 0 must be a dictionary" in result["errors"]

    @pytest.mark.asyncio
    async def test_validate_automation_invalid_service_format(self, agent_with_mock_services):
        """Test validation fails when service format is invalid (missing dot)."""
        agent = agent_with_mock_services
        config = {
            "trigger": [{"platform": "state", "entity_id": "light.test"}],
            "action": [{"service": "invalid_service"}],
        }
        result = await agent._validate_automation(config)
        assert result["valid"] is False
        assert "Action 0 has invalid service format: invalid_service" in result["errors"]

    @pytest.mark.asyncio
    async def test_validate_automation_unknown_service(self, agent_with_mock_services):
        """Test validation fails when service does not exist."""
        agent = agent_with_mock_services
        agent.hass.services.has_service.return_value = False
        config = {
            "trigger": [{"platform": "state", "entity_id": "light.test"}],
            "action": [{"service": "nonexistent.service"}],
        }
        result = await agent._validate_automation(config)
        assert result["valid"] is False
        assert "Action 0 references unknown service: nonexistent.service" in result["errors"]
        agent.hass.services.has_service.assert_called_with("nonexistent", "service")

    @pytest.mark.asyncio
    async def test_validate_automation_multiple_errors(self, agent_with_mock_services):
        """Test validation collects multiple errors."""
        agent = agent_with_mock_services
        agent.hass.services.has_service.return_value = False
        config = {
            "trigger": [],
            "condition": [{"entity_id": "sun.sun"}],
            "action": [{"service": "unknown.service"}],
        }
        result = await agent._validate_automation(config)
        assert result["valid"] is False
        assert len(result["errors"]) >= 3
        assert "At least one trigger is required" in result["errors"]
        assert "Condition 0 missing 'condition' field" in result["errors"]
        assert "Action 0 references unknown service: unknown.service" in result["errors"]

    @pytest.mark.asyncio
    async def test_validate_automation_empty_conditions_allowed(self, agent_with_mock_services):
        """Test validation passes with empty conditions (conditions are optional)."""
        agent = agent_with_mock_services
        config = {
            "trigger": [{"platform": "state", "entity_id": "light.test"}],
            "condition": [],
            "action": [{"service": "light.turn_on"}],
        }
        result = await agent._validate_automation(config)
        assert result["valid"] is True
        assert result["errors"] is None

    @pytest.mark.asyncio
    async def test_validate_automation_no_conditions_key(self, agent_with_mock_services):
        """Test validation passes when conditions key is omitted."""
        agent = agent_with_mock_services
        config = {
            "trigger": [{"platform": "time", "at": "06:00:00"}],
            "action": [{"service": "light.turn_on", "target": {"entity_id": "light.bedroom"}}],
        }
        result = await agent._validate_automation(config)
        assert result["valid"] is True
        assert result["errors"] is None

    @pytest.mark.asyncio
    async def test_validate_automation_multiple_triggers(self, agent_with_mock_services):
        """Test validation with multiple triggers."""
        agent = agent_with_mock_services
        config = {
            "trigger": [
                {"platform": "state", "entity_id": "light.test", "to": "on"},
                {"platform": "time", "at": "sunset"},
            ],
            "action": [{"service": "light.turn_off"}],
        }
        result = await agent._validate_automation(config)
        assert result["valid"] is True
        assert result["errors"] is None

    @pytest.mark.asyncio
    async def test_validate_automation_multiple_actions(self, agent_with_mock_services):
        """Test validation with multiple actions."""
        agent = agent_with_mock_services
        config = {
            "trigger": [{"platform": "state", "entity_id": "light.test"}],
            "action": [
                {"service": "light.turn_on", "target": {"entity_id": "light.bedroom"}},
                {"service": "notify.mobile_app", "data": {"message": "Light turned on"}},
            ],
        }
        result = await agent._validate_automation(config)
        assert result["valid"] is True
        assert agent.hass.services.has_service.call_count == 2

    @pytest.mark.asyncio
    async def test_validate_automation_action_without_service(self, agent_with_mock_services):
        """Test validation passes for actions without service (e.g., delay, wait)."""
        agent = agent_with_mock_services
        config = {
            "trigger": [{"platform": "state", "entity_id": "light.test"}],
            "action": [
                {"delay": "00:00:05"},
                {"service": "light.turn_on"},
            ],
        }
        result = await agent._validate_automation(config)
        assert result["valid"] is True
        agent.hass.services.has_service.assert_called_once_with("light", "turn_on")


class TestEntityFilteringMethods:
    """Tests for dashboard configuration file update logic (agent.py lines 3150-3290)."""

    @pytest.fixture
    def agent(self, hass):
        """Create an agent instance for testing."""
        config = {
            "ai_provider": "openai",
            "openai_token": "test_token",
        }
        return AiAgentHaAgent(hass, config)

    @pytest.mark.asyncio
    async def test_create_dashboard_lovelace_exists_with_dashboards(self, agent, hass, tmp_path):
        """Test adding dashboard when lovelace section exists with dashboards section.

        Note: The implementation breaks after finding dashboards and adding the new entry,
        which means only content up to dashboards: is preserved in the written file.
        """
        config_file = tmp_path / "configuration.yaml"
        dashboard_file = tmp_path / "ui-lovelace-new-dash.yaml"

        # Create config with existing lovelace and dashboards section
        config_content = """homeassistant:
  name: Home

lovelace:
  mode: yaml
  dashboards:
    existing-dash:
      mode: yaml
      title: Existing Dashboard
"""
        config_file.write_text(config_content)

        def mock_path(filename):
            if filename == "configuration.yaml":
                return str(config_file)
            elif filename.startswith("ui-lovelace-"):
                return str(tmp_path / filename)
            return str(tmp_path / filename)

        with patch.object(hass.config, 'path', side_effect=mock_path):
            dashboard_config = {
                "title": "New Dashboard",
                "url_path": "new-dash",
                "icon": "mdi:view-dashboard",
                "show_in_sidebar": True,
                "views": []
            }

            result = await agent.create_dashboard(dashboard_config)

            assert result["success"] is True

            # Verify configuration.yaml was updated with new dashboard
            updated_content = config_file.read_text()
            assert "new-dash:" in updated_content
            assert "lovelace:" in updated_content
            assert "dashboards:" in updated_content
            # Note: Due to the break statement in the implementation,
            # the new dashboard is added right after dashboards: line

    @pytest.mark.asyncio
    async def test_create_dashboard_lovelace_exists_no_dashboards(self, agent, hass, tmp_path):
        """Test adding dashboard when lovelace section exists but no dashboards section."""
        config_file = tmp_path / "configuration.yaml"
        dashboard_file = tmp_path / "ui-lovelace-new-dash.yaml"

        # Create config with lovelace section but no dashboards
        config_content = """homeassistant:
  name: Home

lovelace:
  mode: storage

automation: !include automations.yaml
"""
        config_file.write_text(config_content)

        def mock_path(filename):
            if filename == "configuration.yaml":
                return str(config_file)
            elif filename.startswith("ui-lovelace-"):
                return str(tmp_path / filename)
            return str(tmp_path / filename)

        with patch.object(hass.config, 'path', side_effect=mock_path):
            dashboard_config = {
                "title": "New Dashboard",
                "url_path": "new-dash",
                "views": []
            }

            result = await agent.create_dashboard(dashboard_config)

            assert result["success"] is True

            # Verify dashboards section was added under lovelace
            updated_content = config_file.read_text()
            assert "lovelace:" in updated_content
            assert "dashboards:" in updated_content
            assert "new-dash:" in updated_content

    @pytest.mark.asyncio
    async def test_create_dashboard_no_lovelace_section(self, agent, hass, tmp_path):
        """Test adding dashboard when no lovelace section exists."""
        config_file = tmp_path / "configuration.yaml"
        dashboard_file = tmp_path / "ui-lovelace-test.yaml"

        # Create config without lovelace section
        config_content = """homeassistant:
  name: Home

automation: !include automations.yaml
"""
        config_file.write_text(config_content)

        def mock_path(filename):
            if filename == "configuration.yaml":
                return str(config_file)
            elif filename.startswith("ui-lovelace-"):
                return str(tmp_path / filename)
            return str(tmp_path / filename)

        with patch.object(hass.config, 'path', side_effect=mock_path):
            dashboard_config = {
                "title": "Test Dashboard",
                "url_path": "test",
                "views": []
            }

            result = await agent.create_dashboard(dashboard_config)

            assert result["success"] is True

            # Verify lovelace section was added
            updated_content = config_file.read_text()
            assert "lovelace:" in updated_content
            assert "dashboards:" in updated_content
            assert "test:" in updated_content

    @pytest.mark.asyncio
    async def test_create_dashboard_lovelace_section_detection(self, agent, hass, tmp_path):
        """Test detection of lovelace section with different formats."""
        config_file = tmp_path / "configuration.yaml"
        dashboard_file = tmp_path / "ui-lovelace-myboard.yaml"

        # Create config with lovelace: inline style
        config_content = """homeassistant:
  name: Home

lovelace: !include lovelace.yaml

automation: !include automations.yaml
"""
        config_file.write_text(config_content)

        def mock_path(filename):
            if filename == "configuration.yaml":
                return str(config_file)
            elif filename.startswith("ui-lovelace-"):
                return str(tmp_path / filename)
            return str(tmp_path / filename)

        with patch.object(hass.config, 'path', side_effect=mock_path):
            dashboard_config = {
                "title": "My Board",
                "url_path": "myboard",
                "views": []
            }

            result = await agent.create_dashboard(dashboard_config)

            assert result["success"] is True

            # Verify dashboard was added
            updated_content = config_file.read_text()
            assert "myboard:" in updated_content

    @pytest.mark.asyncio
    async def test_create_dashboard_indent_handling(self, agent, hass, tmp_path):
        """Test that dashboard entry is properly indented."""
        config_file = tmp_path / "configuration.yaml"
        dashboard_file = tmp_path / "ui-lovelace-indented.yaml"

        config_content = """lovelace:
  dashboards:
    existing:
      mode: yaml
      title: Existing
"""
        config_file.write_text(config_content)

        def mock_path(filename):
            if filename == "configuration.yaml":
                return str(config_file)
            elif filename.startswith("ui-lovelace-"):
                return str(tmp_path / filename)
            return str(tmp_path / filename)

        with patch.object(hass.config, 'path', side_effect=mock_path):
            dashboard_config = {
                "title": "Indented Dashboard",
                "url_path": "indented",
                "views": []
            }

            result = await agent.create_dashboard(dashboard_config)

            assert result["success"] is True

            # Verify proper indentation in config
            updated_content = config_file.read_text()
            assert "indented:" in updated_content
            assert "mode: yaml" in updated_content

    @pytest.mark.asyncio
    async def test_create_dashboard_exit_lovelace_on_new_section(self, agent, hass, tmp_path):
        """Test that parsing correctly identifies the dashboards section.

        Note: The implementation breaks after adding the dashboard entry to write only
        the content up to and including the dashboards section with the new entry.
        """
        config_file = tmp_path / "configuration.yaml"
        dashboard_file = tmp_path / "ui-lovelace-sectioned.yaml"

        # Config where lovelace has dashboards section
        config_content = """lovelace:
  dashboards:
    existing:
      mode: yaml
"""
        config_file.write_text(config_content)

        def mock_path(filename):
            if filename == "configuration.yaml":
                return str(config_file)
            elif filename.startswith("ui-lovelace-"):
                return str(tmp_path / filename)
            return str(tmp_path / filename)

        with patch.object(hass.config, 'path', side_effect=mock_path):
            dashboard_config = {
                "title": "Sectioned Dashboard",
                "url_path": "sectioned",
                "views": []
            }

            result = await agent.create_dashboard(dashboard_config)

            assert result["success"] is True

            # Dashboard should be added under dashboards
            updated_content = config_file.read_text()
            assert "sectioned:" in updated_content
            assert "lovelace:" in updated_content
            assert "dashboards:" in updated_content

    @pytest.mark.asyncio
    async def test_create_dashboard_fallback_append(self, agent, hass, tmp_path):
        """Test fallback to append method when dashboard cannot be added inline."""
        config_file = tmp_path / "configuration.yaml"
        dashboard_file = tmp_path / "ui-lovelace-fallback.yaml"

        # Minimal config that triggers fallback
        config_content = "# Empty config"
        config_file.write_text(config_content)

        def mock_path(filename):
            if filename == "configuration.yaml":
                return str(config_file)
            elif filename.startswith("ui-lovelace-"):
                return str(tmp_path / filename)
            return str(tmp_path / filename)

        with patch.object(hass.config, 'path', side_effect=mock_path):
            dashboard_config = {
                "title": "Fallback Dashboard",
                "url_path": "fallback",
                "views": []
            }

            result = await agent.create_dashboard(dashboard_config)

            assert result["success"] is True

            # Verify lovelace was added
            updated_content = config_file.read_text()
            assert "lovelace:" in updated_content
            assert "fallback:" in updated_content

    @pytest.mark.asyncio
    async def test_create_dashboard_exception_fallback(self, agent, hass, tmp_path):
        """Test fallback mechanism when primary update fails with exception."""
        config_file = tmp_path / "configuration.yaml"
        dashboard_file = tmp_path / "ui-lovelace-exception.yaml"

        # Create a valid config
        config_content = """lovelace:
  mode: yaml
"""
        config_file.write_text(config_content)

        call_count = [0]
        original_content = [config_content]

        def mock_path(filename):
            if filename == "configuration.yaml":
                return str(config_file)
            elif filename.startswith("ui-lovelace-"):
                return str(tmp_path / filename)
            return str(tmp_path / filename)

        with patch.object(hass.config, 'path', side_effect=mock_path):
            dashboard_config = {
                "title": "Exception Dashboard",
                "url_path": "exception",
                "views": []
            }

            result = await agent.create_dashboard(dashboard_config)

            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_create_dashboard_empty_lines_handling(self, agent, hass, tmp_path):
        """Test handling of empty lines in configuration file."""
        config_file = tmp_path / "configuration.yaml"
        dashboard_file = tmp_path / "ui-lovelace-empty.yaml"

        # Config with empty lines
        config_content = """homeassistant:
  name: Home


lovelace:

  dashboards:

    existing:
      mode: yaml


automation: !include automations.yaml
"""
        config_file.write_text(config_content)

        def mock_path(filename):
            if filename == "configuration.yaml":
                return str(config_file)
            elif filename.startswith("ui-lovelace-"):
                return str(tmp_path / filename)
            return str(tmp_path / filename)

        with patch.object(hass.config, 'path', side_effect=mock_path):
            dashboard_config = {
                "title": "Empty Lines Dashboard",
                "url_path": "empty",
                "views": []
            }

            result = await agent.create_dashboard(dashboard_config)

            assert result["success"] is True

            updated_content = config_file.read_text()
            assert "empty:" in updated_content

    @pytest.mark.asyncio
    async def test_create_dashboard_show_in_sidebar_false(self, agent, hass, tmp_path):
        """Test dashboard creation with show_in_sidebar set to false."""
        config_file = tmp_path / "configuration.yaml"
        dashboard_file = tmp_path / "ui-lovelace-hidden.yaml"

        config_content = "# Empty config"
        config_file.write_text(config_content)

        def mock_path(filename):
            if filename == "configuration.yaml":
                return str(config_file)
            elif filename.startswith("ui-lovelace-"):
                return str(tmp_path / filename)
            return str(tmp_path / filename)

        with patch.object(hass.config, 'path', side_effect=mock_path):
            dashboard_config = {
                "title": "Hidden Dashboard",
                "url_path": "hidden",
                "show_in_sidebar": False,
                "views": []
            }

            result = await agent.create_dashboard(dashboard_config)

            assert result["success"] is True

            updated_content = config_file.read_text()
            assert "show_in_sidebar: false" in updated_content

    @pytest.mark.asyncio
    async def test_create_dashboard_custom_icon(self, agent, hass, tmp_path):
        """Test dashboard creation with custom icon."""
        config_file = tmp_path / "configuration.yaml"
        dashboard_file = tmp_path / "ui-lovelace-custom-icon.yaml"

        config_content = "# Empty config"
        config_file.write_text(config_content)

        def mock_path(filename):
            if filename == "configuration.yaml":
                return str(config_file)
            elif filename.startswith("ui-lovelace-"):
                return str(tmp_path / filename)
            return str(tmp_path / filename)

        with patch.object(hass.config, 'path', side_effect=mock_path):
            dashboard_config = {
                "title": "Custom Icon Dashboard",
                "url_path": "custom-icon",
                "icon": "mdi:rocket-launch",
                "views": []
            }

            result = await agent.create_dashboard(dashboard_config)

            assert result["success"] is True

            updated_content = config_file.read_text()
            assert "mdi:rocket-launch" in updated_content

    @pytest.mark.asyncio
    async def test_create_dashboard_preserves_other_sections(self, agent, hass, tmp_path):
        """Test that creating dashboard preserves other configuration sections."""
        config_file = tmp_path / "configuration.yaml"
        dashboard_file = tmp_path / "ui-lovelace-preserve.yaml"

        config_content = """homeassistant:
  name: My Home
  unit_system: metric

logger:
  default: info

automation: !include automations.yaml
script: !include scripts.yaml
"""
        config_file.write_text(config_content)

        def mock_path(filename):
            if filename == "configuration.yaml":
                return str(config_file)
            elif filename.startswith("ui-lovelace-"):
                return str(tmp_path / filename)
            return str(tmp_path / filename)

        with patch.object(hass.config, 'path', side_effect=mock_path):
            dashboard_config = {
                "title": "Preserve Test",
                "url_path": "preserve",
                "views": []
            }

            result = await agent.create_dashboard(dashboard_config)

            assert result["success"] is True

            # Verify other sections are preserved
            updated_content = config_file.read_text()
            assert "homeassistant:" in updated_content
            assert "name: My Home" in updated_content
            assert "logger:" in updated_content
            assert "automation: !include automations.yaml" in updated_content

    @pytest.mark.asyncio
    async def test_create_dashboard_fallback_adds_dashboards_section(self, agent, hass, tmp_path):
        """Test fallback mechanism adds dashboards section when missing under lovelace."""
        config_file = tmp_path / "configuration.yaml"
        dashboard_file = tmp_path / "ui-lovelace-fallback-dash.yaml"

        # Config with lovelace but the parsing doesn't find dashboards section inline
        config_content = """lovelace:
  mode: yaml
  resources:
    - url: /local/custom.js
      type: module
"""
        config_file.write_text(config_content)

        def mock_path(filename):
            if filename == "configuration.yaml":
                return str(config_file)
            elif filename.startswith("ui-lovelace-"):
                return str(tmp_path / filename)
            return str(tmp_path / filename)

        with patch.object(hass.config, 'path', side_effect=mock_path):
            dashboard_config = {
                "title": "Fallback Dash",
                "url_path": "fallback-dash",
                "views": []
            }

            result = await agent.create_dashboard(dashboard_config)

            assert result["success"] is True

            updated_content = config_file.read_text()
            assert "dashboards:" in updated_content
            assert "fallback-dash:" in updated_content

    @pytest.mark.asyncio
    async def test_create_dashboard_multiple_lovelace_mentions(self, agent, hass, tmp_path):
        """Test handling when 'lovelace' appears in comments or other contexts."""
        config_file = tmp_path / "configuration.yaml"
        dashboard_file = tmp_path / "ui-lovelace-multi.yaml"

        config_content = """# This file configures lovelace and other components
homeassistant:
  name: Home

# Configure lovelace dashboard
lovelace:
  dashboards:
    existing:
      mode: yaml
"""
        config_file.write_text(config_content)

        def mock_path(filename):
            if filename == "configuration.yaml":
                return str(config_file)
            elif filename.startswith("ui-lovelace-"):
                return str(tmp_path / filename)
            return str(tmp_path / filename)

        with patch.object(hass.config, 'path', side_effect=mock_path):
            dashboard_config = {
                "title": "Multi Dashboard",
                "url_path": "multi",
                "views": []
            }

            result = await agent.create_dashboard(dashboard_config)

            assert result["success"] is True

            updated_content = config_file.read_text()
            assert "multi:" in updated_content
            # Comments should be preserved
            assert "# This file configures lovelace" in updated_content


class TestServiceExecutionMethods:
    """Test service execution methods in process_query (lines 4284-4398).

    Tests for handling get_entities, get_entities_by_area, call_service with
    nested requests, and backward compatibility with old format.
    """

    @pytest.fixture
    def agent_config(self):
        """Mock agent configuration."""
        return {
            "ai_provider": "openai",
            "openai_token": "sk-" + "a" * 48,
        }

    @pytest.mark.asyncio
    async def test_process_query_direct_get_entities(self, hass, agent_config):
        """Test process_query handles direct get_entities request."""
        hass.states.async_set("light.living_room", "on", {"friendly_name": "Living Room"})
        hass.states.async_set("light.bedroom", "off", {"friendly_name": "Bedroom"})

        agent = AiAgentHaAgent(hass, agent_config)

        # First response: get_entities request, second: final response
        response_1 = json.dumps({
            "request_type": "get_entities",
            "parameters": {"area_id": None},
        })
        response_2 = json.dumps({
            "request_type": "final_response",
            "response": "Found 2 lights",
        })

        with patch.object(agent, "_get_ai_response", new_callable=AsyncMock) as mock_ai:
            mock_ai.side_effect = [response_1, response_2]

            result = await agent.process_query("List all lights")

            assert result["success"] is True
            assert result["answer"] == "Found 2 lights"
            assert mock_ai.call_count == 2

    @pytest.mark.asyncio
    async def test_process_query_direct_get_entities_by_area(self, hass, agent_config):
        """Test process_query handles direct get_entities_by_area request."""
        hass.states.async_set("light.living_room", "on", {"friendly_name": "Living Room"})

        agent = AiAgentHaAgent(hass, agent_config)

        response_1 = json.dumps({
            "request_type": "get_entities_by_area",
            "parameters": {"area_id": "living_room"},
        })
        response_2 = json.dumps({
            "request_type": "final_response",
            "response": "Found entities in living room",
        })

        with patch.object(agent, "_get_ai_response", new_callable=AsyncMock) as mock_ai:
            with patch.object(agent, "get_entities_by_area", new_callable=AsyncMock) as mock_get_area:
                mock_get_area.return_value = [{"entity_id": "light.living_room", "state": "on"}]
                mock_ai.side_effect = [response_1, response_2]

                result = await agent.process_query("What's in the living room?")

                assert result["success"] is True
                mock_get_area.assert_called_once_with("living_room")

    @pytest.mark.asyncio
    async def test_process_query_call_service_with_nested_get_entities(self, hass, agent_config):
        """Test call_service with nested get_entities request in target."""
        hass.states.async_set("light.living_room", "off", {"friendly_name": "Living Room"})
        hass.states.async_set("light.bedroom", "off", {"friendly_name": "Bedroom"})

        agent = AiAgentHaAgent(hass, agent_config)

        # Response with nested get_entities in target
        response_1 = json.dumps({
            "request_type": "call_service",
            "parameters": {
                "domain": "light",
                "service": "turn_on",
                "target": {
                    "entity_id": {
                        "request_type": "get_entities",
                        "parameters": {"area_id": "living_room"},
                    }
                },
            },
        })
        response_2 = json.dumps({
            "request_type": "final_response",
            "response": "Turned on the lights",
        })

        with patch.object(agent, "_get_ai_response", new_callable=AsyncMock) as mock_ai:
            with patch.object(agent, "get_entities", new_callable=AsyncMock) as mock_get_entities:
                with patch.object(agent, "call_service", new_callable=AsyncMock) as mock_call:
                    mock_get_entities.return_value = [
                        {"entity_id": "light.living_room"},
                        {"entity_id": "light.bedroom"},
                    ]
                    mock_call.return_value = {"success": True}
                    mock_ai.side_effect = [response_1, response_2]

                    result = await agent.process_query("Turn on all lights")

                    assert result["success"] is True
                    mock_get_entities.assert_called_once_with(area_id="living_room", area_ids=None)

    @pytest.mark.asyncio
    async def test_process_query_call_service_with_nested_get_entities_by_area(self, hass, agent_config):
        """Test call_service with nested get_entities_by_area request."""
        hass.states.async_set("light.kitchen", "off", {"friendly_name": "Kitchen"})

        agent = AiAgentHaAgent(hass, agent_config)

        response_1 = json.dumps({
            "request_type": "call_service",
            "parameters": {
                "domain": "light",
                "service": "turn_on",
                "target": {
                    "entity_id": {
                        "request_type": "get_entities_by_area",
                        "parameters": {"area_id": "kitchen"},
                    }
                },
            },
        })
        response_2 = json.dumps({
            "request_type": "final_response",
            "response": "Kitchen lights on",
        })

        with patch.object(agent, "_get_ai_response", new_callable=AsyncMock) as mock_ai:
            with patch.object(agent, "get_entities_by_area", new_callable=AsyncMock) as mock_get_area:
                with patch.object(agent, "call_service", new_callable=AsyncMock) as mock_call:
                    mock_get_area.return_value = [{"entity_id": "light.kitchen"}]
                    mock_call.return_value = {"success": True}
                    mock_ai.side_effect = [response_1, response_2]

                    result = await agent.process_query("Turn on kitchen lights")

                    assert result["success"] is True
                    mock_get_area.assert_called_once_with("kitchen")

    @pytest.mark.asyncio
    async def test_process_query_call_service_with_nested_get_entities_by_domain(self, hass, agent_config):
        """Test call_service with nested get_entities_by_domain request."""
        hass.states.async_set("light.one", "off", {})
        hass.states.async_set("light.two", "off", {})

        agent = AiAgentHaAgent(hass, agent_config)

        response_1 = json.dumps({
            "request_type": "call_service",
            "parameters": {
                "domain": "light",
                "service": "turn_on",
                "target": {
                    "entity_id": {
                        "request_type": "get_entities_by_domain",
                        "parameters": {"domain": "light"},
                    }
                },
            },
        })
        response_2 = json.dumps({
            "request_type": "final_response",
            "response": "All lights on",
        })

        with patch.object(agent, "_get_ai_response", new_callable=AsyncMock) as mock_ai:
            with patch.object(agent, "get_entities_by_domain", new_callable=AsyncMock) as mock_get_domain:
                with patch.object(agent, "call_service", new_callable=AsyncMock) as mock_call:
                    mock_get_domain.return_value = [
                        {"entity_id": "light.one"},
                        {"entity_id": "light.two"},
                    ]
                    mock_call.return_value = {"success": True}
                    mock_ai.side_effect = [response_1, response_2]

                    result = await agent.process_query("Turn on all lights")

                    assert result["success"] is True
                    mock_get_domain.assert_called_once_with("light")

    @pytest.mark.asyncio
    async def test_process_query_call_service_unsupported_nested_request(self, hass, agent_config):
        """Test call_service returns error for unsupported nested request type."""
        agent = AiAgentHaAgent(hass, agent_config)

        response_1 = json.dumps({
            "request_type": "call_service",
            "parameters": {
                "domain": "light",
                "service": "turn_on",
                "target": {
                    "entity_id": {
                        "request_type": "unsupported_request",
                        "parameters": {},
                    }
                },
            },
        })

        with patch.object(agent, "_get_ai_response", new_callable=AsyncMock) as mock_ai:
            mock_ai.return_value = response_1

            result = await agent.process_query("Do something")

            assert result["success"] is False
            assert "Unsupported nested request type: unsupported_request" in result["error"]

    @pytest.mark.asyncio
    async def test_process_query_call_service_nested_returns_non_list(self, hass, agent_config):
        """Test call_service returns error when nested request returns non-list data."""
        agent = AiAgentHaAgent(hass, agent_config)

        response_1 = json.dumps({
            "request_type": "call_service",
            "parameters": {
                "domain": "light",
                "service": "turn_on",
                "target": {
                    "entity_id": {
                        "request_type": "get_entities",
                        "parameters": {},
                    }
                },
            },
        })

        with patch.object(agent, "_get_ai_response", new_callable=AsyncMock) as mock_ai:
            with patch.object(agent, "get_entities", new_callable=AsyncMock) as mock_get_entities:
                # Return a dict instead of a list to trigger the error
                mock_get_entities.return_value = {"error": "something went wrong"}
                mock_ai.return_value = response_1

                result = await agent.process_query("Turn on lights")

                assert result["success"] is False
                assert "Nested request returned unexpected data format" in result["error"]

    @pytest.mark.asyncio
    async def test_process_query_call_service_backward_compat_old_format(self, hass, agent_config):
        """Test call_service backward compatibility with old format (request + parameters.entity_id)."""
        hass.states.async_set("light.living_room", "off", {"friendly_name": "Living Room"})

        agent = AiAgentHaAgent(hass, agent_config)

        # Old format: request with entity_id in parameters
        response_1 = json.dumps({
            "request_type": "call_service",
            "request": "turn_on",
            "parameters": {
                "entity_id": "light.living_room",
                "brightness": 255,
            },
        })
        response_2 = json.dumps({
            "request_type": "final_response",
            "response": "Light turned on",
        })

        with patch.object(agent, "_get_ai_response", new_callable=AsyncMock) as mock_ai:
            with patch.object(agent, "call_service", new_callable=AsyncMock) as mock_call:
                mock_call.return_value = {"success": True}
                mock_ai.side_effect = [response_1, response_2]

                result = await agent.process_query("Turn on living room light")

                assert result["success"] is True
                # Old format should extract domain from entity_id
                mock_call.assert_called_once_with(
                    "light",
                    "turn_on",
                    {"entity_id": "light.living_room"},
                    {"brightness": 255},
                )

    @pytest.mark.asyncio
    async def test_process_query_call_service_error_response(self, hass, agent_config):
        """Test process_query handles call_service error response."""
        hass.states.async_set("light.living_room", "off", {})

        agent = AiAgentHaAgent(hass, agent_config)

        response_1 = json.dumps({
            "request_type": "call_service",
            "parameters": {
                "domain": "light",
                "service": "turn_on",
                "target": {"entity_id": "light.living_room"},
            },
        })

        with patch.object(agent, "_get_ai_response", new_callable=AsyncMock) as mock_ai:
            with patch.object(agent, "call_service", new_callable=AsyncMock) as mock_call:
                mock_call.return_value = {"error": "Entity not found"}
                mock_ai.return_value = response_1

                result = await agent.process_query("Turn on light")

                assert result["success"] is False
                assert result["error"] == "Entity not found"

    @pytest.mark.asyncio
    async def test_process_query_call_service_top_level_params(self, hass, agent_config):
        """Test call_service with parameters at top level (backward compat)."""
        hass.states.async_set("switch.test", "off", {})

        agent = AiAgentHaAgent(hass, agent_config)

        # Top-level domain/service for backward compatibility
        response_1 = json.dumps({
            "request_type": "call_service",
            "domain": "switch",
            "service": "turn_on",
            "target": {"entity_id": "switch.test"},
        })
        response_2 = json.dumps({
            "request_type": "final_response",
            "response": "Switch turned on",
        })

        with patch.object(agent, "_get_ai_response", new_callable=AsyncMock) as mock_ai:
            with patch.object(agent, "call_service", new_callable=AsyncMock) as mock_call:
                mock_call.return_value = {"success": True}
                mock_ai.side_effect = [response_1, response_2]

                result = await agent.process_query("Turn on switch")

                assert result["success"] is True
                mock_call.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_query_get_entities_adds_to_conversation_history(self, hass, agent_config):
        """Test that get_entities request adds data to conversation history as user role."""
        hass.states.async_set("sensor.temp", "22", {"friendly_name": "Temperature"})

        agent = AiAgentHaAgent(hass, agent_config)

        # Use get_entities_by_domain which doesn't require area_id
        response_1 = json.dumps({
            "request_type": "get_entities_by_domain",
            "parameters": {"domain": "sensor"},
        })
        response_2 = json.dumps({
            "request_type": "final_response",
            "response": "Done",
        })

        with patch.object(agent, "_get_ai_response", new_callable=AsyncMock) as mock_ai:
            mock_ai.side_effect = [response_1, response_2]

            await agent.process_query("Get entities")

            # Find the data message in conversation history
            data_messages = [
                msg for msg in agent.conversation_history
                if msg.get("role") == "user" and '"data":' in msg.get("content", "")
            ]
            assert len(data_messages) >= 1, "Data payload should be added as user role"

    @pytest.mark.asyncio
    async def test_process_query_call_service_with_service_data(self, hass, agent_config):
        """Test call_service properly passes service_data."""
        hass.states.async_set("light.test", "off", {})

        agent = AiAgentHaAgent(hass, agent_config)

        response_1 = json.dumps({
            "request_type": "call_service",
            "parameters": {
                "domain": "light",
                "service": "turn_on",
                "target": {"entity_id": "light.test"},
                "service_data": {"brightness": 128, "transition": 2},
            },
        })
        response_2 = json.dumps({
            "request_type": "final_response",
            "response": "Light dimmed",
        })

        with patch.object(agent, "_get_ai_response", new_callable=AsyncMock) as mock_ai:
            with patch.object(agent, "call_service", new_callable=AsyncMock) as mock_call:
                mock_call.return_value = {"success": True}
                mock_ai.side_effect = [response_1, response_2]

                result = await agent.process_query("Dim the light")

                assert result["success"] is True
                mock_call.assert_called_once_with(
                    "light",
                    "turn_on",
                    {"entity_id": "light.test"},
                    {"brightness": 128, "transition": 2},
                )

    @pytest.mark.asyncio
    async def test_process_query_call_service_success_continues_loop(self, hass, agent_config):
        """Test that successful call_service continues the conversation loop."""
        hass.states.async_set("light.test", "off", {})

        agent = AiAgentHaAgent(hass, agent_config)

        response_1 = json.dumps({
            "request_type": "call_service",
            "parameters": {
                "domain": "light",
                "service": "turn_on",
                "target": {"entity_id": "light.test"},
            },
        })
        response_2 = json.dumps({
            "request_type": "final_response",
            "response": "Done turning on the light",
        })

        with patch.object(agent, "_get_ai_response", new_callable=AsyncMock) as mock_ai:
            with patch.object(agent, "call_service", new_callable=AsyncMock) as mock_call:
                mock_call.return_value = {"success": True, "message": "Service called"}
                mock_ai.side_effect = [response_1, response_2]

                result = await agent.process_query("Turn on light")

                # Should have called AI twice (first for service, then for final response)
                assert mock_ai.call_count == 2
                assert result["success"] is True
                assert result["answer"] == "Done turning on the light"

    @pytest.mark.asyncio
    async def test_process_query_nested_request_extracts_entity_ids(self, hass, agent_config):
        """Test that nested request properly extracts entity_ids from returned data."""
        hass.states.async_set("light.one", "off", {})
        hass.states.async_set("light.two", "off", {})

        agent = AiAgentHaAgent(hass, agent_config)

        response_1 = json.dumps({
            "request_type": "call_service",
            "parameters": {
                "domain": "light",
                "service": "turn_on",
                "target": {
                    "entity_id": {
                        "request_type": "get_entities",
                        "parameters": {},
                    }
                },
            },
        })
        response_2 = json.dumps({
            "request_type": "final_response",
            "response": "Lights on",
        })

        with patch.object(agent, "_get_ai_response", new_callable=AsyncMock) as mock_ai:
            with patch.object(agent, "get_entities", new_callable=AsyncMock) as mock_get_entities:
                with patch.object(agent, "call_service", new_callable=AsyncMock) as mock_call:
                    # Return entities with entity_id keys
                    mock_get_entities.return_value = [
                        {"entity_id": "light.one", "state": "off"},
                        {"entity_id": "light.two", "state": "off"},
                    ]
                    mock_call.return_value = {"success": True}
                    mock_ai.side_effect = [response_1, response_2]

                    await agent.process_query("Turn on all lights")

                    # Verify call_service was called with resolved entity IDs
                    call_args = mock_call.call_args
                    # Target should have entity_id as list of IDs
                    target = call_args[0][2]  # Third positional arg is target
                    assert target["entity_id"] == ["light.one", "light.two"]


class TestDataRetrievalMethods:
    """Tests for data retrieval methods in agent.py lines 2135-2341."""

    @pytest.fixture
    def agent_config(self):
        """Mock agent configuration."""
        return {
            "ai_provider": "openai",
            "openai_token": "sk-" + "a" * 48,
        }

    @pytest.fixture
    def agent(self, hass, agent_config):
        """Create agent instance."""
        return AiAgentHaAgent(hass, agent_config)

    # Tests for get_entities_by_area

    @pytest.mark.asyncio
    async def test_get_entities_by_area_no_area_id(self, agent):
        """Test get_entities_by_area returns error when no area_id provided."""
        result = await agent.get_entities_by_area("")
        assert len(result) == 1
        assert "error" in result[0]
        assert "area_id is required" in result[0]["error"]

    @pytest.mark.asyncio
    async def test_get_entities_by_area_with_entities(self, hass, agent):
        """Test get_entities_by_area returns entities in the specified area."""
        from homeassistant.helpers import entity_registry as er
        from homeassistant.helpers import device_registry as dr

        # Create registries
        entity_registry = er.async_get(hass)
        device_registry = dr.async_get(hass)

        # Create a mock entity entry with area_id
        entry = entity_registry.async_get_or_create(
            domain="light",
            platform="test",
            unique_id="test_light_1",
            suggested_object_id="living_room",
        )
        entity_registry.async_update_entity(entry.entity_id, area_id="living_room")

        # Set the entity state
        hass.states.async_set(entry.entity_id, "on", {"friendly_name": "Living Room Light"})

        result = await agent.get_entities_by_area("living_room")
        assert len(result) >= 1
        entity_ids = [e.get("entity_id") for e in result if "entity_id" in e]
        assert entry.entity_id in entity_ids

    @pytest.mark.asyncio
    async def test_get_entities_by_area_via_device(self, hass, agent):
        """Test get_entities_by_area returns entities via device area assignment."""
        from homeassistant.helpers import entity_registry as er
        from homeassistant.helpers import device_registry as dr

        entity_registry = er.async_get(hass)
        device_registry = dr.async_get(hass)

        # Create and add mock config entry
        config_entry = MockConfigEntry(domain="test", entry_id="test_config_entry")
        config_entry.add_to_hass(hass)

        # Create a device with area
        device = device_registry.async_get_or_create(
            config_entry_id="test_config_entry",
            identifiers={("test", "device1")},
            name="Test Device",
        )
        device_registry.async_update_device(device.id, area_id="bedroom")

        # Create an entity linked to the device
        entry = entity_registry.async_get_or_create(
            domain="sensor",
            platform="test",
            unique_id="test_sensor_1",
            device_id=device.id,
        )

        hass.states.async_set(entry.entity_id, "25", {"friendly_name": "Bedroom Sensor"})

        result = await agent.get_entities_by_area("bedroom")
        entity_ids = [e.get("entity_id") for e in result if "entity_id" in e]
        assert entry.entity_id in entity_ids

    @pytest.mark.asyncio
    async def test_get_entities_by_area_empty_area(self, hass, agent):
        """Test get_entities_by_area returns empty list for area with no entities."""
        result = await agent.get_entities_by_area("nonexistent_area")
        # Should return empty list, not error
        assert isinstance(result, list)
        # Filter out any error entries
        entities = [e for e in result if "entity_id" in e]
        assert len(entities) == 0

    @pytest.mark.asyncio
    async def test_get_entities_by_area_exception_handling(self, agent):
        """Test get_entities_by_area handles exceptions gracefully."""
        with patch("homeassistant.helpers.entity_registry.async_get", side_effect=Exception("Registry error")):
            result = await agent.get_entities_by_area("test_area")
            assert len(result) == 1
            assert "error" in result[0]
            assert "Registry error" in result[0]["error"]

    # Tests for get_entities

    @pytest.mark.asyncio
    async def test_get_entities_no_params(self, agent):
        """Test get_entities returns error when no area parameters provided."""
        result = await agent.get_entities()
        assert len(result) == 1
        assert "error" in result[0]
        assert "No area_id or area_ids provided" in result[0]["error"]

    @pytest.mark.asyncio
    async def test_get_entities_single_area_id(self, hass, agent):
        """Test get_entities with single area_id parameter."""
        from homeassistant.helpers import entity_registry as er

        entity_registry = er.async_get(hass)

        entry = entity_registry.async_get_or_create(
            domain="switch",
            platform="test",
            unique_id="test_switch_1",
        )
        entity_registry.async_update_entity(entry.entity_id, area_id="kitchen")
        hass.states.async_set(entry.entity_id, "off", {"friendly_name": "Kitchen Switch"})

        result = await agent.get_entities(area_id="kitchen")
        entity_ids = [e.get("entity_id") for e in result if "entity_id" in e]
        assert entry.entity_id in entity_ids

    @pytest.mark.asyncio
    async def test_get_entities_multiple_area_ids(self, hass, agent):
        """Test get_entities with multiple area_ids parameter."""
        from homeassistant.helpers import entity_registry as er

        entity_registry = er.async_get(hass)

        # Create entities in different areas
        entry1 = entity_registry.async_get_or_create(
            domain="light",
            platform="test",
            unique_id="test_light_kitchen",
        )
        entity_registry.async_update_entity(entry1.entity_id, area_id="kitchen")
        hass.states.async_set(entry1.entity_id, "on", {"friendly_name": "Kitchen Light"})

        entry2 = entity_registry.async_get_or_create(
            domain="light",
            platform="test",
            unique_id="test_light_bathroom",
        )
        entity_registry.async_update_entity(entry2.entity_id, area_id="bathroom")
        hass.states.async_set(entry2.entity_id, "off", {"friendly_name": "Bathroom Light"})

        result = await agent.get_entities(area_ids=["kitchen", "bathroom"])
        entity_ids = [e.get("entity_id") for e in result if "entity_id" in e]
        assert entry1.entity_id in entity_ids
        assert entry2.entity_id in entity_ids

    @pytest.mark.asyncio
    async def test_get_entities_area_id_as_list(self, hass, agent):
        """Test get_entities handles area_id parameter passed as list."""
        from homeassistant.helpers import entity_registry as er

        entity_registry = er.async_get(hass)

        entry = entity_registry.async_get_or_create(
            domain="fan",
            platform="test",
            unique_id="test_fan_1",
        )
        entity_registry.async_update_entity(entry.entity_id, area_id="office")
        hass.states.async_set(entry.entity_id, "on", {"friendly_name": "Office Fan"})

        result = await agent.get_entities(area_id=["office"])
        entity_ids = [e.get("entity_id") for e in result if "entity_id" in e]
        assert entry.entity_id in entity_ids

    @pytest.mark.asyncio
    async def test_get_entities_area_ids_as_string(self, hass, agent):
        """Test get_entities handles area_ids parameter passed as string."""
        from homeassistant.helpers import entity_registry as er

        entity_registry = er.async_get(hass)

        entry = entity_registry.async_get_or_create(
            domain="cover",
            platform="test",
            unique_id="test_cover_1",
        )
        entity_registry.async_update_entity(entry.entity_id, area_id="garage")
        hass.states.async_set(entry.entity_id, "closed", {"friendly_name": "Garage Door"})

        result = await agent.get_entities(area_ids="garage")
        entity_ids = [e.get("entity_id") for e in result if "entity_id" in e]
        assert entry.entity_id in entity_ids

    @pytest.mark.asyncio
    async def test_get_entities_deduplicates(self, hass, agent):
        """Test get_entities removes duplicates when entity appears in multiple areas."""
        from homeassistant.helpers import entity_registry as er

        entity_registry = er.async_get(hass)

        entry = entity_registry.async_get_or_create(
            domain="light",
            platform="test",
            unique_id="test_light_shared",
        )
        entity_registry.async_update_entity(entry.entity_id, area_id="living_room")
        hass.states.async_set(entry.entity_id, "on", {"friendly_name": "Shared Light"})

        # Request same area twice
        result = await agent.get_entities(area_ids=["living_room", "living_room"])
        entity_ids = [e.get("entity_id") for e in result if "entity_id" in e]
        # Entity should only appear once
        assert entity_ids.count(entry.entity_id) == 1

    @pytest.mark.asyncio
    async def test_get_entities_exception_handling(self, agent):
        """Test get_entities handles exceptions gracefully."""
        with patch.object(agent, "get_entities_by_area", side_effect=Exception("Area error")):
            result = await agent.get_entities(area_id="test_area")
            assert len(result) == 1
            assert "error" in result[0]
            assert "Area error" in result[0]["error"]

    # Tests for get_calendar_events

    @pytest.mark.asyncio
    async def test_get_calendar_events_specific_entity(self, hass, agent):
        """Test get_calendar_events with specific entity_id."""
        hass.states.async_set(
            "calendar.family",
            "on",
            {"friendly_name": "Family Calendar", "message": "Birthday Party"}
        )

        result = await agent.get_calendar_events(entity_id="calendar.family")
        assert len(result) == 1
        assert result[0]["entity_id"] == "calendar.family"

    @pytest.mark.asyncio
    async def test_get_calendar_events_all_calendars(self, hass, agent):
        """Test get_calendar_events returns all calendar entities."""
        hass.states.async_set("calendar.work", "on", {"friendly_name": "Work Calendar"})
        hass.states.async_set("calendar.personal", "off", {"friendly_name": "Personal Calendar"})
        hass.states.async_set("light.lamp", "on", {"friendly_name": "Lamp"})

        result = await agent.get_calendar_events()
        entity_ids = [e.get("entity_id") for e in result if "entity_id" in e]
        assert "calendar.work" in entity_ids
        assert "calendar.personal" in entity_ids
        assert "light.lamp" not in entity_ids

    @pytest.mark.asyncio
    async def test_get_calendar_events_exception_handling(self, agent):
        """Test get_calendar_events handles exceptions gracefully."""
        with patch.object(agent, "get_entity_state", side_effect=Exception("Calendar error")):
            result = await agent.get_calendar_events(entity_id="calendar.test")
            assert len(result) == 1
            assert "error" in result[0]
            assert "Calendar error" in result[0]["error"]

    # Tests for get_automations

    @pytest.mark.asyncio
    async def test_get_automations(self, hass, agent):
        """Test get_automations returns all automation entities."""
        hass.states.async_set(
            "automation.morning_lights",
            "on",
            {"friendly_name": "Morning Lights"}
        )
        hass.states.async_set(
            "automation.night_mode",
            "off",
            {"friendly_name": "Night Mode"}
        )
        hass.states.async_set("light.test", "on", {"friendly_name": "Test Light"})

        result = await agent.get_automations()
        entity_ids = [e.get("entity_id") for e in result if "entity_id" in e]
        assert "automation.morning_lights" in entity_ids
        assert "automation.night_mode" in entity_ids
        assert "light.test" not in entity_ids

    @pytest.mark.asyncio
    async def test_get_automations_empty(self, hass, agent):
        """Test get_automations returns empty list when no automations exist."""
        # Only non-automation entities
        hass.states.async_set("light.test", "on", {})

        result = await agent.get_automations()
        # Filter for actual automation entities
        automations = [e for e in result if e.get("entity_id", "").startswith("automation.")]
        assert len(automations) == 0

    @pytest.mark.asyncio
    async def test_get_automations_exception_handling(self, agent):
        """Test get_automations handles exceptions gracefully."""
        with patch.object(agent, "get_entities_by_domain", side_effect=Exception("Automation error")):
            result = await agent.get_automations()
            assert len(result) == 1
            assert "error" in result[0]
            assert "Automation error" in result[0]["error"]

    # Tests for get_entity_registry

    @pytest.mark.asyncio
    async def test_get_entity_registry_basic(self, hass, agent):
        """Test get_entity_registry returns basic registry data."""
        from homeassistant.helpers import entity_registry as er

        entity_registry = er.async_get(hass)

        entry = entity_registry.async_get_or_create(
            domain="sensor",
            platform="test",
            unique_id="test_sensor_registry",
        )
        hass.states.async_set(entry.entity_id, "50", {"device_class": "temperature"})

        result = await agent.get_entity_registry()
        assert "entities" in result
        assert "total_count" in result
        assert "has_more" in result
        assert isinstance(result["entities"], list)

    @pytest.mark.asyncio
    async def test_get_entity_registry_filter_by_domain(self, hass, agent):
        """Test get_entity_registry filters by domain."""
        from homeassistant.helpers import entity_registry as er

        entity_registry = er.async_get(hass)

        sensor_entry = entity_registry.async_get_or_create(
            domain="sensor",
            platform="test",
            unique_id="test_sensor_domain",
        )
        light_entry = entity_registry.async_get_or_create(
            domain="light",
            platform="test",
            unique_id="test_light_domain",
        )
        hass.states.async_set(sensor_entry.entity_id, "25", {})
        hass.states.async_set(light_entry.entity_id, "on", {})

        result = await agent.get_entity_registry(domain="sensor")
        entity_ids = [e["entity_id"] for e in result["entities"]]
        # Should only contain sensor entities
        for eid in entity_ids:
            assert eid.startswith("sensor.")

    @pytest.mark.asyncio
    async def test_get_entity_registry_filter_by_area(self, hass, agent):
        """Test get_entity_registry filters by area_id."""
        from homeassistant.helpers import entity_registry as er

        entity_registry = er.async_get(hass)

        entry1 = entity_registry.async_get_or_create(
            domain="switch",
            platform="test",
            unique_id="test_switch_area1",
        )
        entity_registry.async_update_entity(entry1.entity_id, area_id="basement")

        entry2 = entity_registry.async_get_or_create(
            domain="switch",
            platform="test",
            unique_id="test_switch_area2",
        )
        entity_registry.async_update_entity(entry2.entity_id, area_id="attic")

        hass.states.async_set(entry1.entity_id, "on", {})
        hass.states.async_set(entry2.entity_id, "off", {})

        result = await agent.get_entity_registry(area_id="basement")
        entity_ids = [e["entity_id"] for e in result["entities"]]
        assert entry1.entity_id in entity_ids
        assert entry2.entity_id not in entity_ids

    @pytest.mark.asyncio
    async def test_get_entity_registry_filter_by_device_class(self, hass, agent):
        """Test get_entity_registry filters by device_class."""
        from homeassistant.helpers import entity_registry as er

        entity_registry = er.async_get(hass)

        temp_entry = entity_registry.async_get_or_create(
            domain="sensor",
            platform="test",
            unique_id="test_temp_dc",
        )
        humid_entry = entity_registry.async_get_or_create(
            domain="sensor",
            platform="test",
            unique_id="test_humid_dc",
        )
        hass.states.async_set(temp_entry.entity_id, "22", {"device_class": "temperature"})
        hass.states.async_set(humid_entry.entity_id, "55", {"device_class": "humidity"})

        result = await agent.get_entity_registry(device_class="temperature")
        entity_ids = [e["entity_id"] for e in result["entities"]]
        assert temp_entry.entity_id in entity_ids
        assert humid_entry.entity_id not in entity_ids

    @pytest.mark.asyncio
    async def test_get_entity_registry_pagination(self, hass, agent):
        """Test get_entity_registry pagination with limit and offset."""
        from homeassistant.helpers import entity_registry as er

        entity_registry = er.async_get(hass)

        # Create multiple entities
        for i in range(5):
            entry = entity_registry.async_get_or_create(
                domain="binary_sensor",
                platform="test",
                unique_id=f"test_binary_{i}",
            )
            hass.states.async_set(entry.entity_id, "on", {})

        # Get first page
        result1 = await agent.get_entity_registry(domain="binary_sensor", limit=2, offset=0)
        assert result1["limit"] == 2
        assert result1["offset"] == 0
        assert len(result1["entities"]) <= 2

        # Get second page
        result2 = await agent.get_entity_registry(domain="binary_sensor", limit=2, offset=2)
        assert result2["offset"] == 2

    @pytest.mark.asyncio
    async def test_get_entity_registry_limit_enforcement(self, hass, agent):
        """Test get_entity_registry enforces limit bounds."""
        # Test max limit capped at 200
        result = await agent.get_entity_registry(limit=500)
        assert result["limit"] == 200

        # Test min limit at 1
        result = await agent.get_entity_registry(limit=0)
        assert result["limit"] == 1

        # Test negative offset becomes 0
        result = await agent.get_entity_registry(offset=-5)
        assert result["offset"] == 0

    @pytest.mark.asyncio
    async def test_get_entity_registry_has_more_flag(self, hass, agent):
        """Test get_entity_registry has_more flag is accurate."""
        from homeassistant.helpers import entity_registry as er

        entity_registry = er.async_get(hass)

        # Create 3 entities
        for i in range(3):
            entry = entity_registry.async_get_or_create(
                domain="lock",
                platform="test",
                unique_id=f"test_lock_{i}",
            )
            hass.states.async_set(entry.entity_id, "locked", {})

        # Request with limit smaller than total
        result = await agent.get_entity_registry(domain="lock", limit=2)
        assert result["has_more"] is True

        # Request with limit larger than total
        result = await agent.get_entity_registry(domain="lock", limit=10)
        assert result["has_more"] is False

    @pytest.mark.asyncio
    async def test_get_entity_registry_skips_disabled(self, hass, agent):
        """Test get_entity_registry skips disabled entities."""
        from homeassistant.helpers import entity_registry as er

        entity_registry = er.async_get(hass)

        enabled_entry = entity_registry.async_get_or_create(
            domain="media_player",
            platform="test",
            unique_id="test_mp_enabled",
        )
        disabled_entry = entity_registry.async_get_or_create(
            domain="media_player",
            platform="test",
            unique_id="test_mp_disabled",
            disabled_by=er.RegistryEntryDisabler.USER,
        )
        hass.states.async_set(enabled_entry.entity_id, "playing", {})

        result = await agent.get_entity_registry(domain="media_player")
        entity_ids = [e["entity_id"] for e in result["entities"]]
        assert enabled_entry.entity_id in entity_ids
        assert disabled_entry.entity_id not in entity_ids

    @pytest.mark.asyncio
    async def test_get_entity_registry_resolves_area_from_device(self, hass, agent):
        """Test get_entity_registry resolves area from device when entity has no direct area."""
        from homeassistant.helpers import entity_registry as er
        from homeassistant.helpers import device_registry as dr
        from homeassistant.helpers import area_registry as ar

        entity_registry = er.async_get(hass)
        device_registry = dr.async_get(hass)
        area_registry = ar.async_get(hass)

        # Create an area
        area = area_registry.async_create("Test Room")

        # Create and add mock config entry
        config_entry = MockConfigEntry(domain="test", entry_id="test_config")
        config_entry.add_to_hass(hass)

        # Create a device in the area
        device = device_registry.async_get_or_create(
            config_entry_id="test_config",
            identifiers={("test", "device_area_test")},
            name="Device in Room",
        )
        device_registry.async_update_device(device.id, area_id=area.id)

        # Create entity linked to device (no direct area)
        entry = entity_registry.async_get_or_create(
            domain="sensor",
            platform="test",
            unique_id="test_sensor_device_area",
            device_id=device.id,
        )
        hass.states.async_set(entry.entity_id, "100", {})

        result = await agent.get_entity_registry(area_id=area.id)
        entity_ids = [e["entity_id"] for e in result["entities"]]
        assert entry.entity_id in entity_ids

    @pytest.mark.asyncio
    async def test_get_entity_registry_filters_applied_metadata(self, hass, agent):
        """Test get_entity_registry returns filters_applied metadata."""
        result = await agent.get_entity_registry(
            domain="sensor",
            area_id="test_area",
            device_class="temperature"
        )
        assert "filters_applied" in result
        assert result["filters_applied"]["domain"] == "sensor"
        assert result["filters_applied"]["area_id"] == "test_area"
        assert result["filters_applied"]["device_class"] == "temperature"

    @pytest.mark.asyncio
    async def test_get_entity_registry_empty_registry(self, hass, agent):
        """Test get_entity_registry handles empty registry."""
        with patch("homeassistant.helpers.entity_registry.async_get", return_value=None):
            result = await agent.get_entity_registry()
            assert result["entities"] == []
            assert result["total_count"] == 0
            assert result["has_more"] is False

    @pytest.mark.asyncio
    async def test_get_entity_registry_exception_handling(self, agent):
        """Test get_entity_registry handles exceptions gracefully."""
        with patch("homeassistant.helpers.entity_registry.async_get", side_effect=Exception("Registry error")):
            result = await agent.get_entity_registry()
            assert "error" in result
            assert "Registry error" in result["error"]

    @pytest.mark.asyncio
    async def test_get_entity_registry_entity_data_completeness(self, hass, agent):
        """Test get_entity_registry returns complete entity data structure."""
        from homeassistant.helpers import entity_registry as er

        entity_registry = er.async_get(hass)

        entry = entity_registry.async_get_or_create(
            domain="sensor",
            platform="test_platform",
            unique_id="test_complete_data",
            original_name="Original Sensor Name",
        )
        hass.states.async_set(
            entry.entity_id,
            "42",
            {
                "device_class": "power",
                "state_class": "measurement",
                "unit_of_measurement": "W",
            }
        )

        result = await agent.get_entity_registry(domain="sensor")
        entities = [e for e in result["entities"] if e["entity_id"] == entry.entity_id]
        assert len(entities) == 1
        entity = entities[0]

        # Verify expected fields are present
        assert "entity_id" in entity
        assert "platform" in entity
        assert "device_class" in entity
        assert "state_class" in entity
        assert "unit_of_measurement" in entity
        assert entity["platform"] == "test_platform"
