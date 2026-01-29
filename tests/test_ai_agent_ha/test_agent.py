"""Tests for the AI Agent core functionality."""

import asyncio
import json
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import homeassistant
from homeassistant.core import HomeAssistant

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