"""Tests for EntityManager.

Tests entity operations extracted from the God Class.
Uses TDD approach - tests written first.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock, patch

from custom_components.ai_agent_ha.managers.entity_manager import EntityManager


class MockState:
    """Mock Home Assistant state object."""

    def __init__(
        self,
        entity_id: str,
        state: str,
        attributes: dict = None,
        last_changed: datetime = None,
    ):
        self.entity_id = entity_id
        self.state = state
        self.attributes = attributes or {}
        self.last_changed = last_changed or datetime.now(timezone.utc)


@pytest.fixture
def mock_hass():
    """Create a mock Home Assistant instance."""
    hass = MagicMock()
    hass.states = MagicMock()
    return hass


@pytest.fixture
def entity_manager(mock_hass):
    """Create an EntityManager with mocked hass."""
    return EntityManager(mock_hass)


@pytest.fixture
def sample_states():
    """Create sample entity states for testing."""
    return [
        MockState(
            "light.living_room",
            "on",
            {"friendly_name": "Living Room Light", "brightness": 255},
        ),
        MockState(
            "light.bedroom",
            "off",
            {"friendly_name": "Bedroom Light", "brightness": 0},
        ),
        MockState(
            "sensor.temperature",
            "22.5",
            {"friendly_name": "Temperature Sensor", "unit_of_measurement": "C", "device_class": "temperature"},
        ),
        MockState(
            "sensor.humidity",
            "45",
            {"friendly_name": "Humidity Sensor", "unit_of_measurement": "%", "device_class": "humidity"},
        ),
        MockState(
            "switch.fan",
            "on",
            {"friendly_name": "Fan Switch"},
        ),
    ]


class TestGetEntityState:
    """Tests for get_entity_state method."""

    def test_get_entity_state_returns_entity_dict(self, entity_manager, mock_hass):
        """Test that get_entity_state returns a proper entity state dict."""
        mock_state = MockState(
            "light.living_room",
            "on",
            {"friendly_name": "Living Room Light", "brightness": 255},
        )
        mock_hass.states.get.return_value = mock_state

        result = entity_manager.get_entity_state("light.living_room")

        assert result is not None
        assert result["entity_id"] == "light.living_room"
        assert result["state"] == "on"
        assert result["attributes"]["friendly_name"] == "Living Room Light"
        assert result["attributes"]["brightness"] == 255
        assert "last_changed" in result
        mock_hass.states.get.assert_called_once_with("light.living_room")

    def test_get_entity_state_not_found_returns_none(self, entity_manager, mock_hass):
        """Test that get_entity_state returns None when entity not found."""
        mock_hass.states.get.return_value = None

        result = entity_manager.get_entity_state("nonexistent.entity")

        assert result is None
        mock_hass.states.get.assert_called_once_with("nonexistent.entity")

    def test_get_entity_state_with_empty_entity_id(self, entity_manager, mock_hass):
        """Test that get_entity_state handles empty entity_id."""
        result = entity_manager.get_entity_state("")

        assert result is None
        # Should not call hass.states.get with empty string
        mock_hass.states.get.assert_not_called()

    def test_get_entity_state_includes_last_changed(self, entity_manager, mock_hass):
        """Test that get_entity_state includes last_changed timestamp."""
        test_time = datetime(2025, 1, 29, 12, 0, 0, tzinfo=timezone.utc)
        mock_state = MockState("light.test", "on", {}, test_time)
        mock_hass.states.get.return_value = mock_state

        result = entity_manager.get_entity_state("light.test")

        assert result["last_changed"] == test_time.isoformat()


class TestGetEntitiesByDomain:
    """Tests for get_entities_by_domain method."""

    def test_get_entities_by_domain_returns_list(
        self, entity_manager, mock_hass, sample_states
    ):
        """Test that get_entities_by_domain returns list of entities."""
        mock_hass.states.async_all.return_value = sample_states

        result = entity_manager.get_entities_by_domain("light")

        assert isinstance(result, list)
        assert len(result) == 2
        entity_ids = [e["entity_id"] for e in result]
        assert "light.living_room" in entity_ids
        assert "light.bedroom" in entity_ids

    def test_get_entities_by_domain_empty_result(
        self, entity_manager, mock_hass, sample_states
    ):
        """Test that get_entities_by_domain returns empty list for non-existent domain."""
        mock_hass.states.async_all.return_value = sample_states

        result = entity_manager.get_entities_by_domain("camera")

        assert isinstance(result, list)
        assert len(result) == 0

    def test_get_entities_by_domain_sensor(
        self, entity_manager, mock_hass, sample_states
    ):
        """Test that get_entities_by_domain correctly filters sensor domain."""
        mock_hass.states.async_all.return_value = sample_states

        result = entity_manager.get_entities_by_domain("sensor")

        assert len(result) == 2
        entity_ids = [e["entity_id"] for e in result]
        assert "sensor.temperature" in entity_ids
        assert "sensor.humidity" in entity_ids


class TestGetEntityIdsByDomain:
    """Tests for get_entity_ids_by_domain method."""

    def test_get_entity_ids_by_domain_returns_list_of_ids(
        self, entity_manager, mock_hass, sample_states
    ):
        """Test that get_entity_ids_by_domain returns only entity IDs."""
        mock_hass.states.async_all.return_value = sample_states

        result = entity_manager.get_entity_ids_by_domain("light")

        assert isinstance(result, list)
        assert len(result) == 2
        assert "light.living_room" in result
        assert "light.bedroom" in result
        # Ensure we only get strings (IDs), not dicts
        assert all(isinstance(id, str) for id in result)

    def test_get_entity_ids_by_domain_empty_result(
        self, entity_manager, mock_hass, sample_states
    ):
        """Test that get_entity_ids_by_domain returns empty list for non-existent domain."""
        mock_hass.states.async_all.return_value = sample_states

        result = entity_manager.get_entity_ids_by_domain("vacuum")

        assert isinstance(result, list)
        assert len(result) == 0


class TestFilterEntities:
    """Tests for filter_entities method."""

    def test_filter_entities_by_domain(
        self, entity_manager, mock_hass, sample_states
    ):
        """Test filtering entities by domain only."""
        mock_hass.states.async_all.return_value = sample_states

        result = entity_manager.filter_entities(domain="sensor")

        assert len(result) == 2
        for entity in result:
            assert entity["entity_id"].startswith("sensor.")

    def test_filter_entities_by_state(
        self, entity_manager, mock_hass, sample_states
    ):
        """Test filtering entities by state value."""
        mock_hass.states.async_all.return_value = sample_states

        result = entity_manager.filter_entities(state="on")

        assert len(result) == 2  # light.living_room and switch.fan
        for entity in result:
            assert entity["state"] == "on"

    def test_filter_entities_by_attribute(
        self, entity_manager, mock_hass, sample_states
    ):
        """Test filtering entities by attribute existence and value."""
        mock_hass.states.async_all.return_value = sample_states

        result = entity_manager.filter_entities(
            attribute="device_class", value="temperature"
        )

        assert len(result) == 1
        assert result[0]["entity_id"] == "sensor.temperature"

    def test_filter_entities_by_domain_and_state(
        self, entity_manager, mock_hass, sample_states
    ):
        """Test filtering entities by both domain and state."""
        mock_hass.states.async_all.return_value = sample_states

        result = entity_manager.filter_entities(domain="light", state="on")

        assert len(result) == 1
        assert result[0]["entity_id"] == "light.living_room"

    def test_filter_entities_no_filters_returns_all(
        self, entity_manager, mock_hass, sample_states
    ):
        """Test that no filters returns all entities."""
        mock_hass.states.async_all.return_value = sample_states

        result = entity_manager.filter_entities()

        assert len(result) == len(sample_states)


class TestGetEntityByFriendlyName:
    """Tests for get_entity_by_friendly_name method."""

    def test_get_entity_by_friendly_name_found(
        self, entity_manager, mock_hass, sample_states
    ):
        """Test finding entity by friendly name."""
        mock_hass.states.async_all.return_value = sample_states

        result = entity_manager.get_entity_by_friendly_name("Living Room Light")

        assert result is not None
        assert result["entity_id"] == "light.living_room"

    def test_get_entity_by_friendly_name_not_found(
        self, entity_manager, mock_hass, sample_states
    ):
        """Test that non-existent friendly name returns None."""
        mock_hass.states.async_all.return_value = sample_states

        result = entity_manager.get_entity_by_friendly_name("Non-existent Entity")

        assert result is None

    def test_get_entity_by_friendly_name_case_insensitive(
        self, entity_manager, mock_hass, sample_states
    ):
        """Test that friendly name search is case-insensitive."""
        mock_hass.states.async_all.return_value = sample_states

        result = entity_manager.get_entity_by_friendly_name("living room light")

        assert result is not None
        assert result["entity_id"] == "light.living_room"

    def test_get_entity_by_friendly_name_partial_match(
        self, entity_manager, mock_hass, sample_states
    ):
        """Test that partial matches do not return results (exact match only)."""
        mock_hass.states.async_all.return_value = sample_states

        # "Living Room" is only part of "Living Room Light"
        result = entity_manager.get_entity_by_friendly_name("Living Room")

        assert result is None


class TestEntityManagerInitialization:
    """Tests for EntityManager initialization."""

    def test_init_stores_hass(self, mock_hass):
        """Test that EntityManager stores hass reference."""
        manager = EntityManager(mock_hass)

        assert manager.hass is mock_hass

    def test_init_with_none_hass_raises(self):
        """Test that EntityManager raises error with None hass."""
        with pytest.raises(ValueError, match="hass is required"):
            EntityManager(None)
