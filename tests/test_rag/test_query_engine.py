"""Tests for query engine."""

import pytest
from unittest.mock import AsyncMock, MagicMock

# Note: homeassistant mocks are set up in conftest.py

from custom_components.ai_agent_ha.rag.chroma_store import SearchResult
from custom_components.ai_agent_ha.rag.query_engine import (
    QueryEngine,
    MAX_CONTEXT_LENGTH,
)


class TestQueryEngine:
    """Tests for QueryEngine class."""

    @pytest.fixture
    def mock_store(self):
        """Return a mock ChromaStore."""
        store = MagicMock()
        store.search = AsyncMock(return_value=[])
        return store

    @pytest.fixture
    def mock_embedding_provider(self):
        """Return a mock embedding provider."""
        provider = MagicMock()
        provider.get_embeddings = AsyncMock(return_value=[[0.1] * 768])
        return provider

    @pytest.fixture
    def query_engine(self, mock_store, mock_embedding_provider):
        """Return a QueryEngine instance."""
        return QueryEngine(
            store=mock_store,
            embedding_provider=mock_embedding_provider,
        )

    @pytest.fixture
    def sample_results(self):
        """Return sample search results."""
        return [
            SearchResult(
                id="light.bedroom_lamp",
                text="Bedroom Lamp light in Bedroom",
                metadata={
                    "domain": "light",
                    "friendly_name": "Bedroom Lamp",
                    "area_name": "Bedroom",
                    "state": "on",
                },
                distance=0.1,
            ),
            SearchResult(
                id="switch.kitchen_outlet",
                text="Kitchen Outlet switch in Kitchen",
                metadata={
                    "domain": "switch",
                    "friendly_name": "Kitchen Outlet",
                    "area_name": "Kitchen",
                    "device_class": "outlet",
                    "state": "off",
                },
                distance=0.3,
            ),
            SearchResult(
                id="sensor.temperature",
                text="Living Room Temperature sensor",
                metadata={
                    "domain": "sensor",
                    "friendly_name": "Living Room Temperature",
                    "device_class": "temperature",
                    "state": "22.5",
                },
                distance=0.5,
            ),
        ]

    @pytest.mark.asyncio
    async def test_search_entities(
        self, query_engine, mock_store, mock_embedding_provider, sample_results
    ):
        """Test searching for entities."""
        mock_store.search.return_value = sample_results

        results = await query_engine.search_entities("bedroom light", top_k=10)

        assert len(results) == 3
        mock_embedding_provider.get_embeddings.assert_called_once_with(["bedroom light"])
        mock_store.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_entities_with_domain_filter(
        self, query_engine, mock_store, mock_embedding_provider
    ):
        """Test searching with domain filter."""
        await query_engine.search_entities("light", domain_filter="light")

        # Verify filter was passed
        call_args = mock_store.search.call_args
        assert call_args[1]["where"] == {"domain": "light"}

    @pytest.mark.asyncio
    async def test_search_entities_error_handling(
        self, query_engine, mock_embedding_provider
    ):
        """Test that errors return empty results."""
        mock_embedding_provider.get_embeddings = AsyncMock(side_effect=Exception("API error"))

        results = await query_engine.search_entities("test query")

        assert results == []

    def test_build_compressed_context(self, query_engine, sample_results):
        """Test building compressed context."""
        context = query_engine.build_compressed_context(sample_results)

        assert "Potentially relevant entities" in context
        assert "light.bedroom_lamp" in context
        assert "switch.kitchen_outlet" in context
        assert "Bedroom" in context  # Area should be included
        assert "on" in context  # State for actionable entities

    def test_build_compressed_context_empty(self, query_engine):
        """Test building context with no results."""
        context = query_engine.build_compressed_context([])
        assert context == ""

    def test_build_compressed_context_max_length(self, query_engine, sample_results):
        """Test that context respects max length."""
        # Use very short max length
        context = query_engine.build_compressed_context(sample_results, max_length=100)

        assert len(context) <= 100

    def test_format_entity_basic(self, query_engine):
        """Test basic entity formatting."""
        result = SearchResult(
            id="light.test",
            text="Test Light",
            metadata={"domain": "light"},
            distance=0.1,
        )

        formatted = query_engine._format_entity(result)

        assert "light.test" in formatted
        assert "(light)" in formatted

    def test_format_entity_with_friendly_name(self, query_engine):
        """Test formatting with friendly name."""
        result = SearchResult(
            id="light.test",
            text="Test Light",
            metadata={"domain": "light", "friendly_name": "My Special Light"},
            distance=0.1,
        )

        formatted = query_engine._format_entity(result)

        assert '"My Special Light"' in formatted

    def test_format_entity_with_area(self, query_engine):
        """Test formatting with area."""
        result = SearchResult(
            id="light.test",
            text="Test Light",
            metadata={"domain": "light", "area_name": "Bedroom"},
            distance=0.1,
        )

        formatted = query_engine._format_entity(result)

        assert "in Bedroom" in formatted

    def test_format_entity_with_learned_category(self, query_engine):
        """Test formatting with learned category."""
        result = SearchResult(
            id="switch.lamp",
            text="Lamp Switch",
            metadata={"domain": "switch", "learned_category": "light"},
            distance=0.1,
        )

        formatted = query_engine._format_entity(result)

        assert "<light>" in formatted

    def test_format_entity_with_state(self, query_engine):
        """Test formatting actionable entity with state."""
        result = SearchResult(
            id="light.test",
            text="Test Light",
            metadata={"domain": "light", "state": "on"},
            distance=0.1,
        )

        formatted = query_engine._format_entity(result)

        assert "state:on" in formatted

    def test_format_entity_sensor_no_state(self, query_engine):
        """Test that sensors don't show state in compact format."""
        result = SearchResult(
            id="sensor.test",
            text="Test Sensor",
            metadata={"domain": "sensor", "state": "22.5"},
            distance=0.1,
        )

        formatted = query_engine._format_entity(result)

        # Sensors don't show state in compact format (not actionable)
        assert "state:" not in formatted

    @pytest.mark.asyncio
    async def test_search_and_format(
        self, query_engine, mock_store, sample_results
    ):
        """Test search and format convenience method."""
        mock_store.search.return_value = sample_results

        context = await query_engine.search_and_format("bedroom light")

        assert "Potentially relevant entities" in context
        assert "light.bedroom_lamp" in context

    @pytest.mark.asyncio
    async def test_search_by_criteria(
        self, query_engine, mock_store, mock_embedding_provider
    ):
        """Test searching with multiple criteria."""
        await query_engine.search_by_criteria(
            query="temperature",
            domain="sensor",
            area="Living Room",
            device_class="temperature",
        )

        call_args = mock_store.search.call_args
        where = call_args[1]["where"]
        assert where["domain"] == "sensor"
        assert where["area_name"] == "Living Room"
        assert where["device_class"] == "temperature"


class TestQueryIntentExtraction:
    """Tests for query intent extraction."""

    @pytest.fixture
    def query_engine(self):
        """Return a QueryEngine instance."""
        return QueryEngine(
            store=MagicMock(),
            embedding_provider=MagicMock(),
        )

    def test_extract_domain_light(self, query_engine):
        """Test extracting light domain."""
        intent = query_engine.extract_query_intent("turn on the bedroom light")
        assert intent.get("domain") == "light"

    def test_extract_domain_switch(self, query_engine):
        """Test extracting switch domain."""
        intent = query_engine.extract_query_intent("check the kitchen outlet")
        assert intent.get("domain") == "switch"

    def test_extract_domain_sensor(self, query_engine):
        """Test extracting sensor domain."""
        intent = query_engine.extract_query_intent("what's the temperature")
        assert intent.get("domain") == "sensor"

    def test_extract_device_class_temperature(self, query_engine):
        """Test extracting temperature device class."""
        intent = query_engine.extract_query_intent("show me temperature sensors")
        assert intent.get("device_class") == "temperature"

    def test_extract_device_class_motion(self, query_engine):
        """Test extracting motion device class."""
        intent = query_engine.extract_query_intent("any motion detected?")
        assert intent.get("device_class") == "motion"

    def test_extract_area_bedroom(self, query_engine):
        """Test extracting bedroom area."""
        intent = query_engine.extract_query_intent("lights in the bedroom")
        assert intent.get("area") == "bedroom"

    def test_extract_area_living_room(self, query_engine):
        """Test extracting living room area."""
        intent = query_engine.extract_query_intent("turn off living room lights")
        assert intent.get("area") == "living room"

    def test_extract_multiple_intents(self, query_engine):
        """Test extracting multiple intents."""
        intent = query_engine.extract_query_intent("temperature in the bedroom")
        assert intent.get("device_class") == "temperature"
        assert intent.get("area") == "bedroom"

    def test_extract_no_intent(self, query_engine):
        """Test query with no extractable intent."""
        intent = query_engine.extract_query_intent("hello there")
        assert intent == {}
