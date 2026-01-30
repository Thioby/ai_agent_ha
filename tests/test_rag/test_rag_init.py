"Unit tests for the RAGManager facade."
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from homeassistant.core import HomeAssistant
from custom_components.ai_agent_ha.rag import RAGManager


@pytest.fixture
def mock_dependencies():
    """Mock all internal RAG dependencies."""
    with patch("custom_components.ai_agent_ha.rag.sqlite_store.SqliteStore") as mock_store_cls, \
         patch("custom_components.ai_agent_ha.rag.embeddings.create_embedding_provider") as mock_create_emb, \
         patch("custom_components.ai_agent_ha.rag.entity_indexer.EntityIndexer") as mock_indexer_cls, \
         patch("custom_components.ai_agent_ha.rag.query_engine.QueryEngine") as mock_query_cls, \
         patch("custom_components.ai_agent_ha.rag.intent_detector.IntentDetector") as mock_intent_cls, \
         patch("custom_components.ai_agent_ha.rag.semantic_learner.SemanticLearner") as mock_learner_cls, \
         patch("custom_components.ai_agent_ha.rag.event_handlers.EntityRegistryEventHandler") as mock_handler_cls:

        # Setup mocks
        mock_store = mock_store_cls.return_value
        mock_store.async_initialize = AsyncMock()
        mock_store.get_document_count = AsyncMock(return_value=10)
        mock_store.async_shutdown = AsyncMock()

        mock_emb = MagicMock()
        mock_emb.provider_name = "test_provider"
        mock_emb.dimension = 1536
        mock_create_emb.return_value = mock_emb

        mock_indexer = mock_indexer_cls.return_value
        mock_indexer.full_reindex = AsyncMock()
        mock_indexer.index_entity = AsyncMock()
        mock_indexer.remove_entity = AsyncMock()

        mock_query = mock_query_cls.return_value
        mock_query.search_entities = AsyncMock(return_value=[])
        mock_query.search_by_criteria = AsyncMock(return_value=[])
        mock_query.build_compressed_context = MagicMock(return_value="context")

        mock_intent = mock_intent_cls.return_value
        mock_intent.async_initialize = AsyncMock()
        mock_intent.detect_intent = AsyncMock(return_value={})  # No intent by default

        mock_learner = mock_learner_cls.return_value
        mock_learner.async_load = AsyncMock()
        mock_learner.detect_and_persist = AsyncMock()
        mock_learner.async_save = AsyncMock()
        mock_learner.categories = {"test": "category"}

        mock_handler = mock_handler_cls.return_value
        mock_handler.async_start = AsyncMock()
        mock_handler.async_stop = AsyncMock()

        yield {
            "store_cls": mock_store_cls,
            "store": mock_store,
            "create_emb": mock_create_emb,
            "emb": mock_emb,
            "indexer_cls": mock_indexer_cls,
            "indexer": mock_indexer,
            "query_cls": mock_query_cls,
            "query": mock_query,
            "intent_cls": mock_intent_cls,
            "intent": mock_intent,
            "learner_cls": mock_learner_cls,
            "learner": mock_learner,
            "handler_cls": mock_handler_cls,
            "handler": mock_handler,
        }


def test_get_persist_directory(hass):
    """Test _get_persist_directory returns correct path."""
    rag = RAGManager(hass, {})
    path = rag._get_persist_directory()
    assert path.endswith("ai_agent_ha/rag_db")


@pytest.mark.asyncio
async def test_async_initialize_success(hass, mock_dependencies):
    """Test successful initialization of all components."""
    rag = RAGManager(hass, {})
    
    await rag.async_initialize()
    
    assert rag.is_initialized
    
    # Verify components initialization
    mock_dependencies["store"].async_initialize.assert_called_once()
    mock_dependencies["create_emb"].assert_called_once()
    mock_dependencies["indexer_cls"].assert_called_once()
    mock_dependencies["query_cls"].assert_called_once()
    mock_dependencies["learner"].async_load.assert_called_once()
    mock_dependencies["handler"].async_start.assert_called_once()
    
    # Verify we checked for reindex but didn't run it (doc_count=10 by default in fixture)
    mock_dependencies["store"].get_document_count.assert_called_once()
    mock_dependencies["indexer"].full_reindex.assert_not_called()


@pytest.mark.asyncio
async def test_async_initialize_already_initialized(hass, mock_dependencies):
    """Test initialization is skipped if already initialized."""
    rag = RAGManager(hass, {})
    rag._initialized = True
    
    await rag.async_initialize()
    
    mock_dependencies["store_cls"].assert_not_called()


@pytest.mark.asyncio
async def test_async_initialize_empty_store_reindex(hass, mock_dependencies):
    """Test full reindex is triggered if store is empty."""
    mock_dependencies["store"].get_document_count.return_value = 0
    rag = RAGManager(hass, {})
    
    await rag.async_initialize()
    
    mock_dependencies["indexer"].full_reindex.assert_called_once()


@pytest.mark.asyncio
async def test_async_initialize_exception(hass, mock_dependencies):
    """Test exception during initialization is logged and raised."""
    mock_dependencies["store"].async_initialize.side_effect = Exception("DB Error")
    rag = RAGManager(hass, {})
    
    with pytest.raises(Exception, match="DB Error"):
        await rag.async_initialize()
        
    assert not rag.is_initialized


def test_ensure_initialized_raises(hass):
    """Test _ensure_initialized raises RuntimeError if not initialized."""
    rag = RAGManager(hass, {})
    
    with pytest.raises(RuntimeError, match="not initialized"):
        rag._ensure_initialized()


@pytest.mark.asyncio
async def test_get_relevant_context_success(hass, mock_dependencies):
    """Test fetching relevant context successfully."""
    rag = RAGManager(hass, {})
    await rag.async_initialize()
    
    # Setup mock results
    mock_result = MagicMock()
    mock_result.id = "light.living_room"
    mock_dependencies["query"].search_entities.return_value = [mock_result]
    
    # Set HA state to simulate entity exists
    hass.states.async_set("light.living_room", "on")
    
    context = await rag.get_relevant_context("turn on light")
    
    assert context == "context"
    mock_dependencies["query"].search_entities.assert_called_with(
        query="turn on light", top_k=10
    )
    mock_dependencies["query"].build_compressed_context.assert_called_with([mock_result])


@pytest.mark.asyncio
async def test_get_relevant_context_stale_entity(hass, mock_dependencies):
    """Test stale entities are removed from index."""
    rag = RAGManager(hass, {})
    await rag.async_initialize()
    
    # Setup mock result
    mock_result = MagicMock()
    mock_result.id = "light.removed"
    mock_dependencies["query"].search_entities.return_value = [mock_result]
    
    # Do not set HA state -> simulates entity MISSING (get returns None)
    
    context = await rag.get_relevant_context("turn on light")
    
    # Should attempt to remove the stale entity
    mock_dependencies["indexer"].remove_entity.assert_called_with("light.removed")
    
    # Should build context with empty list (since the only result was stale)
    mock_dependencies["query"].build_compressed_context.assert_called_with([])


@pytest.mark.asyncio
async def test_get_relevant_context_exception(hass, mock_dependencies):
    """Test graceful failure when getting context."""
    rag = RAGManager(hass, {})
    await rag.async_initialize()
    
    mock_dependencies["query"].search_entities.side_effect = Exception("Search failed")
    
    context = await rag.get_relevant_context("query")

    assert context == ""


@pytest.mark.asyncio
async def test_get_relevant_context_with_intent(hass, mock_dependencies):
    """Test that intent-based filtering uses search_by_criteria."""
    rag = RAGManager(hass, {})
    await rag.async_initialize()

    # Setup intent detection to return temperature filter (semantic detection)
    mock_dependencies["intent"].detect_intent.return_value = {
        "device_class": "temperature"
    }

    mock_result = MagicMock()
    mock_result.id = "sensor.temperature"
    mock_dependencies["query"].search_by_criteria.return_value = [mock_result]

    # Set HA state
    hass.states.async_set("sensor.temperature", "22.5")

    context = await rag.get_relevant_context("what is the temperature")

    # Should use search_by_criteria when intent detected
    mock_dependencies["query"].search_by_criteria.assert_called_with(
        query="what is the temperature",
        domain=None,
        device_class="temperature",
        area=None,
        top_k=10,
    )
    # Should NOT call search_entities (because search_by_criteria returned results)
    mock_dependencies["query"].search_entities.assert_not_called()


@pytest.mark.asyncio
async def test_get_relevant_context_intent_fallback(hass, mock_dependencies):
    """Test that filtered search falls back to semantic search when no results."""
    rag = RAGManager(hass, {})
    await rag.async_initialize()

    # Setup intent detection
    mock_dependencies["intent"].detect_intent.return_value = {
        "domain": "media_player",
        "area": "living_room"
    }

    # search_by_criteria returns empty (no match for filters)
    mock_dependencies["query"].search_by_criteria.return_value = []

    # search_entities returns results (fallback)
    mock_result = MagicMock()
    mock_result.id = "media_player.tv"
    mock_dependencies["query"].search_entities.return_value = [mock_result]

    # Set HA state
    hass.states.async_set("media_player.tv", "off")

    context = await rag.get_relevant_context("tell me about the TV")

    # Should first try search_by_criteria
    mock_dependencies["query"].search_by_criteria.assert_called_once()
    # Then fall back to search_entities
    mock_dependencies["query"].search_entities.assert_called_with(
        query="tell me about the TV",
        top_k=10,
    )
    # Context should be built from fallback results
    mock_dependencies["query"].build_compressed_context.assert_called_with([mock_result])


@pytest.mark.asyncio
async def test_get_relevant_context_domain_priority_over_device_class(hass, mock_dependencies):
    """Test that domain filter takes priority over device_class (to avoid conflicts)."""
    rag = RAGManager(hass, {})
    await rag.async_initialize()

    # Intent detection returns both domain and device_class (conflicting)
    mock_dependencies["intent"].detect_intent.return_value = {
        "domain": "media_player",
        "device_class": "motion",  # Wrong - should be ignored
        "area": "living_room"
    }

    mock_result = MagicMock()
    mock_result.id = "media_player.tv"
    mock_dependencies["query"].search_by_criteria.return_value = [mock_result]

    # Set HA state
    hass.states.async_set("media_player.tv", "off")

    await rag.get_relevant_context("what about the TV")

    # Should use domain but NOT device_class (domain takes priority)
    mock_dependencies["query"].search_by_criteria.assert_called_with(
        query="what about the TV",
        domain="media_player",
        device_class=None,  # Ignored because domain is set
        area="living_room",
        top_k=10,
    )


@pytest.mark.asyncio
async def test_learn_from_conversation_success(hass, mock_dependencies):
    """Test learning from conversation delegates to learner."""
    rag = RAGManager(hass, {})
    await rag.async_initialize()
    
    await rag.learn_from_conversation("user msg", "assistant msg")
    
    mock_dependencies["learner"].detect_and_persist.assert_called_with(
        user_message="user msg",
        assistant_message="assistant msg"
    )


@pytest.mark.asyncio
async def test_learn_from_conversation_exception(hass, mock_dependencies):
    """Test learning exception doesn't crash."""
    rag = RAGManager(hass, {})
    await rag.async_initialize()
    
    mock_dependencies["learner"].detect_and_persist.side_effect = Exception("Learn error")
    
    # Should not raise
    await rag.learn_from_conversation("user", "ai")


@pytest.mark.asyncio
async def test_reindex_entity(hass, mock_dependencies):
    """Test reindexing a single entity."""
    rag = RAGManager(hass, {})
    await rag.async_initialize()
    
    await rag.reindex_entity("switch.test")
    
    mock_dependencies["indexer"].index_entity.assert_called_with("switch.test")


@pytest.mark.asyncio
async def test_remove_entity(hass, mock_dependencies):
    """Test removing a single entity."""
    rag = RAGManager(hass, {})
    await rag.async_initialize()
    
    await rag.remove_entity("switch.test")
    
    mock_dependencies["indexer"].remove_entity.assert_called_with("switch.test")


@pytest.mark.asyncio
async def test_full_reindex(hass, mock_dependencies):
    """Test triggering a full reindex."""
    rag = RAGManager(hass, {})
    await rag.async_initialize()
    
    await rag.full_reindex()
    
    # Called once during init (if count=0) and once explicitly here?
    # In this test fixture count=10, so init doesn't call it.
    mock_dependencies["indexer"].full_reindex.assert_called_once()


@pytest.mark.asyncio
async def test_get_stats(hass, mock_dependencies):
    """Test retrieving statistics."""
    rag = RAGManager(hass, {})
    await rag.async_initialize()
    
    stats = await rag.get_stats()
    
    assert stats["indexed_entities"] == 10
    assert stats["embedding_provider"] == "test_provider"
    assert stats["embedding_dimension"] == 1536
    assert stats["learned_categories"] == 1  # mocked dict has 1 item


@pytest.mark.asyncio
async def test_async_shutdown_success(hass, mock_dependencies):
    """Test successful shutdown sequence."""
    rag = RAGManager(hass, {})
    await rag.async_initialize()
    
    await rag.async_shutdown()
    
    assert not rag.is_initialized
    mock_dependencies["handler"].async_stop.assert_called_once()
    mock_dependencies["learner"].async_save.assert_called_once()
    mock_dependencies["store"].async_shutdown.assert_called_once()


@pytest.mark.asyncio
async def test_async_shutdown_exception(hass, mock_dependencies):
    """Test exception during shutdown is raised."""
    rag = RAGManager(hass, {})
    await rag.async_initialize()
    
    mock_dependencies["store"].async_shutdown.side_effect = Exception("Shutdown error")
    
    with pytest.raises(Exception, match="Shutdown error"):
        await rag.async_shutdown()
