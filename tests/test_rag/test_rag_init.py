"Unit tests for the RAGManager facade."
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from homeassistant.core import HomeAssistant
from custom_components.ai_agent_ha.rag import RAGManager


@pytest.fixture
def mock_hass(hass):
    """Fixture for HomeAssistant."""
    hass.config.path = MagicMock(side_effect=lambda *args: "/".join(args))
    return hass


@pytest.fixture
def mock_dependencies():
    """Mock all internal RAG dependencies."""
    with patch("custom_components.ai_agent_ha.rag.sqlite_store.SqliteStore") as mock_store_cls, \
         patch("custom_components.ai_agent_ha.rag.embeddings.create_embedding_provider") as mock_create_emb, \
         patch("custom_components.ai_agent_ha.rag.entity_indexer.EntityIndexer") as mock_indexer_cls, \
         patch("custom_components.ai_agent_ha.rag.query_engine.QueryEngine") as mock_query_cls, \
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
        mock_query.build_compressed_context = MagicMock(return_value="context")

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
            "learner_cls": mock_learner_cls,
            "learner": mock_learner,
            "handler_cls": mock_handler_cls,
            "handler": mock_handler,
        }


def test_get_persist_directory(mock_hass):
    """Test _get_persist_directory returns correct path."""
    rag = RAGManager(mock_hass, {})
    path = rag._get_persist_directory()
    assert path == "ai_agent_ha/rag_db"
    mock_hass.config.path.assert_called_with("ai_agent_ha", "rag_db")


@pytest.mark.asyncio
async def test_async_initialize_success(mock_hass, mock_dependencies):
    """Test successful initialization of all components."""
    rag = RAGManager(mock_hass, {})
    
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
async def test_async_initialize_already_initialized(mock_hass, mock_dependencies):
    """Test initialization is skipped if already initialized."""
    rag = RAGManager(mock_hass, {})
    rag._initialized = True
    
    await rag.async_initialize()
    
    mock_dependencies["store_cls"].assert_not_called()


@pytest.mark.asyncio
async def test_async_initialize_empty_store_reindex(mock_hass, mock_dependencies):
    """Test full reindex is triggered if store is empty."""
    mock_dependencies["store"].get_document_count.return_value = 0
    rag = RAGManager(mock_hass, {})
    
    await rag.async_initialize()
    
    mock_dependencies["indexer"].full_reindex.assert_called_once()


@pytest.mark.asyncio
async def test_async_initialize_exception(mock_hass, mock_dependencies):
    """Test exception during initialization is logged and raised."""
    mock_dependencies["store"].async_initialize.side_effect = Exception("DB Error")
    rag = RAGManager(mock_hass, {})
    
    with pytest.raises(Exception, match="DB Error"):
        await rag.async_initialize()
        
    assert not rag.is_initialized


def test_ensure_initialized_raises(mock_hass):
    """Test _ensure_initialized raises RuntimeError if not initialized."""
    rag = RAGManager(mock_hass, {})
    
    with pytest.raises(RuntimeError, match="not initialized"):
        rag._ensure_initialized()


@pytest.mark.asyncio
async def test_get_relevant_context_success(mock_hass, mock_dependencies):
    """Test fetching relevant context successfully."""
    rag = RAGManager(mock_hass, {})
    await rag.async_initialize()
    
    # Setup mock results
    mock_result = MagicMock()
    mock_result.id = "light.living_room"
    mock_dependencies["query"].search_entities.return_value = [mock_result]
    
    # Set HA state to simulate entity exists
    mock_hass.states.async_set("light.living_room", "on")
    
    context = await rag.get_relevant_context("turn on light")
    
    assert context == "context"
    mock_dependencies["query"].search_entities.assert_called_with(
        query="turn on light", top_k=10
    )
    mock_dependencies["query"].build_compressed_context.assert_called_with([mock_result])


@pytest.mark.asyncio
async def test_get_relevant_context_stale_entity(mock_hass, mock_dependencies):
    """Test stale entities are removed from index."""
    rag = RAGManager(mock_hass, {})
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
async def test_get_relevant_context_exception(mock_hass, mock_dependencies):
    """Test graceful failure when getting context."""
    rag = RAGManager(mock_hass, {})
    await rag.async_initialize()
    
    mock_dependencies["query"].search_entities.side_effect = Exception("Search failed")
    
    context = await rag.get_relevant_context("query")
    
    assert context == ""


@pytest.mark.asyncio
async def test_learn_from_conversation_success(mock_hass, mock_dependencies):
    """Test learning from conversation delegates to learner."""
    rag = RAGManager(mock_hass, {})
    await rag.async_initialize()
    
    await rag.learn_from_conversation("user msg", "assistant msg")
    
    mock_dependencies["learner"].detect_and_persist.assert_called_with(
        user_message="user msg",
        assistant_message="assistant msg"
    )


@pytest.mark.asyncio
async def test_learn_from_conversation_exception(mock_hass, mock_dependencies):
    """Test learning exception doesn't crash."""
    rag = RAGManager(mock_hass, {})
    await rag.async_initialize()
    
    mock_dependencies["learner"].detect_and_persist.side_effect = Exception("Learn error")
    
    # Should not raise
    await rag.learn_from_conversation("user", "ai")


@pytest.mark.asyncio
async def test_reindex_entity(mock_hass, mock_dependencies):
    """Test reindexing a single entity."""
    rag = RAGManager(mock_hass, {})
    await rag.async_initialize()
    
    await rag.reindex_entity("switch.test")
    
    mock_dependencies["indexer"].index_entity.assert_called_with("switch.test")


@pytest.mark.asyncio
async def test_remove_entity(mock_hass, mock_dependencies):
    """Test removing a single entity."""
    rag = RAGManager(mock_hass, {})
    await rag.async_initialize()
    
    await rag.remove_entity("switch.test")
    
    mock_dependencies["indexer"].remove_entity.assert_called_with("switch.test")


@pytest.mark.asyncio
async def test_full_reindex(mock_hass, mock_dependencies):
    """Test triggering a full reindex."""
    rag = RAGManager(mock_hass, {})
    await rag.async_initialize()
    
    await rag.full_reindex()
    
    # Called once during init (if count=0) and once explicitly here?
    # In this test fixture count=10, so init doesn't call it.
    mock_dependencies["indexer"].full_reindex.assert_called_once()


@pytest.mark.asyncio
async def test_get_stats(mock_hass, mock_dependencies):
    """Test retrieving statistics."""
    rag = RAGManager(mock_hass, {})
    await rag.async_initialize()
    
    stats = await rag.get_stats()
    
    assert stats["indexed_entities"] == 10
    assert stats["embedding_provider"] == "test_provider"
    assert stats["embedding_dimension"] == 1536
    assert stats["learned_categories"] == 1  # mocked dict has 1 item


@pytest.mark.asyncio
async def test_async_shutdown_success(mock_hass, mock_dependencies):
    """Test successful shutdown sequence."""
    rag = RAGManager(mock_hass, {})
    await rag.async_initialize()
    
    await rag.async_shutdown()
    
    assert not rag.is_initialized
    mock_dependencies["handler"].async_stop.assert_called_once()
    mock_dependencies["learner"].async_save.assert_called_once()
    mock_dependencies["store"].async_shutdown.assert_called_once()


@pytest.mark.asyncio
async def test_async_shutdown_exception(mock_hass, mock_dependencies):
    """Test exception during shutdown is raised."""
    rag = RAGManager(mock_hass, {})
    await rag.async_initialize()
    
    mock_dependencies["store"].async_shutdown.side_effect = Exception("Shutdown error")
    
    with pytest.raises(Exception, match="Shutdown error"):
        await rag.async_shutdown()
