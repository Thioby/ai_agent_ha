"""Tests for the ChromaDB store wrapper."""

import json
import logging
import sys
from unittest.mock import MagicMock, patch

import pytest

from custom_components.ai_agent_ha.rag.chroma_store import (
    DEFAULT_COLLECTION_NAME,
    ChromaStore,
    SearchResult,
)


@pytest.fixture
def mock_chromadb():
    """Mock the chromadb module."""
    mock = MagicMock()
    mock.PersistentClient = MagicMock()
    # Create a separate mock for Settings class
    mock_settings_cls = MagicMock()
    
    # Create a module mock that has both Settings (class) and the rest of chromadb
    module_mock = MagicMock()
    module_mock.PersistentClient = mock.PersistentClient
    
    # We need to structure the mocks so that 'chromadb' and 'chromadb.config' are available
    return module_mock, mock_settings_cls


@pytest.fixture
def store():
    """Create a ChromaStore instance."""
    return ChromaStore(persist_directory="/tmp/chroma_test")


def test_search_result_init():
    """Test SearchResult initialization."""
    result = SearchResult(
        id="test_id",
        text="test text",
        metadata={"key": "value"},
        distance=0.5,
    )
    assert result.id == "test_id"
    assert result.text == "test text"
    assert result.metadata == {"key": "value"}
    assert result.distance == 0.5


@pytest.mark.asyncio
async def test_async_initialize(store, mock_chromadb):
    """Test initialization of ChromaDB client."""
    mock_db, mock_settings_cls = mock_chromadb
    mock_collection = MagicMock()
    mock_client = mock_db.PersistentClient.return_value
    mock_client.get_or_create_collection.return_value = mock_collection

    # Mocking sys.modules to simulate successful import
    with patch.dict(sys.modules, {"chromadb": mock_db, "chromadb.config": MagicMock(Settings=mock_settings_cls)}):
        await store.async_initialize()

    assert store._initialized
    assert store._client == mock_client
    assert store._collection == mock_collection
    
    mock_db.PersistentClient.assert_called_once()
    mock_client.get_or_create_collection.assert_called_once_with(
        name=DEFAULT_COLLECTION_NAME,
        metadata={"description": "Home Assistant entity embeddings"},
    )


@pytest.mark.asyncio
async def test_async_initialize_already_initialized(store):
    """Test initialization skip if already initialized."""
    store._initialized = True
    
    # Should not attempt to import or create client
    with patch.dict(sys.modules, {}):
        await store.async_initialize()


@pytest.mark.asyncio
async def test_async_initialize_import_error(store):
    """Test initialization failure when chromadb is missing."""
    # Simulate ImportError
    with patch.dict(sys.modules, {}):
        # We need to ensure 'chromadb' is NOT in sys.modules and force import error
        # patch.dict(sys.modules) replaces the dict, but keys might still be found if not carefully removed or if we mock import.
        # However, builtins.__import__ is safer here.
        with patch("builtins.__import__", side_effect=ImportError("No module named 'chromadb'")):
            with pytest.raises(ImportError):
                await store.async_initialize()


@pytest.mark.asyncio
async def test_ensure_initialized_error(store):
    """Test error when accessing uninitialized store."""
    with pytest.raises(RuntimeError, match="not initialized"):
        store._ensure_initialized()


@pytest.mark.asyncio
async def test_add_documents(store):
    """Test adding documents."""
    store._initialized = True
    store._collection = MagicMock()

    ids = ["1", "2"]
    texts = ["a", "b"]
    embeddings = [[0.1], [0.2]]
    metadatas = [{"k": "v"}, {"k": "v2"}]

    await store.add_documents(ids, texts, embeddings, metadatas)

    store._collection.add.assert_called_once_with(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )


@pytest.mark.asyncio
async def test_add_documents_empty(store):
    """Test adding empty list of documents."""
    store._initialized = True
    store._collection = MagicMock()

    await store.add_documents([], [], [])

    store._collection.add.assert_not_called()


@pytest.mark.asyncio
async def test_add_documents_metadata_filtering(store):
    """Test metadata filtering during addition."""
    store._initialized = True
    store._collection = MagicMock()

    ids = ["1"]
    texts = ["a"]
    embeddings = [[0.1]]
    # Mixed types in metadata
    metadatas = [{
        "str": "val",
        "int": 1,
        "float": 1.1,
        "bool": True,
        "none": None,
        "list": [1, 2],
        "dict": {"a": 1},
        "obj": object(), # Should be converted to str
    }]

    await store.add_documents(ids, texts, embeddings, metadatas)

    call_args = store._collection.add.call_args
    assert call_args is not None
    
    passed_metadatas = call_args.kwargs["metadatas"]
    assert len(passed_metadatas) == 1
    m = passed_metadatas[0]
    
    assert m["str"] == "val"
    assert m["int"] == 1
    assert m["float"] == 1.1
    assert m["bool"] is True
    assert "none" not in m
    assert m["list"] == json.dumps([1, 2])
    assert m["dict"] == json.dumps({"a": 1})
    assert isinstance(m["obj"], str)


@pytest.mark.asyncio
async def test_upsert_documents(store):
    """Test upserting documents."""
    store._initialized = True
    store._collection = MagicMock()

    ids = ["1"]
    texts = ["a"]
    embeddings = [[0.1]]

    await store.upsert_documents(ids, texts, embeddings)

    store._collection.upsert.assert_called_once_with(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=None,
    )


@pytest.mark.asyncio
async def test_upsert_documents_empty(store):
    """Test upserting empty list."""
    store._initialized = True
    store._collection = MagicMock()

    await store.upsert_documents([], [], [])

    store._collection.upsert.assert_not_called()


@pytest.mark.asyncio
async def test_filter_metadata_direct(store):
    """Direct test for _filter_metadata."""
    metadata = {
        "simple": "value",
        "nested": {"key": "val"},
        "invalid": None
    }
    filtered = store._filter_metadata(metadata)
    
    assert filtered["simple"] == "value"
    assert filtered["nested"] == '{"key": "val"}'
    assert "invalid" not in filtered


@pytest.mark.asyncio
async def test_search(store):
    """Test search functionality."""
    store._initialized = True
    store._collection = MagicMock()

    # Mock response from ChromaDB
    store._collection.query.return_value = {
        "ids": [["1", "2"]],
        "documents": [["text1", "text2"]],
        "metadatas": [[{"m": 1}, {"m": 2}]],
        "distances": [[0.1, 0.2]],
    }

    query_embedding = [0.1, 0.1]
    results = await store.search(query_embedding)

    assert len(results) == 2
    assert results[0].id == "1"
    assert results[0].text == "text1"
    assert results[0].distance == 0.1
    assert results[1].id == "2"

    store._collection.query.assert_called_once()
    args = store._collection.query.call_args.kwargs
    assert args["query_embeddings"] == [query_embedding]


@pytest.mark.asyncio
async def test_search_with_where(store):
    """Test search with filter."""
    store._initialized = True
    store._collection = MagicMock()
    
    store._collection.query.return_value = {
        "ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]
    }

    where = {"field": "value"}
    await store.search([0.1], where=where)

    args = store._collection.query.call_args.kwargs
    assert args["where"] == where


@pytest.mark.asyncio
async def test_search_no_results(store):
    """Test search with no results."""
    store._initialized = True
    store._collection = MagicMock()

    store._collection.query.return_value = {
        "ids": [[]],
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
    }

    results = await store.search([0.1])
    assert results == []


@pytest.mark.asyncio
async def test_delete_documents(store):
    """Test deleting documents."""
    store._initialized = True
    store._collection = MagicMock()

    ids = ["1", "2"]
    await store.delete_documents(ids)

    store._collection.delete.assert_called_once_with(ids=ids)


@pytest.mark.asyncio
async def test_delete_documents_empty(store):
    """Test deleting empty list."""
    store._initialized = True
    store._collection = MagicMock()

    await store.delete_documents([])

    store._collection.delete.assert_not_called()


@pytest.mark.asyncio
async def test_get_document_count(store):
    """Test getting document count."""
    store._initialized = True
    store._collection = MagicMock()
    store._collection.count.return_value = 42

    count = await store.get_document_count()
    assert count == 42


@pytest.mark.asyncio
async def test_get_document(store):
    """Test getting single document."""
    store._initialized = True
    store._collection = MagicMock()

    store._collection.get.return_value = {
        "ids": ["1"],
        "documents": ["text"],
        "metadatas": [{"k": "v"}],
    }

    result = await store.get_document("1")
    
    assert result is not None
    assert result.id == "1"
    assert result.text == "text"
    assert result.metadata == {"k": "v"}
    assert result.distance == 0.0


@pytest.mark.asyncio
async def test_get_document_not_found(store):
    """Test getting non-existent document."""
    store._initialized = True
    store._collection = MagicMock()

    store._collection.get.return_value = {
        "ids": [],
        "documents": [],
        "metadatas": [],
    }

    result = await store.get_document("999")
    assert result is None


@pytest.mark.asyncio
async def test_clear_collection(store):
    """Test clearing collection."""
    store._initialized = True
    store._collection = MagicMock()

    store._collection.get.return_value = {"ids": ["1", "2"]}

    await store.clear_collection()

    store._collection.get.assert_called_once()
    store._collection.delete.assert_called_once_with(ids=["1", "2"])


@pytest.mark.asyncio
async def test_async_shutdown(store):
    """Test shutdown."""
    store._initialized = True
    store._client = MagicMock()
    store._collection = MagicMock()

    await store.async_shutdown()

    assert not store._initialized
    assert store._client is None
    assert store._collection is None