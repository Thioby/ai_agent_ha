import os
import json
import sqlite3
import pytest
from custom_components.ai_agent_ha.rag.sqlite_store import (
    SqliteStore,
    SearchResult,
    _cosine_similarity,
    _cosine_distance,
    DEFAULT_TABLE_NAME,
)

# Test _cosine_similarity
def test_cosine_similarity():
    # Identical vectors
    vec1 = [1.0, 0.0]
    vec2 = [1.0, 0.0]
    assert _cosine_similarity(vec1, vec2) == 1.0

    # Perpendicular vectors
    vec1 = [1.0, 0.0]
    vec2 = [0.0, 1.0]
    assert _cosine_similarity(vec1, vec2) == 0.0

    # Different lengths
    vec1 = [1.0, 0.0]
    vec2 = [1.0]
    assert _cosine_similarity(vec1, vec2) == 0.0

    # Zero vector
    vec1 = [0.0, 0.0]
    vec2 = [1.0, 1.0]
    assert _cosine_similarity(vec1, vec2) == 0.0

    # Opposite vectors
    vec1 = [1.0, 0.0]
    vec2 = [-1.0, 0.0]
    assert _cosine_similarity(vec1, vec2) == -1.0


# Test _cosine_distance
def test_cosine_distance():
    vec1 = [1.0, 0.0]
    vec2 = [0.0, 1.0]
    similarity = _cosine_similarity(vec1, vec2)
    assert _cosine_distance(vec1, vec2) == 1.0 - similarity

    vec1 = [1.0, 0.0]
    vec2 = [1.0, 0.0]
    assert _cosine_distance(vec1, vec2) == 0.0


# Test SearchResult
def test_search_result_instantiation():
    res = SearchResult(
        id="test_id",
        text="test_text",
        metadata={"key": "value"},
        distance=0.5
    )
    assert res.id == "test_id"
    assert res.text == "test_text"
    assert res.metadata == {"key": "value"}
    assert res.distance == 0.5


# Test SqliteStore
@pytest.mark.asyncio
async def test_sqlite_store_initialization(tmp_path):
    store = SqliteStore(persist_directory=str(tmp_path))
    
    assert store._initialized is False
    assert store._conn is None

    await store.async_initialize()

    assert store._initialized is True
    assert store._conn is not None
    assert os.path.exists(os.path.join(tmp_path, "vectors.db"))
    
    # Test double initialization (should skip)
    await store.async_initialize()
    assert store._initialized is True
    
    await store.async_shutdown()


@pytest.mark.asyncio
async def test_ensure_initialized(tmp_path):
    store = SqliteStore(persist_directory=str(tmp_path))
    
    with pytest.raises(RuntimeError):
        store._ensure_initialized()
        
    await store.async_initialize()
    store._ensure_initialized() # Should not raise
    await store.async_shutdown()


@pytest.mark.asyncio
async def test_add_documents(tmp_path):
    store = SqliteStore(persist_directory=str(tmp_path))
    await store.async_initialize()

    ids = ["1", "2"]
    texts = ["text1", "text2"]
    embeddings = [[1.0, 0.0], [0.0, 1.0]]
    metadatas = [{"source": "a"}, {"source": "b"}]

    await store.add_documents(ids, texts, embeddings, metadatas)

    count = await store.get_document_count()
    assert count == 2

    # Test adding empty list
    await store.add_documents([], [], [])
    assert await store.get_document_count() == 2
    
    await store.async_shutdown()


@pytest.mark.asyncio
async def test_add_documents_fallback_to_upsert(tmp_path):
    store = SqliteStore(persist_directory=str(tmp_path))
    await store.async_initialize()

    ids = ["1"]
    texts = ["text1"]
    embeddings = [[1.0, 0.0]]

    await store.add_documents(ids, texts, embeddings)
    
    # Try adding same ID again
    texts_new = ["text1_updated"]
    await store.add_documents(ids, texts_new, embeddings)

    doc = await store.get_document("1")
    assert doc.text == "text1_updated"

    await store.async_shutdown()


@pytest.mark.asyncio
async def test_upsert_documents(tmp_path):
    store = SqliteStore(persist_directory=str(tmp_path))
    await store.async_initialize()

    ids = ["1"]
    texts = ["text1"]
    embeddings = [[1.0, 0.0]]

    # Insert
    await store.upsert_documents(ids, texts, embeddings)
    count = await store.get_document_count()
    assert count == 1
    
    # Update
    texts_updated = ["text1_updated"]
    await store.upsert_documents(ids, texts_updated, embeddings)
    
    doc = await store.get_document("1")
    assert doc.text == "text1_updated"
    
    # Empty list
    await store.upsert_documents([], [], [])
    assert await store.get_document_count() == 1

    await store.async_shutdown()


@pytest.mark.asyncio
async def test_search(tmp_path):
    store = SqliteStore(persist_directory=str(tmp_path))
    await store.async_initialize()

    ids = ["1", "2", "3"]
    texts = ["apple", "banana", "cherry"]
    # 1: [1, 0], 2: [0, 1], 3: [1, 1] (normalized approx [0.7, 0.7])
    embeddings = [[1.0, 0.0], [0.0, 1.0], [0.707, 0.707]]
    metadatas = [{"type": "fruit"}, {"type": "fruit"}, {"type": "berry"}]

    await store.add_documents(ids, texts, embeddings, metadatas)

    # Search query close to "apple" [1, 0]
    results = await store.search([1.0, 0.0], n_results=3)
    
    assert len(results) == 3
    assert results[0].id == "1" # Closest
    assert results[0].distance == 0.0

    # Search with filter
    results = await store.search([1.0, 0.0], n_results=3, where={"type": "berry"})
    assert len(results) == 1
    assert results[0].id == "3"

    # Search with no results
    results = await store.search([1.0, 0.0], n_results=3, where={"type": "vegetable"})
    assert len(results) == 0

    await store.async_shutdown()


@pytest.mark.asyncio
async def test_delete_documents(tmp_path):
    store = SqliteStore(persist_directory=str(tmp_path))
    await store.async_initialize()

    ids = ["1", "2"]
    texts = ["a", "b"]
    embeddings = [[1.0], [1.0]]
    await store.add_documents(ids, texts, embeddings)

    await store.delete_documents(["1"])
    assert await store.get_document_count() == 1
    assert await store.get_document("1") is None
    assert await store.get_document("2") is not None

    # Empty list
    await store.delete_documents([])
    assert await store.get_document_count() == 1

    await store.async_shutdown()


@pytest.mark.asyncio
async def test_get_document(tmp_path):
    store = SqliteStore(persist_directory=str(tmp_path))
    await store.async_initialize()
    
    ids = ["1"]
    texts = ["a"]
    embeddings = [[1.0]]
    await store.add_documents(ids, texts, embeddings)
    
    doc = await store.get_document("1")
    assert doc is not None
    assert doc.id == "1"
    
    doc = await store.get_document("nonexistent")
    assert doc is None

    await store.async_shutdown()


@pytest.mark.asyncio
async def test_clear_collection(tmp_path):
    store = SqliteStore(persist_directory=str(tmp_path))
    await store.async_initialize()
    
    ids = ["1", "2"]
    texts = ["a", "b"]
    embeddings = [[1.0], [1.0]]
    await store.add_documents(ids, texts, embeddings)
    
    assert await store.get_document_count() == 2
    
    await store.clear_collection()
    assert await store.get_document_count() == 0

    await store.async_shutdown()


def test_filter_metadata():
    store = SqliteStore(persist_directory="dummy")
    
    class Unserializable:
        def __str__(self):
            return "converted"
            
    metadata = {
        "str": "value",
        "int": 1,
        "float": 1.5,
        "bool": True,
        "none": None,
        "list": [1, 2],
        "dict": {"a": 1},
        "unserializable": Unserializable()
    }
    
    filtered = store._filter_metadata(metadata)
    
    assert filtered["str"] == "value"
    assert filtered["int"] == 1
    assert filtered["float"] == 1.5
    assert filtered["bool"] is True
    assert "none" not in filtered
    assert filtered["list"] == [1, 2]
    assert filtered["dict"] == {"a": 1}
    assert filtered["unserializable"] == "converted"
