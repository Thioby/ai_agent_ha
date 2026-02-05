"""SQLite-based vector store for RAG system.

This module provides a simple SQLite-based storage for entity embeddings
with cosine similarity search. No external vector database dependencies required.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
from dataclasses import dataclass, field
from typing import Any

_LOGGER = logging.getLogger(__name__)

# Default table name for entity embeddings
DEFAULT_TABLE_NAME = "ha_entities"


@dataclass
class SearchResult:
    """Result from a vector search."""

    id: str
    text: str
    metadata: dict[str, Any]
    distance: float


def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        vec1: First vector.
        vec2: Second vector.

    Returns:
        Cosine similarity score (0-1, higher is more similar).
    """
    if len(vec1) != len(vec2):
        return 0.0

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def _cosine_distance(vec1: list[float], vec2: list[float]) -> float:
    """Compute cosine distance between two vectors.

    Args:
        vec1: First vector.
        vec2: Second vector.

    Returns:
        Cosine distance (0-2, lower is more similar).
    """
    return 1.0 - _cosine_similarity(vec1, vec2)


@dataclass
class SqliteStore:
    """SQLite-based vector store for entity embeddings.

    Provides async-compatible methods for storing and searching embeddings.
    Uses SQLite with JSON for embedding storage and pure Python for
    cosine similarity computation.
    """

    persist_directory: str
    table_name: str = DEFAULT_TABLE_NAME
    _db_path: str = field(default="", repr=False)
    _conn: sqlite3.Connection | None = field(default=None, repr=False)
    _initialized: bool = field(default=False, repr=False)

    async def async_initialize(self) -> None:
        """Initialize SQLite database and create tables.

        Creates the persist directory if it doesn't exist and initializes
        the SQLite database with required tables.
        """
        if self._initialized:
            _LOGGER.debug("SqliteStore already initialized")
            return

        try:
            # Ensure persist directory exists
            os.makedirs(self.persist_directory, exist_ok=True)
            _LOGGER.debug("SQLite persist directory: %s", self.persist_directory)

            # Set up database path
            self._db_path = os.path.join(self.persist_directory, "vectors.db")

            # Initialize SQLite connection
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row

            # Create tables
            self._create_tables()

            self._initialized = True
            count = await self.get_document_count()
            _LOGGER.info("SQLite vector store initialized with %d documents", count)

        except Exception as e:
            _LOGGER.error("Failed to initialize SQLite store: %s", e)
            raise

    def _create_tables(self) -> None:
        """Create the required database tables."""
        if self._conn is None:
            return

        cursor = self._conn.cursor()

        # Main documents table
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                embedding TEXT NOT NULL,
                metadata TEXT
            )
        """)

        # Index for faster lookups
        cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{self.table_name}_id
            ON {self.table_name}(id)
        """)

        # Metadata table for tracking configuration (embedding provider, versions, etc.)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rag_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at REAL NOT NULL
            )
        """)

        self._conn.commit()
        _LOGGER.debug("Database tables created/verified")

    def _ensure_initialized(self) -> None:
        """Ensure the store is initialized before operations."""
        if not self._initialized or self._conn is None:
            raise RuntimeError(
                "SqliteStore not initialized. Call async_initialize() first."
            )

    async def add_documents(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """Add documents with their embeddings to the store.

        Args:
            ids: Unique identifiers for each document.
            texts: Text content of each document.
            embeddings: Pre-computed embeddings for each document.
            metadatas: Optional metadata dictionaries for each document.
        """
        self._ensure_initialized()

        if not ids:
            _LOGGER.debug("No documents to add")
            return

        try:
            cursor = self._conn.cursor()  # type: ignore[union-attr]

            for i, doc_id in enumerate(ids):
                text = texts[i] if i < len(texts) else ""
                embedding = embeddings[i] if i < len(embeddings) else []
                metadata = metadatas[i] if metadatas and i < len(metadatas) else {}

                cursor.execute(
                    f"""
                    INSERT INTO {self.table_name} (id, text, embedding, metadata)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        doc_id,
                        text,
                        json.dumps(embedding),
                        json.dumps(self._filter_metadata(metadata)),
                    ),
                )

            self._conn.commit()  # type: ignore[union-attr]
            _LOGGER.debug("Added %d documents to SQLite store", len(ids))

        except sqlite3.IntegrityError:
            # Document already exists, use upsert instead
            _LOGGER.debug("Some documents already exist, using upsert")
            await self.upsert_documents(ids, texts, embeddings, metadatas)
        except Exception as e:
            _LOGGER.error("Failed to add documents: %s", e)
            raise

    async def upsert_documents(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """Upsert documents (add or update) in the store.

        Args:
            ids: Unique identifiers for each document.
            texts: Text content of each document.
            embeddings: Pre-computed embeddings for each document.
            metadatas: Optional metadata dictionaries for each document.
        """
        self._ensure_initialized()

        if not ids:
            _LOGGER.debug("No documents to upsert")
            return

        try:
            cursor = self._conn.cursor()  # type: ignore[union-attr]

            for i, doc_id in enumerate(ids):
                text = texts[i] if i < len(texts) else ""
                embedding = embeddings[i] if i < len(embeddings) else []
                metadata = metadatas[i] if metadatas and i < len(metadatas) else {}

                cursor.execute(
                    f"""
                    INSERT OR REPLACE INTO {self.table_name} (id, text, embedding, metadata)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        doc_id,
                        text,
                        json.dumps(embedding),
                        json.dumps(self._filter_metadata(metadata)),
                    ),
                )

            self._conn.commit()  # type: ignore[union-attr]
            _LOGGER.debug("Upserted %d documents to SQLite store", len(ids))

        except Exception as e:
            _LOGGER.error("Failed to upsert documents: %s", e)
            raise

    def _filter_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Filter metadata to ensure JSON-serializable types.

        Args:
            metadata: Original metadata dictionary.

        Returns:
            Filtered metadata with only serializable types.
        """
        filtered = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                filtered[key] = value
            elif value is None:
                continue
            elif isinstance(value, (list, dict)):
                try:
                    json.dumps(value)  # Test if serializable
                    filtered[key] = value
                except (TypeError, ValueError):
                    _LOGGER.debug("Skipping non-serializable metadata key: %s", key)
            else:
                filtered[key] = str(value)
        return filtered

    async def search(
        self,
        query_embedding: list[float],
        n_results: int = 10,
        where: dict[str, Any] | None = None,
        min_similarity: float | None = None,
    ) -> list[SearchResult]:
        """Search for similar documents using query embedding.

        Args:
            query_embedding: The embedding vector to search with.
            n_results: Maximum number of results to return.
            where: Optional filter conditions for metadata (simple equality only).
            min_similarity: Minimum cosine similarity (0-1) for results.
                          If specified, only results with similarity >= this value are returned.
                          E.g., 0.5 = 50% similarity. Lower values are more lenient.

        Returns:
            List of SearchResult objects sorted by similarity (lowest distance first).
        """
        self._ensure_initialized()

        try:
            cursor = self._conn.cursor()  # type: ignore[union-attr]

            # Fetch all documents (for small datasets this is fine)
            # For larger datasets, consider chunking or using approximate methods
            cursor.execute(
                f"SELECT id, text, embedding, metadata FROM {self.table_name}"
            )
            rows = cursor.fetchall()

            # Compute similarities
            results_with_distance = []
            for row in rows:
                doc_id = row["id"]
                text = row["text"]
                embedding = json.loads(row["embedding"])
                metadata = json.loads(row["metadata"]) if row["metadata"] else {}

                # Apply where filter if provided
                if where:
                    match = True
                    for key, value in where.items():
                        if metadata.get(key) != value:
                            match = False
                            break
                    if not match:
                        continue

                # Compute cosine distance
                distance = _cosine_distance(query_embedding, embedding)
                results_with_distance.append(
                    SearchResult(
                        id=doc_id,
                        text=text,
                        metadata=metadata,
                        distance=distance,
                    )
                )

            # Apply similarity threshold filter if specified
            filtered_results = []
            if min_similarity is not None:
                for result in results_with_distance:
                    similarity = 1.0 - result.distance
                    if similarity >= min_similarity:
                        filtered_results.append(result)

                # Log filtering stats
                _LOGGER.debug(
                    "RAG similarity filter: %d/%d results above threshold %.2f",
                    len(filtered_results),
                    len(results_with_distance),
                    min_similarity,
                )
            else:
                filtered_results = results_with_distance

            # Sort by distance (ascending) and limit results
            filtered_results.sort(key=lambda x: x.distance)
            search_results = filtered_results[:n_results]

            _LOGGER.debug("Search returned %d results", len(search_results))
            return search_results

        except Exception as e:
            _LOGGER.error("Failed to search: %s", e)
            raise

    async def delete_documents(self, ids: list[str]) -> None:
        """Delete documents from the store by their IDs.

        Args:
            ids: List of document IDs to delete.
        """
        self._ensure_initialized()

        if not ids:
            _LOGGER.debug("No documents to delete")
            return

        try:
            cursor = self._conn.cursor()  # type: ignore[union-attr]
            placeholders = ",".join("?" * len(ids))
            cursor.execute(
                f"DELETE FROM {self.table_name} WHERE id IN ({placeholders})",
                ids,
            )
            self._conn.commit()  # type: ignore[union-attr]
            _LOGGER.debug("Deleted %d documents from SQLite store", cursor.rowcount)

        except Exception as e:
            _LOGGER.error("Failed to delete documents: %s", e)
            raise

    async def get_document_count(self) -> int:
        """Get the total number of documents in the store.

        Returns:
            Number of documents in the store.
        """
        self._ensure_initialized()

        try:
            cursor = self._conn.cursor()  # type: ignore[union-attr]
            cursor.execute(f"SELECT COUNT(*) FROM {self.table_name}")
            result = cursor.fetchone()
            return result[0] if result else 0
        except Exception as e:
            _LOGGER.error("Failed to get document count: %s", e)
            raise

    async def get_document(self, doc_id: str) -> SearchResult | None:
        """Get a specific document by its ID.

        Args:
            doc_id: The document ID to retrieve.

        Returns:
            SearchResult if found, None otherwise.
        """
        self._ensure_initialized()

        try:
            cursor = self._conn.cursor()  # type: ignore[union-attr]
            cursor.execute(
                f"SELECT id, text, metadata FROM {self.table_name} WHERE id = ?",
                (doc_id,),
            )
            row = cursor.fetchone()

            if row:
                return SearchResult(
                    id=row["id"],
                    text=row["text"],
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                    distance=0.0,
                )
            return None

        except Exception as e:
            _LOGGER.error("Failed to get document %s: %s", doc_id, e)
            raise

    async def clear_collection(self) -> None:
        """Clear all documents from the store."""
        self._ensure_initialized()

        try:
            cursor = self._conn.cursor()  # type: ignore[union-attr]
            cursor.execute(f"DELETE FROM {self.table_name}")
            self._conn.commit()  # type: ignore[union-attr]
            _LOGGER.info("Cleared all documents from SQLite store")

        except Exception as e:
            _LOGGER.error("Failed to clear collection: %s", e)
            raise

    async def get_metadata(self, key: str) -> str | None:
        """Get metadata value by key.

        Args:
            key: The metadata key to retrieve.

        Returns:
            The metadata value if found, None otherwise.
        """
        self._ensure_initialized()

        try:
            cursor = self._conn.cursor()  # type: ignore[union-attr]
            cursor.execute("SELECT value FROM rag_metadata WHERE key = ?", (key,))
            row = cursor.fetchone()
            return row["value"] if row else None
        except Exception as e:
            _LOGGER.error("Failed to get metadata for key %s: %s", key, e)
            return None

    async def set_metadata(self, key: str, value: str) -> None:
        """Set metadata value.

        Args:
            key: The metadata key.
            value: The metadata value to store.
        """
        self._ensure_initialized()

        try:
            import time

            cursor = self._conn.cursor()  # type: ignore[union-attr]
            cursor.execute(
                """
                INSERT OR REPLACE INTO rag_metadata (key, value, updated_at)
                VALUES (?, ?, ?)
                """,
                (key, value, time.time()),
            )
            self._conn.commit()  # type: ignore[union-attr]
            _LOGGER.debug("Set metadata: %s = %s", key, value)
        except Exception as e:
            _LOGGER.error("Failed to set metadata %s: %s", key, e)
            raise

    async def async_shutdown(self) -> None:
        """Shutdown the SQLite connection gracefully."""
        if self._conn:
            _LOGGER.debug("Shutting down SQLite store")
            self._conn.close()
            self._conn = None
            self._initialized = False


# Alias for backwards compatibility
ChromaStore = SqliteStore
