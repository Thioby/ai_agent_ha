"""ChromaDB wrapper for RAG system.

This module provides a simple wrapper around ChromaDB for storing and
searching entity embeddings.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from chromadb import Collection
    from chromadb.api import ClientAPI

_LOGGER = logging.getLogger(__name__)

# Default collection name for entity embeddings
DEFAULT_COLLECTION_NAME = "ha_entities"


@dataclass
class SearchResult:
    """Result from a ChromaDB search."""

    id: str
    text: str
    metadata: dict[str, Any]
    distance: float


@dataclass
class ChromaStore:
    """ChromaDB wrapper for entity embeddings storage.

    Provides async-compatible methods for storing and searching embeddings.
    Uses ChromaDB's persistent storage in the HA config directory.
    """

    persist_directory: str
    collection_name: str = DEFAULT_COLLECTION_NAME
    _client: ClientAPI | None = field(default=None, repr=False)
    _collection: Collection | None = field(default=None, repr=False)
    _initialized: bool = field(default=False, repr=False)

    async def async_initialize(self) -> None:
        """Initialize ChromaDB client and collection.

        Creates the persist directory if it doesn't exist and initializes
        the ChromaDB client with persistent storage.
        """
        if self._initialized:
            _LOGGER.debug("ChromaStore already initialized")
            return

        try:
            # Import chromadb here to allow graceful failure if not installed
            import chromadb
            from chromadb.config import Settings

            # Ensure persist directory exists
            os.makedirs(self.persist_directory, exist_ok=True)
            _LOGGER.debug("ChromaDB persist directory: %s", self.persist_directory)

            # Initialize ChromaDB with persistent storage
            settings = Settings(
                anonymized_telemetry=False,
                allow_reset=True,
                is_persistent=True,
            )

            self._client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=settings,
            )

            # Get or create the collection
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Home Assistant entity embeddings"},
            )

            self._initialized = True
            _LOGGER.info(
                "ChromaDB initialized with %d documents",
                self._collection.count(),
            )

        except ImportError:
            _LOGGER.error(
                "chromadb package not installed. "
                "Install with: pip install chromadb>=0.4.0"
            )
            raise
        except Exception as e:
            _LOGGER.error("Failed to initialize ChromaDB: %s", e)
            raise

    def _ensure_initialized(self) -> None:
        """Ensure the store is initialized before operations."""
        if not self._initialized or self._collection is None:
            raise RuntimeError(
                "ChromaStore not initialized. Call async_initialize() first."
            )

    async def add_documents(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """Add documents with their embeddings to the collection.

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
            # Filter metadata to only include ChromaDB-compatible types
            filtered_metadatas = None
            if metadatas:
                filtered_metadatas = [
                    self._filter_metadata(m) for m in metadatas
                ]

            self._collection.add(  # type: ignore[union-attr]
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=filtered_metadatas,
            )
            _LOGGER.debug("Added %d documents to ChromaDB", len(ids))

        except Exception as e:
            _LOGGER.error("Failed to add documents to ChromaDB: %s", e)
            raise

    async def upsert_documents(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """Upsert documents (add or update) in the collection.

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
            # Filter metadata to only include ChromaDB-compatible types
            filtered_metadatas = None
            if metadatas:
                filtered_metadatas = [
                    self._filter_metadata(m) for m in metadatas
                ]

            self._collection.upsert(  # type: ignore[union-attr]
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=filtered_metadatas,
            )
            _LOGGER.debug("Upserted %d documents to ChromaDB", len(ids))

        except Exception as e:
            _LOGGER.error("Failed to upsert documents to ChromaDB: %s", e)
            raise

    def _filter_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Filter metadata to only include ChromaDB-compatible types.

        ChromaDB only supports str, int, float, and bool values.

        Args:
            metadata: Original metadata dictionary.

        Returns:
            Filtered metadata with only compatible types.
        """
        filtered = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                filtered[key] = value
            elif value is None:
                # Skip None values
                continue
            elif isinstance(value, (list, dict)):
                # Convert complex types to JSON string
                import json
                try:
                    filtered[key] = json.dumps(value)
                except (TypeError, ValueError):
                    _LOGGER.debug(
                        "Skipping non-serializable metadata key: %s", key
                    )
            else:
                # Convert other types to string
                filtered[key] = str(value)
        return filtered

    async def search(
        self,
        query_embedding: list[float],
        n_results: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for similar documents using query embedding.

        Args:
            query_embedding: The embedding vector to search with.
            n_results: Maximum number of results to return.
            where: Optional filter conditions for metadata.

        Returns:
            List of SearchResult objects sorted by similarity.
        """
        self._ensure_initialized()

        try:
            results = self._collection.query(  # type: ignore[union-attr]
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"],
            )

            search_results = []
            if results["ids"] and results["ids"][0]:
                ids = results["ids"][0]
                documents = results["documents"][0] if results["documents"] else []
                metadatas = results["metadatas"][0] if results["metadatas"] else []
                distances = results["distances"][0] if results["distances"] else []

                for i, doc_id in enumerate(ids):
                    search_results.append(
                        SearchResult(
                            id=doc_id,
                            text=documents[i] if i < len(documents) else "",
                            metadata=metadatas[i] if i < len(metadatas) else {},
                            distance=distances[i] if i < len(distances) else 0.0,
                        )
                    )

            _LOGGER.debug(
                "Search returned %d results", len(search_results)
            )
            return search_results

        except Exception as e:
            _LOGGER.error("Failed to search ChromaDB: %s", e)
            raise

    async def delete_documents(self, ids: list[str]) -> None:
        """Delete documents from the collection by their IDs.

        Args:
            ids: List of document IDs to delete.
        """
        self._ensure_initialized()

        if not ids:
            _LOGGER.debug("No documents to delete")
            return

        try:
            self._collection.delete(ids=ids)  # type: ignore[union-attr]
            _LOGGER.debug("Deleted %d documents from ChromaDB", len(ids))

        except Exception as e:
            _LOGGER.error("Failed to delete documents from ChromaDB: %s", e)
            raise

    async def get_document_count(self) -> int:
        """Get the total number of documents in the collection.

        Returns:
            Number of documents in the collection.
        """
        self._ensure_initialized()

        try:
            return self._collection.count()  # type: ignore[union-attr]
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
            results = self._collection.get(  # type: ignore[union-attr]
                ids=[doc_id],
                include=["documents", "metadatas"],
            )

            if results["ids"]:
                return SearchResult(
                    id=results["ids"][0],
                    text=results["documents"][0] if results["documents"] else "",
                    metadata=results["metadatas"][0] if results["metadatas"] else {},
                    distance=0.0,
                )
            return None

        except Exception as e:
            _LOGGER.error("Failed to get document %s: %s", doc_id, e)
            raise

    async def clear_collection(self) -> None:
        """Clear all documents from the collection."""
        self._ensure_initialized()

        try:
            # Get all IDs and delete them
            all_ids = self._collection.get()["ids"]  # type: ignore[union-attr]
            if all_ids:
                self._collection.delete(ids=all_ids)  # type: ignore[union-attr]
            _LOGGER.info("Cleared %d documents from collection", len(all_ids))

        except Exception as e:
            _LOGGER.error("Failed to clear collection: %s", e)
            raise

    async def async_shutdown(self) -> None:
        """Shutdown the ChromaDB client gracefully."""
        if self._client:
            _LOGGER.debug("Shutting down ChromaDB client")
            # ChromaDB PersistentClient doesn't require explicit shutdown
            # but we reset our state
            self._collection = None
            self._client = None
            self._initialized = False
