"""RAG (Retrieval-Augmented Generation) system for AI Agent HA.

This module provides semantic search capabilities for Home Assistant entities,
learning from conversations to improve entity categorization over time.

Uses SQLite for vector storage - no external dependencies required.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)

# RAG search configuration
RAG_MIN_SIMILARITY = 0.5  # Minimum cosine similarity (0-1) for search results
# 0.5 = 50% similarity, reasonable default
# Lower = more results (higher recall, lower precision)
# Higher = fewer but more relevant results (lower recall, higher precision)

# Re-export for external use
__all__ = [
    "RAGManager",
]


@dataclass
class RAGManager:
    """Facade for the RAG system.

    Orchestrates all RAG components: SQLite storage, embeddings,
    entity indexing, query engine, and semantic learning.

    Usage:
        rag = RAGManager(hass, config, config_entry)
        await rag.async_initialize()
        context = await rag.get_relevant_context("turn on bedroom light")
        await rag.learn_from_conversation(user_msg, assistant_msg)
        await rag.async_shutdown()
    """

    hass: HomeAssistant
    config: dict[str, Any]
    config_entry: ConfigEntry | None = None
    _store: Any | None = field(default=None, repr=False)  # SqliteStore
    _embedding_provider: Any | None = field(
        default=None, repr=False
    )  # EmbeddingProvider
    _indexer: Any | None = field(default=None, repr=False)  # EntityIndexer
    _query_engine: Any | None = field(default=None, repr=False)  # QueryEngine
    _intent_detector: Any | None = field(default=None, repr=False)  # IntentDetector
    _learner: Any | None = field(default=None, repr=False)  # SemanticLearner
    _event_handlers: Any | None = field(default=None, repr=False)  # EventHandlers
    _initialized: bool = field(default=False, repr=False)

    def _get_persist_directory(self) -> str:
        """Get the SQLite persist directory path."""
        # Use HA config directory: /config/ai_agent_ha/rag_db/
        config_dir = self.hass.config.path("ai_agent_ha", "rag_db")
        return config_dir

    async def async_initialize(self) -> None:
        """Initialize all RAG components.

        This initializes:
        1. SQLite vector storage
        2. Embedding provider
        3. Entity indexer (imports lazily to avoid circular imports)
        4. Query engine
        5. Semantic learner
        6. Event handlers for entity registry updates
        """
        if self._initialized:
            _LOGGER.debug("RAGManager already initialized")
            return

        try:
            _LOGGER.info("Initializing RAG system...")

            # 1. Initialize SQLite storage
            from .sqlite_store import SqliteStore

            persist_dir = self._get_persist_directory()
            self._store = SqliteStore(persist_directory=persist_dir)
            await self._store.async_initialize()

            # 2. Initialize embedding provider
            from .embeddings import create_embedding_provider

            self._embedding_provider = create_embedding_provider(
                self.hass, self.config, self.config_entry
            )
            _LOGGER.info(
                "RAG using embedding provider: %s",
                self._embedding_provider.provider_name,
            )

            # 3. Initialize entity indexer
            from .entity_indexer import EntityIndexer

            self._indexer = EntityIndexer(
                hass=self.hass,
                store=self._store,
                embedding_provider=self._embedding_provider,
            )

            # 4. Initialize query engine
            from .query_engine import QueryEngine

            self._query_engine = QueryEngine(
                store=self._store,
                embedding_provider=self._embedding_provider,
            )

            # 5. Initialize intent detector (semantic intent detection with cached embeddings)
            from .intent_detector import IntentDetector

            self._intent_detector = IntentDetector(
                embedding_provider=self._embedding_provider,
            )
            await self._intent_detector.async_initialize()

            # 6. Initialize semantic learner
            from .semantic_learner import SemanticLearner

            learner_storage_path = self.hass.config.path(
                "ai_agent_ha", "learned_categories.json"
            )
            self._learner = SemanticLearner(
                hass=self.hass,
                indexer=self._indexer,
                storage_path=learner_storage_path,
            )
            await self._learner.async_load()

            # 7. Initialize event handlers
            from .event_handlers import EntityRegistryEventHandler

            self._event_handlers = EntityRegistryEventHandler(
                hass=self.hass,
                indexer=self._indexer,
            )
            await self._event_handlers.async_start()

            # 8. Check if embedding provider or configuration changed (auto-reindex)
            provider_name = self._embedding_provider.provider_name
            stored_provider = await self._store.get_metadata("embedding_provider")

            reindex_needed = False

            if stored_provider and stored_provider != provider_name:
                _LOGGER.warning(
                    "Embedding provider changed from %s to %s, triggering full reindex",
                    stored_provider,
                    provider_name,
                )
                reindex_needed = True
            elif not stored_provider:
                # First run, store current provider
                await self._store.set_metadata("embedding_provider", provider_name)
                _LOGGER.info("Stored embedding provider: %s", provider_name)

            # Check if Gemini task type configuration changed (only for Gemini)
            if provider_name == "gemini":
                task_type_version = (
                    "v2_query_document_split"  # Increment when task type logic changes
                )
                stored_version = await self._store.get_metadata(
                    "gemini_task_type_version"
                )

                if stored_version and stored_version != task_type_version:
                    _LOGGER.warning(
                        "Gemini task type configuration changed (%s -> %s), triggering full reindex",
                        stored_version,
                        task_type_version,
                    )
                    reindex_needed = True
                elif not stored_version:
                    await self._store.set_metadata(
                        "gemini_task_type_version", task_type_version
                    )
                    _LOGGER.info(
                        "Stored Gemini task type version: %s", task_type_version
                    )

            # 9. Perform full reindex if needed
            doc_count = await self._store.get_document_count()
            if reindex_needed:
                _LOGGER.info("Reindexing all entities due to configuration change...")
                await self._indexer.full_reindex()
                # Update metadata after successful reindex
                await self._store.set_metadata("embedding_provider", provider_name)
                if provider_name == "gemini":
                    await self._store.set_metadata(
                        "gemini_task_type_version", "v2_query_document_split"
                    )
            elif doc_count == 0:
                _LOGGER.info("No indexed entities found, performing full reindex...")
                await self._indexer.full_reindex()
            else:
                _LOGGER.info("RAG system has %d indexed entities", doc_count)

            self._initialized = True
            _LOGGER.info("RAG system initialized successfully")

        except Exception as e:
            _LOGGER.exception("Failed to initialize RAG system: %s", e)
            raise

    def _ensure_initialized(self) -> None:
        """Ensure RAG is initialized before operations."""
        if not self._initialized:
            raise RuntimeError(
                "RAGManager not initialized. Call async_initialize() first."
            )

    async def get_relevant_context(
        self,
        query: str,
        top_k: int = 10,
    ) -> str:
        """Get relevant entity context for a user query.

        Uses semantic search to find entities related to the query
        and returns a compressed context string suitable for LLM.

        Includes self-healing: removes stale entities from the index
        if they no longer exist in Home Assistant.

        Args:
            query: The user's query text.
            top_k: Maximum number of entities to include.

        Returns:
            Compressed context string for the LLM, or empty string if no results.
        """
        self._ensure_initialized()

        try:
            # Extract intent from query using semantic similarity (cached embeddings)
            intent = await self._intent_detector.detect_intent(query)

            # Pre-filter: Skip RAG if query clearly not HA-related
            if not intent:
                # Check for basic HA keywords (English + Polish)
                query_lower = query.lower()
                ha_keywords = [
                    # English
                    "light",
                    "turn",
                    "switch",
                    "temperature",
                    "sensor",
                    "automation",
                    "scene",
                    "device",
                    "home",
                    "room",
                    "cover",
                    "blind",
                    "lock",
                    "fan",
                    "climate",
                    "thermostat",
                    # Polish
                    "światło",
                    "światła",
                    "włącz",
                    "wyłącz",
                    "temperatura",
                    "czujnik",
                    "urządzenie",
                    "dom",
                    "pokój",
                    "roleta",
                ]

                if not any(keyword in query_lower for keyword in ha_keywords):
                    _LOGGER.debug(
                        "RAG pre-filter: Query doesn't appear HA-related, skipping search: %s",
                        query[:100],
                    )
                    return ""

            # Determine which filters to apply
            # Priority: domain > device_class (they often conflict, e.g. media_player vs motion)
            # Area is always safe to combine
            use_domain = intent.get("domain")
            use_device_class = intent.get("device_class") if not use_domain else None
            use_area = intent.get("area")

            # Use filtered search if intent detected, otherwise plain search
            if use_domain or use_device_class or use_area:
                _LOGGER.debug(
                    "RAG using intent-based search: domain=%s, device_class=%s, area=%s (raw: %s)",
                    use_domain,
                    use_device_class,
                    use_area,
                    intent,
                )
                results = await self._query_engine.search_by_criteria(
                    query=query,
                    domain=use_domain,
                    device_class=use_device_class,
                    area=use_area,
                    top_k=top_k,
                    min_similarity=RAG_MIN_SIMILARITY,
                )
                # Fallback: if no results with filters, try semantic search without filters
                if not results:
                    _LOGGER.debug(
                        "RAG filtered search returned 0 results, falling back to semantic search"
                    )
                    results = await self._query_engine.search_entities(
                        query=query,
                        top_k=top_k,
                        min_similarity=RAG_MIN_SIMILARITY,
                    )
            else:
                results = await self._query_engine.search_entities(
                    query=query,
                    top_k=top_k,
                    min_similarity=RAG_MIN_SIMILARITY,
                )

            # Early return if no results pass threshold
            if not results:
                _LOGGER.debug(
                    "RAG search returned no results above similarity threshold"
                )
                return ""

            # Log top similarity scores
            top_similarities = [1.0 - r.distance for r in results[:3]]
            if len(top_similarities) >= 1:
                _LOGGER.debug(
                    "RAG top similarity scores: %s",
                    ", ".join(f"{s:.3f}" for s in top_similarities),
                )

            # Validate entities exist and remove stale ones
            valid_results = []
            stale_entities = []

            for result in results:
                entity_id = result.id
                # Check if entity still exists in Home Assistant
                if self.hass.states.get(entity_id):
                    valid_results.append(result)
                else:
                    stale_entities.append(entity_id)

            # Remove stale entities from index (self-healing)
            if stale_entities:
                _LOGGER.warning(
                    "Found %d stale entities in RAG index, removing: %s",
                    len(stale_entities),
                    stale_entities,
                )
                for entity_id in stale_entities:
                    try:
                        await self._indexer.remove_entity(entity_id)
                    except Exception as e:
                        _LOGGER.error(
                            "Failed to remove stale entity %s: %s", entity_id, e
                        )

            # Build context from valid results only
            context = self._query_engine.build_compressed_context(valid_results)

            if context:
                _LOGGER.debug(
                    "RAG context generated (%d chars) for query: %s...",
                    len(context),
                    query[:50],
                )
            return context

        except Exception as e:
            _LOGGER.warning("RAG context retrieval failed: %s", e)
            # Graceful degradation - return empty context
            return ""

    async def learn_from_conversation(
        self,
        user_message: str,
        assistant_message: str,
    ) -> None:
        """Learn semantic corrections from a conversation.

        Analyzes the conversation for patterns like:
        - "switch.lamp is actually a light"
        - "treat sensor.xyz as a temperature sensor"

        Args:
            user_message: The user's message.
            assistant_message: The assistant's response.
        """
        self._ensure_initialized()

        try:
            await self._learner.detect_and_persist(
                user_message=user_message,
                assistant_message=assistant_message,
            )
        except Exception as e:
            _LOGGER.warning("Failed to learn from conversation: %s", e)
            # Graceful degradation - don't fail the main flow

    async def reindex_entity(self, entity_id: str) -> None:
        """Reindex a single entity.

        Args:
            entity_id: The entity ID to reindex.
        """
        self._ensure_initialized()
        await self._indexer.index_entity(entity_id)

    async def remove_entity(self, entity_id: str) -> None:
        """Remove an entity from the index.

        Args:
            entity_id: The entity ID to remove.
        """
        self._ensure_initialized()
        await self._indexer.remove_entity(entity_id)

    async def full_reindex(self) -> None:
        """Perform a full reindex of all entities."""
        self._ensure_initialized()
        await self._indexer.full_reindex()

    async def get_stats(self) -> dict[str, Any]:
        """Get RAG system statistics.

        Returns:
            Dictionary with stats like document count, learned categories, etc.
        """
        self._ensure_initialized()

        return {
            "indexed_entities": await self._store.get_document_count(),
            "embedding_provider": self._embedding_provider.provider_name,
            "embedding_dimension": self._embedding_provider.dimension,
            "learned_categories": len(self._learner.categories) if self._learner else 0,
        }

    async def async_shutdown(self) -> None:
        """Shutdown all RAG components gracefully."""
        _LOGGER.info("Shutting down RAG system...")

        try:
            if self._event_handlers:
                await self._event_handlers.async_stop()
                self._event_handlers = None

            if self._learner:
                await self._learner.async_save()
                self._learner = None

            if self._store:
                await self._store.async_shutdown()
                self._store = None

            self._indexer = None
            self._query_engine = None
            self._embedding_provider = None
            self._initialized = False

            _LOGGER.info("RAG system shut down successfully")

        except Exception as e:
            _LOGGER.error("Error during RAG shutdown: %s", e)
            raise

    @property
    def is_initialized(self) -> bool:
        """Check if RAG system is initialized."""
        return self._initialized
