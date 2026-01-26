"""Query engine for RAG system.

This module handles semantic search and context compression for
providing relevant entity information to the LLM.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .sqlite_store import SqliteStore, SearchResult
from .embeddings import EmbeddingProvider, get_embedding_for_query

_LOGGER = logging.getLogger(__name__)

# Maximum context length to avoid token overflow
MAX_CONTEXT_LENGTH = 2000


@dataclass
class QueryEngine:
    """Handles semantic search and context formatting.

    Searches for relevant entities and builds a compressed context
    string suitable for injection into the LLM prompt.
    """

    store: SqliteStore
    embedding_provider: EmbeddingProvider

    async def search_entities(
        self,
        query: str,
        top_k: int = 10,
        domain_filter: str | None = None,
    ) -> list[SearchResult]:
        """Search for entities semantically similar to the query.

        Args:
            query: The user's query text.
            top_k: Maximum number of results to return.
            domain_filter: Optional domain to filter by (e.g., "light").

        Returns:
            List of SearchResult objects sorted by relevance.
        """
        try:
            # Generate embedding for the query
            query_embedding = await get_embedding_for_query(
                self.embedding_provider, query
            )

            # Build filter if domain specified
            where_filter = None
            if domain_filter:
                where_filter = {"domain": domain_filter}

            # Search the vector store
            results = await self.store.search(
                query_embedding=query_embedding,
                n_results=top_k,
                where=where_filter,
            )

            _LOGGER.debug(
                "Search for '%s' returned %d results", query[:50], len(results)
            )
            if results:
                # Log found entity IDs for debugging
                entity_ids = [r.id for r in results[:5]]  # First 5
                _LOGGER.info(
                    "RAG search found entities: %s%s",
                    entity_ids,
                    f" (+{len(results)-5} more)" if len(results) > 5 else "",
                )
            return results

        except Exception as e:
            _LOGGER.error("Entity search failed: %s", e)
            return []

    def build_compressed_context(
        self,
        results: list[SearchResult],
        max_length: int = MAX_CONTEXT_LENGTH,
    ) -> str:
        """Build a compressed context string from search results.

        Formats results into a compact representation that provides
        relevant entity information without wasting tokens.

        Args:
            results: Search results to format.
            max_length: Maximum character length for the context.

        Returns:
            Formatted context string.
        """
        if not results:
            return ""

        lines = []
        current_length = 0

        # Header with instruction to use exact entity_ids
        header = "Relevant entities (IMPORTANT: use exact entity_id values, do not shorten or modify them):"
        lines.append(header)
        current_length += len(header)

        for result in results:
            # Format each entity compactly
            line = self._format_entity(result)

            # Check if adding this line would exceed max length
            if current_length + len(line) + 1 > max_length:
                break

            lines.append(line)
            current_length += len(line) + 1  # +1 for newline

        return "\n".join(lines)

    def _format_entity(self, result: SearchResult) -> str:
        """Format a single entity result compactly.

        Args:
            result: The search result to format.

        Returns:
            Compact string representation.
        """
        metadata = result.metadata
        entity_id = result.id

        # Build compact representation
        parts = [f"- {entity_id}"]

        # Add friendly name if different from entity_id
        friendly_name = metadata.get("friendly_name")
        if friendly_name and friendly_name.lower() not in entity_id.lower():
            parts.append(f'"{friendly_name}"')

        # Add domain if not obvious from entity_id
        domain = metadata.get("domain")
        if domain:
            parts.append(f"({domain})")

        # Add area if available
        area_name = metadata.get("area_name")
        if area_name:
            parts.append(f"in {area_name}")

        # Add device class if useful
        device_class = metadata.get("device_class")
        if device_class and device_class != domain:
            parts.append(f"[{device_class}]")

        # Add learned category if available
        learned_cat = metadata.get("learned_category")
        if learned_cat:
            parts.append(f"<{learned_cat}>")

        # Add current state for actionable entities
        state = metadata.get("state")
        if state and domain in ("light", "switch", "cover", "lock", "fan"):
            parts.append(f"state:{state}")

        return " ".join(parts)

    async def search_and_format(
        self,
        query: str,
        top_k: int = 10,
        max_context_length: int = MAX_CONTEXT_LENGTH,
    ) -> str:
        """Search for entities and return formatted context.

        Convenience method that combines search and formatting.

        Args:
            query: The user's query text.
            top_k: Maximum number of results.
            max_context_length: Maximum context length.

        Returns:
            Formatted context string, or empty string if no results.
        """
        results = await self.search_entities(query, top_k)
        return self.build_compressed_context(results, max_context_length)

    async def search_by_criteria(
        self,
        query: str,
        domain: str | None = None,
        area: str | None = None,
        device_class: str | None = None,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Search with additional filter criteria.

        Args:
            query: The semantic query text.
            domain: Filter by domain (e.g., "light", "sensor").
            area: Filter by area name.
            device_class: Filter by device class.
            top_k: Maximum number of results.

        Returns:
            Filtered search results.
        """
        # Build the where filter
        where_filter: dict[str, Any] = {}

        if domain:
            where_filter["domain"] = domain
        if area:
            where_filter["area_name"] = area
        if device_class:
            where_filter["device_class"] = device_class

        try:
            query_embedding = await get_embedding_for_query(
                self.embedding_provider, query
            )

            return await self.store.search(
                query_embedding=query_embedding,
                n_results=top_k,
                where=where_filter if where_filter else None,
            )

        except Exception as e:
            _LOGGER.error("Filtered search failed: %s", e)
            return []

    def extract_query_intent(self, query: str) -> dict[str, Any]:
        """Extract intent and filters from a natural language query.

        Simple rule-based extraction of common patterns like:
        - "lights in bedroom" → domain=light, area=bedroom
        - "temperature sensors" → domain=sensor, device_class=temperature

        Args:
            query: The user's query.

        Returns:
            Dictionary with extracted filters.
        """
        query_lower = query.lower()
        intent: dict[str, Any] = {}

        # Domain extraction
        domain_keywords = {
            "light": ["light", "lamp", "bulb"],
            "switch": ["switch", "outlet", "plug"],
            "sensor": ["sensor", "temperature", "humidity", "motion"],
            "cover": ["cover", "blind", "curtain", "shade"],
            "climate": ["climate", "thermostat", "hvac", "ac", "heating"],
            "lock": ["lock", "door lock"],
            "fan": ["fan"],
            "media_player": ["media", "speaker", "tv", "television"],
            "camera": ["camera"],
        }

        for domain, keywords in domain_keywords.items():
            if any(kw in query_lower for kw in keywords):
                intent["domain"] = domain
                break

        # Device class extraction (subset)
        device_class_keywords = {
            "temperature": ["temperature", "temp"],
            "humidity": ["humidity"],
            "motion": ["motion", "movement"],
            "door": ["door"],
            "window": ["window"],
            "battery": ["battery"],
            "power": ["power", "energy", "watt"],
        }

        for device_class, keywords in device_class_keywords.items():
            if any(kw in query_lower for kw in keywords):
                intent["device_class"] = device_class
                break

        # Common room/area names
        area_keywords = [
            "bedroom",
            "living room",
            "kitchen",
            "bathroom",
            "office",
            "garage",
            "basement",
            "attic",
            "hallway",
            "dining room",
            "garden",
            "patio",
            "backyard",
            "front yard",
        ]

        for area in area_keywords:
            if area in query_lower:
                intent["area"] = area
                break

        return intent
