"""Event handlers for RAG system.

This module listens to Home Assistant entity registry events
and triggers reindexing when entities are added, updated, or removed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from homeassistant.core import Event, HomeAssistant

from .entity_indexer import EntityIndexer

_LOGGER = logging.getLogger(__name__)

# Home Assistant entity registry event type
EVENT_ENTITY_REGISTRY_UPDATED = "entity_registry_updated"


@dataclass
class EntityRegistryEventHandler:
    """Handles entity registry events for RAG reindexing.

    Listens to entity added/removed/updated events and triggers
    appropriate reindexing operations.
    """

    hass: HomeAssistant
    indexer: EntityIndexer
    _unsub_listener: Callable[[], None] | None = field(default=None, repr=False)
    _started: bool = field(default=False, repr=False)

    async def async_start(self) -> None:
        """Start listening to entity registry events."""
        if self._started:
            _LOGGER.debug("Event handler already started")
            return

        # Subscribe to entity registry updated events
        self._unsub_listener = self.hass.bus.async_listen(
            EVENT_ENTITY_REGISTRY_UPDATED,
            self._handle_entity_registry_updated,
        )

        self._started = True
        _LOGGER.info("RAG event handlers started")

    async def async_stop(self) -> None:
        """Stop listening to events."""
        if self._unsub_listener:
            self._unsub_listener()
            self._unsub_listener = None

        self._started = False
        _LOGGER.info("RAG event handlers stopped")

    async def _handle_entity_registry_updated(self, event: Event) -> None:
        """Handle entity registry update events.

        Event data contains:
        - action: "create", "remove", or "update"
        - entity_id: The affected entity ID
        - changes: Dict of changed attributes (for updates)

        Args:
            event: The entity registry update event.
        """
        try:
            action = event.data.get("action")
            entity_id = event.data.get("entity_id")

            if not entity_id:
                _LOGGER.debug("Entity registry event missing entity_id")
                return

            _LOGGER.debug(
                "Entity registry event: action=%s, entity_id=%s",
                action,
                entity_id,
            )

            if action == "create":
                # New entity added - index it
                await self._on_entity_added(entity_id)

            elif action == "remove":
                # Entity removed - remove from index
                await self._on_entity_removed(entity_id)

            elif action == "update":
                # Entity updated - reindex it
                changes = event.data.get("changes", {})
                await self._on_entity_updated(entity_id, changes)

        except Exception as e:
            _LOGGER.error("Error handling entity registry event: %s", e)

    async def _on_entity_added(self, entity_id: str) -> None:
        """Handle entity added event.

        Args:
            entity_id: The new entity ID.
        """
        _LOGGER.debug("Entity added, indexing: %s", entity_id)
        try:
            await self.indexer.index_entity(entity_id)
        except Exception as e:
            _LOGGER.error("Failed to index new entity %s: %s", entity_id, e)

    async def _on_entity_removed(self, entity_id: str) -> None:
        """Handle entity removed event.

        Args:
            entity_id: The removed entity ID.
        """
        _LOGGER.debug("Entity removed, removing from index: %s", entity_id)
        try:
            await self.indexer.remove_entity(entity_id)
        except Exception as e:
            _LOGGER.error("Failed to remove entity %s from index: %s", entity_id, e)

    async def _on_entity_updated(
        self,
        entity_id: str,
        changes: dict[str, Any],
    ) -> None:
        """Handle entity updated event.

        Reindexes if relevant attributes changed (name, area, device_class).

        Args:
            entity_id: The updated entity ID.
            changes: Dictionary of changed attributes.
        """
        # Only reindex if searchable attributes changed
        relevant_changes = {
            "name",
            "area_id",
            "device_class",
            "original_device_class",
            "original_name",
        }

        if changes and not relevant_changes.intersection(changes.keys()):
            _LOGGER.debug(
                "Entity %s updated but no relevant changes, skipping reindex",
                entity_id,
            )
            return

        _LOGGER.debug("Entity updated, reindexing: %s", entity_id)
        try:
            await self.indexer.index_entity(entity_id)
        except Exception as e:
            _LOGGER.error("Failed to reindex updated entity %s: %s", entity_id, e)


@dataclass
class StateChangeHandler:
    """Optional handler for state change events.

    This can be used to update embeddings when entity states change,
    but is typically not needed since state is not the primary search criteria.

    Disabled by default to reduce overhead.
    """

    hass: HomeAssistant
    indexer: EntityIndexer
    enabled: bool = False
    _unsub_listener: Callable[[], None] | None = field(default=None, repr=False)

    async def async_start(self) -> None:
        """Start listening to state change events."""
        if not self.enabled:
            return

        from homeassistant.const import EVENT_STATE_CHANGED

        self._unsub_listener = self.hass.bus.async_listen(
            EVENT_STATE_CHANGED,
            self._handle_state_changed,
        )
        _LOGGER.info("RAG state change handler started")

    async def async_stop(self) -> None:
        """Stop listening to state changes."""
        if self._unsub_listener:
            self._unsub_listener()
            self._unsub_listener = None
        _LOGGER.info("RAG state change handler stopped")

    async def _handle_state_changed(self, event: Event) -> None:
        """Handle state change events.

        Args:
            event: The state changed event.
        """
        # This is intentionally minimal - state changes are frequent
        # and typically don't affect search relevance
        entity_id = event.data.get("entity_id")
        if not entity_id:
            return

        # Only reindex if it's a significant state change
        # (e.g., friendly_name in attributes changed)
        old_state = event.data.get("old_state")
        new_state = event.data.get("new_state")

        if old_state and new_state:
            old_name = old_state.attributes.get("friendly_name")
            new_name = new_state.attributes.get("friendly_name")

            if old_name != new_name:
                _LOGGER.debug(
                    "Entity %s friendly_name changed, reindexing",
                    entity_id,
                )
                await self.indexer.index_entity(entity_id)
