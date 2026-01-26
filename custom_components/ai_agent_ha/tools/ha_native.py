"""Native Home Assistant tools for AI Agent.

These tools wrap standard Home Assistant functionality to make it accessible
to the AI agent via native function calling.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from .base import Tool, ToolRegistry, ToolResult, ToolParameter, ToolCategory

_LOGGER = logging.getLogger(__name__)


@ToolRegistry.register
class GetEntityState(Tool):
    id = "get_entity_state"
    description = "Get the state and attributes of a specific entity."
    category = ToolCategory.HOME_ASSISTANT
    parameters = [
        ToolParameter(
            name="entity_id",
            type="string",
            description="The entity ID (e.g. light.living_room)",
            required=True,
        )
    ]

    async def execute(self, entity_id: str, **kwargs) -> ToolResult:
        if not entity_id:
            return ToolResult(output="Entity ID is required", error="Missing entity_id", success=False)

        state = self.hass.states.get(entity_id)
        if not state:
            return ToolResult(
                output=f"Entity {entity_id} not found", 
                error=f"Entity {entity_id} not found", 
                success=False
            )

        # Basic state info
        result = {
            "entity_id": state.entity_id,
            "state": state.state,
            "attributes": dict(state.attributes),
            "last_changed": state.last_changed.isoformat() if state.last_changed else None,
            "last_updated": state.last_updated.isoformat() if state.last_updated else None,
        }
        
        # Format as JSON string for output
        return ToolResult(output=json.dumps(result, default=str), metadata=result)


@ToolRegistry.register
class GetEntitiesByDomain(Tool):
    id = "get_entities_by_domain"
    description = "Get all entities for a specific domain (e.g., light, switch, sensor)."
    category = ToolCategory.HOME_ASSISTANT
    parameters = [
        ToolParameter(
            name="domain",
            type="string",
            description="The domain to filter by (e.g. 'light', 'sensor')",
            required=True,
        )
    ]

    async def execute(self, domain: str, **kwargs) -> ToolResult:
        if not domain:
            return ToolResult(output="Domain is required", error="Missing domain", success=False)

        states = [
            state
            for state in self.hass.states.async_all()
            if state.entity_id.startswith(f"{domain}.")
        ]
        
        results = []
        for state in states:
            results.append({
                "entity_id": state.entity_id,
                "state": state.state,
                "attributes": dict(state.attributes)
            })

        return ToolResult(
            output=json.dumps(results, default=str),
            metadata={"count": len(results), "entities": results}
        )


@ToolRegistry.register
class GetEntityRegistrySummary(Tool):
    id = "get_entity_registry_summary"
    description = "Get a summary of all entities in the system, counted by domain, area, and device_class."
    category = ToolCategory.HOME_ASSISTANT
    parameters = []  # No parameters

    async def execute(self, **kwargs) -> ToolResult:
        # This mirrors agent.py implementation logic
        from homeassistant.helpers import area_registry as ar
        from homeassistant.helpers import device_registry as dr
        from homeassistant.helpers import entity_registry as er

        entity_registry = er.async_get(self.hass)
        if not entity_registry:
            return ToolResult(output="{}", metadata={})

        device_registry = dr.async_get(self.hass)
        area_registry = ar.async_get(self.hass)

        area_names = {}
        if area_registry:
            for area in area_registry.areas.values():
                area_names[area.id] = area.name

        by_domain = {}
        by_area = {}
        by_device_class = {}
        total = 0

        for entry in entity_registry.entities.values():
            if entry.disabled:
                continue

            total += 1
            domain = entry.entity_id.split(".")[0]
            by_domain[domain] = by_domain.get(domain, 0) + 1

            area_id = entry.area_id
            if not area_id and entry.device_id and device_registry:
                device_entry = device_registry.async_get(entry.device_id)
                if device_entry:
                    area_id = device_entry.area_id

            if area_id:
                area_name = area_names.get(area_id, area_id)
                by_area[area_name] = by_area.get(area_name, 0) + 1
            else:
                by_area["unassigned"] = by_area.get("unassigned", 0) + 1

            state = self.hass.states.get(entry.entity_id)
            if state:
                device_class = state.attributes.get("device_class")
                if device_class:
                    by_device_class[device_class] = by_device_class.get(device_class, 0) + 1

        summary = {
            "total_entities": total,
            "by_domain": dict(sorted(by_domain.items(), key=lambda x: x[1], reverse=True)),
            "by_area": dict(sorted(by_area.items(), key=lambda x: x[1], reverse=True)),
            "by_device_class": dict(sorted(by_device_class.items(), key=lambda x: x[1], reverse=True)),
        }

        return ToolResult(output=json.dumps(summary, default=str), metadata=summary)


@ToolRegistry.register
class GetEntityRegistry(Tool):
    id = "get_entity_registry"
    description = "Get list of entities filtered by domain, area, or device_class."
    category = ToolCategory.HOME_ASSISTANT
    parameters = [
        ToolParameter(name="domain", type="string", description="Filter by domain", required=False),
        ToolParameter(name="area_id", type="string", description="Filter by area ID", required=False),
        ToolParameter(name="device_class", type="string", description="Filter by device class", required=False),
        ToolParameter(name="limit", type="integer", description="Max results (default 50)", required=False, default=50),
        ToolParameter(name="offset", type="integer", description="Pagination offset", required=False, default=0),
    ]

    async def execute(self, domain: Optional[str] = None, area_id: Optional[str] = None, 
                      device_class: Optional[str] = None, limit: int = 50, offset: int = 0, **kwargs) -> ToolResult:
        # Simplified implementation that returns basic list, real logic is complex
        # This is primarily for schema generation so Gemini knows how to call it
        # If execution falls through to here, we return a placeholder or simple logic
        
        # NOTE: agent.py logic is complex (resolves names, areas, etc). 
        # Ideally we should call agent.get_entity_registry if possible, but we don't have access to agent here.
        # For now, this tool definition primarily serves to provide the SCHEMA to Gemini.
        # agent.py process_query loop will intercept the request_type and run its own logic.
        
        return ToolResult(
            output="Please use the built-in agent handler for this complex query.",
            success=True
        )


@ToolRegistry.register
class CallService(Tool):
    id = "call_service"
    description = "Call a Home Assistant service to control devices."
    category = ToolCategory.HOME_ASSISTANT
    parameters = [
        ToolParameter(name="domain", type="string", description="Service domain (e.g., light)", required=True),
        ToolParameter(name="service", type="string", description="Service name (e.g., turn_on)", required=True),
        ToolParameter(name="target", type="dict", description="Target entities (e.g., {'entity_id': 'light.kitchen'})", required=False),
        ToolParameter(name="service_data", type="dict", description="Service data (e.g., {'brightness': 255})", required=False),
    ]

    async def execute(self, domain: str, service: str, target: Optional[Dict] = None, 
                      service_data: Optional[Dict] = None, **kwargs) -> ToolResult:
        # Schema-only definition for Gemini
        return ToolResult(output="Service called via agent logic", success=True)
