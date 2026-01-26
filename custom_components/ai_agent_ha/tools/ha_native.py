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


@ToolRegistry.register
class GetHistory(Tool):
    id = "get_history"
    description = "Get historical state changes for an entity over a specified time period."
    category = ToolCategory.HOME_ASSISTANT
    parameters = [
        ToolParameter(
            name="entity_id",
            type="string",
            description="The entity ID to get history for (e.g. sensor.temperature)",
            required=True,
        ),
        ToolParameter(
            name="hours",
            type="integer",
            description="Number of hours of history to retrieve (default: 24)",
            required=False,
            default=24,
        ),
    ]

    async def execute(self, entity_id: str, hours: int = 24, **kwargs) -> ToolResult:
        """Get historical state changes for an entity."""
        if not entity_id:
            return ToolResult(output="Entity ID is required", error="Missing entity_id", success=False)

        try:
            from datetime import datetime, timedelta
            from homeassistant.components.recorder.history import get_significant_states

            start_time = datetime.now() - timedelta(hours=hours)
            end_time = datetime.now()

            # Get history using the recorder history module
            history_data = await self.hass.async_add_executor_job(
                get_significant_states,
                self.hass,
                start_time,
                end_time,
                [entity_id],
            )

            results = []
            for entity_id_key, states in history_data.items():
                for state in states:
                    results.append({
                        "state": state.state,
                        "timestamp": state.last_changed.isoformat() if state.last_changed else None,
                        "attributes": dict(state.attributes) if state.attributes else {},
                    })

            _LOGGER.debug("Retrieved %d historical states for %s", len(results), entity_id)
            return ToolResult(
                output=json.dumps(results, default=str),
                metadata={"entity_id": entity_id, "count": len(results), "hours": hours}
            )

        except Exception as e:
            _LOGGER.error("Error getting history for %s: %s", entity_id, e)
            return ToolResult(
                output=f"Error getting history: {str(e)}",
                error=str(e),
                success=False
            )


@ToolRegistry.register
class GetEntitiesByDeviceClass(Tool):
    id = "get_entities_by_device_class"
    description = "Get all entities with a specific device_class (e.g., temperature, humidity, motion)."
    category = ToolCategory.HOME_ASSISTANT
    parameters = [
        ToolParameter(
            name="device_class",
            type="string",
            description="The device class to filter by (e.g., 'temperature', 'humidity', 'motion')",
            required=True,
        ),
        ToolParameter(
            name="domain",
            type="string",
            description="Optional domain to restrict search (e.g., 'sensor', 'binary_sensor')",
            required=False,
        ),
    ]

    async def execute(self, device_class: str, domain: Optional[str] = None, **kwargs) -> ToolResult:
        if not device_class:
            return ToolResult(output="Device class is required", error="Missing device_class", success=False)

        matching_entities = []
        for state in self.hass.states.async_all():
            if domain and not state.entity_id.startswith(f"{domain}."):
                continue
            entity_device_class = state.attributes.get("device_class")
            if entity_device_class == device_class:
                matching_entities.append({
                    "entity_id": state.entity_id,
                    "state": state.state,
                    "attributes": dict(state.attributes),
                })

        return ToolResult(
            output=json.dumps(matching_entities, default=str),
            metadata={"count": len(matching_entities), "device_class": device_class}
        )


@ToolRegistry.register
class GetEntitiesByArea(Tool):
    id = "get_entities_by_area"
    description = "Get all entities for a specific area."
    category = ToolCategory.HOME_ASSISTANT
    parameters = [
        ToolParameter(
            name="area_id",
            type="string",
            description="The area ID to filter by",
            required=True,
        ),
    ]

    async def execute(self, area_id: str, **kwargs) -> ToolResult:
        if not area_id:
            return ToolResult(output="Area ID is required", error="Missing area_id", success=False)

        from homeassistant.helpers import device_registry as dr
        from homeassistant.helpers import entity_registry as er

        entity_registry = er.async_get(self.hass)
        device_registry = dr.async_get(self.hass)

        entities_in_area = []
        for entity in entity_registry.entities.values():
            if entity.area_id == area_id:
                entities_in_area.append(entity.entity_id)
            elif entity.device_id:
                device = device_registry.devices.get(entity.device_id)
                if device and device.area_id == area_id:
                    entities_in_area.append(entity.entity_id)

        results = []
        for entity_id in entities_in_area:
            state = self.hass.states.get(entity_id)
            if state:
                results.append({
                    "entity_id": state.entity_id,
                    "state": state.state,
                    "attributes": dict(state.attributes),
                })

        return ToolResult(
            output=json.dumps(results, default=str),
            metadata={"count": len(results), "area_id": area_id}
        )


@ToolRegistry.register
class GetEntities(Tool):
    id = "get_entities"
    description = "Get entities by area(s) - supports single area or multiple areas."
    category = ToolCategory.HOME_ASSISTANT
    parameters = [
        ToolParameter(
            name="area_id",
            type="string",
            description="Single area ID to filter by",
            required=False,
        ),
        ToolParameter(
            name="area_ids",
            type="array",
            description="List of area IDs to filter by",
            required=False,
        ),
    ]

    async def execute(self, area_id: Optional[str] = None, area_ids: Optional[List[str]] = None, **kwargs) -> ToolResult:
        # Schema-only definition - agent.py handles the complex logic
        return ToolResult(output="Please use the built-in agent handler for this query.", success=True)


@ToolRegistry.register
class GetClimateRelatedEntities(Tool):
    id = "get_climate_related_entities"
    description = "Get all climate-related entities including thermostats, temperature sensors, and humidity sensors."
    category = ToolCategory.HOME_ASSISTANT
    parameters = []

    async def execute(self, **kwargs) -> ToolResult:
        # Schema-only definition - agent.py handles the logic
        return ToolResult(output="Please use the built-in agent handler for this query.", success=True)


@ToolRegistry.register
class GetStatistics(Tool):
    id = "get_statistics"
    description = "Get statistics (mean, min, max, sum) for an entity from the recorder."
    category = ToolCategory.HOME_ASSISTANT
    parameters = [
        ToolParameter(
            name="entity_id",
            type="string",
            description="The entity ID to get statistics for",
            required=True,
        ),
    ]

    async def execute(self, entity_id: str, **kwargs) -> ToolResult:
        if not entity_id:
            return ToolResult(output="Entity ID is required", error="Missing entity_id", success=False)

        try:
            from homeassistant.components import recorder
            import homeassistant.components.recorder.statistics as stats_module

            if not self.hass.data.get(recorder.DATA_INSTANCE):
                return ToolResult(output="Recorder component is not available", error="No recorder", success=False)

            stats = await self.hass.async_add_executor_job(
                stats_module.get_last_short_term_statistics,
                self.hass,
                1,
                entity_id,
                True,
                set(),
            )

            if entity_id in stats:
                stat_data = stats[entity_id][0] if stats[entity_id] else {}
                result = {
                    "entity_id": entity_id,
                    "start": stat_data.get("start"),
                    "mean": stat_data.get("mean"),
                    "min": stat_data.get("min"),
                    "max": stat_data.get("max"),
                    "last_reset": stat_data.get("last_reset"),
                    "state": stat_data.get("state"),
                    "sum": stat_data.get("sum"),
                }
                return ToolResult(output=json.dumps(result, default=str), metadata=result)
            else:
                return ToolResult(
                    output=f"No statistics available for entity {entity_id}",
                    error="No statistics",
                    success=False
                )
        except Exception as e:
            _LOGGER.error("Error getting statistics for %s: %s", entity_id, e)
            return ToolResult(output=f"Error getting statistics: {str(e)}", error=str(e), success=False)


@ToolRegistry.register
class GetDeviceRegistrySummary(Tool):
    id = "get_device_registry_summary"
    description = "Get a summary of all devices in the system, counted by manufacturer, area, and integration."
    category = ToolCategory.HOME_ASSISTANT
    parameters = []

    async def execute(self, **kwargs) -> ToolResult:
        # Schema-only definition - agent.py handles the complex logic
        return ToolResult(output="Please use the built-in agent handler for this query.", success=True)


@ToolRegistry.register
class GetDeviceRegistry(Tool):
    id = "get_device_registry"
    description = "Get device registry entries with filtering and pagination."
    category = ToolCategory.HOME_ASSISTANT
    parameters = [
        ToolParameter(name="area_id", type="string", description="Filter by area ID", required=False),
        ToolParameter(name="manufacturer", type="string", description="Filter by manufacturer name", required=False),
        ToolParameter(name="limit", type="integer", description="Max results (default 50, max 200)", required=False, default=50),
        ToolParameter(name="offset", type="integer", description="Pagination offset", required=False, default=0),
    ]

    async def execute(self, area_id: Optional[str] = None, manufacturer: Optional[str] = None,
                      limit: int = 50, offset: int = 0, **kwargs) -> ToolResult:
        # Schema-only definition - agent.py handles the complex logic
        return ToolResult(output="Please use the built-in agent handler for this query.", success=True)


@ToolRegistry.register
class GetAreaRegistry(Tool):
    id = "get_area_registry"
    description = "Get all areas defined in Home Assistant with their details."
    category = ToolCategory.HOME_ASSISTANT
    parameters = []

    async def execute(self, **kwargs) -> ToolResult:
        from homeassistant.helpers import area_registry as ar

        registry = ar.async_get(self.hass)
        if not registry:
            return ToolResult(output="{}", metadata={})

        result = {}
        for area in registry.areas.values():
            result[area.id] = {
                "name": area.name,
                "normalized_name": area.normalized_name,
                "picture": area.picture,
                "icon": area.icon,
                "floor_id": area.floor_id,
                "labels": list(area.labels) if area.labels else [],
            }

        return ToolResult(output=json.dumps(result, default=str), metadata=result)


@ToolRegistry.register
class GetWeatherData(Tool):
    id = "get_weather_data"
    description = "Get current weather data and forecast from available weather entities."
    category = ToolCategory.HOME_ASSISTANT
    parameters = []

    async def execute(self, **kwargs) -> ToolResult:
        weather_entities = [
            state for state in self.hass.states.async_all()
            if state.domain == "weather"
        ]

        if not weather_entities:
            return ToolResult(
                output="No weather entities found in the system.",
                error="No weather entities",
                success=False
            )

        state = weather_entities[0]
        attrs = state.attributes
        forecast = attrs.get("forecast", [])

        processed_forecast = []
        for day in forecast:
            entry = {
                "datetime": day.get("datetime"),
                "temperature": day.get("temperature"),
                "condition": day.get("condition"),
                "precipitation": day.get("precipitation"),
                "precipitation_probability": day.get("precipitation_probability"),
                "humidity": day.get("humidity"),
                "wind_speed": day.get("wind_speed"),
            }
            if any(v is not None for v in entry.values()):
                processed_forecast.append(entry)

        current = {
            "entity_id": state.entity_id,
            "temperature": attrs.get("temperature"),
            "humidity": attrs.get("humidity"),
            "pressure": attrs.get("pressure"),
            "wind_speed": attrs.get("wind_speed"),
            "wind_bearing": attrs.get("wind_bearing"),
            "condition": state.state,
            "forecast_available": len(processed_forecast) > 0,
        }

        result = {"current": current, "forecast": processed_forecast}
        return ToolResult(output=json.dumps(result, default=str), metadata=result)


@ToolRegistry.register
class GetCalendarEvents(Tool):
    id = "get_calendar_events"
    description = "Get calendar events from calendar entities."
    category = ToolCategory.HOME_ASSISTANT
    parameters = [
        ToolParameter(
            name="entity_id",
            type="string",
            description="Optional specific calendar entity ID to query",
            required=False,
        ),
    ]

    async def execute(self, entity_id: Optional[str] = None, **kwargs) -> ToolResult:
        # Schema-only definition - agent.py handles calendar logic
        return ToolResult(output="Please use the built-in agent handler for this query.", success=True)


@ToolRegistry.register
class GetAutomations(Tool):
    id = "get_automations"
    description = "Get all automations in the system."
    category = ToolCategory.HOME_ASSISTANT
    parameters = []

    async def execute(self, **kwargs) -> ToolResult:
        states = [
            state for state in self.hass.states.async_all()
            if state.entity_id.startswith("automation.")
        ]

        results = []
        for state in states:
            results.append({
                "entity_id": state.entity_id,
                "state": state.state,
                "friendly_name": state.attributes.get("friendly_name"),
                "last_triggered": state.attributes.get("last_triggered"),
            })

        return ToolResult(
            output=json.dumps(results, default=str),
            metadata={"count": len(results)}
        )


@ToolRegistry.register
class GetScenes(Tool):
    id = "get_scenes"
    description = "Get all scenes in the system."
    category = ToolCategory.HOME_ASSISTANT
    parameters = []

    async def execute(self, **kwargs) -> ToolResult:
        results = []
        for state in self.hass.states.async_all("scene"):
            results.append({
                "entity_id": state.entity_id,
                "name": state.attributes.get("friendly_name", state.entity_id),
                "last_activated": state.attributes.get("last_activated"),
                "icon": state.attributes.get("icon"),
                "last_changed": state.last_changed.isoformat() if state.last_changed else None,
            })

        return ToolResult(
            output=json.dumps(results, default=str),
            metadata={"count": len(results)}
        )


@ToolRegistry.register
class GetPersonData(Tool):
    id = "get_person_data"
    description = "Get person tracking information including location data."
    category = ToolCategory.HOME_ASSISTANT
    parameters = []

    async def execute(self, **kwargs) -> ToolResult:
        results = []
        for state in self.hass.states.async_all("person"):
            results.append({
                "entity_id": state.entity_id,
                "name": state.attributes.get("friendly_name", state.entity_id),
                "state": state.state,
                "latitude": state.attributes.get("latitude"),
                "longitude": state.attributes.get("longitude"),
                "source": state.attributes.get("source"),
                "gps_accuracy": state.attributes.get("gps_accuracy"),
                "last_changed": state.last_changed.isoformat() if state.last_changed else None,
            })

        return ToolResult(
            output=json.dumps(results, default=str),
            metadata={"count": len(results)}
        )


@ToolRegistry.register
class GetDashboards(Tool):
    id = "get_dashboards"
    description = "Get list of all Lovelace dashboards."
    category = ToolCategory.HOME_ASSISTANT
    parameters = []

    async def execute(self, **kwargs) -> ToolResult:
        # Schema-only definition - agent.py handles the Lovelace API complexity
        return ToolResult(output="Please use the built-in agent handler for this query.", success=True)


@ToolRegistry.register
class GetDashboardConfig(Tool):
    id = "get_dashboard_config"
    description = "Get configuration of a specific dashboard."
    category = ToolCategory.HOME_ASSISTANT
    parameters = [
        ToolParameter(
            name="dashboard_url",
            type="string",
            description="Dashboard URL path (None for default dashboard)",
            required=False,
        ),
    ]

    async def execute(self, dashboard_url: Optional[str] = None, **kwargs) -> ToolResult:
        # Schema-only definition - agent.py handles the Lovelace API complexity
        return ToolResult(output="Please use the built-in agent handler for this query.", success=True)
