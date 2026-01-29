"""Compatibility layer for migration from old AiAgentHaAgent to new Agent."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from .core.agent import Agent
from .core.conversation import ConversationManager
from .providers.registry import ProviderRegistry

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant
    from homeassistant.config_entries import ConfigEntry

_LOGGER = logging.getLogger(__name__)


class AiAgentHaAgent:
    """Compatibility wrapper that delegates to new modular Agent.

    This class maintains the same interface as the old monolithic agent
    while internally using the new architecture.
    """

    def __init__(
        self,
        hass: HomeAssistant,
        config: dict[str, Any],
        config_entry: ConfigEntry | None = None,
    ) -> None:
        """Initialize with same signature as old agent."""
        self.hass = hass
        self.config = config
        self.config_entry = config_entry

        # Determine provider and create new architecture components
        self._provider_name = config.get("ai_provider", "openai")
        self._setup_provider()
        self._setup_agent()

        # RAG manager (set externally)
        self._rag_manager = None

        _LOGGER.info("AiAgentHaAgent initialized with new architecture (provider: %s)", self._provider_name)

    def _setup_provider(self) -> None:
        """Create AI provider from config."""
        # Map old config keys to new provider config
        provider_config = self._build_provider_config()

        try:
            self._provider = ProviderRegistry.create(
                self._provider_name,
                self.hass,
                provider_config
            )
        except ValueError:
            # Fallback for OAuth providers or unknown providers
            # They map to base providers
            base_provider = self._get_base_provider_name()
            self._provider = ProviderRegistry.create(
                base_provider,
                self.hass,
                provider_config
            )

    def _get_base_provider_name(self) -> str:
        """Map OAuth provider names to base provider names.

        Note: gemini_oauth is now a registered provider, so it doesn't fall back,
        but we still need the mapping for model/token lookups.
        """
        mapping = {
            "anthropic_oauth": "anthropic",
            "gemini_oauth": "gemini",
            "openai_oauth": "openai",
        }
        return mapping.get(self._provider_name, "openai")

    def _build_provider_config(self) -> dict[str, Any]:
        """Build provider config from old config format."""
        provider = self._provider_name
        base_provider = self._get_base_provider_name()

        # OAuth providers use config_entry for tokens
        is_oauth_provider = provider.endswith("_oauth")

        # Get token for this provider (API key providers)
        token_keys = {
            "openai": "openai_token",
            "gemini": "gemini_token",
            "anthropic": "anthropic_token",
            "groq": "groq_token",
            "openrouter": "openrouter_token",
            "local": None,  # No token needed
        }

        token_key = token_keys.get(base_provider, f"{base_provider}_token")
        token = self.config.get(token_key, "") if token_key else ""

        # Get model for this provider
        models = self.config.get("models", {})

        # For OAuth providers, try provider name first (e.g., gemini_oauth),
        # then fall back to base provider (e.g., gemini)
        if is_oauth_provider:
            model = models.get(provider, models.get(base_provider, self._get_default_model(provider)))
        else:
            model = models.get(provider, self._get_default_model(provider))

        config: dict[str, Any] = {
            "token": token,
            "model": model,
            "api_url": self.config.get(f"{base_provider}_api_url"),
        }

        # OAuth providers need config_entry for token management
        if is_oauth_provider and self.config_entry:
            config["config_entry"] = self.config_entry

        return config

    def _get_default_model(self, provider: str) -> str:
        """Get default model for provider."""
        defaults = {
            "openai": "gpt-4",
            "gemini": "gemini-2.5-flash",
            "gemini_oauth": "gemini-3-pro-preview",
            "anthropic": "claude-sonnet-4-5-20250929",
            "anthropic_oauth": "claude-sonnet-4-5-20250929",
            "groq": "llama-3.3-70b-versatile",
            "openrouter": "openai/gpt-4",
            "local": "llama2",
        }
        return defaults.get(provider, "gpt-4")

    def _setup_agent(self) -> None:
        """Create new Agent orchestrator."""
        from .managers.entity_manager import EntityManager
        from .managers.registry_manager import RegistryManager
        from .managers.automation_manager import AutomationManager
        from .managers.dashboard_manager import DashboardManager
        from .managers.control_manager import ControlManager

        self._agent = Agent(
            hass=self.hass,
            provider=self._provider,
            entity_manager=EntityManager(self.hass),
            registry_manager=RegistryManager(self.hass),
            automation_manager=AutomationManager(self.hass),
            dashboard_manager=DashboardManager(self.hass),
            control_manager=ControlManager(self.hass),
        )

    # === PUBLIC API (same signatures as old agent) ===

    def _get_tools_for_provider(self) -> list[dict[str, Any]] | None:
        """Get tools in OpenAI format for the current provider.

        Returns:
            List of tools in OpenAI format, or None if provider doesn't support tools.
        """
        if not self._provider.supports_tools:
            return None

        try:
            from .tools import ToolRegistry
            from .function_calling import ToolSchemaConverter

            tools = ToolRegistry.get_all_tools(
                hass=self.hass,
                config=self.config,
                enabled_only=True,
            )

            if not tools:
                _LOGGER.debug("No tools available for native function calling")
                return None

            openai_tools = ToolSchemaConverter.to_openai_format(tools)
            _LOGGER.debug("Retrieved %d tools for native function calling", len(tools))
            return openai_tools

        except Exception as e:
            _LOGGER.warning("Failed to get tools for native function calling: %s", e)
            return None

    async def process_query(
        self,
        user_query: str,
        provider: str | None = None,
        debug: bool = False,
        conversation_history: list[dict] | None = None,
    ) -> dict[str, Any]:
        """Process a user query through the AI provider.

        Same signature as old AiAgentHaAgent.process_query().
        """
        # If external conversation history provided, use it
        if conversation_history:
            # Inject history into conversation manager
            self._agent._conversation.clear()
            for msg in conversation_history:
                self._agent._conversation.add_message(
                    msg.get("role", "user"),
                    msg.get("content", "")
                )

        # Build kwargs
        kwargs = {}
        if debug:
            kwargs["debug"] = debug

        # Add tools for native function calling
        tools = self._get_tools_for_provider()
        if tools:
            kwargs["tools"] = tools

        # Add RAG context if available
        if self._rag_manager:
            rag_context = await self._get_rag_context(user_query)
            if rag_context:
                kwargs["rag_context"] = rag_context

        try:
            result = await self._agent.process_query(user_query, **kwargs)

            # Transform to old response format
            return {
                "success": result.get("success", True),
                "answer": result.get("response", ""),
                "automation": result.get("automation"),
                "dashboard": result.get("dashboard"),
                "debug": result.get("debug") if debug else None,
            }
        except Exception as e:
            _LOGGER.error("Error processing query: %s", e)
            return {
                "success": False,
                "error": str(e),
            }

    async def create_automation(self, automation_config: dict) -> dict[str, Any]:
        """Create a new automation."""
        return await self._agent.create_automation(automation_config)

    async def create_dashboard(self, dashboard_config: dict) -> dict[str, Any]:
        """Create a new dashboard."""
        return await self._agent.create_dashboard(dashboard_config)

    async def update_dashboard(
        self,
        dashboard_url: str,
        dashboard_config: dict,
    ) -> dict[str, Any]:
        """Update an existing dashboard."""
        return await self._agent._get_dashboard_manager().update_dashboard(
            dashboard_url, dashboard_config
        )

    async def save_user_prompt_history(
        self,
        user_id: str,
        history: list[str],
    ) -> dict[str, Any]:
        """Save user prompt history."""
        # This is a storage operation - delegate to storage module
        try:
            from .storage import save_prompt_history
            await save_prompt_history(self.hass, user_id, history)
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def load_user_prompt_history(self, user_id: str) -> dict[str, Any]:
        """Load user prompt history."""
        try:
            from .storage import load_prompt_history
            history = await load_prompt_history(self.hass, user_id)
            return {"success": True, "history": history}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def set_rag_manager(self, rag_manager) -> None:
        """Set the RAG manager for semantic search."""
        self._rag_manager = rag_manager

    async def _get_rag_context(self, query: str) -> str | None:
        """Get relevant context from RAG system."""
        if not self._rag_manager:
            return None
        try:
            results = await self._rag_manager.query(query)
            if results:
                return "\n".join(str(r) for r in results[:5])
        except Exception as e:
            _LOGGER.warning("RAG query failed: %s", e)
        return None

    # === ENTITY OPERATIONS (delegate to EntityManager) ===

    def get_entity_state(self, entity_id: str) -> dict | None:
        """Get entity state."""
        return self._agent.get_entity_state(entity_id)

    def get_entities_by_domain(self, domain: str) -> list[dict]:
        """Get entities by domain."""
        return self._agent.get_entities_by_domain(domain)

    # === CONTROL OPERATIONS (delegate to ControlManager) ===

    async def call_service(
        self,
        domain: str,
        service: str,
        **kwargs,
    ) -> dict[str, Any]:
        """Call a Home Assistant service."""
        return await self._agent.call_service(domain, service, **kwargs)

    # === CONVERSATION ===

    def clear_conversation_history(self) -> None:
        """Clear conversation history."""
        self._agent.clear_conversation()
