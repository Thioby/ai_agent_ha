"""Tests for compatibility layer."""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from custom_components.ai_agent_ha.agent_compat import AiAgentHaAgent


class TestAiAgentHaAgentCompat:
    """Test compatibility wrapper."""

    @pytest.fixture
    def mock_hass(self):
        return MagicMock()

    @pytest.fixture
    def config(self):
        return {
            "ai_provider": "openai",
            "openai_token": "sk-test",
            "models": {"openai": "gpt-4"},
        }

    @pytest.fixture
    def patch_managers(self):
        """Patch all managers at their import locations."""
        with patch("custom_components.ai_agent_ha.managers.entity_manager.EntityManager") as mock_entity, \
             patch("custom_components.ai_agent_ha.managers.registry_manager.RegistryManager") as mock_registry_mgr, \
             patch("custom_components.ai_agent_ha.managers.automation_manager.AutomationManager") as mock_automation, \
             patch("custom_components.ai_agent_ha.managers.dashboard_manager.DashboardManager") as mock_dashboard, \
             patch("custom_components.ai_agent_ha.managers.control_manager.ControlManager") as mock_control:
            yield {
                "entity": mock_entity,
                "registry": mock_registry_mgr,
                "automation": mock_automation,
                "dashboard": mock_dashboard,
                "control": mock_control,
            }

    @patch("custom_components.ai_agent_ha.agent_compat.ProviderRegistry")
    def test_init_creates_agent(self, mock_registry, patch_managers, mock_hass, config):
        """Test initialization creates new Agent."""
        mock_provider = MagicMock()
        mock_registry.create.return_value = mock_provider

        agent = AiAgentHaAgent(mock_hass, config)

        assert agent.hass == mock_hass
        assert agent._provider == mock_provider
        mock_registry.create.assert_called_once()

    @patch("custom_components.ai_agent_ha.agent_compat.ProviderRegistry")
    def test_init_with_config_entry(self, mock_registry, patch_managers, mock_hass, config):
        """Test initialization with config entry."""
        config_entry = MagicMock()
        mock_registry.create.return_value = MagicMock()

        agent = AiAgentHaAgent(mock_hass, config, config_entry)
        assert agent.config_entry == config_entry

    @pytest.mark.asyncio
    @patch("custom_components.ai_agent_ha.agent_compat.ProviderRegistry")
    async def test_process_query_delegates(self, mock_registry, patch_managers, mock_hass, config):
        """Test process_query delegates to new Agent."""
        mock_registry.create.return_value = MagicMock()
        agent = AiAgentHaAgent(mock_hass, config)
        agent._agent.process_query = AsyncMock(return_value={
            "success": True,
            "response": "Hello!",
        })

        result = await agent.process_query("Hi")

        assert result["success"] is True
        assert result["answer"] == "Hello!"

    @patch("custom_components.ai_agent_ha.agent_compat.ProviderRegistry")
    def test_set_rag_manager(self, mock_registry, patch_managers, mock_hass, config):
        """Test RAG manager injection."""
        mock_registry.create.return_value = MagicMock()
        agent = AiAgentHaAgent(mock_hass, config)
        rag = MagicMock()

        agent.set_rag_manager(rag)

        assert agent._rag_manager == rag

    @patch("custom_components.ai_agent_ha.agent_compat.ProviderRegistry")
    def test_clear_conversation_history(self, mock_registry, patch_managers, mock_hass, config):
        """Test clearing conversation history."""
        mock_registry.create.return_value = MagicMock()
        agent = AiAgentHaAgent(mock_hass, config)
        agent._agent.clear_conversation = MagicMock()

        agent.clear_conversation_history()

        agent._agent.clear_conversation.assert_called_once()

    @pytest.mark.asyncio
    @patch("custom_components.ai_agent_ha.agent_compat.ProviderRegistry")
    async def test_create_automation_delegates(self, mock_registry, patch_managers, mock_hass, config):
        """Test create_automation delegates to new Agent."""
        mock_registry.create.return_value = MagicMock()
        agent = AiAgentHaAgent(mock_hass, config)
        agent._agent.create_automation = AsyncMock(return_value={
            "success": True,
            "id": "automation.test",
        })

        result = await agent.create_automation({"alias": "Test"})

        assert result["success"] is True
        agent._agent.create_automation.assert_called_once_with({"alias": "Test"})

    @pytest.mark.asyncio
    @patch("custom_components.ai_agent_ha.agent_compat.ProviderRegistry")
    async def test_create_dashboard_delegates(self, mock_registry, patch_managers, mock_hass, config):
        """Test create_dashboard delegates to new Agent."""
        mock_registry.create.return_value = MagicMock()
        agent = AiAgentHaAgent(mock_hass, config)
        agent._agent.create_dashboard = AsyncMock(return_value={
            "success": True,
        })

        result = await agent.create_dashboard({"title": "Test"})

        assert result["success"] is True
        agent._agent.create_dashboard.assert_called_once_with({"title": "Test"})

    @pytest.mark.asyncio
    @patch("custom_components.ai_agent_ha.agent_compat.ProviderRegistry")
    async def test_call_service_delegates(self, mock_registry, patch_managers, mock_hass, config):
        """Test call_service delegates to new Agent."""
        mock_registry.create.return_value = MagicMock()
        agent = AiAgentHaAgent(mock_hass, config)
        agent._agent.call_service = AsyncMock(return_value={
            "success": True,
        })

        result = await agent.call_service("light", "turn_on", entity_id="light.test")

        assert result["success"] is True
        agent._agent.call_service.assert_called_once_with(
            "light", "turn_on", entity_id="light.test"
        )

    @patch("custom_components.ai_agent_ha.agent_compat.ProviderRegistry")
    def test_get_entity_state_delegates(self, mock_registry, patch_managers, mock_hass, config):
        """Test get_entity_state delegates to new Agent."""
        mock_registry.create.return_value = MagicMock()
        agent = AiAgentHaAgent(mock_hass, config)
        agent._agent.get_entity_state = MagicMock(return_value={"state": "on"})

        result = agent.get_entity_state("light.test")

        assert result == {"state": "on"}
        agent._agent.get_entity_state.assert_called_once_with("light.test")

    @patch("custom_components.ai_agent_ha.agent_compat.ProviderRegistry")
    def test_get_entities_by_domain_delegates(self, mock_registry, patch_managers, mock_hass, config):
        """Test get_entities_by_domain delegates to new Agent."""
        mock_registry.create.return_value = MagicMock()
        agent = AiAgentHaAgent(mock_hass, config)
        agent._agent.get_entities_by_domain = MagicMock(return_value=[{"entity_id": "light.test"}])

        result = agent.get_entities_by_domain("light")

        assert result == [{"entity_id": "light.test"}]
        agent._agent.get_entities_by_domain.assert_called_once_with("light")

    @pytest.mark.asyncio
    @patch("custom_components.ai_agent_ha.agent_compat.ProviderRegistry")
    async def test_process_query_with_conversation_history(self, mock_registry, patch_managers, mock_hass, config):
        """Test process_query with conversation history injection."""
        mock_registry.create.return_value = MagicMock()
        agent = AiAgentHaAgent(mock_hass, config)
        agent._agent._conversation.clear = MagicMock()
        agent._agent._conversation.add_message = MagicMock()
        agent._agent.process_query = AsyncMock(return_value={
            "success": True,
            "response": "Response with context",
        })

        history = [
            {"role": "user", "content": "Previous message"},
            {"role": "assistant", "content": "Previous response"},
        ]

        result = await agent.process_query("New question", conversation_history=history)

        assert result["success"] is True
        agent._agent._conversation.clear.assert_called_once()
        assert agent._agent._conversation.add_message.call_count == 2

    @pytest.mark.asyncio
    @patch("custom_components.ai_agent_ha.agent_compat.ProviderRegistry")
    async def test_process_query_error_handling(self, mock_registry, patch_managers, mock_hass, config):
        """Test process_query handles errors gracefully."""
        mock_registry.create.return_value = MagicMock()
        agent = AiAgentHaAgent(mock_hass, config)
        agent._agent.process_query = AsyncMock(side_effect=Exception("API Error"))

        result = await agent.process_query("Test query")

        assert result["success"] is False
        assert "API Error" in result["error"]

    @patch("custom_components.ai_agent_ha.agent_compat.ProviderRegistry")
    def test_get_base_provider_name_mapping(self, mock_registry, patch_managers, mock_hass, config):
        """Test OAuth provider name mapping."""
        mock_registry.create.return_value = MagicMock()

        agent = AiAgentHaAgent(mock_hass, config)

        assert agent._get_base_provider_name() == "openai"

        agent._provider_name = "anthropic_oauth"
        assert agent._get_base_provider_name() == "anthropic"

        agent._provider_name = "gemini_oauth"
        assert agent._get_base_provider_name() == "gemini"

        agent._provider_name = "unknown"
        assert agent._get_base_provider_name() == "openai"

    @patch("custom_components.ai_agent_ha.agent_compat.ProviderRegistry")
    def test_get_default_model(self, mock_registry, patch_managers, mock_hass, config):
        """Test default model selection for providers."""
        mock_registry.create.return_value = MagicMock()

        agent = AiAgentHaAgent(mock_hass, config)

        assert agent._get_default_model("openai") == "gpt-4"
        assert agent._get_default_model("anthropic") == "claude-sonnet-4-5-20250929"
        assert agent._get_default_model("gemini") == "gemini-2.5-flash"
        assert agent._get_default_model("groq") == "llama-3.3-70b-versatile"
        assert agent._get_default_model("unknown") == "gpt-4"

    @patch("custom_components.ai_agent_ha.agent_compat.ProviderRegistry")
    def test_build_provider_config(self, mock_registry, patch_managers, mock_hass, config):
        """Test provider config building from old config format."""
        mock_registry.create.return_value = MagicMock()

        agent = AiAgentHaAgent(mock_hass, config)

        provider_config = agent._build_provider_config()

        assert provider_config["token"] == "sk-test"
        assert provider_config["model"] == "gpt-4"

    @patch("custom_components.ai_agent_ha.agent_compat.ProviderRegistry")
    def test_provider_fallback_for_oauth(self, mock_registry, patch_managers, mock_hass, config):
        """Test provider fallback when OAuth provider is not directly registered."""
        # First call raises ValueError (unknown provider), second call succeeds
        mock_provider = MagicMock()
        mock_registry.create.side_effect = [ValueError("Unknown provider"), mock_provider]

        config["ai_provider"] = "anthropic_oauth"
        agent = AiAgentHaAgent(mock_hass, config)

        assert agent._provider == mock_provider
        # Should have been called twice: once for anthropic_oauth, once for anthropic
        assert mock_registry.create.call_count == 2
