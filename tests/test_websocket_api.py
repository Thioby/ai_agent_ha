"""Tests for the WebSocket API module."""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ============================================================================
# Mock Home Assistant dependencies
# ============================================================================


class MockStore:
    """Mock Home Assistant Store for testing."""

    _stores: dict[str, "MockStore"] = {}

    def __init__(self, hass: Any, version: int, key: str) -> None:
        self.hass = hass
        self.version = version
        self.key = key
        if key in MockStore._stores:
            self._data = MockStore._stores[key]._data
        else:
            self._data: dict[str, Any] | None = None
        MockStore._stores[key] = self

    async def async_load(self) -> dict[str, Any] | None:
        return self._data

    async def async_save(self, data: dict[str, Any]) -> None:
        self._data = data

    @classmethod
    def reset_stores(cls) -> None:
        cls._stores.clear()


class MockUser:
    """Mock Home Assistant User."""

    def __init__(self, user_id: str = "test_user_123") -> None:
        self.id = user_id


class MockConnection:
    """Mock WebSocket ActiveConnection."""

    def __init__(self, user: MockUser | None = None) -> None:
        self.user = user or MockUser()
        self.results: list[tuple[int, Any]] = []
        self.errors: list[tuple[int, str, str]] = []

    def send_result(self, msg_id: int, result: Any) -> None:
        self.results.append((msg_id, result))

    def send_error(self, msg_id: int, code: str, message: str) -> None:
        self.errors.append((msg_id, code, message))


def _setup_homeassistant_mocks() -> dict[str, Any]:
    """Set up mock modules for Home Assistant."""

    def _create_mock_module(name: str) -> ModuleType:
        mod = ModuleType(name)
        mod.__path__ = []
        return mod

    mocks = {
        "homeassistant": _create_mock_module("homeassistant"),
        "homeassistant.core": _create_mock_module("homeassistant.core"),
        "homeassistant.helpers": _create_mock_module("homeassistant.helpers"),
        "homeassistant.helpers.storage": _create_mock_module(
            "homeassistant.helpers.storage"
        ),
        "homeassistant.components": _create_mock_module("homeassistant.components"),
        "homeassistant.components.websocket_api": _create_mock_module(
            "homeassistant.components.websocket_api"
        ),
        "homeassistant.exceptions": _create_mock_module("homeassistant.exceptions"),
        "voluptuous": MagicMock(),
    }

    # Add Store class
    setattr(mocks["homeassistant.helpers.storage"], "Store", MockStore)

    # Add HomeAssistant mock
    setattr(mocks["homeassistant.core"], "HomeAssistant", MagicMock)

    # Add HomeAssistantError
    class HomeAssistantError(Exception):
        pass

    setattr(mocks["homeassistant.exceptions"], "HomeAssistantError", HomeAssistantError)

    # Add websocket_api decorators and classes
    ws_api_mock = mocks["homeassistant.components.websocket_api"]

    def websocket_command(schema: dict) -> Any:
        def decorator(func: Any) -> Any:
            func._ws_schema = schema
            return func

        return decorator

    def async_response(func: Any) -> Any:
        return func

    def async_register_command(hass: Any, handler: Any) -> None:
        pass

    setattr(ws_api_mock, "websocket_command", websocket_command)
    setattr(ws_api_mock, "async_response", async_response)
    setattr(ws_api_mock, "async_register_command", async_register_command)
    setattr(ws_api_mock, "ActiveConnection", MockConnection)

    for name, mock in mocks.items():
        sys.modules[name] = mock

    return mocks


_mocks = _setup_homeassistant_mocks()


def _import_modules():
    """Import modules directly without going through __init__.py."""
    base_path = Path(__file__).parent.parent / "custom_components" / "ai_agent_ha"

    # Import const first
    const_path = base_path / "const.py"
    spec = importlib.util.spec_from_file_location("const", const_path)
    const_module = importlib.util.module_from_spec(spec)
    sys.modules["const"] = const_module
    spec.loader.exec_module(const_module)

    # Import storage
    storage_path = base_path / "storage.py"
    spec = importlib.util.spec_from_file_location("storage", storage_path)
    storage_module = importlib.util.module_from_spec(spec)
    sys.modules["storage"] = storage_module
    spec.loader.exec_module(storage_module)

    # Read and modify websocket_api source to use absolute imports
    ws_path = base_path / "websocket_api.py"
    with open(ws_path) as f:
        ws_source = f.read()

    # Replace relative imports with our mock modules
    ws_source = ws_source.replace(
        "from .const import DOMAIN, VALID_PROVIDERS",
        "from const import DOMAIN, VALID_PROVIDERS",
    )
    ws_source = ws_source.replace(
        "from .storage import MAX_MESSAGE_LENGTH, Message, SessionStorage",
        "from storage import MAX_MESSAGE_LENGTH, Message, SessionStorage",
    )

    # Execute modified source
    ws_module = ModuleType("websocket_api")
    ws_module.__file__ = str(ws_path)
    sys.modules["websocket_api"] = ws_module
    exec(compile(ws_source, ws_path, "exec"), ws_module.__dict__)

    return const_module, storage_module, ws_module


_const, _storage, _ws_api = _import_modules()

# Export what we need
DOMAIN = _const.DOMAIN
VALID_PROVIDERS = _const.VALID_PROVIDERS
SessionStorage = _storage.SessionStorage
Message = _storage.Message

ws_list_sessions = _ws_api.ws_list_sessions
ws_get_session = _ws_api.ws_get_session
ws_create_session = _ws_api.ws_create_session
ws_delete_session = _ws_api.ws_delete_session
ws_rename_session = _ws_api.ws_rename_session
ws_send_message = _ws_api.ws_send_message
_get_user_id = _ws_api._get_user_id

ERR_SESSION_NOT_FOUND = _ws_api.ERR_SESSION_NOT_FOUND
ERR_STORAGE_ERROR = _ws_api.ERR_STORAGE_ERROR


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def reset_stores():
    """Reset mock stores before each test."""
    MockStore.reset_stores()
    yield
    MockStore.reset_stores()


@pytest.fixture
def mock_hass() -> MagicMock:
    """Create a mock Home Assistant instance."""
    hass = MagicMock()
    hass.data = {}
    return hass


@pytest.fixture
def mock_connection() -> MockConnection:
    """Create a mock WebSocket connection."""
    return MockConnection(MockUser("test_user_123"))


@pytest.fixture
def mock_connection_no_user() -> MockConnection:
    """Create a mock WebSocket connection without user."""
    conn = MockConnection()
    conn.user = None
    return conn


# ============================================================================
# Test Cases
# ============================================================================


class TestGetUserId:
    """Tests for _get_user_id helper."""

    def test_get_user_id_with_user(self, mock_connection: MockConnection) -> None:
        """Test extracting user ID from connection."""
        user_id = _get_user_id(mock_connection)
        assert user_id == "test_user_123"

    def test_get_user_id_no_user(self, mock_connection_no_user: MockConnection) -> None:
        """Test fallback when no user present."""
        user_id = _get_user_id(mock_connection_no_user)
        assert user_id == "default"

    def test_get_user_id_user_no_id(self) -> None:
        """Test fallback when user has no ID."""
        conn = MockConnection()
        conn.user = MagicMock()
        conn.user.id = None
        user_id = _get_user_id(conn)
        assert user_id == "default"


class TestWsListSessions:
    """Tests for ws_list_sessions command."""

    @pytest.mark.asyncio
    async def test_list_sessions_empty(
        self, mock_hass: MagicMock, mock_connection: MockConnection
    ) -> None:
        """Test listing sessions when none exist."""
        msg = {"id": 1, "type": "ai_agent_ha/sessions/list"}

        await ws_list_sessions(mock_hass, mock_connection, msg)

        assert len(mock_connection.results) == 1
        msg_id, result = mock_connection.results[0]
        assert msg_id == 1
        assert result == {"sessions": []}

    @pytest.mark.asyncio
    async def test_list_sessions_with_data(
        self, mock_hass: MagicMock, mock_connection: MockConnection
    ) -> None:
        """Test listing sessions with existing data."""
        # Create some sessions first
        storage = SessionStorage(mock_hass, "test_user_123")
        await storage.create_session(provider="anthropic", title="Session 1")
        await storage.create_session(provider="openai", title="Session 2")

        msg = {"id": 2, "type": "ai_agent_ha/sessions/list"}
        await ws_list_sessions(mock_hass, mock_connection, msg)

        assert len(mock_connection.results) == 1
        msg_id, result = mock_connection.results[0]
        assert msg_id == 2
        assert len(result["sessions"]) == 2
        assert result["sessions"][0]["title"] == "Session 2"  # Most recent first


class TestWsGetSession:
    """Tests for ws_get_session command."""

    @pytest.mark.asyncio
    async def test_get_session_success(
        self, mock_hass: MagicMock, mock_connection: MockConnection
    ) -> None:
        """Test getting a session with messages."""
        storage = SessionStorage(mock_hass, "test_user_123")
        session = await storage.create_session(provider="anthropic", title="Test")
        message = Message(
            message_id="msg-1",
            session_id=session.session_id,
            role="user",
            content="Hello",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        await storage.add_message(session.session_id, message)

        msg = {
            "id": 1,
            "type": "ai_agent_ha/sessions/get",
            "session_id": session.session_id,
        }
        await ws_get_session(mock_hass, mock_connection, msg)

        assert len(mock_connection.results) == 1
        msg_id, result = mock_connection.results[0]
        assert msg_id == 1
        assert result["session"]["session_id"] == session.session_id
        assert len(result["messages"]) == 1
        assert result["messages"][0]["content"] == "Hello"

    @pytest.mark.asyncio
    async def test_get_session_not_found(
        self, mock_hass: MagicMock, mock_connection: MockConnection
    ) -> None:
        """Test getting a non-existent session."""
        msg = {
            "id": 1,
            "type": "ai_agent_ha/sessions/get",
            "session_id": "non-existent",
        }
        await ws_get_session(mock_hass, mock_connection, msg)

        assert len(mock_connection.errors) == 1
        msg_id, code, message = mock_connection.errors[0]
        assert msg_id == 1
        assert code == ERR_SESSION_NOT_FOUND


class TestWsCreateSession:
    """Tests for ws_create_session command."""

    @pytest.mark.asyncio
    async def test_create_session_success(
        self, mock_hass: MagicMock, mock_connection: MockConnection
    ) -> None:
        """Test creating a new session."""
        msg = {
            "id": 1,
            "type": "ai_agent_ha/sessions/create",
            "provider": "anthropic",
        }
        await ws_create_session(mock_hass, mock_connection, msg)

        assert len(mock_connection.results) == 1
        msg_id, result = mock_connection.results[0]
        assert msg_id == 1
        assert result["provider"] == "anthropic"
        assert result["title"] == "New Conversation"
        assert "session_id" in result

    @pytest.mark.asyncio
    async def test_create_session_with_title(
        self, mock_hass: MagicMock, mock_connection: MockConnection
    ) -> None:
        """Test creating a session with custom title."""
        msg = {
            "id": 1,
            "type": "ai_agent_ha/sessions/create",
            "provider": "openai",
            "title": "My Custom Chat",
        }
        await ws_create_session(mock_hass, mock_connection, msg)

        assert len(mock_connection.results) == 1
        _, result = mock_connection.results[0]
        assert result["title"] == "My Custom Chat"


class TestWsDeleteSession:
    """Tests for ws_delete_session command."""

    @pytest.mark.asyncio
    async def test_delete_session_success(
        self, mock_hass: MagicMock, mock_connection: MockConnection
    ) -> None:
        """Test deleting an existing session."""
        storage = SessionStorage(mock_hass, "test_user_123")
        session = await storage.create_session(provider="anthropic")

        msg = {
            "id": 1,
            "type": "ai_agent_ha/sessions/delete",
            "session_id": session.session_id,
        }
        await ws_delete_session(mock_hass, mock_connection, msg)

        assert len(mock_connection.results) == 1
        _, result = mock_connection.results[0]
        assert result == {"success": True}

        # Verify session is deleted
        sessions = await storage.list_sessions()
        assert len(sessions) == 0

    @pytest.mark.asyncio
    async def test_delete_session_not_found(
        self, mock_hass: MagicMock, mock_connection: MockConnection
    ) -> None:
        """Test deleting a non-existent session (should succeed silently)."""
        msg = {
            "id": 1,
            "type": "ai_agent_ha/sessions/delete",
            "session_id": "non-existent",
        }
        await ws_delete_session(mock_hass, mock_connection, msg)

        # Should still return success
        assert len(mock_connection.results) == 1
        _, result = mock_connection.results[0]
        assert result == {"success": True}


class TestWsRenameSession:
    """Tests for ws_rename_session command."""

    @pytest.mark.asyncio
    async def test_rename_session_success(
        self, mock_hass: MagicMock, mock_connection: MockConnection
    ) -> None:
        """Test renaming an existing session."""
        storage = SessionStorage(mock_hass, "test_user_123")
        session = await storage.create_session(provider="anthropic")

        msg = {
            "id": 1,
            "type": "ai_agent_ha/sessions/rename",
            "session_id": session.session_id,
            "title": "Renamed Session",
        }
        await ws_rename_session(mock_hass, mock_connection, msg)

        assert len(mock_connection.results) == 1
        _, result = mock_connection.results[0]
        assert result == {"success": True}

        # Verify rename
        renamed = await storage.get_session(session.session_id)
        assert renamed.title == "Renamed Session"

    @pytest.mark.asyncio
    async def test_rename_session_not_found(
        self, mock_hass: MagicMock, mock_connection: MockConnection
    ) -> None:
        """Test renaming a non-existent session."""
        msg = {
            "id": 1,
            "type": "ai_agent_ha/sessions/rename",
            "session_id": "non-existent",
            "title": "New Title",
        }
        await ws_rename_session(mock_hass, mock_connection, msg)

        assert len(mock_connection.errors) == 1
        _, code, _ = mock_connection.errors[0]
        assert code == ERR_SESSION_NOT_FOUND


class TestWsSendMessage:
    """Tests for ws_send_message command."""

    @pytest.mark.asyncio
    async def test_send_message_success(
        self, mock_hass: MagicMock, mock_connection: MockConnection
    ) -> None:
        """Test sending a message with successful AI response."""
        # Set up session
        storage = SessionStorage(mock_hass, "test_user_123")
        session = await storage.create_session(provider="anthropic")

        # Set up mock AI agent
        mock_agent = AsyncMock()
        mock_agent.process_query = AsyncMock(
            return_value={
                "success": True,
                "answer": "Hello! I can help with that.",
                "automation": None,
                "dashboard": None,
                "debug": None,
            }
        )
        mock_hass.data = {
            DOMAIN: {
                "agents": {"anthropic": mock_agent},
            }
        }

        msg = {
            "id": 1,
            "type": "ai_agent_ha/chat/send",
            "session_id": session.session_id,
            "message": "Hello, AI!",
        }
        await ws_send_message(mock_hass, mock_connection, msg)

        assert len(mock_connection.results) == 1
        _, result = mock_connection.results[0]
        assert result["success"] is True
        assert result["user_message"]["content"] == "Hello, AI!"
        assert result["user_message"]["role"] == "user"
        assert result["assistant_message"]["content"] == "Hello! I can help with that."
        assert result["assistant_message"]["role"] == "assistant"
        assert result["assistant_message"]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_send_message_ai_error(
        self, mock_hass: MagicMock, mock_connection: MockConnection
    ) -> None:
        """Test sending a message when AI returns error."""
        storage = SessionStorage(mock_hass, "test_user_123")
        session = await storage.create_session(provider="anthropic")

        # Set up mock AI agent that raises error
        mock_agent = AsyncMock()
        mock_agent.process_query = AsyncMock(side_effect=Exception("AI service down"))
        mock_hass.data = {
            DOMAIN: {
                "agents": {"anthropic": mock_agent},
            }
        }

        msg = {
            "id": 1,
            "type": "ai_agent_ha/chat/send",
            "session_id": session.session_id,
            "message": "Hello",
        }
        await ws_send_message(mock_hass, mock_connection, msg)

        assert len(mock_connection.results) == 1
        _, result = mock_connection.results[0]
        assert result["success"] is False
        assert result["user_message"]["status"] == "completed"
        assert result["assistant_message"]["status"] == "error"
        assert "AI service down" in result["assistant_message"]["error_message"]

    @pytest.mark.asyncio
    async def test_send_message_provider_not_configured(
        self, mock_hass: MagicMock, mock_connection: MockConnection
    ) -> None:
        """Test sending message when provider is not configured."""
        storage = SessionStorage(mock_hass, "test_user_123")
        session = await storage.create_session(provider="anthropic")

        # No agents configured
        mock_hass.data = {}

        msg = {
            "id": 1,
            "type": "ai_agent_ha/chat/send",
            "session_id": session.session_id,
            "message": "Hello",
        }
        await ws_send_message(mock_hass, mock_connection, msg)

        assert len(mock_connection.results) == 1
        _, result = mock_connection.results[0]
        assert result["success"] is False
        assert result["assistant_message"]["status"] == "error"
        assert "not configured" in result["assistant_message"]["error_message"]

    @pytest.mark.asyncio
    async def test_send_message_session_not_found(
        self, mock_hass: MagicMock, mock_connection: MockConnection
    ) -> None:
        """Test sending message to non-existent session."""
        msg = {
            "id": 1,
            "type": "ai_agent_ha/chat/send",
            "session_id": "non-existent",
            "message": "Hello",
        }
        await ws_send_message(mock_hass, mock_connection, msg)

        assert len(mock_connection.errors) == 1
        _, code, _ = mock_connection.errors[0]
        assert code == ERR_SESSION_NOT_FOUND

    @pytest.mark.asyncio
    async def test_send_message_with_conversation_history(
        self, mock_hass: MagicMock, mock_connection: MockConnection
    ) -> None:
        """Test that conversation history is passed to AI."""
        storage = SessionStorage(mock_hass, "test_user_123")
        session = await storage.create_session(provider="anthropic")

        # Add existing messages
        msg1 = Message(
            message_id="msg-1",
            session_id=session.session_id,
            role="user",
            content="Turn on lights",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        msg2 = Message(
            message_id="msg-2",
            session_id=session.session_id,
            role="assistant",
            content="Done, lights are on",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        await storage.add_message(session.session_id, msg1)
        await storage.add_message(session.session_id, msg2)

        # Set up mock agent
        mock_agent = AsyncMock()
        mock_agent.process_query = AsyncMock(
            return_value={"answer": "Lights turned off"}
        )
        mock_hass.data = {DOMAIN: {"agents": {"anthropic": mock_agent}}}

        msg = {
            "id": 1,
            "type": "ai_agent_ha/chat/send",
            "session_id": session.session_id,
            "message": "Now turn them off",
        }
        await ws_send_message(mock_hass, mock_connection, msg)

        # Verify conversation history was passed
        call_kwargs = mock_agent.process_query.call_args.kwargs
        history = call_kwargs.get("conversation_history", [])
        assert len(history) == 3  # 2 existing + 1 new user message
        assert history[0]["content"] == "Turn on lights"
        assert history[1]["content"] == "Done, lights are on"
        assert history[2]["content"] == "Now turn them off"


class TestUserIsolation:
    """Tests for user isolation in WebSocket API."""

    @pytest.mark.asyncio
    async def test_users_see_only_their_sessions(self, mock_hass: MagicMock) -> None:
        """Test that users can only see their own sessions."""
        # Create sessions for user A
        conn_a = MockConnection(MockUser("user_a"))
        storage_a = SessionStorage(mock_hass, "user_a")
        await storage_a.create_session(provider="anthropic", title="A's session")

        # Create sessions for user B
        conn_b = MockConnection(MockUser("user_b"))
        storage_b = SessionStorage(mock_hass, "user_b")
        await storage_b.create_session(provider="anthropic", title="B's session")

        # User A lists sessions
        await ws_list_sessions(mock_hass, conn_a, {"id": 1, "type": "..."})
        _, result_a = conn_a.results[0]

        # User B lists sessions
        await ws_list_sessions(mock_hass, conn_b, {"id": 1, "type": "..."})
        _, result_b = conn_b.results[0]

        assert len(result_a["sessions"]) == 1
        assert result_a["sessions"][0]["title"] == "A's session"

        assert len(result_b["sessions"]) == 1
        assert result_b["sessions"][0]["title"] == "B's session"
