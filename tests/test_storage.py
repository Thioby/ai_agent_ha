"""Tests for the session storage module.

These tests run independently of Home Assistant by mocking its dependencies.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock

import pytest


# ============================================================================
# Mock Home Assistant dependencies
# ============================================================================


class MockStore:
    """Mock Home Assistant Store for testing.

    This class simulates the behavior of homeassistant.helpers.storage.Store.
    """

    # Class-level storage for persistence testing
    _stores: dict[str, "MockStore"] = {}

    def __init__(
        self,
        hass: Any,
        version: int,
        key: str,
    ) -> None:
        self.hass = hass
        self.version = version
        self.key = key
        # Check if we have existing data for this key
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
        """Reset all stored data between tests."""
        cls._stores.clear()


def _setup_homeassistant_mocks() -> None:
    """Set up mock modules for Home Assistant."""

    def _create_mock_module(name: str) -> ModuleType:
        mod = ModuleType(name)
        mod.__path__ = []
        return mod

    # Create mock modules
    mocks = {
        "homeassistant": _create_mock_module("homeassistant"),
        "homeassistant.core": _create_mock_module("homeassistant.core"),
        "homeassistant.helpers": _create_mock_module("homeassistant.helpers"),
        "homeassistant.helpers.storage": _create_mock_module(
            "homeassistant.helpers.storage"
        ),
    }

    # Add Store class to storage module
    setattr(mocks["homeassistant.helpers.storage"], "Store", MockStore)

    # Add HomeAssistant mock
    setattr(mocks["homeassistant.core"], "HomeAssistant", MagicMock)

    for name, mock in mocks.items():
        sys.modules[name] = mock


# Set up mocks before importing storage module
_setup_homeassistant_mocks()


def _import_storage_module():
    """Import storage module directly without going through __init__.py."""
    storage_path = (
        Path(__file__).parent.parent
        / "custom_components"
        / "ai_agent_ha"
        / "storage.py"
    )
    spec = importlib.util.spec_from_file_location("storage", storage_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load storage module from {storage_path}")

    storage_module = importlib.util.module_from_spec(spec)
    sys.modules["storage"] = storage_module
    spec.loader.exec_module(storage_module)
    return storage_module


_storage = _import_storage_module()

# Export the classes and constants from the storage module
MAX_MESSAGES_PER_SESSION = _storage.MAX_MESSAGES_PER_SESSION
MAX_SESSIONS = _storage.MAX_SESSIONS
SESSION_RETENTION_DAYS = _storage.SESSION_RETENTION_DAYS
STORAGE_VERSION = _storage.STORAGE_VERSION
Message = _storage.Message
Session = _storage.Session
SessionStorage = _storage.SessionStorage


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
def storage(mock_hass: MagicMock) -> SessionStorage:
    """Create a SessionStorage instance for testing."""
    return SessionStorage(mock_hass, "test_user")


@pytest.fixture
def storage_factory(mock_hass: MagicMock):
    """Factory to create SessionStorage instances for different users."""

    def create(user_id: str) -> SessionStorage:
        return SessionStorage(mock_hass, user_id)

    return create


# ============================================================================
# Test Cases
# ============================================================================


class TestMessage:
    """Tests for the Message dataclass."""

    def test_message_creation(self) -> None:
        """Test basic message creation."""
        msg = Message(
            message_id="msg-123",
            session_id="session-456",
            role="user",
            content="Hello, world!",
            timestamp="2026-01-23T10:00:00+00:00",
        )
        assert msg.message_id == "msg-123"
        assert msg.role == "user"
        assert msg.status == "completed"
        assert msg.error_message == ""
        assert msg.metadata == {}

    def test_message_with_all_fields(self) -> None:
        """Test message creation with all optional fields."""
        msg = Message(
            message_id="msg-123",
            session_id="session-456",
            role="assistant",
            content="Response text",
            timestamp="2026-01-23T10:00:00+00:00",
            status="completed",
            error_message="",
            metadata={"token_usage": 150},
        )
        assert msg.metadata["token_usage"] == 150

    def test_message_invalid_role(self) -> None:
        """Test that invalid role raises ValueError."""
        with pytest.raises(ValueError, match="Invalid role"):
            Message(
                message_id="msg-123",
                session_id="session-456",
                role="invalid",
                content="Hello",
                timestamp="2026-01-23T10:00:00+00:00",
            )

    def test_message_invalid_status(self) -> None:
        """Test that invalid status raises ValueError."""
        with pytest.raises(ValueError, match="Invalid status"):
            Message(
                message_id="msg-123",
                session_id="session-456",
                role="user",
                content="Hello",
                timestamp="2026-01-23T10:00:00+00:00",
                status="invalid",
            )

    def test_message_content_truncation(self) -> None:
        """Test that long content is truncated."""
        long_content = "x" * 60000
        msg = Message(
            message_id="msg-123",
            session_id="session-456",
            role="user",
            content=long_content,
            timestamp="2026-01-23T10:00:00+00:00",
        )
        assert len(msg.content) == 50000


class TestSession:
    """Tests for the Session dataclass."""

    def test_session_creation(self) -> None:
        """Test basic session creation."""
        session = Session(
            session_id="session-123",
            title="Test Session",
            created_at="2026-01-23T10:00:00+00:00",
            updated_at="2026-01-23T10:00:00+00:00",
            provider="anthropic",
        )
        assert session.session_id == "session-123"
        assert session.message_count == 0
        assert session.preview == ""


class TestSessionStorageBasics:
    """Basic tests for SessionStorage."""

    @pytest.mark.asyncio
    async def test_create_session(self, storage: SessionStorage) -> None:
        """Test creating a new session."""
        session = await storage.create_session(provider="anthropic")

        assert session.session_id is not None
        assert len(session.session_id) == 36  # UUID length
        assert session.provider == "anthropic"
        assert session.title == "New Conversation"
        assert session.message_count == 0

    @pytest.mark.asyncio
    async def test_create_session_with_title(self, storage: SessionStorage) -> None:
        """Test creating a session with custom title."""
        session = await storage.create_session(
            provider="openai", title="My Custom Chat"
        )

        assert session.title == "My Custom Chat"

    @pytest.mark.asyncio
    async def test_list_sessions_empty(self, storage: SessionStorage) -> None:
        """Test listing sessions when none exist."""
        sessions = await storage.list_sessions()
        assert sessions == []

    @pytest.mark.asyncio
    async def test_list_sessions_sorted(self, storage: SessionStorage) -> None:
        """Test that sessions are sorted by updated_at descending."""
        # Create sessions with delays to ensure different timestamps
        session1 = await storage.create_session(provider="anthropic", title="First")
        session2 = await storage.create_session(provider="anthropic", title="Second")
        session3 = await storage.create_session(provider="anthropic", title="Third")

        sessions = await storage.list_sessions()

        assert len(sessions) == 3
        # Most recently created should be first
        assert sessions[0].title == "Third"
        assert sessions[1].title == "Second"
        assert sessions[2].title == "First"

    @pytest.mark.asyncio
    async def test_get_session(self, storage: SessionStorage) -> None:
        """Test getting a session by ID."""
        created = await storage.create_session(provider="anthropic", title="Test")

        session = await storage.get_session(created.session_id)

        assert session is not None
        assert session.session_id == created.session_id
        assert session.title == "Test"

    @pytest.mark.asyncio
    async def test_get_session_not_found(self, storage: SessionStorage) -> None:
        """Test getting a non-existent session."""
        session = await storage.get_session("non-existent-id")
        assert session is None

    @pytest.mark.asyncio
    async def test_delete_session(self, storage: SessionStorage) -> None:
        """Test deleting a session."""
        session = await storage.create_session(provider="anthropic")

        result = await storage.delete_session(session.session_id)

        assert result is True
        sessions = await storage.list_sessions()
        assert len(sessions) == 0

    @pytest.mark.asyncio
    async def test_delete_session_not_found(self, storage: SessionStorage) -> None:
        """Test deleting a non-existent session (should succeed silently)."""
        result = await storage.delete_session("non-existent-id")
        assert result is True

    @pytest.mark.asyncio
    async def test_rename_session(self, storage: SessionStorage) -> None:
        """Test renaming a session."""
        session = await storage.create_session(provider="anthropic")

        result = await storage.rename_session(session.session_id, "New Title")

        assert result is True
        renamed = await storage.get_session(session.session_id)
        assert renamed is not None
        assert renamed.title == "New Title"

    @pytest.mark.asyncio
    async def test_rename_session_not_found(self, storage: SessionStorage) -> None:
        """Test renaming a non-existent session."""
        result = await storage.rename_session("non-existent-id", "New Title")
        assert result is False


class TestSessionStorageMessages:
    """Tests for message operations in SessionStorage."""

    @pytest.mark.asyncio
    async def test_add_message(self, storage: SessionStorage) -> None:
        """Test adding a message to a session."""
        session = await storage.create_session(provider="anthropic")
        message = Message(
            message_id="msg-1",
            session_id=session.session_id,
            role="user",
            content="Hello, AI!",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        await storage.add_message(session.session_id, message)

        messages = await storage.get_session_messages(session.session_id)
        assert len(messages) == 1
        assert messages[0].content == "Hello, AI!"

    @pytest.mark.asyncio
    async def test_add_message_updates_session_metadata(
        self, storage: SessionStorage
    ) -> None:
        """Test that adding a user message updates session preview and title."""
        session = await storage.create_session(provider="anthropic")
        message = Message(
            message_id="msg-1",
            session_id=session.session_id,
            role="user",
            content="How do I turn on the kitchen lights?",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        await storage.add_message(session.session_id, message)

        updated_session = await storage.get_session(session.session_id)
        assert updated_session is not None
        assert updated_session.message_count == 1
        assert updated_session.preview == "How do I turn on the kitchen lights?"
        assert updated_session.title == "How do I turn on the kitchen lights?"

    @pytest.mark.asyncio
    async def test_add_message_auto_title_truncation(
        self, storage: SessionStorage
    ) -> None:
        """Test that auto-generated titles are truncated properly."""
        session = await storage.create_session(provider="anthropic")
        long_content = (
            "This is a very long message that should be truncated for the title"
        )
        message = Message(
            message_id="msg-1",
            session_id=session.session_id,
            role="user",
            content=long_content,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        await storage.add_message(session.session_id, message)

        updated_session = await storage.get_session(session.session_id)
        assert updated_session is not None
        assert len(updated_session.title) <= 43  # 40 chars + "..."
        assert updated_session.title.endswith("...")

    @pytest.mark.asyncio
    async def test_add_message_to_nonexistent_session(
        self, storage: SessionStorage
    ) -> None:
        """Test that adding a message to a non-existent session raises error."""
        message = Message(
            message_id="msg-1",
            session_id="non-existent",
            role="user",
            content="Hello",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        with pytest.raises(ValueError, match="Session non-existent not found"):
            await storage.add_message("non-existent", message)

    @pytest.mark.asyncio
    async def test_get_session_messages_not_found(
        self, storage: SessionStorage
    ) -> None:
        """Test getting messages from a non-existent session raises error."""
        with pytest.raises(ValueError, match="not found"):
            await storage.get_session_messages("non-existent")

    @pytest.mark.asyncio
    async def test_update_message(self, storage: SessionStorage) -> None:
        """Test updating an existing message."""
        session = await storage.create_session(provider="anthropic")
        message = Message(
            message_id="msg-1",
            session_id=session.session_id,
            role="assistant",
            content="",
            timestamp=datetime.now(timezone.utc).isoformat(),
            status="pending",
        )
        await storage.add_message(session.session_id, message)

        result = await storage.update_message(
            session.session_id,
            "msg-1",
            content="Updated response",
            status="completed",
        )

        assert result is True
        messages = await storage.get_session_messages(session.session_id)
        assert messages[0].content == "Updated response"
        assert messages[0].status == "completed"

    @pytest.mark.asyncio
    async def test_update_message_not_found(self, storage: SessionStorage) -> None:
        """Test updating a non-existent message returns False."""
        session = await storage.create_session(provider="anthropic")

        result = await storage.update_message(
            session.session_id, "non-existent", content="test"
        )

        assert result is False


class TestSessionStorageLimits:
    """Tests for storage limits enforcement."""

    @pytest.mark.asyncio
    async def test_session_limit_enforcement(self, storage: SessionStorage) -> None:
        """Test that session limit is enforced by removing oldest."""
        # Create MAX_SESSIONS sessions
        for i in range(MAX_SESSIONS):
            await storage.create_session(provider="anthropic", title=f"Session {i}")

        sessions = await storage.list_sessions()
        assert len(sessions) == MAX_SESSIONS

        # Create one more - should remove oldest
        await storage.create_session(provider="anthropic", title="New Session")

        sessions = await storage.list_sessions()
        assert len(sessions) == MAX_SESSIONS
        assert "Session 0" not in [s.title for s in sessions]
        assert "New Session" in [s.title for s in sessions]

    @pytest.mark.asyncio
    async def test_message_limit_enforcement(self, storage: SessionStorage) -> None:
        """Test that message limit is enforced by removing oldest."""
        session = await storage.create_session(provider="anthropic")

        # Add MAX_MESSAGES_PER_SESSION messages
        for i in range(MAX_MESSAGES_PER_SESSION):
            message = Message(
                message_id=f"msg-{i}",
                session_id=session.session_id,
                role="user",
                content=f"Message {i}",
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            await storage.add_message(session.session_id, message)

        messages = await storage.get_session_messages(session.session_id)
        assert len(messages) == MAX_MESSAGES_PER_SESSION
        assert messages[0].content == "Message 0"

        # Add one more - should remove oldest
        new_message = Message(
            message_id="msg-new",
            session_id=session.session_id,
            role="user",
            content="New message",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        await storage.add_message(session.session_id, new_message)

        messages = await storage.get_session_messages(session.session_id)
        assert len(messages) == MAX_MESSAGES_PER_SESSION
        assert messages[0].content == "Message 1"  # Message 0 was removed
        assert messages[-1].content == "New message"


class TestSessionStorageUserIsolation:
    """Tests for user isolation in SessionStorage."""

    @pytest.mark.asyncio
    async def test_user_isolation(self, storage_factory) -> None:
        """Test that users have isolated storage."""
        storage_a = storage_factory("user_a")
        storage_b = storage_factory("user_b")

        await storage_a.create_session(provider="anthropic", title="A's session")
        await storage_b.create_session(provider="anthropic", title="B's session")

        a_sessions = await storage_a.list_sessions()
        b_sessions = await storage_b.list_sessions()

        assert len(a_sessions) == 1
        assert len(b_sessions) == 1
        assert a_sessions[0].title == "A's session"
        assert b_sessions[0].title == "B's session"


class TestSessionStorageCleanup:
    """Tests for session cleanup functionality."""

    @pytest.mark.asyncio
    async def test_cleanup_old_sessions(self, mock_hass: MagicMock) -> None:
        """Test that old sessions are cleaned up."""
        # Pre-populate store with old and recent sessions
        old_date = (
            datetime.now(timezone.utc) - timedelta(days=SESSION_RETENTION_DAYS + 1)
        ).isoformat()
        recent_date = datetime.now(timezone.utc).isoformat()

        # Create storage and manually set data
        storage = SessionStorage(mock_hass, "cleanup_test_user")

        # Directly set data in the mock store
        store_key = f"ai_agent_ha_user_data_cleanup_test_user"
        if store_key in MockStore._stores:
            MockStore._stores[store_key]._data = {
                "version": STORAGE_VERSION,
                "sessions": [
                    {
                        "session_id": "old-session",
                        "title": "Old Session",
                        "created_at": old_date,
                        "updated_at": old_date,
                        "provider": "anthropic",
                        "message_count": 0,
                        "preview": "",
                    },
                    {
                        "session_id": "recent-session",
                        "title": "Recent Session",
                        "created_at": recent_date,
                        "updated_at": recent_date,
                        "provider": "anthropic",
                        "message_count": 0,
                        "preview": "",
                    },
                ],
                "messages": {
                    "old-session": [],
                    "recent-session": [],
                },
            }

        # Force reload to trigger cleanup
        storage._data = None
        sessions = await storage.list_sessions()

        assert len(sessions) == 1
        assert sessions[0].session_id == "recent-session"


class TestSessionStorageMigration:
    """Tests for legacy data migration."""

    @pytest.mark.asyncio
    async def test_migrate_legacy_data(self, mock_hass: MagicMock) -> None:
        """Test migration of legacy prompt history."""
        # Create the legacy store with data
        legacy_key = "ai_agent_ha_history_migration_user"
        new_key = "ai_agent_ha_user_data_migration_user"

        # Pre-create stores
        legacy_store = MockStore(mock_hass, 1, legacy_key)
        legacy_store._data = {
            "prompts": ["Hello", "How are you?", "Turn on lights"],
        }
        MockStore._stores[legacy_key] = legacy_store

        # Create storage which should trigger migration
        storage = SessionStorage(mock_hass, "migration_user")
        sessions = await storage.list_sessions()

        # Should have created one session with migrated messages
        assert len(sessions) == 1
        assert sessions[0].title == "Imported History"
        assert sessions[0].message_count == 3

        # Legacy data should be marked as migrated
        assert MockStore._stores[legacy_key]._data["_migrated"] is True

    @pytest.mark.asyncio
    async def test_migrate_legacy_data_already_migrated(
        self, mock_hass: MagicMock
    ) -> None:
        """Test that already migrated data is not migrated again."""
        legacy_key = "ai_agent_ha_history_already_migrated_user"

        # Pre-create legacy store with migrated flag
        legacy_store = MockStore(mock_hass, 1, legacy_key)
        legacy_store._data = {
            "prompts": ["Hello"],
            "_migrated": True,
        }
        MockStore._stores[legacy_key] = legacy_store

        # Create storage
        storage = SessionStorage(mock_hass, "already_migrated_user")
        sessions = await storage.list_sessions()

        # No sessions should be created
        assert len(sessions) == 0


class TestSessionStoragePersistence:
    """Tests for data persistence."""

    @pytest.mark.asyncio
    async def test_data_persists_across_instances(self, mock_hass: MagicMock) -> None:
        """Test that data persists when creating new storage instances."""
        # Create session with first instance
        storage1 = SessionStorage(mock_hass, "persist_user")
        session = await storage1.create_session(
            provider="anthropic", title="Persistent"
        )
        message = Message(
            message_id="msg-1",
            session_id=session.session_id,
            role="user",
            content="Test persistence",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        await storage1.add_message(session.session_id, message)

        # Create new instance (simulates restart)
        storage2 = SessionStorage(mock_hass, "persist_user")
        storage2._data = None  # Force reload

        sessions = await storage2.list_sessions()
        messages = await storage2.get_session_messages(session.session_id)

        assert len(sessions) == 1
        assert sessions[0].title == "Persistent"
        assert len(messages) == 1
        assert messages[0].content == "Test persistence"


class TestSessionStorageClearAll:
    """Tests for clear all functionality."""

    @pytest.mark.asyncio
    async def test_clear_all_sessions(self, storage: SessionStorage) -> None:
        """Test clearing all sessions for a user."""
        # Create some sessions
        await storage.create_session(provider="anthropic", title="Session 1")
        await storage.create_session(provider="anthropic", title="Session 2")

        sessions = await storage.list_sessions()
        assert len(sessions) == 2

        # Clear all
        await storage.clear_all_sessions()

        sessions = await storage.list_sessions()
        assert len(sessions) == 0
