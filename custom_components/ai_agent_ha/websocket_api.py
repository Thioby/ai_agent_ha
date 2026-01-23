"""WebSocket API for AI Agent HA chat sessions."""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import voluptuous as vol
from homeassistant.components import websocket_api
from homeassistant.exceptions import HomeAssistantError

from .const import DOMAIN, VALID_PROVIDERS
from .storage import MAX_MESSAGE_LENGTH, Message, SessionStorage

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)

# Error codes
ERR_SESSION_NOT_FOUND = "session_not_found"
ERR_INVALID_INPUT = "invalid_input"
ERR_STORAGE_ERROR = "storage_error"
ERR_AI_ERROR = "ai_error"
ERR_RATE_LIMITED = "rate_limited"

# Validation constants
MAX_TITLE_LENGTH = 200
UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE
)

# Storage cache key prefix
_STORAGE_CACHE_PREFIX = f"{DOMAIN}_storage_"


def _validate_session_id(value: Any) -> str:
    """Validate that session_id is a valid UUID format."""
    if not isinstance(value, str):
        raise vol.Invalid("Session ID must be a string")
    if not UUID_PATTERN.match(value):
        raise vol.Invalid("Session ID must be a valid UUID format")
    return value


def _validate_title(value: Any) -> str:
    """Validate and truncate title to max length."""
    if not isinstance(value, str):
        raise vol.Invalid("Title must be a string")
    if len(value) == 0:
        raise vol.Invalid("Title cannot be empty")
    # Truncate to max length
    return value[:MAX_TITLE_LENGTH]


def _validate_message(value: Any) -> str:
    """Validate message content with length limit."""
    if not isinstance(value, str):
        raise vol.Invalid("Message must be a string")
    if len(value) == 0:
        raise vol.Invalid("Message cannot be empty")
    if len(value) > MAX_MESSAGE_LENGTH:
        raise vol.Invalid(f"Message exceeds maximum length of {MAX_MESSAGE_LENGTH}")
    return value


def async_register_websocket_commands(hass: HomeAssistant) -> None:
    """Register all WebSocket API commands.

    This function should only be called once per Home Assistant instance.
    Use the guard in __init__.py to prevent multiple registrations.
    """
    websocket_api.async_register_command(hass, ws_list_sessions)
    websocket_api.async_register_command(hass, ws_get_session)
    websocket_api.async_register_command(hass, ws_create_session)
    websocket_api.async_register_command(hass, ws_delete_session)
    websocket_api.async_register_command(hass, ws_rename_session)
    websocket_api.async_register_command(hass, ws_send_message)


def _get_user_id(connection: websocket_api.ActiveConnection) -> str:
    """Extract user ID from WebSocket connection."""
    if connection.user and connection.user.id:
        return connection.user.id
    return "default"


def _get_storage(hass: HomeAssistant, user_id: str) -> SessionStorage:
    """Get or create a cached SessionStorage instance for a user.

    Caching storage instances improves performance by avoiding repeated
    migration and cleanup checks, and ensures consistent state across requests.
    """
    cache_key = f"{_STORAGE_CACHE_PREFIX}{user_id}"
    if cache_key not in hass.data:
        hass.data[cache_key] = SessionStorage(hass, user_id)
    return hass.data[cache_key]


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ai_agent_ha/sessions/list",
    }
)
@websocket_api.async_response
async def ws_list_sessions(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """List all sessions for current user."""
    user_id = _get_user_id(connection)

    try:
        storage = _get_storage(hass, user_id)
        sessions = await storage.list_sessions()
        connection.send_result(msg["id"], {"sessions": [asdict(s) for s in sessions]})
    except Exception as err:
        _LOGGER.exception("Failed to list sessions for user %s", user_id)
        connection.send_error(msg["id"], ERR_STORAGE_ERROR, "Failed to load sessions")


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ai_agent_ha/sessions/get",
        vol.Required("session_id"): _validate_session_id,
    }
)
@websocket_api.async_response
async def ws_get_session(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Get a session with all messages."""
    user_id = _get_user_id(connection)
    session_id = msg["session_id"]

    try:
        storage = _get_storage(hass, user_id)
        session = await storage.get_session(session_id)
        if session is None:
            connection.send_error(msg["id"], ERR_SESSION_NOT_FOUND, "Session not found")
            return

        messages = await storage.get_session_messages(session_id)
        connection.send_result(
            msg["id"],
            {
                "session": asdict(session),
                "messages": [asdict(m) for m in messages],
            },
        )
    except ValueError:
        connection.send_error(msg["id"], ERR_SESSION_NOT_FOUND, "Session not found")
    except Exception as err:
        _LOGGER.exception("Failed to get session %s for user %s", session_id, user_id)
        connection.send_error(msg["id"], ERR_STORAGE_ERROR, "Failed to load session")


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ai_agent_ha/sessions/create",
        vol.Required("provider"): vol.In(VALID_PROVIDERS),
        vol.Optional("title"): _validate_title,
    }
)
@websocket_api.async_response
async def ws_create_session(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Create a new session."""
    user_id = _get_user_id(connection)
    provider = msg["provider"]

    try:
        storage = _get_storage(hass, user_id)
        session = await storage.create_session(
            provider=provider, title=msg.get("title")
        )
        connection.send_result(msg["id"], asdict(session))
    except Exception as err:
        _LOGGER.exception("Failed to create session for user %s", user_id)
        connection.send_error(msg["id"], ERR_STORAGE_ERROR, "Failed to create session")


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ai_agent_ha/sessions/delete",
        vol.Required("session_id"): _validate_session_id,
    }
)
@websocket_api.async_response
async def ws_delete_session(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Delete a session."""
    user_id = _get_user_id(connection)

    try:
        storage = _get_storage(hass, user_id)
        await storage.delete_session(msg["session_id"])
        connection.send_result(msg["id"], {"success": True})
    except Exception as err:
        _LOGGER.exception(
            "Failed to delete session %s for user %s", msg["session_id"], user_id
        )
        connection.send_error(msg["id"], ERR_STORAGE_ERROR, "Failed to delete session")


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ai_agent_ha/sessions/rename",
        vol.Required("session_id"): _validate_session_id,
        vol.Required("title"): _validate_title,
    }
)
@websocket_api.async_response
async def ws_rename_session(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Rename a session."""
    user_id = _get_user_id(connection)

    try:
        storage = _get_storage(hass, user_id)
        success = await storage.rename_session(msg["session_id"], msg["title"])
        if success:
            connection.send_result(msg["id"], {"success": True})
        else:
            connection.send_error(msg["id"], ERR_SESSION_NOT_FOUND, "Session not found")
    except Exception as err:
        _LOGGER.exception(
            "Failed to rename session %s for user %s", msg["session_id"], user_id
        )
        connection.send_error(msg["id"], ERR_STORAGE_ERROR, "Failed to rename session")


@websocket_api.websocket_command(
    {
        vol.Required("type"): "ai_agent_ha/chat/send",
        vol.Required("session_id"): _validate_session_id,
        vol.Required("message"): _validate_message,
        vol.Optional("provider"): str,
        vol.Optional("debug"): vol.Coerce(bool),
    }
)
@websocket_api.async_response
async def ws_send_message(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Send a chat message and get AI response."""
    user_id = _get_user_id(connection)
    session_id = msg["session_id"]

    try:
        storage = _get_storage(hass, user_id)

        # Verify session exists
        session = await storage.get_session(session_id)
        if session is None:
            connection.send_error(msg["id"], ERR_SESSION_NOT_FOUND, "Session not found")
            return

        now = datetime.now(timezone.utc).isoformat()

        # Create and save user message
        user_message = Message(
            message_id=str(uuid.uuid4()),
            session_id=session_id,
            role="user",
            content=msg["message"],
            timestamp=now,
            status="completed",
        )
        await storage.add_message(session_id, user_message)

        # Build conversation history for AI context
        all_messages = await storage.get_session_messages(session_id)
        conversation_history = [
            {"role": m.role, "content": m.content} for m in all_messages
        ]

        # Determine provider
        provider = msg.get("provider") or session.provider or "anthropic"

        # Create assistant message (will be updated with response)
        assistant_message = Message(
            message_id=str(uuid.uuid4()),
            session_id=session_id,
            role="assistant",
            content="",
            timestamp=datetime.now(timezone.utc).isoformat(),
            status="pending",
        )

        # Call AI agent
        try:
            if DOMAIN in hass.data and provider in hass.data[DOMAIN].get("agents", {}):
                agent = hass.data[DOMAIN]["agents"][provider]
                result = await agent.process_query(
                    msg["message"],
                    provider=provider,
                    debug=msg.get("debug", False),
                    conversation_history=conversation_history,
                )

                assistant_message.content = result.get("answer", "")
                assistant_message.status = "completed"
                assistant_message.metadata = {
                    k: v
                    for k, v in {
                        "automation": result.get("automation"),
                        "dashboard": result.get("dashboard"),
                        "debug": result.get("debug"),
                    }.items()
                    if v is not None
                }
            else:
                raise HomeAssistantError(f"Provider {provider} not configured")

        except Exception as ai_err:
            assistant_message.status = "error"
            assistant_message.error_message = str(ai_err)
            _LOGGER.error("AI error for session %s: %s", session_id, ai_err)

        # Save assistant message
        await storage.add_message(session_id, assistant_message)

        connection.send_result(
            msg["id"],
            {
                "user_message": asdict(user_message),
                "assistant_message": asdict(assistant_message),
                "success": assistant_message.status == "completed",
            },
        )

    except ValueError:
        connection.send_error(msg["id"], ERR_SESSION_NOT_FOUND, "Session not found")
    except Exception as err:
        _LOGGER.exception(
            "Failed to send message in session %s for user %s", session_id, user_id
        )
        connection.send_error(msg["id"], ERR_STORAGE_ERROR, "Failed to send message")
