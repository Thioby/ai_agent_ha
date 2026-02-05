"""Gemini OAuth provider implementation using Cloud Code Assist API."""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import aiohttp

from .registry import AIProvider, ProviderRegistry

if TYPE_CHECKING:
    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)

# Cloud Code Assist API constants (from gemini-cli / opencode-gemini-auth)
GEMINI_CODE_ASSIST_ENDPOINT = "https://cloudcode-pa.googleapis.com/v1internal"

GEMINI_CODE_ASSIST_HEADERS = {
    "User-Agent": "google-api-nodejs-client/9.15.1",
    "X-Goog-Api-Client": "gl-node/22.17.0",
    "Client-Metadata": "ideType=IDE_UNSPECIFIED,platform=PLATFORM_UNSPECIFIED,pluginType=GEMINI",
}

GEMINI_CODE_ASSIST_METADATA = {
    "ideType": "IDE_UNSPECIFIED",
    "platform": "PLATFORM_UNSPECIFIED",
    "pluginType": "GEMINI",
}

# Load models from config file
MODELS_CONFIG_PATH = Path(__file__).parent.parent / "models_config.json"


def _load_gemini_oauth_models() -> tuple[list[str], str]:
    """Load available models and default from config file."""
    try:
        with open(MODELS_CONFIG_PATH, encoding="utf-8") as f:
            config = json.load(f)
        provider_config = config.get("gemini_oauth", {})
        models = provider_config.get("models", [])
        model_ids = [m["id"] for m in models]
        default = next((m["id"] for m in models if m.get("default")), None)
        return model_ids, default or (
            model_ids[0] if model_ids else "gemini-3-pro-preview"
        )
    except (FileNotFoundError, json.JSONDecodeError, KeyError, StopIteration):
        return ["gemini-3-pro-preview"], "gemini-3-pro-preview"


GEMINI_AVAILABLE_MODELS, _DEFAULT_MODEL = _load_gemini_oauth_models()


@ProviderRegistry.register("gemini_oauth")
class GeminiOAuthProvider(AIProvider):
    """Gemini provider using OAuth authentication via Cloud Code Assist API.

    Uses cloudcode-pa.googleapis.com endpoint which requires OAuth Bearer token
    and a managed project ID. On first use, the provider will automatically
    onboard the user to obtain a project ID for the FREE tier.
    """

    # Retry configuration (from gemini-cli)
    MAX_ATTEMPTS = 10
    INITIAL_DELAY_MS = 5000
    MAX_DELAY_MS = 30000
    DEFAULT_MODEL = _DEFAULT_MODEL

    def __init__(self, hass: HomeAssistant, config: dict[str, Any]) -> None:
        """Initialize the Gemini OAuth provider.

        Args:
            hass: Home Assistant instance.
            config: Provider configuration containing:
                - config_entry: ConfigEntry with OAuth tokens
                - model: Optional model name (default: gemini-3-pro-preview)
        """
        super().__init__(hass, config)
        self._model = config.get("model", self.DEFAULT_MODEL)
        self._config_entry: ConfigEntry | None = config.get("config_entry")
        self._oauth_data: dict[str, Any] = {}
        self._refresh_lock = asyncio.Lock()
        self._project_lock = asyncio.Lock()

        # Load OAuth data from config entry
        if self._config_entry:
            self._oauth_data = dict(self._config_entry.data.get("gemini_oauth", {}))

    @property
    def supports_tools(self) -> bool:
        """Return True as Gemini supports function calling."""
        return True

    async def _get_valid_token(self) -> str:
        """Get a valid access token, refreshing if necessary.

        Returns:
            Valid access token string.

        Raises:
            Exception: If no token available or refresh fails.
        """
        from ..gemini_oauth import refresh_token, GeminiOAuthRefreshError

        async with self._refresh_lock:
            # Check if token is still valid (with 5 minute buffer)
            if time.time() < self._oauth_data.get("expires_at", 0) - 300:
                access_token = self._oauth_data.get("access_token")
                if not access_token:
                    raise GeminiOAuthRefreshError(
                        "No access token available - re-authentication required"
                    )
                return access_token

            _LOGGER.debug("Refreshing Gemini OAuth token")

            # Check if refresh token exists
            refresh_tok = self._oauth_data.get("refresh_token")
            if not refresh_tok:
                raise GeminiOAuthRefreshError(
                    "No refresh token available - re-authentication required"
                )

            try:
                async with aiohttp.ClientSession() as session:
                    new_tokens = await refresh_token(session, refresh_tok)
            except GeminiOAuthRefreshError as e:
                _LOGGER.error("Gemini OAuth refresh failed: %s", e)
                raise

            self._oauth_data.update(new_tokens)

            # Persist refreshed tokens to config entry
            if self._config_entry:
                new_data = {
                    **self._config_entry.data,
                    "gemini_oauth": self._oauth_data,
                }
                self.hass.config_entries.async_update_entry(
                    self._config_entry, data=new_data
                )

            return new_tokens["access_token"]

    async def _save_project_id(self, project_id: str) -> None:
        """Persist managed project ID to config entry."""
        self._oauth_data["managed_project_id"] = project_id
        if self._config_entry:
            new_data = {
                **self._config_entry.data,
                "gemini_oauth": self._oauth_data,
            }
            self.hass.config_entries.async_update_entry(
                self._config_entry, data=new_data
            )
        _LOGGER.info("Saved Gemini managed project ID: %s", project_id)

    async def _ensure_project_id(
        self, session: aiohttp.ClientSession, access_token: str
    ) -> str:
        """Ensure we have a valid project ID, onboarding if necessary.

        Cloud Code Assist API requires a project ID for all requests.
        For FREE tier users, Google provides a managed project automatically.

        Returns:
            The managed project ID.

        Raises:
            Exception: If project ID cannot be obtained.
        """
        async with self._project_lock:
            # 1. Check if we have cached project ID
            project_id = self._oauth_data.get("managed_project_id")
            if project_id:
                _LOGGER.debug("Using cached Gemini project ID: %s", project_id)
                return project_id

            _LOGGER.info("No Gemini project ID cached, resolving...")

            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
                **GEMINI_CODE_ASSIST_HEADERS,
            }

            # 2. Try loadCodeAssist to get existing project
            try:
                async with session.post(
                    f"{GEMINI_CODE_ASSIST_ENDPOINT}:loadCodeAssist",
                    headers=headers,
                    json={"metadata": GEMINI_CODE_ASSIST_METADATA},
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    response_text = await resp.text()
                    _LOGGER.debug(
                        "loadCodeAssist response (%d): %s",
                        resp.status,
                        response_text[:500],
                    )
                    if resp.status == 200:
                        data = json.loads(response_text)
                        project_id = data.get("cloudaicompanionProject")
                        if project_id:
                            _LOGGER.info(
                                "Found existing Gemini project: %s", project_id
                            )
                            await self._save_project_id(project_id)
                            return project_id

                        # Check tier - enterprise users need their own project
                        current_tier = data.get("currentTier", {}).get("id")
                        if current_tier and current_tier != "FREE":
                            raise Exception(
                                f"Gemini tier '{current_tier}' requires manual project "
                                "configuration."
                            )

                        _LOGGER.info(
                            "loadCodeAssist returned no project, will try onboarding"
                        )
                    else:
                        _LOGGER.warning(
                            "loadCodeAssist failed (%d): %s",
                            resp.status,
                            response_text[:200],
                        )
            except aiohttp.ClientError as e:
                _LOGGER.warning("loadCodeAssist request failed: %s", e)

            # 3. Onboard user for FREE tier (with retry loop)
            _LOGGER.info("Onboarding new Gemini FREE tier user...")

            max_attempts = 10
            delay_seconds = 5

            for attempt in range(max_attempts):
                try:
                    async with session.post(
                        f"{GEMINI_CODE_ASSIST_ENDPOINT}:onboardUser",
                        headers=headers,
                        json={
                            "tierId": "FREE",
                            "metadata": GEMINI_CODE_ASSIST_METADATA,
                        },
                        timeout=aiohttp.ClientTimeout(total=30),
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            _LOGGER.debug(
                                "onboardUser response (attempt %d): %s",
                                attempt + 1,
                                json.dumps(data)[:300],
                            )

                            # Check if onboarding is complete
                            if data.get("done"):
                                project_id = (
                                    data.get("response", {})
                                    .get("cloudaicompanionProject", {})
                                    .get("id")
                                )
                                if project_id:
                                    _LOGGER.info(
                                        "Gemini onboarding complete, project: %s",
                                        project_id,
                                    )
                                    await self._save_project_id(project_id)
                                    return project_id
                        else:
                            error_text = await resp.text()
                            _LOGGER.warning(
                                "onboardUser failed (%d): %s",
                                resp.status,
                                error_text[:200],
                            )
                except aiohttp.ClientError as e:
                    _LOGGER.warning(
                        "onboardUser request failed (attempt %d): %s", attempt + 1, e
                    )

                if attempt < max_attempts - 1:
                    _LOGGER.debug(
                        "Onboarding in progress, waiting %ds (attempt %d/%d)...",
                        delay_seconds,
                        attempt + 1,
                        max_attempts,
                    )
                    await asyncio.sleep(delay_seconds)

            raise Exception(
                "Failed to obtain Gemini project ID after onboarding. "
                "Please try again later or check your Google account permissions."
            )

    def _convert_messages(
        self, messages: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Convert OpenAI-style messages to Gemini format.

        Handles:
        - System messages -> systemInstruction
        - User messages -> user role
        - Assistant messages -> model role (including function calls)
        - Function results -> user role with functionResponse

        Returns:
            Tuple of (contents list, system instruction text or None).
        """
        contents = []
        system_instruction = None

        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")

            if role == "system":
                if system_instruction is None:
                    system_instruction = content
                else:
                    system_instruction += "\n\n" + content
            elif role == "user" and content:
                contents.append({"role": "user", "parts": [{"text": content}]})
            elif role == "assistant" and content:
                # Check if this is a function call response (JSON with functionCall)
                try:
                    parsed = json.loads(content)
                    if "functionCall" in parsed:
                        contents.append({"role": "model", "parts": [parsed]})
                        continue
                except (ValueError, TypeError):
                    pass
                contents.append({"role": "model", "parts": [{"text": content}]})
            elif role == "function":
                # Tool result - Gemini uses functionResponse in user role
                func_name = message.get("name", "unknown")
                try:
                    result_data = json.loads(content)
                except (ValueError, TypeError):
                    result_data = {"result": content}
                contents.append(
                    {
                        "role": "user",
                        "parts": [
                            {
                                "functionResponse": {
                                    "name": func_name,
                                    "response": result_data,
                                }
                            }
                        ],
                    }
                )

        return contents, system_instruction

    def _convert_tools(
        self, openai_tools: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Convert OpenAI tool format to Gemini functionDeclarations format."""
        if not openai_tools:
            return []

        function_declarations = []
        for tool in openai_tools:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                function_declarations.append(
                    {
                        "name": func.get("name", ""),
                        "description": func.get("description", ""),
                        "parameters": func.get("parameters", {}),
                    }
                )

        return (
            [{"functionDeclarations": function_declarations}]
            if function_declarations
            else []
        )

    async def _retry_with_backoff(self, func, *args, **kwargs):
        """Retry with exponential backoff for 429 and 5xx errors."""
        attempt = 0
        current_delay = self.INITIAL_DELAY_MS / 1000

        while attempt < self.MAX_ATTEMPTS:
            attempt += 1
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                error_str = str(e)

                # Check if retryable (429 or 5xx)
                is_429 = "429" in error_str
                is_5xx = any(f"{code}" in error_str for code in range(500, 600))

                if not (is_429 or is_5xx):
                    raise

                if attempt >= self.MAX_ATTEMPTS:
                    _LOGGER.error(
                        "Max retry attempts (%d) reached for Gemini API",
                        self.MAX_ATTEMPTS,
                    )
                    raise

                # Check for "Please retry in Xs" in error message
                retry_match = re.search(
                    r"retry in (\d+(?:\.\d+)?)\s*(s|ms)",
                    error_str,
                    re.IGNORECASE,
                )
                if retry_match:
                    delay_value = float(retry_match.group(1))
                    delay_unit = retry_match.group(2).lower()
                    delay = delay_value if delay_unit == "s" else delay_value / 1000
                else:
                    # Exponential backoff with jitter (±30%)
                    jitter = current_delay * 0.3 * (random.random() * 2 - 1)
                    delay = max(0, current_delay + jitter)

                _LOGGER.warning(
                    "Gemini API %s (attempt %d/%d). Retrying in %.1fs...",
                    "429 rate limited" if is_429 else "server error",
                    attempt,
                    self.MAX_ATTEMPTS,
                    delay,
                )

                await asyncio.sleep(delay)
                current_delay = min(self.MAX_DELAY_MS / 1000, current_delay * 2)

        raise Exception("Retry attempts exhausted")

    async def _do_request(
        self,
        session: aiohttp.ClientSession,
        url: str,
        headers: dict,
        wrapped_payload: dict,
    ) -> str:
        """Execute the HTTP request to Gemini API."""
        # Log request details (without sensitive data)
        request_contents = wrapped_payload.get("request", {}).get("contents", [])
        _LOGGER.debug(
            "Gemini OAuth request: model=%s, contents_count=%d, has_tools=%s",
            wrapped_payload.get("model"),
            len(request_contents),
            "tools" in wrapped_payload.get("request", {}),
        )
        # Log first user message for context (truncated)
        if request_contents:
            last_content = request_contents[-1]
            parts = last_content.get("parts", [])
            if parts and "text" in parts[0]:
                text_preview = parts[0]["text"][:100]
                _LOGGER.debug("Gemini OAuth request last message: %s...", text_preview)

        async with session.post(
            url,
            headers=headers,
            json=wrapped_payload,
            timeout=aiohttp.ClientTimeout(total=300),
        ) as resp:
            response_text = await resp.text()

            _LOGGER.debug(
                "Gemini OAuth response: status=%d, length=%d",
                resp.status,
                len(response_text),
            )

            if resp.status != 200:
                _LOGGER.error(
                    "Gemini OAuth API error %d: %s", resp.status, response_text[:500]
                )
                raise Exception(
                    f"Gemini OAuth API error {resp.status}: {response_text[:500]}"
                )

            try:
                data = json.loads(response_text)
            except json.JSONDecodeError as e:
                _LOGGER.error(
                    "Gemini OAuth JSON decode error: %s, response: %s",
                    e,
                    response_text[:500],
                )
                raise Exception(f"Invalid JSON response from Gemini: {e}")

            # Log raw response structure for debugging
            _LOGGER.debug(
                "Gemini OAuth response keys: %s",
                list(data.keys()) if isinstance(data, dict) else type(data),
            )

            # Unwrap response if it contains 'response' key (Cloud Code API)
            if "response" in data:
                data = data["response"]
                _LOGGER.debug(
                    "Gemini OAuth unwrapped response keys: %s", list(data.keys())
                )

            # Check for error in response
            if "error" in data:
                error_info = data["error"]
                _LOGGER.error(
                    "Gemini OAuth API returned error: %s",
                    json.dumps(error_info)[:500],
                )
                raise Exception(f"Gemini API error: {error_info}")

            # Extract response from Gemini format
            candidates = data.get("candidates", [])
            _LOGGER.debug("Gemini OAuth candidates count: %d", len(candidates))

            if candidates:
                # Log finish reason if present
                finish_reason = candidates[0].get("finishReason")
                if finish_reason:
                    _LOGGER.debug("Gemini OAuth finish reason: %s", finish_reason)

                content = candidates[0].get("content", {})
                parts = content.get("parts", [])
                _LOGGER.debug("Gemini OAuth parts count: %d", len(parts))

                if parts:
                    first_part = parts[0]
                    # Check for function call first
                    if "functionCall" in first_part:
                        func_name = first_part["functionCall"].get("name", "unknown")
                        _LOGGER.debug(
                            "Gemini OAuth function call detected: %s", func_name
                        )
                        # Return JSON with functionCall for agentic loop to detect
                        return json.dumps(first_part)
                    # Otherwise return text
                    if "text" in first_part:
                        text_response = first_part["text"]
                        _LOGGER.debug(
                            "Gemini OAuth text response length: %d, preview: %s...",
                            len(text_response),
                            text_response[:100],
                        )
                        return text_response
                    # Log unexpected part format
                    _LOGGER.warning(
                        "Gemini OAuth unexpected part format: %s",
                        list(first_part.keys()),
                    )
                    # Fallback to JSON representation
                    return json.dumps(data)
            else:
                # No candidates - check for promptFeedback
                prompt_feedback = data.get("promptFeedback")
                if prompt_feedback:
                    _LOGGER.warning(
                        "Gemini OAuth prompt feedback (possibly blocked): %s",
                        json.dumps(prompt_feedback),
                    )

            _LOGGER.warning(
                "Unexpected Gemini response format: %s", response_text[:500]
            )
            return json.dumps(data)

    async def get_response(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        """Get a response from Gemini using OAuth token.

        Args:
            messages: List of message dictionaries with role and content.
            **kwargs: Additional arguments:
                - tools: List of tools for function calling.
                - model: Optional model override (must be in GEMINI_AVAILABLE_MODELS).

        Returns:
            The AI response as a string.
        """
        # Allow per-request model override
        model = kwargs.get("model") or self._model
        if model not in GEMINI_AVAILABLE_MODELS:
            _LOGGER.warning(
                "Model '%s' not in available models, using default '%s'",
                model,
                self.DEFAULT_MODEL,
            )
            model = self.DEFAULT_MODEL

        _LOGGER.info(
            "GeminiOAuthProvider.get_response called. managed_project_id: %s, model: %s",
            self._oauth_data.get("managed_project_id", "NOT_SET"),
            model,
        )
        access_token = await self._get_valid_token()

        _LOGGER.debug("Making OAuth request to Gemini API with model: %s", model)

        # Convert messages to Gemini format
        gemini_contents, system_instruction = self._convert_messages(messages)
        _LOGGER.debug(
            "Gemini OAuth message conversion: %d contents, system_instruction: %s",
            len(gemini_contents),
            "YES" if system_instruction else "NO",
        )

        # Build request payload
        request_payload: dict[str, Any] = {
            "contents": gemini_contents,
            "generationConfig": {
                "temperature": self.config.get("temperature", 0.7),
                "maxOutputTokens": 8192,
            },
        }

        # Add system instruction if present
        if system_instruction:
            request_payload["systemInstruction"] = {
                "parts": [{"text": system_instruction}]
            }
            _LOGGER.debug(
                "Gemini OAuth system instruction added (%d chars), preview: %s...",
                len(system_instruction),
                system_instruction[:200],
            )

        # Add tools for function calling if provided
        tools = kwargs.get("tools")
        if tools:
            gemini_tools = self._convert_tools(tools)
            if gemini_tools:
                request_payload["tools"] = gemini_tools
                _LOGGER.debug(
                    "Added %d tools to Gemini OAuth request",
                    len(gemini_tools[0].get("functionDeclarations", [])),
                )

        async with aiohttp.ClientSession() as session:
            # Ensure we have a valid project ID (will onboard if necessary)
            project_id = await self._ensure_project_id(session, access_token)

            # Wrap payload as per Cloud Code API expectation
            wrapped_payload = {
                "project": project_id,
                "model": model,
                "request": request_payload,
            }

            # URL construction: endpoint + :generateContent
            url = f"{GEMINI_CODE_ASSIST_ENDPOINT}:generateContent"

            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
                **GEMINI_CODE_ASSIST_HEADERS,
            }

            _LOGGER.debug("Gemini OAuth request URL: %s", url)
            _LOGGER.debug(
                "Gemini OAuth wrapped payload: project=%s, model=%s",
                project_id,
                model,
            )

            # Execute request with retry for 429 and 5xx errors
            return await self._retry_with_backoff(
                self._do_request, session, url, headers, wrapped_payload
            )
