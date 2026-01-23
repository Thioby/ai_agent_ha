# Anthropic OAuth Implementation Reference

> **Target**: LLM agents implementing OAuth in ai_agent_ha
> **Source**: `opencode-anthropic-auth/index.mjs`, `CodebuffAI/codebuff`
> **Last Updated**: 2025-01-23

---

## Quick Reference

| Component | Value |
|-----------|-------|
| Client ID | `9d1c250a-e61b-44d9-88ed-5944d1962f5e` |
| Auth URL (Max) | `https://claude.ai/oauth/authorize` |
| Token URL | `https://console.anthropic.com/v1/oauth/token` |
| API URL | `https://api.anthropic.com/v1/messages` |
| Redirect URI | `https://console.anthropic.com/oauth/code/callback` |
| Scopes | `org:create_api_key user:profile user:inference` |

---

## CRITICAL: Required System Prompt Prefix

**Without this prefix, requests fail with "This credential is only authorized for use with Claude Code".**

```python
CLAUDE_CODE_SYSTEM_PREFIX = "You are Claude Code, Anthropic's official CLI for Claude."

# System prompt MUST start with this prefix
system_prompt = f"{CLAUDE_CODE_SYSTEM_PREFIX}\n\n{your_actual_system_prompt}"
```

**Sources**:
- `CodebuffAI/codebuff/common/src/constants/claude-oauth.ts` - documents this as `CLAUDE_CODE_SYSTEM_PROMPT_PREFIX`
- `opencode/packages/opencode/src/session/prompt/anthropic_spoof.txt` - contains the prefix
- `opencode/packages/opencode/src/session/system.ts` - adds prefix via `header()` function for all Anthropic requests

---

## Required Headers

```python
headers = {
    "authorization": f"Bearer {access_token}",
    "content-type": "application/json",
    "anthropic-version": "2023-06-01",
    "anthropic-beta": "oauth-2025-04-20,claude-code-20250219,interleaved-thinking-2025-05-14,fine-grained-tool-streaming-2025-05-14",
}
```

### Beta Flags Explained

| Flag | Purpose | Required |
|------|---------|----------|
| `oauth-2025-04-20` | Enables OAuth token authentication | YES |
| `claude-code-20250219` | Identifies as Claude Code client | YES |
| `interleaved-thinking-2025-05-14` | Extended thinking features | Optional |
| `fine-grained-tool-streaming-2025-05-14` | Tool streaming | Optional |

---

## OAuth Endpoints

| Endpoint | URL | Method |
|----------|-----|--------|
| Auth (Max) | `https://claude.ai/oauth/authorize` | GET (browser) |
| Auth (Console) | `https://console.anthropic.com/oauth/authorize` | GET (browser) |
| Token | `https://console.anthropic.com/v1/oauth/token` | POST |
| API | `https://api.anthropic.com/v1/messages` | POST |

---

## PKCE Flow

### 1. Generate Challenge

```python
import secrets, hashlib, base64

def generate_pkce() -> tuple[str, str]:
    code_verifier = secrets.token_urlsafe(64)  # 64 bytes = 86 chars
    code_challenge = base64.urlsafe_b64encode(
        hashlib.sha256(code_verifier.encode()).digest()
    ).decode().rstrip("=")
    return code_verifier, code_challenge
```

### 2. Build Auth URL

**CRITICAL**: `state` parameter MUST equal `code_verifier` (not a separate value)

```python
def build_auth_url(challenge: str, state: str, mode: str = "max") -> str:
    base = "https://claude.ai" if mode == "max" else "https://console.anthropic.com"
    params = {
        "code": "true",
        "client_id": CLIENT_ID,
        "response_type": "code",
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPES,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": state,  # MUST be code_verifier
    }
    return f"{base}/oauth/authorize?{urlencode(params)}"
```

### 3. Exchange Code for Tokens

**CRITICAL**: Use `application/json` Content-Type (NOT form-urlencoded)

```python
async def exchange_code(code: str, verifier: str) -> dict:
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://console.anthropic.com/v1/oauth/token",
            json={  # JSON, not data=
                "code": code,
                "grant_type": "authorization_code",
                "client_id": CLIENT_ID,
                "redirect_uri": REDIRECT_URI,
                "code_verifier": verifier,
            },
            headers={"Content-Type": "application/json"},
        ) as resp:
            return await resp.json()
```

### 4. Token Response

```json
{
  "token_type": "Bearer",
  "access_token": "sk-ant-oat01-...",
  "expires_in": 28800,
  "refresh_token": "sk-ant-ort01-...",
  "scope": "user:inference user:profile"
}
```

### 5. Refresh Token

```python
async def refresh_token(refresh: str) -> dict:
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://console.anthropic.com/v1/oauth/token",
            json={
                "grant_type": "refresh_token",
                "refresh_token": refresh,
                "client_id": CLIENT_ID,
            },
            headers={"Content-Type": "application/json"},
        ) as resp:
            return await resp.json()
```

---

## Complete API Request Example

```python
async def make_oauth_request(access_token: str, messages: list, system_prompt: str, model: str):
    # CRITICAL: Prepend required prefix
    CLAUDE_CODE_PREFIX = "You are Claude Code, Anthropic's official CLI for Claude."
    full_system = f"{CLAUDE_CODE_PREFIX}\n\n{system_prompt}"
    
    headers = {
        "authorization": f"Bearer {access_token}",
        "content-type": "application/json",
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "oauth-2025-04-20,claude-code-20250219,interleaved-thinking-2025-05-14,fine-grained-tool-streaming-2025-05-14",
    }
    
    payload = {
        "model": model,
        "max_tokens": 8192,
        "messages": messages,
        "system": full_system,
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload,
        ) as resp:
            return await resp.json()
```

---

## Token Lifecycle

| Token | Prefix | Lifetime | Storage |
|-------|--------|----------|---------|
| Access | `sk-ant-oat01-` | 8 hours (28800s) | Memory/ConfigEntry |
| Refresh | `sk-ant-ort01-` | Long-lived | ConfigEntry.data (encrypted) |

### Refresh Logic

```python
async def get_valid_token(oauth_data: dict, hass, entry) -> str:
    if time.time() < oauth_data.get("expires_at", 0) - 300:  # 5 min buffer
        return oauth_data["access_token"]
    
    new_tokens = await refresh_token(oauth_data["refresh_token"])
    
    oauth_data.update({
        "access_token": new_tokens["access_token"],
        "refresh_token": new_tokens.get("refresh_token", oauth_data["refresh_token"]),
        "expires_at": time.time() + new_tokens["expires_in"],
    })
    
    # Persist to HA config entry
    hass.config_entries.async_update_entry(entry, data={**entry.data, "anthropic_oauth": oauth_data})
    
    return new_tokens["access_token"]
```

---

## Error Handling

| HTTP Status | Error Message | Cause | Fix |
|-------------|---------------|-------|-----|
| 400 | `anthropic-version: header is required` | Missing header | Add `anthropic-version: 2023-06-01` |
| 400 | `This credential is only authorized for use with Claude Code` | Missing system prefix OR beta flags | Add system prefix + `claude-code-20250219` beta |
| 401 | Token expired | Access token expired | Refresh token |
| 403 | Insufficient scopes | Wrong scopes | Re-authorize |

---

## Key Differences: OAuth vs API Key

| Aspect | API Key | OAuth |
|--------|---------|-------|
| Header | `x-api-key: sk-ant-api...` | `authorization: Bearer sk-ant-oat01-...` |
| Beta header | Not required | `oauth-2025-04-20,claude-code-20250219,...` |
| System prefix | Not required | **REQUIRED**: `"You are Claude Code..."` |
| Billing | Per-token usage | Flat subscription (Pro/Max) |

---

## Implementation Files

| File | Purpose |
|------|---------|
| `oauth.py` | PKCE generation, token exchange, refresh |
| `agent.py` | `AnthropicOAuthClient` class with system prefix |
| `config_flow.py` | Single-step OAuth form (link + code input) |
| `translations/en.json` | UI strings for OAuth step |
| `frontend/ai_agent_ha-panel.js` | `PROVIDERS` map includes `anthropic_oauth` |

---

## Config Entry Data Schema

```python
{
    "ai_provider": "anthropic_oauth",
    "anthropic_oauth": {
        "access_token": "sk-ant-oat01-...",
        "refresh_token": "sk-ant-ort01-...",
        "expires_at": 1737500000.0,  # Unix timestamp
    },
}
```

---

## Upstream Sync Checklist

Monitor these repos for changes:

### `opencode-anthropic-auth`
- `CLIENT_ID` - unlikely to change
- `anthropic-beta` header values - may add new betas
- OAuth endpoints - stable

### `opencode/packages/opencode/src/session/`
- `prompt/anthropic_spoof.txt` - system prefix
- `system.ts` - header injection logic

### `CodebuffAI/codebuff`
- `common/src/constants/claude-oauth.ts` - documents all requirements

**Check frequency**: Before major releases or if auth stops working.
