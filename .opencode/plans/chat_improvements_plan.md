# Chat UX Improvement Plan - ai_agent_ha

**Date**: 2026-02-05  
**Status**: Ready for Implementation  
**Priority**: HIGH - Fixes "chat się urywa" issue

---

## Executive Summary

Based on analysis of OpenClaw, CodeBuff, and OpenCode, identified 3 critical gaps causing poor chat UX:

1. ❌ **No Streaming** - responses appear all at once after 10-30s wait
2. ❌ **No Connection Status** - users don't know if system is working
3. ❌ **No Retry UI** - must retype messages after errors

**Impact**: Users experience "chat urywania się" (hanging/freezing)

---

## Current State Assessment

### ✅ What We Have (Good News)

| Feature | Status | Location | Notes |
|---------|--------|----------|-------|
| Retry Logic (Backend) | ✅ Implemented | `providers/gemini_oauth.py:390-442` | 10 attempts, exp backoff, smart errors |
| Error Classification | ✅ Exists | `error_handler.py` | Not integrated! |
| Error Display | ✅ Works | `frontend/src/lib/components/Chat/ErrorMessage.svelte` | Shows errors to user |
| Base Retry | ✅ Basic | `providers/base_client.py:114-152` | Dumb retry (all errors) |

### ❌ What's Missing (Root Causes)

| Feature | Impact | User Experience |
|---------|--------|-----------------|
| **Streaming** | 🔴 CRITICAL | 10-30s blank screen, UI frozen, looks broken |
| Connection Status | 🟡 HIGH | No feedback, users refresh page |
| Frontend Retry | 🟡 HIGH | Must retype after transient errors |
| Error Handler Integration | 🟡 MEDIUM | Wastes retries on permanent errors |

---

## Problem Analysis

### Problem #1: "Chat Hangs During Long Responses"

**Symptoms**:
- Loading spinner for 10-30 seconds
- No feedback during AI processing
- UI appears frozen
- Users think it's broken

**Root Cause**:
```typescript
// websocket_api.py:294 - Request-response pattern
async def ws_send_message(hass, connection, msg):
    # Process entire message
    response = await agent.process_query(message)  # BLOCKS for 10-30s
    # Return full response at once
    connection.send_result(msg["id"], response)    # All or nothing
```

**Why This Fails**:
- User sees nothing until AI finishes completely
- No token-by-token rendering
- Feels like system crashed
- Poor UX compared to ChatGPT/Claude

---

### Problem #2: "No Feedback When Connection Drops"

**Symptoms**:
- WebSocket disconnects silently
- Generic "error" appears
- Users don't know if it's their WiFi or server issue

**Root Cause**:
```typescript
// appState.ts - No connection tracking
interface AppStateType {
  hass: HomeAssistant | null;
  isLoading: boolean;
  error: string | null;
  // MISSING: connectionStatus, lastPing, reconnectAttempts
}
```

**Why This Fails**:
- Home Assistant's core handles reconnection, but ai_agent_ha doesn't expose this
- No visual indicator of connection health
- Errors look identical (network vs API vs logic)

---

### Problem #3: "Must Retype Messages After Errors"

**Symptoms**:
- Transient error (429 rate limit, network blip)
- Message disappears
- Must type again from scratch

**Root Cause**:
```typescript
// InputArea.svelte:92
catch (error: any) {
  console.error('WebSocket error:', error);
  const errorMessage = error.message || 'An error occurred';
  // NO: retry logic, message preservation, smart classification
  // User must retype manually
}
```

**Why This Fails**:
- No distinction between retryable vs permanent errors
- No "Retry" button on error messages
- Frontend doesn't queue failed messages

---

## Solution: 3-Phase Implementation

---

## Phase 1: Add Streaming (CRITICAL)

**Impact**: 🔴 Massive UX improvement  
**Effort**: 🟡 Medium (2-3 hours)  
**Priority**: P0 - Do First

### Backend Changes

**File**: `custom_components/ai_agent_ha/websocket_api.py`

**Current Code**:
```python
async def ws_send_message(hass, connection, msg):
    # Get full response
    response = await agent.process_query(message)
    # Send all at once
    connection.send_result(msg["id"], response)
```

**New Code - Streaming**:
```python
async def ws_send_message(hass, connection, msg):
    message_id = msg["id"]
    user_message = msg.get("message", "")
    
    # Send initial acknowledgment
    connection.send_message({
        "id": message_id,
        "type": "stream_start",
        "message_id": generate_uuid()
    })
    
    # Stream tokens
    async for chunk in agent.process_query_stream(user_message):
        connection.send_message({
            "id": message_id,
            "type": "stream_chunk",
            "chunk": chunk,  # Can be token, tool call, or metadata
        })
    
    # Send completion
    connection.send_message({
        "id": message_id,
        "type": "stream_end",
        "final_message": final_response
    })
```

**File**: `custom_components/ai_agent_ha/core/query_processor.py`

**Add Streaming Method**:
```python
async def process_query_stream(
    self, 
    query: str, 
    **kwargs
) -> AsyncGenerator[dict, None]:
    """Stream response chunks as they're generated."""
    
    # ... existing preprocessing ...
    
    # Stream from provider
    async for chunk in self.provider.get_response_stream(
        messages=built_messages,
        tools=tools,
        **provider_kwargs
    ):
        # Yield token chunks
        if chunk.get("type") == "text":
            yield {
                "type": "text",
                "content": chunk["content"]
            }
        
        # Yield tool calls
        elif chunk.get("type") == "tool_call":
            yield {
                "type": "tool",
                "name": chunk["name"],
                "status": "running"
            }
        
        # Yield tool results
        elif chunk.get("type") == "tool_result":
            yield {
                "type": "tool",
                "name": chunk["name"],
                "status": "complete",
                "result": chunk["result"]
            }
```

**File**: `custom_components/ai_agent_ha/providers/gemini_oauth.py`

**Add Streaming Support**:
```python
async def get_response_stream(
    self,
    messages: list[dict],
    tools: list[dict] | None = None,
    **kwargs
) -> AsyncGenerator[dict, None]:
    """Stream response from Gemini API."""
    
    # Build request with streaming enabled
    request_payload = self._build_request(messages, tools, **kwargs)
    request_payload["generationConfig"]["enableStreaming"] = True
    
    # Make streaming request
    async with aiohttp.ClientSession() as session:
        async with session.post(
            self._api_url,
            headers=self._headers,
            json=request_payload,
            timeout=aiohttp.ClientTimeout(total=300)  # 5 min
        ) as response:
            # Stream chunks
            async for line in response.content:
                if not line:
                    continue
                
                try:
                    chunk = json.loads(line.decode('utf-8'))
                    
                    # Parse Gemini response format
                    if "candidates" in chunk:
                        candidate = chunk["candidates"][0]
                        content = candidate.get("content", {})
                        parts = content.get("parts", [])
                        
                        for part in parts:
                            if "text" in part:
                                yield {
                                    "type": "text",
                                    "content": part["text"]
                                }
                            elif "functionCall" in part:
                                yield {
                                    "type": "tool_call",
                                    "name": part["functionCall"]["name"],
                                    "args": part["functionCall"]["args"]
                                }
                
                except json.JSONDecodeError:
                    continue
```

### Frontend Changes

**File**: `frontend/src/lib/services/websocket.service.ts`

**Add Streaming Handler**:
```typescript
export class WebSocketService {
  private streamHandlers = new Map<string, (chunk: any) => void>();
  
  sendMessageStream(
    message: string,
    onChunk: (chunk: StreamChunk) => void,
    onComplete: (response: Response) => void,
    onError: (error: Error) => void
  ) {
    const messageId = generateUUID();
    
    // Register stream handler
    this.streamHandlers.set(messageId, (data) => {
      if (data.type === 'stream_start') {
        onChunk({ type: 'start', messageId: data.message_id });
      }
      else if (data.type === 'stream_chunk') {
        onChunk({ type: 'chunk', content: data.chunk });
      }
      else if (data.type === 'stream_end') {
        this.streamHandlers.delete(messageId);
        onComplete(data.final_message);
      }
    });
    
    // Send message
    this.hass.callWS({
      type: 'ai_agent_ha/send_message',
      message_id: messageId,
      message: message
    });
  }
  
  // Handle incoming stream messages
  private handleMessage(message: any) {
    const handler = this.streamHandlers.get(message.id);
    if (handler) {
      handler(message);
    }
  }
}
```

**File**: `frontend/src/lib/components/Chat/ChatMessage.svelte`

**Add Streaming UI**:
```svelte
<script lang="ts">
  import { onMount } from 'svelte';
  
  export let message: Message;
  
  let displayedContent = '';
  let isStreaming = message.status === 'streaming';
  
  // Stream text letter by letter (with throttling)
  $: if (isStreaming) {
    let index = 0;
    const interval = setInterval(() => {
      if (index < message.content.length) {
        displayedContent = message.content.slice(0, index + 1);
        index++;
      } else {
        clearInterval(interval);
        isStreaming = false;
      }
    }, 20); // 20ms per char = ~50 chars/sec
  }
</script>

<div class="message assistant">
  <div class="content">
    {#if isStreaming}
      {displayedContent}<span class="cursor">▋</span>
    {:else}
      {message.content}
    {/if}
  </div>
</div>

<style>
  .cursor {
    animation: blink 1s infinite;
  }
  
  @keyframes blink {
    0%, 50% { opacity: 1; }
    51%, 100% { opacity: 0; }
  }
</style>
```

**File**: `frontend/src/lib/stores/chatStore.ts`

**Add Streaming State**:
```typescript
interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  status: 'pending' | 'streaming' | 'complete' | 'error';
  chunks?: string[];  // Store chunks during streaming
  timestamp: number;
}

export const chatStore = writable<Message[]>([]);

export function addStreamingMessage(messageId: string) {
  chatStore.update(messages => [
    ...messages,
    {
      id: messageId,
      role: 'assistant',
      content: '',
      status: 'streaming',
      chunks: [],
      timestamp: Date.now()
    }
  ]);
}

export function appendChunk(messageId: string, chunk: string) {
  chatStore.update(messages => 
    messages.map(msg => 
      msg.id === messageId
        ? { ...msg, content: msg.content + chunk, chunks: [...msg.chunks!, chunk] }
        : msg
    )
  );
}

export function completeStream(messageId: string) {
  chatStore.update(messages =>
    messages.map(msg =>
      msg.id === messageId
        ? { ...msg, status: 'complete' }
        : msg
    )
  );
}
```

**Testing**:
1. Send message "Write a long paragraph about Poland"
2. Verify tokens appear gradually (not all at once)
3. Verify cursor animation during streaming
4. Verify completion state after stream ends

---

## Phase 2: Add Connection Status UI

**Impact**: 🟡 High - User feedback  
**Effort**: 🟢 Low (1 hour)  
**Priority**: P1 - Do Second

### Frontend Changes

**File**: `frontend/src/lib/stores/connectionStore.ts` (NEW)

```typescript
import { writable, derived } from 'svelte/store';

interface ConnectionState {
  status: 'connected' | 'connecting' | 'disconnected' | 'reconnecting';
  lastPing: number;
  reconnectAttempts: number;
  error: string | null;
}

export const connectionState = writable<ConnectionState>({
  status: 'connected',
  lastPing: Date.now(),
  reconnectAttempts: 0,
  error: null
});

// Derived - is connection healthy?
export const isHealthy = derived(
  connectionState,
  $state => $state.status === 'connected' && Date.now() - $state.lastPing < 30000
);

// Monitor Home Assistant connection
export function initConnectionMonitor(hass: HomeAssistant) {
  // Listen to HA connection events
  hass.connection.addEventListener('ready', () => {
    connectionState.update(s => ({
      ...s,
      status: 'connected',
      lastPing: Date.now(),
      reconnectAttempts: 0,
      error: null
    }));
  });
  
  hass.connection.addEventListener('disconnected', () => {
    connectionState.update(s => ({
      ...s,
      status: 'disconnected',
      error: 'Lost connection to Home Assistant'
    }));
  });
  
  hass.connection.addEventListener('reconnect-error', () => {
    connectionState.update(s => ({
      ...s,
      status: 'reconnecting',
      reconnectAttempts: s.reconnectAttempts + 1
    }));
  });
  
  // Heartbeat - ping every 10s
  setInterval(() => {
    if (hass.connection.connected) {
      connectionState.update(s => ({ ...s, lastPing: Date.now() }));
    }
  }, 10000);
}
```

**File**: `frontend/src/lib/components/ConnectionStatus.svelte` (NEW)

```svelte
<script lang="ts">
  import { connectionState } from '$lib/stores/connectionStore';
  
  $: statusColor = {
    connected: 'green',
    connecting: 'yellow',
    disconnected: 'red',
    reconnecting: 'orange'
  }[$connectionState.status];
  
  $: statusText = {
    connected: 'Connected',
    connecting: 'Connecting...',
    disconnected: 'Disconnected',
    reconnecting: `Reconnecting (attempt ${$connectionState.reconnectAttempts})`
  }[$connectionState.status];
</script>

{#if $connectionState.status !== 'connected'}
  <div class="connection-banner {statusColor}">
    <span class="dot"></span>
    <span>{statusText}</span>
    {#if $connectionState.error}
      <span class="error-detail">{$connectionState.error}</span>
    {/if}
  </div>
{/if}

<style>
  .connection-banner {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    padding: 8px 16px;
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 14px;
    z-index: 1000;
  }
  
  .green { background: #4caf50; color: white; }
  .yellow { background: #ff9800; color: white; }
  .red { background: #f44336; color: white; }
  .orange { background: #ff5722; color: white; }
  
  .dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: currentColor;
    animation: pulse 2s infinite;
  }
  
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }
</style>
```

**File**: `frontend/src/App.svelte`

**Add Connection Monitor**:
```svelte
<script lang="ts">
  import ConnectionStatus from '$lib/components/ConnectionStatus.svelte';
  import { initConnectionMonitor } from '$lib/stores/connectionStore';
  import { onMount } from 'svelte';
  
  export let hass: HomeAssistant;
  
  onMount(() => {
    initConnectionMonitor(hass);
  });
</script>

<ConnectionStatus />
<main>
  <!-- Existing app content -->
</main>
```

**Testing**:
1. Disconnect WiFi
2. Verify red "Disconnected" banner appears
3. Reconnect WiFi
4. Verify banner shows "Reconnecting" then disappears

---

## Phase 3: Add Retry UI & Smart Error Handling

**Impact**: 🟡 Medium - Quality of life  
**Effort**: 🟡 Medium (2 hours)  
**Priority**: P2 - Do Third

### Backend Changes

**File**: `custom_components/ai_agent_ha/websocket_api.py`

**Integrate Error Classifier**:
```python
from ..error_handler import ErrorClassifier, ErrorType

async def ws_send_message(hass, connection, msg):
    try:
        # ... existing code ...
    except Exception as e:
        # Classify error
        error_type = ErrorClassifier.classify_error(e)
        
        connection.send_error(msg["id"], {
            "code": error_type.name,
            "message": str(e),
            "retryable": error_type == ErrorType.TRANSIENT,
            "details": {
                "provider": "gemini_oauth",
                "timestamp": time.time()
            }
        })
```

### Frontend Changes

**File**: `frontend/src/lib/components/Chat/ErrorMessage.svelte`

**Add Retry Button**:
```svelte
<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  
  export let message: Message;
  export let error: ErrorDetails;
  
  const dispatch = createEventDispatcher();
  
  function retry() {
    dispatch('retry', { messageId: message.id });
  }
</script>

<div class="error-message">
  <div class="icon">⚠️</div>
  <div class="content">
    <p class="title">Error: {error.code}</p>
    <p class="detail">{error.message}</p>
    
    {#if error.retryable}
      <button class="retry-btn" on:click={retry}>
        🔄 Retry
      </button>
    {:else}
      <p class="help">
        This error cannot be automatically retried. Please check your message and try again.
      </p>
    {/if}
  </div>
  <button class="dismiss" on:click={() => dispatch('dismiss')}>✕</button>
</div>

<style>
  .error-message {
    background: #fff3cd;
    border: 1px solid #ffc107;
    border-radius: 8px;
    padding: 12px;
    display: flex;
    gap: 12px;
    margin: 8px 0;
  }
  
  .retry-btn {
    margin-top: 8px;
    padding: 6px 12px;
    background: #2196f3;
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
  }
  
  .retry-btn:hover {
    background: #1976d2;
  }
</style>
```

**File**: `frontend/src/lib/components/Input/InputArea.svelte`

**Add Retry Handler**:
```typescript
async function sendMessage() {
  const userMessage = message.trim();
  if (!userMessage) return;
  
  // Add user message immediately (optimistic UI)
  chatStore.addMessage({
    role: 'user',
    content: userMessage,
    status: 'complete'
  });
  
  // Clear input
  message = '';
  
  // Send to backend with retry
  await sendWithRetry(userMessage, 3);
}

async function sendWithRetry(content: string, maxAttempts: number) {
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      await websocketService.sendMessageStream(
        content,
        (chunk) => chatStore.appendChunk(chunk),
        (response) => chatStore.completeStream(response),
        (error) => {
          if (error.retryable && attempt < maxAttempts) {
            // Auto-retry transient errors
            const delay = Math.min(2 ** attempt * 1000, 10000);
            setTimeout(() => sendWithRetry(content, maxAttempts - attempt), delay);
          } else {
            // Show error with retry button
            chatStore.addError(error);
          }
        }
      );
      return; // Success
      
    } catch (error: any) {
      if (attempt === maxAttempts) {
        // Final attempt failed
        chatStore.addError({
          code: error.code || 'UNKNOWN',
          message: error.message,
          retryable: error.retryable !== false
        });
      }
      // Wait before retry
      await new Promise(resolve => setTimeout(resolve, 2 ** attempt * 1000));
    }
  }
}

function handleRetry(event: CustomEvent) {
  const { messageId } = event.detail;
  const originalMessage = chatStore.getMessageContent(messageId);
  sendWithRetry(originalMessage, 3);
}
```

**Testing**:
1. Simulate 429 error (mock API)
2. Verify auto-retry happens (3 attempts)
3. Verify retry button appears if all fail
4. Click retry button - message re-sent

---

## Implementation Order

1. **Phase 1: Streaming** (P0)
   - Backend: Add `get_response_stream()` to providers
   - Backend: Add `process_query_stream()` to query_processor
   - Backend: Update `ws_send_message()` to stream
   - Frontend: Add streaming WebSocket handler
   - Frontend: Add streaming UI components
   - Test: Send long message, verify gradual appearance

2. **Phase 2: Connection Status** (P1)
   - Frontend: Create `connectionStore.ts`
   - Frontend: Create `ConnectionStatus.svelte`
   - Frontend: Init monitor in `App.svelte`
   - Test: Disconnect/reconnect WiFi

3. **Phase 3: Retry UI** (P2)
   - Backend: Integrate ErrorClassifier
   - Frontend: Update ErrorMessage with retry button
   - Frontend: Add auto-retry logic to InputArea
   - Test: Simulate errors, verify retry

---

## Testing Plan

### Test Case 1: Streaming Works
**Steps**:
1. Send message: "Write 3 paragraphs about Warsaw"
2. Observe UI during processing

**Expected**:
- Loading spinner appears immediately
- After 1-2s, first words appear
- Words stream in gradually (not all at once)
- Cursor blinks during streaming
- Final message marked as complete

**Metrics**:
- Time to first token: <2s
- Tokens per second: ~20-50
- UI responsive during streaming: YES

---

### Test Case 2: Connection Status Visible
**Steps**:
1. Start chat
2. Disable WiFi/Ethernet
3. Wait 5s
4. Re-enable network

**Expected**:
- Yellow "Connecting..." banner appears within 5s
- Banner changes to orange "Reconnecting (attempt 1)"
- After reconnect, banner disappears
- No errors shown to user

---

### Test Case 3: Retry Button Works
**Steps**:
1. Mock 429 error from API
2. Send message

**Expected**:
- Auto-retry happens (3 attempts, exp backoff)
- After 3 failures, error message appears
- Error has "Retry" button
- Click retry → message re-sent
- Success → error disappears

---

## Success Metrics

### Before (Current State)
- Time to first feedback: 10-30s
- User knows system is working: ❌ No
- Recovery from errors: ❌ Manual retype
- Connection issues visible: ❌ No

### After (With Improvements)
- Time to first feedback: <2s ✅
- User knows system is working: ✅ Streaming + status indicator
- Recovery from errors: ✅ Auto-retry + retry button
- Connection issues visible: ✅ Banner with reconnect status

**User Experience Rating**:
- Before: 3/10 (feels broken)
- After: 8/10 (modern chat UX)

---

## Rollback Plan

If issues arise:

1. **Disable Streaming**:
   ```python
   # In websocket_api.py, comment out streaming code
   # Revert to old request-response
   ```

2. **Disable Connection Status**:
   ```svelte
   <!-- In App.svelte, comment out ConnectionStatus -->
   <!-- {#if false}<ConnectionStatus />{/if} -->
   ```

3. **Disable Auto-Retry**:
   ```typescript
   // In InputArea.svelte, set maxAttempts = 1
   await sendWithRetry(content, 1);
   ```

All features are opt-in and can be disabled independently.

---

## Future Enhancements (Out of Scope)

1. **Optimistic UI** - Show user message before backend confirms
2. **Message Queue** - Queue messages when offline, send when reconnected
3. **Typing Indicators** - Show "AI is thinking..."
4. **Read Receipts** - Mark messages as seen
5. **Message Editing** - Edit sent messages
6. **Voice Input** - Speak instead of type
7. **Markdown Rendering** - Rich text in responses
8. **Code Highlighting** - Syntax highlighting for code blocks

---

## Estimated Timeline

| Phase | Tasks | Effort | Duration |
|-------|-------|--------|----------|
| Phase 1: Streaming | Backend + Frontend | 2-3 hours | 1 day |
| Phase 2: Connection | Frontend only | 1 hour | 2 hours |
| Phase 3: Retry UI | Backend + Frontend | 2 hours | 3 hours |
| **Testing** | All test cases | 1-2 hours | 2 hours |
| **TOTAL** | - | **6-8 hours** | **1-2 days** |

---

## Dependencies

**Required**:
- Home Assistant 2024.1+ (WebSocket API)
- Python 3.11+ (async generators)
- Svelte 4+ (reactive stores)

**Optional**:
- Gemini API streaming support (check docs)
- Anthropic API streaming (if using Claude)

---

## Conclusion

This plan addresses the root causes of "chat urywania się":

1. ✅ Streaming eliminates 10-30s blank screen
2. ✅ Connection status gives user feedback
3. ✅ Retry UI prevents message loss

**Implementation is straightforward** - follows patterns from OpenClaw/CodeBuff/OpenCode.

**Ready to proceed** - all technical details specified.

---

**End of Plan**
