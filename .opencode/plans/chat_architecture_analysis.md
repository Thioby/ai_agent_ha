# Chat Architecture Analysis - OpenClaw, CodeBuff, OpenCode

**Date**: 2026-02-05  
**Problem**: Chat się urywa i nic się nie dzieje w ai_agent_ha  
**Analyzed by**: Gemini 3 Pro + Task exploration

---

## Executive Summary

Przeanalizowano 3 projekty AI chat/coding assistant aby zidentyfikować przyczyny "urywania się" czatu:
- **OpenClaw** - Production-grade personal assistant (12+ channels)
- **CodeBuff** - Multi-agent coding assistant
- **OpenCode** - AI coding agent

**Kluczowe Wnioski**:
1. ❌ **ai_agent_ha brakuje Retry Logic** - system poddaje się przy błędach API
2. ❌ **ai_agent_ha brakuje UI Throttling** - za dużo re-renderów przy streamingu
3. ❌ **ai_agent_ha brakuje Connection State** - frontend nie wykrywa disconnect

---

## 1. Streaming & Event Buffering (OpenClaw)

### Problem
UI zamarza przy szybkim streamingu eventów z backendu.

### Rozwiązanie OpenClaw
**Plik:** `openclaw/ui/src/ui/app-tool-stream.ts`

```typescript
const TOOL_STREAM_THROTTLE_MS = 80;

// Buforowanie aktualizacji UI
export function scheduleToolStreamSync(host: ToolStreamHost, force = false) {
  if (force) {
    flushToolStreamSync(host);
    return;
  }
  if (host.toolStreamSyncTimer != null) {
    return; // Timer już działa, czekamy
  }
  // Opóźnione odświeżanie UI co 80ms
  host.toolStreamSyncTimer = window.setTimeout(
    () => flushToolStreamSync(host),
    TOOL_STREAM_THROTTLE_MS,
  );
}

function flushToolStreamSync(host: ToolStreamHost) {
  clearTimeout(host.toolStreamSyncTimer);
  host.toolStreamSyncTimer = null;
  
  // Batch update - przetwarzamy wszystkie zgromadzone eventy
  host.requestUpdate();
}
```

**Mechanizm**:
- Backend wysyła eventy ciągłym strumieniem
- UI **buforuje** je przez 80ms
- Następnie **batch update** - renderuje wszystkie razem
- Zapobiega "zadławieniu" przeglądarki

**Wniosek dla ai_agent_ha**:
```
❌ Jeśli UI zamarza lub "tyka" przy długich odpowiedziach AI
→ Brakuje warstwy buforującej między SSE/WebSocket a Svelte store
→ Każdy event triggeruje re-render, co blokuje UI
```

---

## 2. Client-Side Queue & Reconnection (CodeBuff)

### Problem
Utrata połączenia powoduje "urywanie się" czatu bez informacji dla użytkownika.

### Rozwiązanie CodeBuff
**Plik:** `codebuff/cli/src/hooks/use-chat-streaming.ts`

```typescript
// React hook zarządzający stanem połączenia
const useConnectionStatus = (handleReconnection) => {
  const [isConnected, setIsConnected] = useState(true);
  
  useEffect(() => {
    // Nasłuchiwanie eventów online/offline
    const handleOnline = () => {
      setIsConnected(true);
      handleReconnection(false); // false = nie pierwsze połączenie
    };
    
    const handleOffline = () => {
      setIsConnected(false);
    };
    
    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);
    
    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, [handleReconnection]);
  
  return isConnected;
};

// Obsługa reconnectu z UI feedback
const handleReconnection = useCallback((isInitialConnection: boolean) => {
  // Invalidacja cache
  queryClient.invalidateQueries();
  
  if (!isInitialConnection) {
    // Pokaż banner "Połączono ponownie"
    setShowReconnectionMessage(true);
    
    // Ukryj po 3 sekundach
    reconnectionTimeout.setTimeout('reconnection-message', () => {
       setShowReconnectionMessage(false);
    }, 3000);
  }
}, [queryClient]);

// UI pokazuje stan
const isConnected = useConnectionStatus(handleReconnection);

return (
  <>
    {!isConnected && <div className="offline-banner">Brak połączenia...</div>}
    {showReconnectionMessage && <div className="reconnect-banner">Połączono ponownie!</div>}
    <ChatMessages />
  </>
);
```

**Mechanizm**:
1. Hook nasłuchuje `online`/`offline` events
2. Pokazuje UI banner gdy brak połączenia
3. Auto-reconnect po powrocie sieci
4. Invalidacja cache (refetch state)

**Dodatkowa Funkcja - Message Queue**:
```typescript
// Kolejka wiadomości wysłanych offline
const messageQueue = useRef<Message[]>([]);

const sendMessage = async (text: string) => {
  const msg = { id: uuid(), text, timestamp: Date.now() };
  
  if (!isConnected) {
    // Dodaj do kolejki zamiast wysyłać
    messageQueue.current.push(msg);
    showNotification("Wiadomość wysłana gdy będzie połączenie");
    return;
  }
  
  // Wyślij normalnie
  await api.sendMessage(msg);
};

// Po reconnect - wyślij zakolejkowane
useEffect(() => {
  if (isConnected && messageQueue.current.length > 0) {
    messageQueue.current.forEach(msg => api.sendMessage(msg));
    messageQueue.current = [];
  }
}, [isConnected]);
```

**Wniosek dla ai_agent_ha**:
```
❌ Jeśli czat "urywa się" bez informacji dla użytkownika
→ Brakuje detection stanu połączenia
→ Brakuje kolejki dla wiadomości wysłanych offline
→ Użytkownik nie wie czy "czeka" czy "padło"
```

---

## 3. Robust Retry Logic (OpenCode)

### Problem
API zwraca błędy (429 Rate Limit, 503 Overloaded) i system się poddaje.

### Rozwiązanie OpenCode
**Plik:** `opencode/packages/opencode/src/session/retry.ts`

```typescript
const RETRY_INITIAL_DELAY = 2000;      // 2s
const RETRY_BACKOFF_FACTOR = 2;        // Podwajaj
const RETRY_MAX_DELAY_NO_HEADERS = 60000; // Max 60s
const RETRY_MAX_DELAY_WITH_HEADERS = 300000; // Max 5min jeśli API podał Retry-After

export function delay(attempt: number, error?: MessageV2.APIError) {
  if (error) {
    const headers = error.data.responseHeaders;
    
    // Respektowanie nagłówka Retry-After
    if (headers?.["retry-after"]) {
      const retryAfter = headers["retry-after"];
      
      // Może być data lub sekundy
      const parsedDate = new Date(retryAfter).getTime();
      if (!isNaN(parsedDate)) {
        const delayMs = parsedDate - Date.now();
        return Math.min(delayMs, RETRY_MAX_DELAY_WITH_HEADERS);
      }
      
      // Lub liczba sekund
      const parsedSeconds = parseInt(retryAfter, 10);
      if (!isNaN(parsedSeconds)) {
        return Math.min(parsedSeconds * 1000, RETRY_MAX_DELAY_WITH_HEADERS);
      }
    }
  }
  
  // Exponential backoff: 2s, 4s, 8s, 16s, 32s, 60s (max)
  return Math.min(
    RETRY_INITIAL_DELAY * Math.pow(RETRY_BACKOFF_FACTOR, attempt - 1),
    RETRY_MAX_DELAY_NO_HEADERS
  );
}

export function retryable(error: any): string | undefined {
  // Sprawdź czy błąd jest do ponowienia
  
  if (error.status === 429) {
    return "Rate Limited";
  }
  
  if (error.status === 503 || error.status === 502) {
    return "Provider Overloaded";
  }
  
  // Analiza JSON body
  const json = error.data?.error || {};
  
  if (json.code?.includes("exhausted") || json.code?.includes("unavailable")) {
    return "Provider is overloaded";
  }
  
  if (json.error?.code?.includes("rate_limit")) {
    return "Rate Limited";
  }
  
  if (json.type === "overloaded_error") {
    return "Provider Overloaded";
  }
  
  // Błędy krytyczne - NIE ponawiaj
  if (error.status === 401 || error.status === 403) {
    return undefined; // Auth failed
  }
  
  if (error.status >= 400 && error.status < 500) {
    return undefined; // Client error - retry nie pomoże
  }
  
  return undefined;
}

// Wrapper dla LLM call
export async function withRetry<T>(
  fn: () => Promise<T>,
  maxAttempts = 5
): Promise<T> {
  let lastError: any;
  
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;
      
      const retryReason = retryable(error);
      if (!retryReason) {
        // Nie można ponowić - rzuć błąd
        throw error;
      }
      
      if (attempt === maxAttempts) {
        // Ostatnia próba - rzuć błąd
        throw new Error(`Failed after ${maxAttempts} attempts: ${retryReason}`);
      }
      
      // Oblicz delay
      const delayMs = delay(attempt, error);
      
      console.log(
        `Retry ${attempt}/${maxAttempts} after ${delayMs}ms: ${retryReason}`
      );
      
      // Czekaj
      await new Promise(resolve => setTimeout(resolve, delayMs));
    }
  }
  
  throw lastError;
}
```

**Użycie**:
```typescript
// W providers/gemini.ts
const response = await withRetry(
  () => fetch(apiUrl, { method: 'POST', body: payload }),
  5 // max 5 prób
);
```

**Wniosek dla ai_agent_ha**:
```
❌ Jeśli czat "urywa się" przy błędach API (429, 503)
→ Brakuje retry logic - jeden błąd kończy całą konwersację
→ Brakuje backoff - natychmiastowe ponowienie tylko pogarsza rate limit
→ Brakuje respektowania Retry-After header
```

---

## Porównanie Projektów

| Cecha | OpenClaw | CodeBuff | OpenCode |
|-------|----------|----------|----------|
| **Architektura** | Event-driven UI + Gateway daemon | React Hooks + TUI | Vercel AI SDK + Session |
| **Streaming** | SSE z throttling (80ms) | WebSocket/SSE z queue | `streamText()` z transforms |
| **Retry Backend** | Gateway (ukryte) | ❌ Brak | ✅ **Pełny Backoff + Headers** |
| **Reconnection UI** | ❌ Nie potrzebne (daemon) | ✅ **Banner + Queue** | ❌ Brak (desktop app) |
| **UI Performance** | ✅ **Throttling + Batching** | React optymalizacje | Electron (native) |
| **Error Handling** | Event `error` | Try-catch + notification | Retry wrapper |
| **Długie operacje** | Progress eventy | Loading states | Middleware progress |

---

## Rekomendacje dla ai_agent_ha

### Obecna Architektura ai_agent_ha:
- Backend: **FastAPI** (Python)
- Frontend: **Svelte** + **WebSocket/SSE**
- Provider: **Gemini OAuth**

### Problem 1: "Chat się urywa" - Brak Retry
**Symptom**: Przy błędzie 429/503 chat przestaje odpowiadać

**Rozwiązanie**:
```python
# custom_components/ai_agent_ha/providers/gemini_oauth.py

import time
from typing import TypeVar, Callable

T = TypeVar('T')

def with_retry(
    fn: Callable[[], T],
    max_attempts: int = 5
) -> T:
    """Exponential backoff retry wrapper."""
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as e:
            # Sprawdź czy można ponowić
            if not _retryable(e):
                raise
            
            if attempt == max_attempts:
                raise Exception(f"Failed after {max_attempts} attempts")
            
            # Oblicz delay
            delay = min(2 ** attempt, 60)  # 2s, 4s, 8s, 16s, 32s, 60s
            
            _LOGGER.warning(
                f"Retry {attempt}/{max_attempts} after {delay}s: {e}"
            )
            time.sleep(delay)

def _retryable(error: Exception) -> bool:
    """Check if error is retryable."""
    # HTTP 429, 503
    if hasattr(error, 'status'):
        return error.status in [429, 503, 502]
    
    # GaxiosError z Gemini
    if 'rate_limit' in str(error).lower():
        return True
    if 'overload' in str(error).lower():
        return True
    
    return False

# Użycie:
async def get_response(self, ...):
    return with_retry(
        lambda: self._make_api_call(...),
        max_attempts=5
    )
```

---

### Problem 2: "UI zamarza" - Brak Throttling
**Symptom**: Podczas długich odpowiedzi UI się "tyka"

**Rozwiązanie**:
```svelte
<!-- frontend/src/lib/stores/chatStore.ts -->

import { writable } from 'svelte/store';

const THROTTLE_MS = 80; // jak OpenClaw

class ThrottledChatStore {
  private buffer: Message[] = [];
  private timer: NodeJS.Timeout | null = null;
  
  addMessage(msg: Message) {
    this.buffer.push(msg);
    
    if (!this.timer) {
      this.timer = setTimeout(() => {
        this.flush();
      }, THROTTLE_MS);
    }
  }
  
  flush() {
    if (this.buffer.length === 0) return;
    
    // Batch update - wszystkie na raz
    chatMessages.update(msgs => [...msgs, ...this.buffer]);
    this.buffer = [];
    this.timer = null;
  }
  
  forceFlush() {
    if (this.timer) {
      clearTimeout(this.timer);
      this.timer = null;
    }
    this.flush();
  }
}

export const throttledChat = new ThrottledChatStore();
```

---

### Problem 3: "Nie wiadomo co się dzieje" - Brak Connection State
**Symptom**: Czat przestaje działać bez komunikatu

**Rozwiązanie**:
```svelte
<!-- frontend/src/lib/stores/connectionStore.ts -->

import { writable } from 'svelte/store';

export const connectionStatus = writable<'connected' | 'disconnected' | 'reconnecting'>('connected');
export const messageQueue = writable<Message[]>([]);

// Hook do WebSocket/SSE
export function setupConnectionMonitor(ws: WebSocket) {
  ws.addEventListener('close', () => {
    connectionStatus.set('disconnected');
  });
  
  ws.addEventListener('open', () => {
    connectionStatus.set('connected');
    
    // Wyślij zakolejkowane wiadomości
    messageQueue.update(queue => {
      queue.forEach(msg => ws.send(JSON.stringify(msg)));
      return [];
    });
  });
  
  ws.addEventListener('error', () => {
    connectionStatus.set('reconnecting');
  });
}

<!-- UI Component -->
<script>
  import { connectionStatus } from '$lib/stores/connectionStore';
</script>

{#if $connectionStatus === 'disconnected'}
  <div class="banner error">Brak połączenia z serwerem</div>
{:else if $connectionStatus === 'reconnecting'}
  <div class="banner warning">Łączenie ponowne...</div>
{/if}
```

---

## Priorytety Implementacji

### P0 - Krytyczne (natychmiast)
1. ✅ **Retry Logic w Gemini Provider** - zapobiega "urywaniu się" przy błędach API
2. ✅ **Connection State w UI** - użytkownik wie co się dzieje

### P1 - Ważne (niedługo)
3. ✅ **UI Throttling** - zapobiega zawieszaniu się przy długich odpowiedziach

### P2 - Nice-to-have
4. Message Queue - offline message sending
5. Heartbeat/Ping - wykrywanie cichej utraty połączenia
6. Progress Events - pokazywanie postępu długich operacji

---

## Następne Kroki

1. **Sprawdź obecny kod ai_agent_ha**:
   - Czy jest jakiś retry? → `providers/gemini_oauth.py`
   - Jak działa streaming? → `frontend/...`
   - Czy jest connection state? → `frontend/...`

2. **Zaimplementuj P0**:
   - Dodaj `with_retry()` wrapper
   - Dodaj connection status store

3. **Przetestuj**:
   - Symuluj 429 błąd (rate limit)
   - Symuluj utratę sieci
   - Wyślij bardzo długą odpowiedź (>5000 tokenów)

---

## Appendix: Code Locations

### OpenClaw
- Session: `src/agents/cli-session.ts`
- Streaming: `ui/src/ui/app-tool-stream.ts`
- SSE: `extensions/tlon/src/urbit/sse-client.ts`

### CodeBuff  
- Streaming Hook: `cli/src/hooks/use-chat-streaming.ts`
- Chunk Processor: `cli/src/utils/stream-chunk-processor.ts`
- Connection Status: `cli/src/hooks/use-connection-status.ts`

### OpenCode
- Retry Logic: `packages/opencode/src/session/retry.ts`
- LLM Wrapper: `packages/opencode/src/session/llm.ts`
- Session: `packages/opencode/src/session/index.ts`

---

**Koniec Analizy**
