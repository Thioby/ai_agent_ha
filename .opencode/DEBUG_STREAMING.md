# Debug Streaming - Instrukcje

## Problem
Streaming nie działa - odpowiedź przychodzi całościowo zamiast token-by-token.

## Co sprawdzić

### 1. Sprawdź logi Home Assistant

```bash
# Otwórz terminal i monitoruj logi:
tail -f /config/home-assistant.log | grep -E "(STREAMING|stream_|WS_SEND)"
```

Lub w Home Assistant UI:
- Settings → System → Logs
- Szukaj: `STREAMING`, `stream_chunk`, `WS_SEND_MESSAGE_STREAM`

### 2. Sprawdź browser console

1. Otwórz AI Agent HA panel
2. Naciśnij F12 (Developer Tools)
3. Zakładka Console
4. Wyślij wiadomość
5. Szukaj logów:
   - `[InputArea] Using STREAMING mode`
   - `[WebSocket] Sending STREAMING message`
   - `[WebSocket] Received streaming event`
   - `[WebSocket] Stream chunk`

### 3. Sprawdź która funkcja jest wywoływana

**Jeśli widzisz:**
- ✅ `WS_SEND_MESSAGE_STREAM CALLED!` → streaming endpoint działa
- ❌ Brak tego loga → frontend NIE wywołuje streamingu

**Jeśli widzisz w konsoli:**
- ✅ `[InputArea] Using STREAMING mode` → frontend próbuje
- ❌ Brak tego → `USE_STREAMING` jest false

### 4. Sprawdź czy Gemini zwraca streaming

**W logach szukaj:**
- `Gemini OAuth STREAMING request to:` → pokazuje URL
- `Gemini streaming response status: 200` → API odpowiada
- `Gemini streaming: starting to read chunks` → zaczyna czytać
- `Gemini streaming: received first chunk` → dostaje dane

**Jeśli NIE ma "received first chunk":**
→ Gemini Cloud Code API NIE wspiera `:streamGenerateContent`

## Możliwe przyczyny

### Przyczyna 1: Frontend nie wywołuje streamingu
**Symptom**: Brak `[InputArea] Using STREAMING mode` w console

**Fix**: 
```typescript
// InputArea.svelte
const USE_STREAMING = true; // Sprawdź czy to jest true!
```

### Przyczyna 2: WebSocket command nie jest zarejestrowany
**Symptom**: Error w console: `Unknown command: ai_agent_ha/chat/send_stream`

**Fix**: Restart Home Assistant

### Przyczyna 3: Gemini API nie wspiera streamingu
**Symptom**: 
- Logi: `Gemini streaming response status: 404` lub `405`
- Brak chunków

**Fix**: Użyj alternatywnego podejścia (chunking on server)

### Przyczyna 4: Agent fallback do non-streaming
**Symptom**: Logi: `Agent doesn't support streaming, using non-streaming`

**Fix**: Problem w `agent_compat.py` - sprawdź czy ma metodę `process_query_stream`

## Tymczasowe obejście

Jeśli Gemini Cloud Code API nie wspiera streamingu, możemy:

**Opcja A**: Server-side chunking
- Backend dzieli odpowiedź na kawałki
- Wysyła chunk co 50ms
- Symuluje streaming

**Opcja B**: Użyj standardowego Gemini API (nie OAuth)
- `generativelanguage.googleapis.com` WSPIERA streaming
- Ale wymaga API key zamiast OAuth

**Opcja C**: Wyłącz streaming
```typescript
const USE_STREAMING = false;
```

## Co zrobić dalej

1. **Wyślij wiadomość przez panel**
2. **Skopiuj logi** (backend + frontend console)
3. **Wklej tutaj** żebym mógł przeanalizować

Albo powiedz mi co widzisz w logach i pomogę zdiagnozować!
