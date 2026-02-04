# 📋 Szczegółowy Plan Migracji: Lit → Svelte + TypeScript

## 🎯 Cel końcowy
Przepisać `ai_agent_ha-panel.js` (1990 linii) na modularny, wydajny kod w **Svelte 5 + TypeScript**, zachowując 100% funkcjonalności.

> **📌 Uwaga:** Ten plan został zaktualizowany na podstawie szczegółowej analizy Gemini AI.  
> Zobacz: `GEMINI_ANALYSIS_MIGRATION_PLAN.md` dla pełnych rekomendacji.

---

## 📊 Analiza obecnego kodu

### Obecna struktura (monolityczna):
```
ai_agent_ha-panel.js (1990 linii)
├── Imports (11 linii) - CDN dependencies
├── CSS Styles (898 linii) - inline w JS
├── Properties (16 properties) - stan komponentu
├── Constructor - inicjalizacja
├── Metody biznesowe (~40 metod):
│   ├── Session management (load, create, delete, select)
│   ├── Message handling (send, receive, parse)
│   ├── WebSocket communication
│   ├── Provider/Model management
│   ├── Automation/Dashboard approval
│   ├── Markdown rendering + cache
│   └── UI helpers (scroll, resize, etc.)
└── Render methods - Lit templates
```

### Funkcjonalności do zachowania:
1. ✅ Sidebar z listą sesji (create, select, delete)
2. ✅ Chat area z wiadomościami (user + assistant)
3. ✅ Markdown rendering z syntax highlighting
4. ✅ Provider/Model selection dropdown
5. ✅ WebSocket komunikacja (send message, receive response)
6. ✅ Session persistence (load/save)
7. ✅ Automation suggestions (approve/reject)
8. ✅ Dashboard suggestions (approve/reject)
9. ✅ Debug mode (thinking panel)
10. ✅ Loading states, error handling
11. ✅ Mobile responsive (sidebar overlay)
12. ✅ Markdown cache dla wydajności
13. ✅ Auto-resize textarea
14. ✅ Empty states (no sessions, no messages)
15. ✅ Keyboard shortcuts (Enter = send)

---

## 🏗️ Nowa architektura (modularna)

```
custom_components/ai_agent_ha/
├── frontend/
│   ├── src/                          # Kod źródłowy Svelte
│   │   ├── lib/
│   │   │   ├── components/           # Komponenty UI
│   │   │   │   ├── AiAgentPanel.svelte       # Root component (kontener)
│   │   │   │   ├── Header.svelte             # Nagłówek z tytułem
│   │   │   │   ├── Sidebar/
│   │   │   │   │   ├── Sidebar.svelte        # Sidebar kontener
│   │   │   │   │   ├── SessionList.svelte    # Lista sesji
│   │   │   │   │   ├── SessionItem.svelte    # Pojedyncza sesja
│   │   │   │   │   └── NewChatButton.svelte  # Przycisk "New Chat"
│   │   │   │   ├── Chat/
│   │   │   │   │   ├── ChatArea.svelte       # Obszar wiadomości
│   │   │   │   │   ├── MessageBubble.svelte  # Pojedyncza wiadomość
│   │   │   │   │   ├── LoadingIndicator.svelte
│   │   │   │   │   ├── EmptyState.svelte     # Pusty stan
│   │   │   │   │   └── ErrorMessage.svelte   # Komunikat błędu
│   │   │   │   ├── Input/
│   │   │   │   │   ├── InputArea.svelte      # Kontener inputu
│   │   │   │   │   ├── MessageInput.svelte   # Textarea
│   │   │   │   │   ├── ProviderSelector.svelte
│   │   │   │   │   ├── ModelSelector.svelte
│   │   │   │   │   ├── SendButton.svelte
│   │   │   │   │   └── ThinkingToggle.svelte
│   │   │   │   ├── Suggestions/
│   │   │   │   │   ├── AutomationSuggestion.svelte
│   │   │   │   │   └── DashboardSuggestion.svelte
│   │   │   │   └── Debug/
│   │   │   │       └── ThinkingPanel.svelte   # Debug panel
│   │   │   ├── stores/               # Svelte 5 Runes (state management)
│   │   │   │   ├── appState.ts       # Global app state ($state)
│   │   │   │   ├── chat.ts           # Stan wiadomości, loading
│   │   │   │   ├── sessions.ts       # Stan sesji
│   │   │   │   ├── providers.ts      # Providers & models
│   │   │   │   └── ui.ts             # UI state (sidebar, errors)
│   │   │   ├── services/             # Logika biznesowa
│   │   │   │   ├── websocket.service.ts   # WebSocket API
│   │   │   │   ├── markdown.service.ts    # Markdown + cache
│   │   │   │   ├── session.service.ts     # Session CRUD
│   │   │   │   └── provider.service.ts    # Provider/model API
│   │   │   ├── types/                # TypeScript types
│   │   │   │   ├── index.ts          # Re-exports
│   │   │   │   ├── message.ts        # Message, Assistant, User
│   │   │   │   ├── session.ts        # Session
│   │   │   │   ├── provider.ts       # Provider, Model
│   │   │   │   └── hass.ts           # Home Assistant types
│   │   │   └── utils/                # Utilities
│   │   │       ├── time.ts           # Formatowanie czasu
│   │   │       ├── json.ts           # JSON parsing
│   │   │       └── dom.ts            # DOM helpers
│   │   ├── main.ts                   # Entry point (Web Component wrapper)
│   │   ├── app.css                   # Global styles
│   │   └── vite-env.d.ts             # Vite types
│   ├── dist/                         # Build output (git ignored)
│   │   └── ai_agent_ha-panel.js      # Bundled file
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   ├── svelte.config.js
│   └── .gitignore
│   
└── ai_agent_ha-panel.js             # STARY PLIK (backup, potem usunąć)
```

---

## 📝 Plan krok po kroku

### **FAZA 0: Przygotowanie (1h)** ⚠️ Zaktualizowana estymacja

#### **Krok 0.1: Backup obecnego pliku**
- [ ] Skopiuj `ai_agent_ha-panel.js` → `ai_agent_ha-panel.js.backup`
- [ ] Git commit: "Backup original Lit panel before Svelte migration"

#### **Krok 0.2: Setup projektu**
```bash
cd custom_components/ai_agent_ha/frontend/
npm init -y

# 🔴 KRYTYCZNE: Użyj Svelte 5 (nie 4!)
npm install -D svelte@latest vite @sveltejs/vite-plugin-svelte
npm install -D typescript @tsconfig/svelte
npm install -D @types/node

# Dependencies
npm install marked marked-highlight @highlightjs/cdn-assets dompurify
npm install -D @types/marked @types/dompurify

# 🆕 DODANE: Home Assistant types
npm install home-assistant-js-websocket
npm install -D @types/home-assistant-js-websocket

# 🆕 DODANE: Linter & Formatter
npm install -D eslint prettier eslint-plugin-svelte
npm install -D @typescript-eslint/eslint-plugin @typescript-eslint/parser
```

#### **Krok 0.3: Pliki konfiguracyjne**
- [ ] `vite.config.ts` - produkcja (build do single file)
- [ ] `vite.config.dev.ts` - 🆕 development z proxy do HA
- [ ] `tsconfig.json` - TypeScript config dla Svelte
- [ ] `svelte.config.js` - Svelte 5 config z `css: 'injected'` dla Shadow DOM
- [ ] `.eslintrc.json` - 🆕 ESLint config
- [ ] `.prettierrc` - 🆕 Prettier config
- [ ] `package.json` - dodać scripts (dev, build, preview, lint, format)
- [ ] `.gitignore` - node_modules, dist, .svelte-kit

---

### **FAZA 1: Fundament (3-4h)** ⚠️ Zaktualizowana estymacja

#### **Krok 1.1: TypeScript types**
🆕 **Wykorzystaj istniejące typy z `home-assistant-js-websocket`:**
- [ ] `types/hass.ts` - **import z `home-assistant-js-websocket`**, rozszerz tylko potrzebne
- [ ] `types/message.ts` - `Message`, `MessageType`, `MessageMetadata`
- [ ] `types/session.ts` - `Session`, `SessionListItem`
- [ ] `types/provider.ts` - `Provider`, `Model`, `ProviderInfo`
- [ ] `types/debug.ts` - `DebugInfo`, `ConversationEntry`
- [ ] `types/index.ts` - re-export wszystkiego

**Przykład `types/hass.ts`:**
```typescript
import type { HassEntity, HassEntities, Connection } from 'home-assistant-js-websocket';

export interface HomeAssistant {
  entities: HassEntities;
  connection: Connection;
  callService: (domain: string, service: string, data?: any) => Promise<any>;
  callWS: (message: any) => Promise<any>;
  // ... tylko to co faktycznie używasz
}
```

**Estymacja:** 30 min

#### **Krok 1.2: Svelte 5 Runes (state management)**
🔴 **KRYTYCZNE: Użyj Svelte 5 Runes ($state), NIE Stores (writable)!**

Przenieść state z Lit properties do Svelte 5 Runes:
- [ ] `stores/appState.ts` - **globalny $state** z `hass`, `messages`, `isLoading`
- [ ] `stores/sessions.ts` - `sessions`, `activeSessionId` ($state)
- [ ] `stores/providers.ts` - `selectedProvider`, `availableProviders` ($state)
- [ ] `stores/ui.ts` - `sidebarOpen`, `showThinking` ($state)

**Przykład `stores/appState.ts`:**
```typescript
export const appState = $state({
  hass: null as HomeAssistant | null,
  messages: [] as Message[],
  isLoading: false,
  error: null as string | null
});

// Derived state
export const hasMessages = $derived(appState.messages.length > 0);
```

**Dlaczego Svelte 5 Runes:**
- ✅ Lepsza wydajność z obiektem `hass` (zmienia się dziesiątki razy/s)
- ✅ Mniej boilerplate'u niż Stores
- ✅ Automatyczna reaktywność
- ✅ Intuicyjne dla programistów React/Lit

**Estymacja:** 1.5h (więcej czasu na poprawną implementację reaktywności)

#### **Krok 1.3: Services (logika biznesowa)**
Wydzielić logikę z metod Lit do services:
- [ ] `services/markdown.service.ts` - `renderMarkdown()`, cache Map
- [ ] `services/websocket.service.ts` - `sendMessage()`, `subscribeToEvents()`
- [ ] `services/session.service.ts` - `loadSessions()`, `createSession()`, `deleteSession()`, `selectSession()`
- [ ] `services/provider.service.ts` - `loadProviders()`, `fetchModels()`

**Estymacja:** 1h

#### **Krok 1.4: Utils**
- [ ] `utils/time.ts` - `formatSessionTime()`, `getCurrentTime()`
- [ ] `utils/json.ts` - `parseJSONResponse()`
- [ ] `utils/dom.ts` - `scrollToBottom()`, `autoResize()`

**Estymacja:** 15 min

---

### **FAZA 2: Komponenty UI - Core (3-4h)**

#### **Krok 2.1: MessageBubble.svelte**
Przepisać rendering pojedyncze wiadomości:
- [ ] Props: `message: Message`
- [ ] Markdown rendering dla assistant messages
- [ ] Style dla `.user-message` i `.assistant-message`
- [ ] Automation/Dashboard suggestions jako sloty

**Estymacja:** 30 min

#### **Krok 2.2: ChatArea.svelte**
Obszar wiadomości:
- [ ] Lista `MessageBubble` components
- [ ] `LoadingIndicator` component
- [ ] `EmptyState` component
- [ ] `ErrorMessage` component
- [ ] Auto-scroll do dołu przy nowych wiadomościach
- [ ] Style scrollbara

**Estymacja:** 45 min

#### **Krok 2.3: MessageInput.svelte**
Textarea z auto-resize:
- [ ] Textarea element
- [ ] Auto-resize on input
- [ ] Enter = send (Shift+Enter = newline)
- [ ] Disabled state gdy loading

**Estymacja:** 20 min

#### **Krok 2.4: InputArea.svelte**
Kontener dla całego input area:
- [ ] `MessageInput` component
- [ ] `ProviderSelector` component
- [ ] `ModelSelector` component (conditional)
- [ ] `ThinkingToggle` component
- [ ] `SendButton` component
- [ ] Footer layout

**Estymacja:** 45 min

#### **Krok 2.5: Header.svelte**
Nagłówek panelu:
- [ ] Tytuł "AI Agent HA"
- [ ] Menu toggle button (mobile)
- [ ] Clear Chat button
- [ ] Ikony

**Estymacja:** 20 min

---

### **FAZA 3: Session Management (2-3h)**

#### **Krok 3.1: SessionItem.svelte**
Pojedyncza sesja w sidebar:
- [ ] Props: `session: SessionListItem`, `isActive: boolean`
- [ ] Click handler → select session
- [ ] Delete button (hover to show)
- [ ] Title, preview, timestamp
- [ ] Active state styling

**Estymacja:** 30 min

#### **Krok 3.2: SessionList.svelte**
Lista sesji:
- [ ] Loop przez `$sessions` store
- [ ] Empty state gdy brak sesji
- [ ] Loading skeleton
- [ ] Virtual scrolling (opcjonalnie dla wydajności)

**Estymacja:** 30 min

#### **Krok 3.3: Sidebar.svelte**
Sidebar kontener:
- [ ] Header z "Conversations"
- [ ] `NewChatButton` component
- [ ] `SessionList` component
- [ ] Open/close animation
- [ ] Overlay dla mobile

**Estymacja:** 45 min

#### **Krok 3.4: Session Service Integration**
Podłączyć service do komponentów:
- [ ] Load sessions on mount
- [ ] Create new session
- [ ] Select session → load messages
- [ ] Delete session
- [ ] Update session title/preview

**Estymacja:** 45 min

---

### **FAZA 4: Advanced Features (2-3h)**

#### **Krok 4.1: Provider & Model Selection**
- [ ] `ProviderSelector.svelte` - dropdown z providerami
- [ ] `ModelSelector.svelte` - dropdown z modelami (conditional render)
- [ ] Load providers on mount
- [ ] Fetch models when provider changes
- [ ] Store selection w stores

**Estymacja:** 45 min

#### **Krok 4.2: Automation/Dashboard Suggestions**
- [ ] `AutomationSuggestion.svelte` - karta z automation YAML
- [ ] `DashboardSuggestion.svelte` - karta z dashboard JSON
- [ ] Approve/Reject buttons
- [ ] Call HA services

**Estymacja:** 45 min

#### **Krok 4.3: ThinkingPanel.svelte**
Debug panel:
- [ ] Toggle expand/collapse
- [ ] Show conversation trace
- [ ] Provider/Model/Endpoint info
- [ ] Conditional render gdy `$showThinking === true`

**Estymacja:** 30 min

#### **Krok 4.4: Error Handling**
- [ ] `ErrorMessage.svelte` - dismissible error banner
- [ ] Error store z timeout auto-clear
- [ ] Try-catch w services
- [ ] Fallback messages

**Estymacja:** 30 min

---

### **FAZA 5: Integration & Root Component (1-2h)**

#### **Krok 5.1: AiAgentPanel.svelte**
Root component łączący wszystko:
- [ ] Layout: Header + Main Container (Sidebar + Chat Area)
- [ ] Podłączenie wszystkich stores
- [ ] Lifecycle hooks (onMount → init)
- [ ] Window resize listener
- [ ] Document click listener (close dropdowns)

**Estymacja:** 1h

#### **Krok 5.2: Web Component Wrapper**
🔴 **KRYTYCZNE: Shadow DOM dla izolacji stylów!**

`main.ts` - wrapper dla Home Assistant:
```typescript
import { mount } from 'svelte';
import AiAgentPanel from './lib/components/AiAgentPanel.svelte';

class AiAgentHaPanel extends HTMLElement {
  private component?: ReturnType<typeof mount>;
  private shadow: ShadowRoot;
  private _lastHass: any;
  
  connectedCallback() {
    // 🔴 KRYTYCZNE: Utwórz Shadow Root dla izolacji stylów
    this.shadow = this.attachShadow({ mode: 'open' });
    
    // Montuj w Shadow DOM (NIE w this!)
    this.component = mount(AiAgentPanel, {
      target: this.shadow, // ✅ Shadow DOM
      props: {
        hass: (this as any).hass,
        narrow: (this as any).narrow,
        panel: (this as any).panel
      }
    });
    
    this._lastHass = (this as any).hass;
  }
  
  disconnectedCallback() {
    if (this.component) {
      // Svelte 5 unmount
      this.component.$destroy?.();
    }
  }
  
  // Wydajny setter - unikaj niepotrzebnych update'ów
  set hass(value: any) {
    if (this.component && value !== this._lastHass) {
      this._lastHass = value;
      this.component.$set?.({ hass: value });
    }
  }
}

customElements.define('ai-agent-ha-panel', AiAgentHaPanel);
```

**Dlaczego Shadow DOM:**
- ✅ Izolacja stylów - globalne style HA nie wpłyną na panel
- ✅ Twoje style nie wyciekną na zewnątrz
- ✅ Kompatybilność z obecnym Lit Element

**Estymacja:** 1h (włącznie z konfiguracją Svelte dla Shadow DOM)

---

### **FAZA 6: Styling & Responsive (4-5h)** ⚠️ Zaktualizowana estymacja

#### **Krok 6.1: Global Styles**
`app.css` - tylko podstawowe style globalne:
- [ ] CSS Variables (theme) - zachować kompatybilność z HA
- [ ] Reset/normalize (minimal)
- [ ] Scrollbar styling
- [ ] Animations (fadeIn, slideIn, bounce)
- [ ] Font imports (jeśli potrzebne w Shadow DOM)

**Estymacja:** 30 min

#### **Krok 6.2: Component Styles (Per-Component)**
🔴 **ZMIANA PODEJŚCIA: NIE rób monolitycznego CSS!**

**Zamiast jednego dużego `app.css` z 900 liniami:**
Każdy komponent ma własną sekcję `<style>` z TYLKO tym co potrzebuje.

**Przykład:**
```svelte
<!-- MessageBubble.svelte -->
<script lang="ts">
  export let message: Message;
</script>

<div class="message" class:user={message.type === 'user'}>
  {@html message.content}
</div>

<style>
  /* Tylko style dla tego komponentu */
  .message {
    padding: 12px 16px;
    border-radius: 12px;
    max-width: 80%;
    line-height: 1.5;
  }
  
  .user {
    background: var(--primary-color);
    color: white;
    margin-left: auto;
  }
</style>
```

**Zalety:**
- ✅ Kolokacja - style przy komponencie
- ✅ Automatyczny scoping
- ✅ Łatwiejsze utrzymanie
- ✅ Tree shaking - niewykorzystane style nie trafią do bundle

**Zadania:**
- [ ] Rozdziel 900 linii CSS na komponenty
- [ ] Użyj CSS variables dla theme compatibility
- [ ] Responsive breakpoints (@media queries) w każdym komponencie
- [ ] Konfiguracja Svelte dla Shadow DOM (`css: 'injected'`)

**Estymacja:** 3h (więcej czasu na rozbicie i testowanie)

#### **Krok 6.3: Mobile Optimization**
- [ ] Sidebar overlay na mobile
- [ ] Compact input footer
- [ ] Hide labels/icons gdzie potrzebne
- [ ] Touch-friendly hit areas (min 44x44px)

**Estymacja:** 45 min

---

### **FAZA 7: Build & Integration (1-2h)**

#### **Krok 7.1: Vite Build Config**
🆕 **Dwa pliki config: produkcja i development**

**`vite.config.ts` (produkcja):**
```typescript
import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';

export default defineConfig({
  plugins: [svelte()],
  build: {
    lib: {
      entry: './src/main.ts',
      name: 'AiAgentHaPanel',
      fileName: () => 'ai_agent_ha-panel.js',
      formats: ['iife']
    },
    outDir: '../',
    emptyOutDir: false, // nie usuwaj innych plików w custom_components
    minify: 'terser',
    sourcemap: false,
    rollupOptions: {
      output: {
        inlineDynamicImports: true,
        manualChunks: undefined // zapobiega splitowaniu
      }
    },
    terserOptions: {
      compress: {
        drop_console: true,
        drop_debugger: true
      }
    }
  }
});
```

**`vite.config.dev.ts` (🆕 development):**
```typescript
import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';

export default defineConfig({
  plugins: [svelte()],
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://homeassistant.local:8123',
      '/auth': 'http://homeassistant.local:8123'
    }
  },
  build: {
    sourcemap: true
  }
});
```

**`svelte.config.js` (dla Shadow DOM):**
```javascript
export default {
  compilerOptions: {
    css: 'injected' // Wstrzykuje style do JS dla Shadow DOM
  }
};
```

**Estymacja:** 45 min

#### **Krok 7.2: Build & Test**
- [ ] `npm run build` → generuje `ai_agent_ha-panel.js`
- [ ] Sprawdzić rozmiar bundle (powinien być ~30-50KB gzipped)
- [ ] Test w Home Assistant:
  - [ ] Panel się ładuje
  - [ ] Providers się ładują
  - [ ] Można wysłać wiadomość
  - [ ] Sesje działają
  - [ ] Mobile responsive

**Estymacja:** 45 min

#### **Krok 7.3: Dev Mode Setup**
🆕 **Rozszerzone scripts w `package.json`:**
```json
{
  "scripts": {
    "dev": "vite --config vite.config.dev.ts",
    "build": "vite build --config vite.config.ts",
    "preview": "vite preview",
    "lint": "eslint src --ext .ts,.svelte",
    "format": "prettier --write src/**/*.{ts,svelte,css}",
    "check": "svelte-check --tsconfig tsconfig.json",
    "bundle-size": "ls -lh ../ai_agent_ha-panel.js && gzip -c ../ai_agent_ha-panel.js | wc -c"
  }
}
```
- [ ] Setup dev proxy do HA w `vite.config.dev.ts`
- [ ] Hot reload podczas developmentu
- [ ] 🆕 Bundle size check script

**Estymacja:** 20 min

---

### **FAZA 8: Testing & Bug Fixes (2-3h)**

#### **Krok 8.1: Functional Testing**
Przetestować wszystkie funkcjonalności:
- [ ] Send message (WebSocket)
- [ ] Create new session
- [ ] Switch between sessions
- [ ] Delete session
- [ ] Change provider
- [ ] Change model
- [ ] Approve automation
- [ ] Approve dashboard
- [ ] Clear chat
- [ ] Toggle thinking panel
- [ ] Mobile sidebar toggle
- [ ] Markdown rendering
- [ ] Code syntax highlighting
- [ ] Error handling
- [ ] Loading states

**Estymacja:** 1h

#### **Krok 8.2: Bug Fixes**
Naprawić wszystkie znalezione błędy.

**Estymacja:** 3-4h ⚠️ (realistyczna estymacja przy przepisywaniu 2000 linii kodu)

#### **Krok 8.3: Performance Check**
- [ ] Sprawdzić szybkość ładowania
- [ ] Profiling (Chrome DevTools)
- [ ] Memory leaks check
- [ ] Markdown cache działa poprawnie

**Estymacja:** 30 min

---

### **FAZA 9: Documentation & Cleanup (1h)**

#### **Krok 9.1: Dokumentacja**
- [ ] `README.md` w `frontend/` - jak buildować, jak developerować
- [ ] Komentarze w kodzie (JSDoc)
- [ ] Type annotations dla wszystkich funkcji

**Estymacja:** 30 min

#### **Krok 9.2: Cleanup**
- [ ] Usunąć `ai_agent_ha-panel.js.backup` (albo przenieść do `/archive`)
- [ ] Sprawdzić `.gitignore`
- [ ] Usunąć unused imports
- [ ] Formatowanie kodu (Prettier)

**Estymacja:** 15 min

#### **Krok 9.3: Git Commit**
```bash
git add .
git commit -m "Refactor: Migrate UI from Lit to Svelte + TypeScript

- Modular architecture with 20+ components
- TypeScript for type safety
- Svelte stores for state management
- Services layer for business logic
- ~60% smaller bundle size
- Improved performance and maintainability
"
```

**Estymacja:** 15 min

---

## ⏱️ Estymacja czasowa

### Oryginalna vs Zaktualizowana estymacja:

| Faza | Czas oryginalny | ⚠️ Czas realistyczny | Uwagi |
|------|----------------|---------------------|-------|
| 0. Przygotowanie | 30 min | **1h** | +30 min: linter, prettier, HA types |
| 1. Fundament | 2-3h | **3-4h** | +1h: Svelte 5 Runes, HA types import |
| 2. Komponenty UI - Core | 3-4h | **3-4h** | = |
| 3. Session Management | 2-3h | **2-3h** | = |
| 4. Advanced Features | 2-3h | **2-3h** | = |
| 5. Integration & Root | 1-2h | **2h** | +30 min: Shadow DOM setup |
| 6. Styling & Responsive | 2-3h | **4-5h** | +2h: per-component styles |
| 7. Build & Integration | 1-2h | **2h** | +30 min: dev/prod configs |
| 8. Testing & Bug Fixes | 2-3h | **4-5h** | +2h: realistyczne debugowanie |
| 9. Documentation & Cleanup | 1h | **1h** | = |
| **TOTAL** | **17-24h** | **24-32h (3-4 dni)** | **+7-8h** |

### 📊 Różnice wynikają z:
1. 🔴 **Shadow DOM** - wymaga dodatkowej konfiguracji
2. 🔴 **Svelte 5 Runes** - nowa API, więcej czasu na naukę
3. 🔴 **Per-component CSS** - więcej czasu na rozbicie 900 linii
4. 🟡 **Realistyczne debugowanie** - przy 2000 linii kodu zawsze jest więcej bugów

---

## 🎯 Checklisty weryfikacyjne

### ✅ Definition of Done (po każdej fazie):
- [ ] Kod się kompiluje bez błędów TypeScript
- [ ] Kod się builduje bez warningów Vite
- [ ] Funkcjonalność działa w przeglądarce
- [ ] Styles są poprawne (nie zepsuty layout)
- [ ] Git commit z opisowym message'm

### ✅ Final Checklist (przed uznaniem za gotowe):
- [ ] Wszystkie funkcjonalności z obecnego UI działają
- [ ] Bundle size < 60KB gzipped
- [ ] Brak błędów w console
- [ ] Brak memory leaks
- [ ] Mobile responsive działa
- [ ] Dark theme z Home Assistant jest zachowany
- [ ] Dokumentacja jest aktualna
- [ ] Kod jest sformatowany (Prettier)
- [ ] Git history jest czyste (sensowne commity)

---

## 🚨 Ryzyka i mitygacja

| Ryzyko | Prawdopodobieństwo | Impact | Mitygacja |
|--------|-------------------|--------|-----------|
| 🔴 Shadow DOM nie izoluje stylów poprawnie | Średnie | Wysokie | Konfiguracja `css: 'injected'` w svelte.config.js + testy |
| 🔴 Obiekt `hass` powoduje zbyt częste re-rendery | Średnie | Wysokie | Svelte 5 Runes + ostrożne $derived/$effect |
| 🟡 Web Component wrapper nie działa z HA | Niskie | Wysokie | Test już w fazie 5.2, fallback do Lit wrapper |
| 🟡 Bundle zbyt duży | Średnie | Średnie | Terser optimization, inline imports, bundle size monitoring |
| 🟡 Markdown rendering wolniejszy | Niskie | Średnie | Cache (już mamy) + virtual scrolling dla długich konwersacji |
| 🟢 TypeScript types dla `hass` | Niskie | Niskie | Import z `home-assistant-js-websocket` |
| 🟢 Breaking changes w HA API | Niskie | Wysokie | Zachować kompatybilność z obecnymi callWS/callService |
| 🟢 Cache busting w HA | Średnie | Średnie | Wersja w nazwie pliku lub query param |

---

## 📦 Deliverables

Po zakończeniu będziesz mieć:

1. **Nowy frontend w Svelte**:
   - `custom_components/ai_agent_ha/frontend/src/` - kod źródłowy
   - `custom_components/ai_agent_ha/ai_agent_ha-panel.js` - zbudowany bundle

2. **Dokumentacja**:
   - `frontend/README.md` - jak buildować i rozwijać
   - Komentarze w kodzie

3. **Narzędzia deweloperskie**:
   - Hot reload podczas developmentu
   - TypeScript type checking
   - Vite dev server

4. **Lepsza wydajność**:
   - ~50% mniejszy bundle
   - Szybsze renderowanie
   - Lepsze cache'owanie

---

## ❓ Pytania do weryfikacji

1. **Czy zgadzasz się z tą strukturą folderów?** Czy coś chciałbyś zmienić?

2. **Czy estymacja 2-3 dni pracy jest OK?** Czy masz tyle czasu?

3. **Czy są jakieś funkcjonalności, które możemy pominąć/uprościć** żeby przyspieszyć migrację?

4. **Czy chcesz zachować plik `.backup`** czy wolisz po prostu git branch?

5. **Czy mam zacząć od razu od Fazy 0**, czy są jakieś pytania/wątpliwości?

---

## 📝 Status realizacji

- [ ] Faza 0: Przygotowanie
- [ ] Faza 1: Fundament
- [ ] Faza 2: Komponenty UI - Core
- [ ] Faza 3: Session Management
- [ ] Faza 4: Advanced Features
- [ ] Faza 5: Integration & Root Component
- [ ] Faza 6: Styling & Responsive
- [ ] Faza 7: Build & Integration
- [ ] Faza 8: Testing & Bug Fixes
- [ ] Faza 9: Documentation & Cleanup

---

---

## 📌 Kluczowe zmiany po analizie Gemini AI

### 🔴 MUST HAVE (krytyczne):
1. ✅ **Svelte 5 (Runes)** zamiast Svelte 4 (Stores) - lepsza wydajność z `hass`
2. ✅ **Shadow DOM** w Web Component wrapper - izolacja stylów
3. ✅ **Per-component CSS** zamiast monolitu - lepsze utrzymanie
4. ✅ **Import typów z `home-assistant-js-websocket`** - oszczędność czasu

### 🟡 SHOULD HAVE (silnie rekomendowane):
5. ✅ **ESLint + Prettier** - spójność kodu
6. ✅ **Vite dev/prod configs** - oddzielne dla developmentu i produkcji
7. ✅ **Realistyczne estymacje** - 24-32h zamiast 17-24h
8. ✅ **Bundle size monitoring** - cel < 60KB gzipped

### 🟢 NICE TO HAVE (opcjonalne):
9. Dev proxy do HA - łatwiejszy development
10. Sourcemaps - lepsze debugowanie
11. Cache busting - aktualizacje w HA

---

**Data utworzenia:** 2026-02-04  
**Wersja:** 2.0 (Zaktualizowana po analizie Gemini AI)  
**Autor:** AI Assistant + anowak  
**Analiza:** Zobacz `GEMINI_ANALYSIS_MIGRATION_PLAN.md`
