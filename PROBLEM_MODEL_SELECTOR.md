# Problem z selektorem modeli - diagnoza

## Symptomy
1. API zwraca 6 poprawnych modeli dla `gemini_oauth`
2. Frontend otrzymuje te 6 modeli (`_availableModels` ma length = 6)
3. **ALE** render pokazuje tylko 4 modele w dropdownie
4. Dropdown modeli czasami znika całkowicie po zmianie providera

## Analiza logów

```javascript
// Render sprawdza availableModels przed fetch
[AI Agent] Render check - availableModels: (4) [{…}, {…}, {…}, {…}]

// Następnie fetch zwraca 6 modeli
[AI Agent] Models response: {provider: 'gemini_oauth', models: Array(6), ...}
[AI Agent] Models list: (6) [{…}, {…}, {…}, {…}, {…}, {…}]

// _availableModels jest ustawiony na 6 elementów
[AI Agent] Available models: 6 Selected: gemini-3-pro-preview
[AI Agent] After fetch - availableModels count: 6
[AI Agent] After fetch - availableModels: (6) [{…}, {…}, {…}, {…}, {…}, {…}]
```

## Główny problem

**Race condition między renderowaniem a fetchowaniem:**

1. Użytkownik zmienia provider na `gemini_oauth`
2. `_selectProvider()` wywołuje `_fetchAvailableModels()` (async)
3. **Lit Element renderuje się NATYCHMIAST** ze starymi modelami (4 elementy)
4. Fetch kończy się i ustawia `_availableModels` na 6 elementów
5. `requestUpdate()` jest wywoływane
6. **ALE Lit nie wykrywa zmiany** - array reference została zmieniona przez spread operator `[...]`, ale Lit już wyrenderował komponent

## Dlaczego `requestUpdate()` nie działa

Lit Element używa **shallow comparison** dla reactive properties. Mimo że tworzymy nową referencję przez `[...result.models]`, Lit może:

1. Nie wykryć zmiany jeśli poprzedni render już się zakończył
2. Mieć cache poprzedniego stanu
3. Nie zauważyć zmiany w async flow

## Lokalizacja kodu

### Frontend: `custom_components/ai_agent_ha/frontend/ai_agent_ha-panel.js`

**Linia 1512-1522: `_selectProvider()`**
```javascript
async _selectProvider(provider) {
  console.log("[AI Agent] Provider selected:", provider);
  this._selectedProvider = provider;
  console.log("[AI Agent] Before fetch - availableModels count:", this._availableModels.length);
  await this._fetchAvailableModels(provider);  // ASYNC!
  console.log("[AI Agent] After fetch - availableModels count:", this._availableModels.length);
  console.log("[AI Agent] After fetch - availableModels:", this._availableModels);
}
```

**Linia 1524-1548: `_fetchAvailableModels()`**
```javascript
async _fetchAvailableModels(provider) {
  // ...
  const result = await this.hass.callWS({
    type: "ai_agent_ha/models/list",
    provider: provider
  });
  
  // Create new array reference to trigger reactivity
  this._availableModels = [...(result.models || [])];
  
  // Select default model
  const defaultModel = this._availableModels.find(m => m.default);
  this._selectedModel = defaultModel ? defaultModel.id : (this._availableModels[0]?.id || null);
  
  this.requestUpdate();  // To nie działa!
}
```

**Linia 1443-1461: Render warunku**
```javascript
${(() => {
  console.log("[AI Agent] Render check - availableModels.length:", this._availableModels.length);
  console.log("[AI Agent] Render check - availableModels:", this._availableModels);
  return this._availableModels.length > 0 ? html`
  <div class="provider-selector">
    <span class="provider-label">Model:</span>
    <select class="provider-button" ...>
      ${this._availableModels.map(model => html`...`)}
    </select>
  </div>
  ` : '';
})()}
```

## Możliwe rozwiązania

### Rozwiązanie 1: Force re-render przez zmianę klucza
Dodaj `key` attribute do dropdown żeby wymusić przebudowanie:

```javascript
<div class="provider-selector" key=${this._selectedProvider}>
```

### Rozwiązanie 2: Użyj setter z notyfikacją
Zamiast bezpośredniego przypisania, użyj setter który wymusza update:

```javascript
setAvailableModels(models) {
  this._availableModels = models;
  this.requestUpdate('_availableModels');
}
```

### Rozwiązanie 3: Wymuszony re-render przez toggle property
Dodaj helper property która zawsze się zmienia:

```javascript
async _fetchAvailableModels(provider) {
  // ...
  this._availableModels = [...(result.models || [])];
  this._modelsFetchCounter = (this._modelsFetchCounter || 0) + 1;
  this.requestUpdate();
}
```

### Rozwiązanie 4: Przebuduj całą sekcję input-footer (ZALECANE)
Dodaj `key` do całego kontenera aby wymusić przebudowanie:

```javascript
<div class="input-footer" key=${`${this._selectedProvider}-${this._availableModels.length}`}>
```

### Rozwiązanie 5: Użyj `until()` directive z Lit
```javascript
import { until } from 'https://unpkg.com/lit@3.1.0/directives/until.js';

// W render:
${until(
  this._fetchAvailableModelsPromise,
  html`<div>Loading models...</div>`
)}
```

## Zalecane natychmiastowe rozwiązanie

**Dodaj `key` attribute do dropdown modeli:**

```javascript
<div class="provider-selector" key=${`models-${this._selectedProvider}-${this._availableModels.length}`}>
  <span class="provider-label">Model:</span>
  <select
    class="provider-button"
    @change=${(e) => this._selectModel(e.target.value)}
    .value=${this._selectedModel || ''}
  >
    ${this._availableModels.map(model => html`
      <option
        value=${model.id}
        ?selected=${model.id === this._selectedModel}
      >
        ${model.name}
      </option>
    `)}
  </select>
</div>
```

`key` attribute zmusza Lit do przebudowania całego elementu gdy wartość się zmienia.

## Dodatkowe problemy

### Problem 1: Console.log w render()
Linię 1443-1445 (IIFE z console.log w render) należy usunąć - to spowalnia każde renderowanie.

### Problem 2: Spread operator może nie wystarczyć
Mimo `[...result.models]`, Lit może nie wykryć zmiany w async context.

### Problem 3: Brak loading state
Użytkownik nie widzi że modele są fetchowane - powinien być spinner lub "Loading models...".

## Pliki do modyfikacji

1. **`custom_components/ai_agent_ha/frontend/ai_agent_ha-panel.js`**
   - Linia 1443-1461: Dodaj `key` attribute
   - Linia 1524-1548: Rozważ dodanie loading state
   - Linia 1443-1445: Usuń console.log z render()

## Testy do wykonania po naprawie

1. Otwórz AI Agent HA w przeglądarce
2. Wybierz provider "Gemini" (API key) - sprawdź czy widać 4 modele
3. Zmień provider na "Gemini (OAuth)" - sprawdź czy widać 6 modeli
4. Zmień z powrotem na "Gemini" - sprawdź czy widać 4 modele
5. Zmień szybko między providerami kilka razy - sprawdź czy dropdown zawsze aktualizuje się poprawnie

## Status

- ✅ Backend działa poprawnie (WebSocket API zwraca prawidłowe modele)
- ✅ Frontend otrzymuje prawidłowe modele
- ❌ Frontend nie renderuje prawidłowo otrzymanych modeli
- ❌ Lit Element reactivity nie działa w async flow

## Priorytet: KRYTYCZNY

Użytkownik nie może wybrać właściwego modelu dla Gemini OAuth.
