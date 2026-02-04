import { get } from 'svelte/store';
import type { HomeAssistant, Provider } from '$lib/types';
import { providerState } from '$lib/stores/providers';
import { appState } from '$lib/stores/appState';
import { PROVIDERS } from '$lib/types';

/**
 * Provider and model management service
 */

/**
 * Load available providers from Home Assistant config
 */
export async function loadProviders(hass: HomeAssistant): Promise<void> {
  const state = get(providerState);
  if (state.providersLoaded) return;

  try {
    // Get all config entries
    const allEntries = await hass.callWS({ type: 'config_entries/get' });

    // Filter for AI Agent HA entries
    const aiAgentEntries = allEntries.filter((entry: any) => entry.domain === 'ai_agent_ha');

    if (aiAgentEntries.length > 0) {
      const providers = aiAgentEntries
        .map((entry: any) => {
          const provider = resolveProviderFromEntry(entry);
          if (!provider) return null;

          return {
            value: provider,
            label: PROVIDERS[provider] || provider,
          };
        })
        .filter(Boolean) as Provider[];

      providerState.update(s => ({ ...s, availableProviders: providers }));

      const currentState = get(providerState);
      
      // Auto-select first provider if none selected
      if (
        (!currentState.selectedProvider ||
          !providers.find((p) => p.value === currentState.selectedProvider)) &&
        providers.length > 0
      ) {
        providerState.update(s => ({ ...s, selectedProvider: providers[0].value }));
      }

      // Fetch models for selected provider
      const updatedState = get(providerState);
      if (updatedState.selectedProvider) {
        await fetchModels(hass, updatedState.selectedProvider);
      }

      providerState.update(s => ({ ...s, providersLoaded: true }));
    } else {
      providerState.update(s => ({ ...s, availableProviders: [] }));
    }
  } catch (error) {
    console.error('Error fetching config entries:', error);
    appState.update(s => ({ ...s, error: 'Failed to load AI provider configurations.' }));
    providerState.update(s => ({ ...s, availableProviders: [] }));
  }
}

/**
 * Fetch available models for a provider
 */
export async function fetchModels(hass: HomeAssistant, provider: string): Promise<void> {
  try {
    const result = await hass.callWS({
      type: 'ai_agent_ha/models/list',
      provider: provider,
    });

    const models = [...(result.models || [])];
    providerState.update(s => ({ ...s, availableModels: models }));

    // Select default model or first available
    const defaultModel = models.find((m: any) => m.default);
    const selectedModel = defaultModel ? defaultModel.id : models[0]?.id || null;
    
    providerState.update(s => ({ ...s, selectedModel }));
  } catch (e) {
    console.warn('Could not fetch available models:', e);
    providerState.update(s => ({ 
      ...s, 
      availableModels: [], 
      selectedModel: null 
    }));
  }
}

/**
 * Resolve provider from config entry
 * (Copied from original Lit code)
 */
function resolveProviderFromEntry(entry: any): string | null {
  if (!entry) return null;

  const providerFromData = entry.data?.ai_provider || entry.options?.ai_provider;
  if (providerFromData && PROVIDERS[providerFromData]) {
    return providerFromData;
  }

  const uniqueId = entry.unique_id || entry.uniqueId;
  if (uniqueId && uniqueId.startsWith('ai_agent_ha_')) {
    const fromUniqueId = uniqueId.replace('ai_agent_ha_', '');
    if (PROVIDERS[fromUniqueId]) {
      return fromUniqueId;
    }
  }

  const titleMap: Record<string, string> = {
    'ai agent ha (openrouter)': 'openrouter',
    'ai agent ha (google gemini)': 'gemini',
    'ai agent ha (openai)': 'openai',
    'ai agent ha (llama)': 'llama',
    'ai agent ha (anthropic (claude))': 'anthropic',
    'ai agent ha (alter)': 'alter',
    'ai agent ha (z.ai)': 'zai',
    'ai agent ha (local model)': 'local',
  };

  if (entry.title) {
    const lowerTitle = entry.title.toLowerCase();
    if (titleMap[lowerTitle]) {
      return titleMap[lowerTitle];
    }

    const match = entry.title.match(/\(([^)]+)\)/);
    if (match && match[1]) {
      const normalized = match[1].toLowerCase().replace(/[^a-z0-9]/g, '');
      const providerKey = Object.keys(PROVIDERS).find(
        (key) => key.replace(/[^a-z0-9]/g, '') === normalized
      );
      if (providerKey) {
        return providerKey;
      }
    }
  }

  return null;
}
