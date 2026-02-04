import type { Provider, Model } from '$lib/types';

/**
 * Provider and model selection state using Svelte 5 Runes
 */
export const providerState = $state({
  availableProviders: [] as Provider[],
  selectedProvider: null as string | null,

  availableModels: [] as Model[],
  selectedModel: null as string | null,

  providersLoaded: false,
});

/**
 * Derived states
 */
export const hasProviders = $derived(providerState.availableProviders.length > 0);
export const hasModels = $derived(providerState.availableModels.length > 0);
export const selectedProviderInfo = $derived(
  providerState.availableProviders.find((p) => p.value === providerState.selectedProvider) || null
);
