import { writable, derived } from 'svelte/store';
import type { Provider, Model } from '$lib/types';

/**
 * Provider and model selection state
 */
export interface ProviderStateType {
  availableProviders: Provider[];
  selectedProvider: string | null;
  availableModels: Model[];
  selectedModel: string | null;
  providersLoaded: boolean;
}

const initialState: ProviderStateType = {
  availableProviders: [],
  selectedProvider: null,
  availableModels: [],
  selectedModel: null,
  providersLoaded: false,
};

export const providerState = writable<ProviderStateType>(initialState);

// Derived stores
export const hasProviders = derived(providerState, $state => $state.availableProviders.length > 0);
export const hasModels = derived(providerState, $state => $state.availableModels.length > 0);
export const selectedProviderInfo = derived(providerState, $state =>
  $state.availableProviders.find(p => p.value === $state.selectedProvider) || null
);
