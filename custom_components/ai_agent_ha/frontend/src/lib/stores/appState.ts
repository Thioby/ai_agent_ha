import type { HomeAssistant, Message } from '$lib/types';

/**
 * Global application state using Svelte 5 Runes
 * This is the main reactive state container
 */
export const appState = $state({
  // Home Assistant instance
  hass: null as HomeAssistant | null,

  // Chat state
  messages: [] as Message[],
  isLoading: false,
  error: null as string | null,

  // Debug info
  debugInfo: null as any,
  showThinking: false,
  thinkingExpanded: false,
});

/**
 * Derived states - automatically computed from appState
 */
export const hasMessages = $derived(appState.messages.length > 0);
export const hasError = $derived(appState.error !== null);
