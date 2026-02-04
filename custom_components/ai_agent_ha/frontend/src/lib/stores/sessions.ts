import type { SessionListItem } from '$lib/types';

/**
 * Session management state using Svelte 5 Runes
 */
export const sessionState = $state({
  sessions: [] as SessionListItem[],
  activeSessionId: null as string | null,
  sessionsLoading: true,
});

/**
 * Derived states
 */
export const hasSessions = $derived(sessionState.sessions.length > 0);
export const activeSession = $derived(
  sessionState.sessions.find((s) => s.session_id === sessionState.activeSessionId) || null
);
