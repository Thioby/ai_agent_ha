import type { HomeAssistant } from '$lib/types';
import { sessionState } from '$lib/stores/sessions';
import { appState } from '$lib/stores/appState';
import { clearSessionCache } from './markdown.service';

/**
 * Session management service
 */

/**
 * Load all sessions from Home Assistant
 */
export async function loadSessions(hass: HomeAssistant): Promise<void> {
  sessionState.sessionsLoading = true;
  try {
    const result = await hass.callWS({
      type: 'ai_agent_ha/sessions/list',
    });

    sessionState.sessions = result.sessions || [];

    // Auto-select first session if none selected
    if (sessionState.sessions.length > 0 && !sessionState.activeSessionId) {
      await selectSession(hass, sessionState.sessions[0].session_id);
    }
  } catch (error) {
    console.error('Failed to load sessions:', error);
    appState.error = 'Could not load conversations';
  } finally {
    sessionState.sessionsLoading = false;
  }
}

/**
 * Select a session and load its messages
 */
export async function selectSession(hass: HomeAssistant, sessionId: string): Promise<void> {
  sessionState.activeSessionId = sessionId;
  appState.isLoading = true;
  appState.error = null;

  try {
    const result = await hass.callWS({
      type: 'ai_agent_ha/sessions/get',
      session_id: sessionId,
    });

    // Map messages from HA format to our format
    appState.messages = (result.messages || []).map((m: any) => ({
      type: m.role === 'user' ? 'user' : 'assistant',
      text: m.content,
      automation: m.metadata?.automation,
      dashboard: m.metadata?.dashboard,
      timestamp: m.timestamp,
      status: m.status,
      error_message: m.error_message,
    }));

    // Close sidebar on mobile
    if (typeof window !== 'undefined' && window.innerWidth <= 768) {
      const { closeSidebar } = await import('$lib/stores/ui');
      closeSidebar();
    }
  } catch (error) {
    console.error('Failed to load session:', error);
    appState.error = 'Could not load conversation';
  } finally {
    appState.isLoading = false;
  }
}

/**
 * Create a new session
 */
export async function createSession(hass: HomeAssistant, provider: string): Promise<void> {
  try {
    const result = await hass.callWS({
      type: 'ai_agent_ha/sessions/create',
      provider: provider,
    });

    sessionState.sessions = [result, ...sessionState.sessions];
    sessionState.activeSessionId = result.session_id;
    appState.messages = [];

    // Close sidebar on mobile
    if (typeof window !== 'undefined' && window.innerWidth <= 768) {
      const { closeSidebar } = await import('$lib/stores/ui');
      closeSidebar();
    }
  } catch (error) {
    console.error('Failed to create session:', error);
    appState.error = 'Could not create new conversation';
  }
}

/**
 * Delete a session
 */
export async function deleteSession(hass: HomeAssistant, sessionId: string): Promise<void> {
  try {
    await hass.callWS({
      type: 'ai_agent_ha/sessions/delete',
      session_id: sessionId,
    });

    // Remove from list
    sessionState.sessions = sessionState.sessions.filter((s: any) => s.session_id !== sessionId);

    // Clear markdown cache for this session
    clearSessionCache(sessionId);

    // If deleted active session, select first available
    if (sessionState.activeSessionId === sessionId) {
      if (sessionState.sessions.length > 0) {
        await selectSession(hass, sessionState.sessions[0].session_id);
      } else {
        sessionState.activeSessionId = null;
        appState.messages = [];
      }
    }
  } catch (error) {
    console.error('Failed to delete session:', error);
    appState.error = 'Could not delete conversation';
  }
}

/**
 * Update session metadata (title/preview)
 */
export function updateSessionInList(sessionId: string, preview?: string, title?: string): void {
  sessionState.sessions = sessionState.sessions.map((s: any) => {
    if (s.session_id === sessionId) {
      return {
        ...s,
        preview: preview ? preview.substring(0, 100) : s.preview,
        title: title || s.title,
        message_count: (s.message_count || 0) + 2,
        updated_at: new Date().toISOString(),
      };
    }
    return s;
  });

  // Re-sort by updated_at
  sessionState.sessions = [...sessionState.sessions].sort(
    (a, b) => new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime()
  );
}
