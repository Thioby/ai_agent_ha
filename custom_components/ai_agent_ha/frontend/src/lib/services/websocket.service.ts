import type { HomeAssistant, Message } from '$lib/types';
import { appState } from '$lib/stores/appState';
import { sessionState } from '$lib/stores/sessions';
import { providerState } from '$lib/stores/providers';

/**
 * WebSocket service for Home Assistant communication
 */

/**
 * Send a message via WebSocket
 */
export async function sendMessage(
  hass: HomeAssistant,
  message: string
): Promise<any> {
  if (!sessionState.activeSessionId) {
    throw new Error('No active session');
  }

  const wsParams: any = {
    type: 'ai_agent_ha/chat/send',
    session_id: sessionState.activeSessionId,
    message: message,
    provider: providerState.selectedProvider,
    debug: appState.showThinking,
  };

  // Add model if selected
  if (providerState.selectedModel) {
    wsParams.model = providerState.selectedModel;
  }

  return hass.callWS(wsParams);
}

/**
 * Subscribe to AI agent response events
 */
export async function subscribeToEvents(
  hass: HomeAssistant,
  callback: (event: any) => void
): Promise<(() => void) | undefined> {
  try {
    return await hass.connection.subscribeEvents(callback, 'ai_agent_ha_response');
  } catch (error) {
    console.error('Failed to subscribe to events:', error);
    return undefined;
  }
}

/**
 * Parse JSON response from AI
 * Handles both pure JSON and markdown with JSON
 */
export function parseAIResponse(content: string): {
  text: string;
  automation?: any;
  dashboard?: any;
} {
  const trimmedContent = content.trim();

  // Only parse if content is pure JSON (starts and ends with braces)
  if (trimmedContent.startsWith('{') && trimmedContent.endsWith('}')) {
    try {
      const parsed = JSON.parse(trimmedContent);

      if (parsed.request_type === 'automation_suggestion') {
        return {
          text: parsed.message || 'I found an automation that might help you.',
          automation: parsed.automation,
        };
      } else if (parsed.request_type === 'dashboard_suggestion') {
        return {
          text: parsed.message || 'I created a dashboard configuration for you.',
          dashboard: parsed.dashboard,
        };
      } else if (parsed.request_type === 'final_response') {
        return {
          text: parsed.response || parsed.message || content,
        };
      } else if (parsed.message) {
        return {
          text: parsed.message,
        };
      } else if (parsed.response) {
        return {
          text: parsed.response,
        };
      }
    } catch (e) {
      // Not valid JSON, use content as-is
    }
  }

  // Default: return content as text
  return { text: content };
}
