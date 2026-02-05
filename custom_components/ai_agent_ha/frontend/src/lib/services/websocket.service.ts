import { get } from 'svelte/store';
import type { HomeAssistant } from '$lib/types';
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
  const session = get(sessionState);
  if (!session.activeSessionId) {
    throw new Error('No active session');
  }

  const provider = get(providerState);
  const app = get(appState);

  const wsParams: any = {
    type: 'ai_agent_ha/chat/send',
    session_id: session.activeSessionId,
    message: message,
    provider: provider.selectedProvider,
    debug: app.showThinking,
  };

  // Add model if selected
  if (provider.selectedModel) {
    wsParams.model = provider.selectedModel;
  }

  return hass.callWS(wsParams);
}

/**
 * Send a message via WebSocket with streaming support
 */
export async function sendMessageStream(
  hass: HomeAssistant,
  message: string,
  callbacks: {
    onStart?: (messageId: string) => void;
    onChunk?: (chunk: string) => void;
    onToolCall?: (name: string, args: any) => void;
    onToolResult?: (name: string, result: any) => void;
    onComplete?: (result: any) => void;
    onError?: (error: string) => void;
  }
): Promise<void> {
  const session = get(sessionState);
  if (!session.activeSessionId) {
    throw new Error('No active session');
  }

  const provider = get(providerState);
  const app = get(appState);

  const wsParams: any = {
    type: 'ai_agent_ha/chat/send_stream',
    session_id: session.activeSessionId,
    message: message,
    provider: provider.selectedProvider,
    debug: app.showThinking,
  };

  // Add model if selected
  if (provider.selectedModel) {
    wsParams.model = provider.selectedModel;
  }

  // Subscribe to events for this request
  const unsubscribe = await hass.connection.subscribeMessage(
    (event: any) => {
      if (event.type === 'event') {
        const eventData = event.event;
        
        switch (eventData.type) {
          case 'stream_start':
            callbacks.onStart?.(eventData.message_id);
            break;
          
          case 'stream_chunk':
            callbacks.onChunk?.(eventData.chunk);
            break;
          
          case 'tool_call':
            callbacks.onToolCall?.(eventData.name, eventData.args);
            break;
          
          case 'tool_result':
            callbacks.onToolResult?.(eventData.name, eventData.result);
            break;
          
          case 'stream_end':
            if (eventData.success) {
              // Will be followed by result message
            } else {
              callbacks.onError?.(eventData.error || 'Unknown error');
            }
            break;
        }
      } else if (event.type === 'result') {
        // Final result with complete messages
        callbacks.onComplete?.(event.result);
        if (unsubscribe) {
          unsubscribe();
        }
      }
    },
    wsParams
  );
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
      }
    } catch (e) {
      // Not valid JSON, return as-is
    }
  }

  // Default: return content as markdown text
  return { text: content };
}
