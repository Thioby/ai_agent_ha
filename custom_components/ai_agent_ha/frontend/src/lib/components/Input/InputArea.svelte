<script lang="ts">
  import { appState } from '$lib/stores/appState';
  import { sessionState } from '$lib/stores/sessions';
  import { providerState } from '$lib/stores/providers';
  import { sendMessage, parseAIResponse } from '$lib/services/websocket.service';
  import { createSession, updateSessionInList } from '$lib/services/session.service';
  import MessageInput from './MessageInput.svelte';
  import ProviderSelector from './ProviderSelector.svelte';
  import ModelSelector from './ModelSelector.svelte';
  import SendButton from './SendButton.svelte';
  import ThinkingToggle from './ThinkingToggle.svelte';

  let messageInput: MessageInput;

  async function handleSend() {
    if (!appState.hass) return;

    const message = messageInput.getValue().trim();
    if (!message || appState.isLoading) return;

    // Clear input
    messageInput.clear();

    appState.isLoading = true;
    appState.error = null;

    // Create session if none active
    if (!sessionState.activeSessionId && providerState.selectedProvider) {
      await createSession(appState.hass, providerState.selectedProvider);
    }

    if (!sessionState.activeSessionId) {
      appState.error = 'No active session';
      appState.isLoading = false;
      return;
    }

    // Add user message
    appState.messages = [...appState.messages, { type: 'user', text: message }];

    try {
      const result = await sendMessage(appState.hass, message);
      appState.isLoading = false;

      if (result.assistant_message) {
        let { text, automation, dashboard } = parseAIResponse(
          result.assistant_message.content || ''
        );

        const assistantMsg: any = {
          type: 'assistant',
          text,
          automation: automation || result.assistant_message.metadata?.automation,
          dashboard: dashboard || result.assistant_message.metadata?.dashboard,
          status: result.assistant_message.status,
          error_message: result.assistant_message.error_message,
        };

        if (result.assistant_message.status === 'error') {
          appState.error = result.assistant_message.error_message;
          assistantMsg.text = `Error: ${result.assistant_message.error_message}`;
        }

        appState.messages = [...appState.messages, assistantMsg];

        // Update session in list
        const session = sessionState.sessions.find(
          (s) => s.session_id === sessionState.activeSessionId
        );
        const isNewConversation = session?.title === 'New Conversation';
        updateSessionInList(
          sessionState.activeSessionId,
          message,
          isNewConversation ? message.substring(0, 40) + (message.length > 40 ? '...' : '') : undefined
        );
      }
    } catch (error: any) {
      console.error('WebSocket error:', error);
      appState.isLoading = false;
      appState.error = error.message || 'An error occurred while processing your request';
      appState.messages = [
        ...appState.messages,
        { type: 'assistant', text: `Error: ${appState.error}` },
      ];
    }
  }
</script>

<div class="input-container">
  <div class="input-main">
    <MessageInput bind:this={messageInput} onSend={handleSend} />
  </div>

  <div class="input-footer">
    <ProviderSelector />
    <ModelSelector />
    <ThinkingToggle />
    <SendButton onclick={handleSend} />
  </div>
</div>

<style>
  .input-container {
    position: relative;
    width: 100%;
    background: var(--card-background-color);
    border: 1px solid var(--divider-color);
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    margin-bottom: 24px;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
  }

  .input-container:focus-within {
    border-color: var(--primary-color);
    box-shadow: 0 0 0 2px rgba(3, 169, 244, 0.1);
  }

  .input-main {
    display: flex;
    align-items: flex-end;
    padding: 12px;
    gap: 12px;
  }

  .input-footer {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 16px 12px 16px;
    border-top: 1px solid var(--divider-color);
    background: var(--card-background-color);
    border-radius: 0 0 12px 12px;
    gap: 12px;
  }

  @media (max-width: 768px) {
    .input-container {
      padding: 12px;
      padding-bottom: calc(12px + env(safe-area-inset-bottom, 0px));
    }

    .input-footer {
      gap: 8px;
    }
  }
</style>
