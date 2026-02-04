<script lang="ts">
  import type { AutomationSuggestion } from '$lib/types';
  import { appState } from '$lib/stores/appState';

  let { automation }: { automation: AutomationSuggestion } = $props();

  async function handleApprove() {
    if (!appState.hass) return;
    if (appState.isLoading) return;

    appState.isLoading = true;
    appState.error = null;

    try {
      const result = await appState.hass.callService('ai_agent_ha', 'create_automation', {
        automation: automation,
      });

      // Add success message
      const successMessage = result?.message || `Automation "${automation.alias}" has been created successfully!`;
      appState.messages = [
        ...appState.messages,
        { type: 'assistant', text: successMessage },
      ];
    } catch (error: any) {
      console.error('Error creating automation:', error);
      appState.error = error.message || 'An error occurred while creating the automation';
      appState.messages = [
        ...appState.messages,
        { type: 'assistant', text: `Error: ${appState.error}` },
      ];
    } finally {
      appState.isLoading = false;
    }
  }

  function handleReject() {
    appState.messages = [
      ...appState.messages,
      {
        type: 'assistant',
        text: 'Automation creation cancelled. Would you like to try something else?',
      },
    ];
  }
</script>

<div class="automation-suggestion">
  <div class="automation-title">{automation.alias}</div>
  {#if automation.description}
    <div class="automation-description">{automation.description}</div>
  {/if}
  
  <div class="automation-details">
    {JSON.stringify(automation, null, 2)}
  </div>
  
  <div class="automation-actions">
    <button class="approve-btn" onclick={handleApprove} disabled={appState.isLoading}>
      <svg viewBox="0 0 24 24" class="icon">
        <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"/>
      </svg>
      Approve
    </button>
    <button class="reject-btn" onclick={handleReject} disabled={appState.isLoading}>
      <svg viewBox="0 0 24 24" class="icon">
        <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/>
      </svg>
      Reject
    </button>
  </div>
</div>

<style>
  .automation-suggestion {
    background: var(--secondary-background-color);
    border: 1px solid var(--primary-color);
    border-radius: 12px;
    padding: 16px;
    margin: 8px 0;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    position: relative;
    z-index: 10;
  }

  .automation-title {
    font-weight: 500;
    margin-bottom: 8px;
    color: var(--primary-color);
    font-size: 16px;
  }

  .automation-description {
    margin-bottom: 16px;
    color: var(--secondary-text-color);
    line-height: 1.4;
    font-size: 14px;
  }

  .automation-details {
    margin-top: 8px;
    padding: 12px;
    background: var(--primary-background-color);
    border-radius: 8px;
    font-family: 'SF Mono', Monaco, Consolas, monospace;
    font-size: 12px;
    white-space: pre-wrap;
    overflow-x: auto;
    max-height: 200px;
    overflow-y: auto;
    border: 1px solid var(--divider-color);
    color: var(--primary-text-color);
  }

  .automation-actions {
    display: flex;
    gap: 8px;
    margin-top: 16px;
    justify-content: flex-end;
  }

  .approve-btn,
  .reject-btn {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 10px 20px;
    border: none;
    border-radius: 20px;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s ease;
    font-family: inherit;
  }

  .icon {
    width: 18px;
    height: 18px;
    fill: currentColor;
  }

  .approve-btn {
    background: var(--success-color, #4caf50);
    color: white;
  }

  .approve-btn:hover:not(:disabled) {
    filter: brightness(1.1);
    transform: translateY(-1px);
  }

  .reject-btn {
    background: var(--error-color);
    color: white;
  }

  .reject-btn:hover:not(:disabled) {
    filter: brightness(1.1);
    transform: translateY(-1px);
  }

  .approve-btn:disabled,
  .reject-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
</style>
