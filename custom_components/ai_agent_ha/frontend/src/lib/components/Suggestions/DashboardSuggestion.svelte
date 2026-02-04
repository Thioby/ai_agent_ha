<script lang="ts">
  import type { DashboardSuggestion } from '$lib/types';
  import { appState } from "$lib/stores/appState"

  let { dashboard }: { dashboard: DashboardSuggestion } = $props();

  async function handleApprove() {
    if (!appState.hass) return;
    if (appState.isLoading) return;

    appState.isLoading = true;
    appState.error = null;

    try {
      const result = await appState.hass.callService('ai_agent_ha', 'create_dashboard', {
        dashboard_config: dashboard,
      });

      // Add success message
      const successMessage = result?.message || `Dashboard "${dashboard.title}" has been created successfully!`;
      appState.messages = [
        ...appState.messages,
        { type: 'assistant', text: successMessage },
      ];
    } catch (error: any) {
      console.error('Error creating dashboard:', error);
      appState.error = error.message || 'An error occurred while creating the dashboard';
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
        text: 'Dashboard creation cancelled. Would you like me to create a different dashboard?',
      },
    ];
  }

  const viewCount = $derived(dashboard.views?.length || 0);
</script>

<div class="dashboard-suggestion">
  <div class="dashboard-title">{dashboard.title}</div>
  <div class="dashboard-description">
    Dashboard with {viewCount} view{viewCount !== 1 ? 's' : ''}
  </div>
  
  <div class="dashboard-details">
    {JSON.stringify(dashboard, null, 2)}
  </div>
  
  <div class="dashboard-actions">
    <button class="approve-btn" onclick={handleApprove} disabled={appState.isLoading}>
      <svg viewBox="0 0 24 24" class="icon">
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm0 16H5V5h14v14z"/>
        <path d="M7 10h2v7H7zm4-3h2v10h-2zm4 6h2v4h-2z"/>
      </svg>
      Create Dashboard
    </button>
    <button class="reject-btn" onclick={handleReject} disabled={appState.isLoading}>
      <svg viewBox="0 0 24 24" class="icon">
        <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/>
      </svg>
      Cancel
    </button>
  </div>
</div>

<style>
  .dashboard-suggestion {
    background: var(--secondary-background-color);
    border: 1px solid var(--info-color, #2196f3);
    border-radius: 12px;
    padding: 16px;
    margin: 8px 0;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    position: relative;
    z-index: 10;
  }

  .dashboard-title {
    font-weight: 500;
    margin-bottom: 8px;
    color: var(--info-color, #2196f3);
    font-size: 16px;
  }

  .dashboard-description {
    margin-bottom: 16px;
    color: var(--secondary-text-color);
    line-height: 1.4;
    font-size: 14px;
  }

  .dashboard-details {
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

  .dashboard-actions {
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
    background: var(--info-color, #2196f3);
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
