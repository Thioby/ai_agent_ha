<script lang="ts">
  import { appState } from "$lib/stores/appState"
  import { sessionState } from "$lib/stores/sessions"
  import { toggleSidebar } from "$lib/stores/ui"
  import { deleteSession } from '$lib/services/session.service';
  import { clearAllCaches } from '$lib/services/markdown.service';

  async function clearChat() {
    if (!appState.hass) return;
    if (!sessionState.activeSessionId) return;

    if (confirm('Clear this conversation?')) {
      await deleteSession(appState.hass, sessionState.activeSessionId);
      clearAllCaches();
    }
  }
</script>

<div class="header">
  <button class="menu-toggle" onclick={toggleSidebar}>
    <svg viewBox="0 0 24 24" class="icon">
      <path d="M3 18h18v-2H3v2zm0-5h18v-2H3v2zm0-7v2h18V6H3z"/>
    </svg>
  </button>
  
  <svg viewBox="0 0 24 24" class="robot-icon">
    <path d="M12 2c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2zm6 16.5V22h-3v-3.5h3zM5.5 22h3v-3.5h-3V22zM19 9h-1.5V7.5c0-1.93-1.57-3.5-3.5-3.5S10.5 5.57 10.5 7.5V9H9c-.55 0-1 .45-1 1v9c0 .55.45 1 1 1h10c.55 0 1-.45 1-1v-9c0-.55-.45-1-1-1zm-7.5-1.5c0-.83.67-1.5 1.5-1.5s1.5.67 1.5 1.5V9h-3V7.5z"/>
  </svg>
  
  <span class="title">AI Agent HA</span>
  
  <button
    class="clear-button"
    onclick={clearChat}
    disabled={appState.isLoading}
  >
    <svg viewBox="0 0 24 24" class="icon">
      <path d="M15 16h4v2h-4zm0-8h7v2h-7zm0 4h6v2h-6zM3 18c0 1.1.9 2 2 2h6c1.1 0 2-.9 2-2V8H3v10zM14 5h-3l-1-1H6L5 5H2v2h12z"/>
    </svg>
    <span>Clear Chat</span>
  </button>
</div>

<style>
  .header {
    background: var(--app-header-background-color, var(--secondary-background-color));
    color: var(--app-header-text-color, var(--primary-text-color));
    padding: 16px 24px;
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 20px;
    font-weight: 500;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    position: relative;
    z-index: 100;
  }

  .menu-toggle {
    display: none;
    width: 44px;
    height: 44px;
    min-width: 44px;
    min-height: 44px;
    border: none;
    background: transparent;
    cursor: pointer;
    border-radius: 50%;
    align-items: center;
    justify-content: center;
    margin-right: 8px;
    padding: 0;
  }

  .menu-toggle:hover {
    background: var(--card-background-color);
  }

  .icon {
    width: 24px;
    height: 24px;
    fill: currentColor;
  }

  .robot-icon {
    width: 24px;
    height: 24px;
    fill: var(--primary-color);
  }

  .title {
    flex: 1;
  }

  .clear-button {
    margin-left: auto;
    border: none;
    border-radius: 16px;
    background: var(--error-color);
    color: #fff;
    cursor: pointer;
    transition: all 0.2s ease;
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 8px 16px;
    font-weight: 500;
    font-size: 13px;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.08);
    min-width: unset;
    width: auto;
    height: 36px;
    flex-shrink: 0;
    font-family: inherit;
  }

  .clear-button .icon {
    width: 16px;
    height: 16px;
    margin-right: 2px;
    fill: white;
  }

  .clear-button span {
    color: #fff;
    font-weight: 500;
  }

  .clear-button:hover:not(:disabled) {
    opacity: 0.92;
    transform: translateY(-1px);
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.13);
  }

  .clear-button:active:not(:disabled) {
    transform: translateY(0);
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.08);
  }

  .clear-button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  @media (max-width: 768px) {
    .menu-toggle {
      display: flex;
    }

    .robot-icon {
      display: none;
    }
  }
</style>
