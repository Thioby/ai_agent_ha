<script lang="ts">
  import { onMount } from 'svelte';
  import type { HomeAssistant } from '../types';
  import { appState } from '../stores/appState';
  import { uiState } from '../stores/ui';
  import { loadProviders } from '../services/provider.service';
  import { loadSessions } from '../services/session.service';
  
  // Components
  import Header from './Header.svelte';
  import Sidebar from './Sidebar/Sidebar.svelte';
  import ChatArea from './Chat/ChatArea.svelte';
  import InputArea from './Input/InputArea.svelte';
  import ThinkingPanel from './Debug/ThinkingPanel.svelte';

  // Props
  let { hass, narrow = false }: { hass: HomeAssistant; narrow?: boolean; panel?: boolean } = $props();

  // Update appState when hass changes
  $effect(() => {
    appState.hass = hass;
  });

  // Lifecycle - Initialize
  onMount(() => {
    console.log('[AiAgentPanel] Mounting...');
    
    // Load providers and sessions in parallel
    (async () => {
      try {
        await Promise.all([
          loadProviders(hass),
          loadSessions(hass)
        ]);
        console.log('[AiAgentPanel] Initialization complete');
      } catch (error) {
        console.error('[AiAgentPanel] Initialization error:', error);
        appState.error = error instanceof Error ? error.message : 'Failed to initialize';
      }
    })();

    // Subscribe to WebSocket events (handled internally by the service)
    // subscribeToEvents is called when sending messages

    // Window resize handler for mobile detection
    const handleResize = () => {
      const isMobile = window.innerWidth <= 768;
      if (!isMobile && uiState.sidebarOpen) {
        // Keep sidebar open on desktop
      }
    };
    window.addEventListener('resize', handleResize);
    handleResize(); // Initial check

    // Cleanup
    return () => {
      window.removeEventListener('resize', handleResize);
    };
  });

  // Computed values
  const isMobile = $derived(narrow || window.innerWidth <= 768);
  const showThinkingPanel = $derived(appState.showThinking && appState.debugInfo.length > 0);
</script>

<div class="ai-agent-panel" class:narrow={isMobile}>
  <Header />
  
  <div class="main-container">
    <Sidebar />
    
    <div class="content-area">
      <div class="chat-container">
        <ChatArea />
        
        {#if showThinkingPanel}
          <ThinkingPanel />
        {/if}
      </div>
      
      <InputArea />
    </div>
  </div>
</div>

<style>
  .ai-agent-panel {
    display: flex;
    flex-direction: column;
    width: 100%;
    height: 100vh;
    overflow: hidden;
    background-color: var(--primary-background-color);
  }

  .main-container {
    display: flex;
    flex: 1;
    overflow: hidden;
    position: relative;
  }

  .content-area {
    display: flex;
    flex-direction: column;
    flex: 1;
    overflow: hidden;
    position: relative;
  }

  .chat-container {
    display: flex;
    flex: 1;
    overflow: hidden;
    position: relative;
  }

  /* Mobile adjustments */
  .ai-agent-panel.narrow .content-area {
    width: 100%;
  }

  /* Responsive */
  @media (max-width: 768px) {
    .ai-agent-panel {
      height: 100vh;
      height: 100dvh; /* Dynamic viewport height for mobile */
    }
  }

  /* Animation */
  .content-area {
    animation: fadeIn 0.3s ease-in-out;
  }

  @keyframes fadeIn {
    from {
      opacity: 0;
    }
    to {
      opacity: 1;
    }
  }
</style>
