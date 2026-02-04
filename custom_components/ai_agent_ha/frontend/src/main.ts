import appCss from './app.css?inline';
import AiAgentPanel from './lib/components/AiAgentPanel.svelte';
import type { HomeAssistant } from './lib/types';

/**
 * AI Agent HA Panel Web Component
 * 
 * Custom element that wraps the Svelte application and integrates with Home Assistant.
 * Uses Shadow DOM for style isolation.
 */
class AiAgentHAPanel extends HTMLElement {
  private _hass?: HomeAssistant;
  private _narrow = false;
  private _panel = true;
  declare shadowRoot: ShadowRoot;
  private svelteApp?: any;
  private mountPoint?: HTMLDivElement;

  constructor() {
    super();
    
    // Attach Shadow DOM for style isolation
    this.shadowRoot = this.attachShadow({ mode: 'open' });
    
    // Add global styles to shadow root
    const style = document.createElement('style');
    style.textContent = appCss;
    this.shadowRoot.appendChild(style);
    
    // Create mount point
    this.mountPoint = document.createElement('div');
    this.mountPoint.id = 'svelte-app';
    this.shadowRoot.appendChild(this.mountPoint);
  }

  connectedCallback() {
    console.log('[AiAgentHAPanel] Connected to DOM');
    this._initializeApp();
  }

  disconnectedCallback() {
    console.log('[AiAgentHAPanel] Disconnected from DOM');
    this._destroyApp();
  }

  /**
   * Initialize Svelte application
   */
  private _initializeApp() {
    if (this.svelteApp) {
      console.warn('[AiAgentHAPanel] App already initialized');
      return;
    }

    if (!this._hass) {
      console.warn('[AiAgentHAPanel] Waiting for hass to be set before mounting');
      return;
    }

    try {
      console.log('[AiAgentHAPanel] Mounting Svelte app...');
      
      this.svelteApp = new (AiAgentPanel as any)({
        target: this.mountPoint!,
        props: {
          hass: this._hass,
          narrow: this._narrow,
          panel: this._panel,
        },
      });

      console.log('[AiAgentHAPanel] Svelte app mounted successfully');
    } catch (error) {
      console.error('[AiAgentHAPanel] Failed to mount Svelte app:', error);
    }
  }

  /**
   * Destroy Svelte application
   */
  private _destroyApp() {
    if (this.svelteApp) {
      try {
        this.svelteApp.$destroy();
        this.svelteApp = null;
        console.log('[AiAgentHAPanel] Svelte app destroyed');
      } catch (error) {
        console.error('[AiAgentHAPanel] Error destroying Svelte app:', error);
      }
    }
  }

  /**
   * Update Svelte component props efficiently
   */
  private _updateProps() {
    if (!this.svelteApp) {
      return;
    }

    try {
      // Update props through Svelte's reactivity system
      this.svelteApp.$set({
        hass: this._hass,
        narrow: this._narrow,
        panel: this._panel,
      });
    } catch (error) {
      console.error('[AiAgentHAPanel] Error updating props:', error);
    }
  }

  /**
   * Home Assistant integration - hass property setter
   */
  set hass(hass: HomeAssistant) {
    const isFirstSet = !this._hass;
    this._hass = hass;

    if (isFirstSet) {
      // First time setting hass - initialize the app
      this._initializeApp();
    } else if (this.svelteApp) {
      // App already mounted - update props
      this._updateProps();
    }
  }

  get hass(): HomeAssistant | undefined {
    return this._hass;
  }

  /**
   * Home Assistant integration - narrow property setter
   */
  set narrow(narrow: boolean) {
    this._narrow = narrow;
    if (this.svelteApp) {
      this._updateProps();
    }
  }

  get narrow(): boolean {
    return this._narrow;
  }

  /**
   * Home Assistant integration - panel property setter
   */
  set panel(panel: boolean) {
    this._panel = panel;
    if (this.svelteApp) {
      this._updateProps();
    }
  }

  get panel(): boolean {
    return this._panel;
  }
}

// Register the custom element
if (!customElements.get('ai-agent-ha-panel')) {
  customElements.define('ai-agent-ha-panel', AiAgentHAPanel);
  console.log('[AiAgentHAPanel] Custom element registered: <ai-agent-ha-panel>');
} else {
  console.warn('[AiAgentHAPanel] Custom element already registered');
}

// Export for potential direct usage
export default AiAgentHAPanel;
