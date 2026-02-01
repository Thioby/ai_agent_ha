import {
  LitElement,
  html,
  css,
} from "https://unpkg.com/lit@3.1.0/index.js?module";
import { unsafeHTML } from "https://unpkg.com/lit@3.1.0/directives/unsafe-html.js?module";
import { marked } from "https://unpkg.com/marked@12.0.0/lib/marked.esm.js";
import { markedHighlight } from "https://unpkg.com/marked-highlight@2.1.1/src/index.js";
import hljs from "https://unpkg.com/@highlightjs/cdn-assets@11.9.0/es/highlight.min.js";
import yaml from "https://unpkg.com/@highlightjs/cdn-assets@11.9.0/es/languages/yaml.min.js";
import DOMPurify from "https://unpkg.com/dompurify@3.0.8/dist/purify.es.mjs";

hljs.registerLanguage('yaml', yaml);

marked.use(markedHighlight({
  langPrefix: 'hljs language-',
  highlight(code, lang) {
    if (lang && hljs.getLanguage(lang)) {
      return hljs.highlight(code, { language: lang }).value;
    }
    return hljs.highlightAuto(code).value;
  }
}));

marked.use({
  gfm: true,
  breaks: true
});

const PROVIDERS = {
  openai: "OpenAI",
  llama: "Llama",
  gemini: "Google Gemini",
  gemini_oauth: "Gemini (OAuth)",
  openrouter: "OpenRouter",
  anthropic: "Anthropic",
  anthropic_oauth: "Claude Pro/Max",
  alter: "Alter",
  zai: "z.ai",
  local: "Local Model",
};

class AiAgentHaPanel extends LitElement {
  static get properties() {
    return {
      hass: { type: Object, reflect: false, attribute: false },
      narrow: { type: Boolean, reflect: false, attribute: false },
      panel: { type: Object, reflect: false, attribute: false },
      _messages: { type: Array, reflect: false, attribute: false },
      _isLoading: { type: Boolean, reflect: false, attribute: false },
      _error: { type: String, reflect: false, attribute: false },
      _pendingAutomation: { type: Object, reflect: false, attribute: false },
      _selectedProvider: { type: String, reflect: false, attribute: false },
      _availableProviders: { type: Array, reflect: false, attribute: false },
      _showProviderDropdown: { type: Boolean, reflect: false, attribute: false },
      _selectedModel: { type: String, reflect: false, attribute: false },
      _availableModels: { type: Array, reflect: false, attribute: false },
      _showThinking: { type: Boolean, reflect: false, attribute: false },
      _thinkingExpanded: { type: Boolean, reflect: false, attribute: false },
      _debugInfo: { type: Object, reflect: false, attribute: false },
      _sessions: { type: Array, reflect: false, attribute: false },
      _activeSessionId: { type: String, reflect: false, attribute: false },
      _sidebarOpen: { type: Boolean, reflect: false, attribute: false },
      _sessionsLoading: { type: Boolean, reflect: false, attribute: false }
    };
  }

  static get styles() {
    return css`
      :host {
        background: var(--primary-background-color);
        -webkit-font-smoothing: antialiased;
        display: flex;
        flex-direction: column;
        height: 100vh;
      }
      .header {
        background: var(--app-header-background-color);
        color: var(--app-header-text-color);
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
        box-shadow: 0 1px 2px rgba(0,0,0,0.08);
        min-width: unset;
        width: auto;
        height: 36px;
        flex-shrink: 0;
        position: relative;
        z-index: 101;
        font-family: inherit;
      }
      .clear-button:hover {
        background: var(--error-color);
        opacity: 0.92;
        transform: translateY(-1px);
        box-shadow: 0 2px 6px rgba(0,0,0,0.13);
      }
      .clear-button:active {
        transform: translateY(0);
        box-shadow: 0 1px 2px rgba(0,0,0,0.08);
      }
      .clear-button ha-icon {
        --mdc-icon-size: 16px;
        margin-right: 2px;
        color: #fff;
      }
      .clear-button span {
        color: #fff;
        font-weight: 500;
      }
      .content {
        flex-grow: 1;
        padding: 24px;
        overflow-y: auto;
        display: flex;
        flex-direction: column;
        justify-content: flex-end;
      }
      .chat-container {
        width: 100%;
        padding: 0;
        display: flex;
        flex-direction: column;
        flex-grow: 1;
        height: 100%;
      }
      .messages {
        overflow-y: auto;
        border: 1px solid var(--divider-color);
        border-radius: 12px;
        margin-bottom: 24px;
        padding: 0;
        background: var(--primary-background-color);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        flex-grow: 1;
        width: 100%;
      }
      .message {
        margin-bottom: 16px;
        padding: 12px 16px;
        border-radius: 12px;
        max-width: 80%;
        line-height: 1.5;
        animation: fadeIn 0.3s ease-out;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
        word-wrap: break-word;
      }
      .user-message {
        background: var(--primary-color);
        color: var(--text-primary-color);
        margin-left: auto;
        border-bottom-right-radius: 4px;
      }
      .assistant-message {
        background: var(--secondary-background-color);
        margin-right: auto;
        border-bottom-left-radius: 4px;
      }
      .message-content {
        word-wrap: break-word;
        overflow-wrap: break-word;
      }
      .message-content h1, .message-content h2, .message-content h3,
      .message-content h4, .message-content h5, .message-content h6 {
        margin: 0.5em 0 0.3em;
        line-height: 1.3;
        color: var(--primary-text-color);
      }
      .message-content h1 { font-size: 1.4em; }
      .message-content h2 { font-size: 1.2em; }
      .message-content h3 { font-size: 1.1em; }
      .message-content p { margin: 0.5em 0; }
      .message-content p:first-child { margin-top: 0; }
      .message-content p:last-child { margin-bottom: 0; }
      .message-content ul, .message-content ol {
        margin: 0.5em 0;
        padding-left: 1.5em;
      }
      .message-content li { margin: 0.25em 0; }
      .message-content strong { font-weight: 600; }
      .message-content code {
        background: rgba(0, 0, 0, 0.06);
        padding: 0.2em 0.4em;
        border-radius: 4px;
        font-family: 'SF Mono', Monaco, Consolas, monospace;
        font-size: 0.85em;
      }
      .message-content pre {
        background: var(--primary-background-color);
        border: 1px solid var(--divider-color);
        padding: 12px;
        border-radius: 8px;
        overflow-x: auto;
        margin: 0.5em 0;
      }
      .message-content pre code {
        background: none;
        padding: 0;
        font-size: 0.85em;
        line-height: 1.5;
      }
      .message-content blockquote {
        border-left: 3px solid var(--primary-color);
        margin: 0.5em 0;
        padding-left: 1em;
        color: var(--secondary-text-color);
      }
      .message-content a {
        color: var(--primary-color);
        text-decoration: none;
      }
      .message-content a:hover {
        text-decoration: underline;
      }
      .message-content hr {
        border: none;
        border-top: 1px solid var(--divider-color);
        margin: 1em 0;
      }
      .hljs { background: transparent; }
      .hljs-keyword { color: #c678dd; }
      .hljs-string { color: #98c379; }
      .hljs-attr { color: #d19a66; }
      .hljs-number { color: #d19a66; }
      .hljs-literal { color: #56b6c2; }
      .hljs-comment { color: #5c6370; font-style: italic; }
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
        box-shadow: 0 0 0 2px rgba(var(--primary-color-rgb), 0.1);
      }
      .input-main {
        display: flex;
        align-items: flex-end;
        padding: 12px;
        gap: 12px;
      }
      .input-wrapper {
        flex-grow: 1;
        position: relative;
        border: 1px solid var(--divider-color);
      }
      textarea {
        width: 100%;
        min-height: 24px;
        max-height: 200px;
        padding: 12px 16px 12px 16px;
        border: none;
        outline: none;
        resize: none;
        font-size: 16px;
        line-height: 1.5;
        background: transparent;
        color: var(--primary-text-color);
        font-family: inherit;
      }
      textarea::placeholder {
        color: var(--secondary-text-color);
      }
      .input-footer {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 8px 16px 12px 16px;
        border-top: 1px solid var(--divider-color);
        background: var(--card-background-color);
        border-radius: 0 0 12px 12px;
      }
      .provider-selector {
        position: relative;
        display: flex;
        align-items: center;
        gap: 8px;
      }
      .provider-button {
        display: flex;
        align-items: center;
        gap: 6px;
        padding: 6px 12px;
        background: var(--secondary-background-color);
        border: 1px solid var(--divider-color);
        border-radius: 8px;
        cursor: pointer;
        font-size: 14px;
        font-weight: 500;
        color: var(--primary-text-color);
        transition: all 0.2s ease;
        min-width: 150px;
        -webkit-appearance: none;
        -moz-appearance: none;
        appearance: none;
        background-image: url('data:image/svg+xml;charset=US-ASCII,<svg width="14" height="14" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M7 10l5 5 5-5H7z" fill="currentColor"/></svg>');
        background-repeat: no-repeat;
        background-position: right 8px center;
        padding-right: 30px;
      }
      .provider-button:hover {
        background-color: var(--primary-background-color);
        border-color: var(--primary-color);
      }
      .provider-button:focus {
        outline: none;
        border-color: var(--primary-color);
        box-shadow: 0 0 0 2px rgba(var(--primary-color-rgb), 0.2);
      }
      .provider-label {
        font-size: 12px;
        color: var(--secondary-text-color);
        margin-right: 8px;
      }
      .thinking-toggle {
        display: flex;
        align-items: center;
        gap: 6px;
        font-size: 12px;
        color: var(--secondary-text-color);
        cursor: pointer;
        user-select: none;
      }
      .thinking-toggle input {
        margin: 0;
      }
      .thinking-panel {
        border: 1px dashed var(--divider-color);
        border-radius: 10px;
        padding: 10px 12px;
        margin: 12px 0;
        background: var(--secondary-background-color);
      }
      .thinking-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        cursor: pointer;
        gap: 10px;
      }
      .thinking-title {
        font-weight: 600;
        color: var(--primary-text-color);
        font-size: 14px;
      }
      .thinking-subtitle {
        display: block;
        font-size: 12px;
        color: var(--secondary-text-color);
        margin-top: 2px;
      }
      .thinking-body {
        margin-top: 10px;
        display: flex;
        flex-direction: column;
        gap: 10px;
        max-height: 240px;
        overflow-y: auto;
      }
      .thinking-entry {
        border: 1px solid var(--divider-color);
        border-radius: 8px;
        padding: 8px;
        background: var(--primary-background-color);
      }
      .thinking-entry .badge {
        display: inline-block;
        background: var(--secondary-background-color);
        color: var(--secondary-text-color);
        font-size: 11px;
        padding: 2px 6px;
        border-radius: 6px;
        margin-bottom: 6px;
      }
      .thinking-entry pre {
        margin: 0;
        white-space: pre-wrap;
        word-break: break-word;
        font-size: 12px;
      }
      .thinking-empty {
        color: var(--secondary-text-color);
        font-size: 12px;
      }
      .send-button {
        --mdc-theme-primary: var(--primary-color);
        --mdc-theme-on-primary: var(--text-primary-color);
        --mdc-typography-button-font-size: 14px;
        --mdc-typography-button-text-transform: none;
        --mdc-typography-button-letter-spacing: 0;
        --mdc-typography-button-font-weight: 500;
        --mdc-button-height: 36px;
        --mdc-button-padding: 0 16px;
        border-radius: 8px;
        transition: all 0.2s ease;
        min-width: 80px;
      }
      .send-button:hover {
        transform: translateY(-1px);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
      }
      .send-button:active {
        transform: translateY(0);
      }
      .send-button:disabled {
        opacity: 0.5;
        cursor: not-allowed;
      }
      .loading {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 16px;
        padding: 12px 16px;
        border-radius: 12px;
        background: var(--secondary-background-color);
        margin-right: auto;
        max-width: 80%;
        animation: fadeIn 0.3s ease-out;
      }
      .loading-dots {
        display: flex;
        gap: 4px;
      }
      .dot {
        width: 8px;
        height: 8px;
        background: var(--primary-color);
        border-radius: 50%;
        animation: bounce 1.4s infinite ease-in-out;
      }
      .dot:nth-child(1) { animation-delay: -0.32s; }
      .dot:nth-child(2) { animation-delay: -0.16s; }
      @keyframes bounce {
        0%, 80%, 100% {
          transform: scale(0);
        }
        40% {
          transform: scale(1.0);
        }
      }
      @keyframes fadeIn {
        from {
          opacity: 0;
          transform: translateY(10px);
        }
        to {
          opacity: 1;
          transform: translateY(0);
        }
      }
      .error {
        color: var(--error-color);
        padding: 12px 16px;
        margin: 8px 16px;
        border-radius: 8px;
        background: rgba(var(--rgb-error-color, 219, 68, 55), 0.1);
        border: 1px solid var(--error-color);
        animation: fadeIn 0.3s ease-out;
        display: flex;
        align-items: center;
        gap: 12px;
        font-size: 14px;
      }
      .error ha-icon {
        --mdc-icon-size: 20px;
        flex-shrink: 0;
      }
      .error-message {
        flex: 1;
      }
      .error-dismiss {
        background: transparent;
        border: none;
        cursor: pointer;
        padding: 4px;
        border-radius: 4px;
        display: flex;
        align-items: center;
        justify-content: center;
      }
      .error-dismiss:hover {
        background: rgba(var(--rgb-error-color, 219, 68, 55), 0.2);
      }
      .error-dismiss ha-icon {
        --mdc-icon-size: 18px;
        color: var(--error-color);
      }
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
      }
      .automation-actions {
        display: flex;
        gap: 8px;
        margin-top: 16px;
        justify-content: flex-end;
      }
      .automation-actions ha-button {
        --mdc-button-height: 40px;
        --mdc-button-padding: 0 20px;
        --mdc-typography-button-font-size: 14px;
        --mdc-typography-button-font-weight: 600;
        border-radius: 20px;
      }
      .automation-actions ha-button:first-child {
        --mdc-theme-primary: var(--success-color, #4caf50);
        --mdc-theme-on-primary: #fff;
      }
      .automation-actions ha-button:last-child {
        --mdc-theme-primary: var(--error-color);
        --mdc-theme-on-primary: #fff;
      }
      .automation-details {
        margin-top: 8px;
        padding: 8px;
        background: var(--primary-background-color);
        border-radius: 8px;
        font-family: monospace;
        font-size: 12px;
        white-space: pre-wrap;
        overflow-x: auto;
        max-height: 200px;
        overflow-y: auto;
        border: 1px solid var(--divider-color);
      }
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
      }
      .dashboard-actions {
        display: flex;
        gap: 8px;
        margin-top: 16px;
        justify-content: flex-end;
      }
      .dashboard-actions ha-button {
        --mdc-button-height: 40px;
        --mdc-button-padding: 0 20px;
        --mdc-typography-button-font-size: 14px;
        --mdc-typography-button-font-weight: 600;
        border-radius: 20px;
      }
      .dashboard-actions ha-button:first-child {
        --mdc-theme-primary: var(--info-color, #2196f3);
        --mdc-theme-on-primary: #fff;
      }
      .dashboard-actions ha-button:last-child {
        --mdc-theme-primary: var(--error-color);
        --mdc-theme-on-primary: #fff;
      }
      .dashboard-details {
        margin-top: 8px;
        padding: 8px;
        background: var(--primary-background-color);
        border-radius: 8px;
        font-family: monospace;
        font-size: 12px;
        white-space: pre-wrap;
        overflow-x: auto;
        max-height: 200px;
        overflow-y: auto;
        border: 1px solid var(--divider-color);
      }
      .no-providers {
        color: var(--error-color);
        font-size: 14px;
        padding: 8px;
      }
      .main-container {
        display: flex;
        flex: 1;
        overflow: hidden;
        height: calc(100vh - 56px);
      }
      .sidebar {
        width: 280px;
        background: var(--secondary-background-color);
        border-right: 1px solid var(--divider-color);
        display: flex;
        flex-direction: column;
        flex-shrink: 0;
        transition: transform 0.3s ease, width 0.3s ease;
      }
      .sidebar.hidden {
        transform: translateX(-100%);
        width: 0;
        border: none;
      }
      .sidebar-header {
        padding: 16px;
        border-bottom: 1px solid var(--divider-color);
        display: flex;
        flex-direction: column;
        gap: 12px;
      }
      .sidebar-title {
        font-size: 16px;
        font-weight: 500;
        color: var(--primary-text-color);
        display: flex;
        align-items: center;
        gap: 8px;
      }
      .new-chat-btn {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
        padding: 14px 16px;
        min-height: 48px;
        background: var(--primary-color);
        color: white;
        border: none;
        border-radius: 8px;
        font-size: 14px;
        font-weight: 500;
        cursor: pointer;
        transition: background-color 0.2s, transform 0.1s;
        font-family: inherit;
      }
      .new-chat-btn:hover {
        filter: brightness(1.1);
      }
      .new-chat-btn:active {
        transform: scale(0.98);
      }
      .session-list {
        flex: 1;
        overflow-y: auto;
        padding: 8px;
      }
      .session-item {
        display: flex;
        flex-direction: column;
        padding: 12px;
        margin-bottom: 4px;
        border-radius: 8px;
        cursor: pointer;
        transition: background-color 0.2s, transform 0.2s;
        position: relative;
        animation: slideIn 0.2s ease-out;
      }
      @keyframes slideIn {
        from {
          opacity: 0;
          transform: translateX(-10px);
        }
        to {
          opacity: 1;
          transform: translateX(0);
        }
      }
      .session-item:active {
        transform: scale(0.98);
      }
      .session-item:hover {
        background: var(--card-background-color);
      }
      .session-item.active {
        background: rgba(var(--rgb-primary-color), 0.15);
        border-left: 3px solid var(--primary-color);
      }
      .session-title {
        font-size: 14px;
        font-weight: 500;
        color: var(--primary-text-color);
        margin-bottom: 4px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        padding-right: 24px;
      }
      .session-preview {
        font-size: 12px;
        color: var(--secondary-text-color);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }
      .session-time {
        font-size: 11px;
        color: var(--disabled-text-color);
        margin-top: 4px;
      }
      .session-delete {
        position: absolute;
        top: 4px;
        right: 4px;
        width: 32px;
        height: 32px;
        min-width: 44px;
        min-height: 44px;
        border: none;
        background: transparent;
        cursor: pointer;
        opacity: 0;
        transition: opacity 0.2s;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 0;
      }
      .session-item:hover .session-delete {
        opacity: 1;
      }
      .session-delete:hover {
        background: rgba(var(--rgb-error-color), 0.2);
      }
      .session-delete ha-icon {
        --mdc-icon-size: 16px;
        color: var(--secondary-text-color);
      }
      .session-delete:hover ha-icon {
        color: var(--error-color);
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
      .menu-toggle ha-icon {
        --mdc-icon-size: 24px;
        color: var(--primary-text-color);
      }
      .sidebar-overlay {
        display: none;
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0, 0, 0, 0.5);
        z-index: 99;
      }
      .sidebar-overlay.show {
        display: block;
      }
      .empty-sessions {
        text-align: center;
        padding: 32px 16px;
        color: var(--secondary-text-color);
      }
      .empty-sessions ha-icon {
        --mdc-icon-size: 48px;
        color: var(--disabled-text-color);
        margin-bottom: 12px;
      }
      .empty-chat {
        flex: 1;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 32px;
        color: var(--secondary-text-color);
      }
      .empty-chat ha-icon {
        --mdc-icon-size: 64px;
        color: var(--disabled-text-color);
        margin-bottom: 16px;
      }
      .empty-chat h3 {
        font-size: 18px;
        font-weight: 500;
        color: var(--primary-text-color);
        margin-bottom: 8px;
      }
      .empty-chat p {
        font-size: 14px;
        margin-bottom: 24px;
        max-width: 300px;
      }
      .empty-chat button {
        padding: 12px 24px;
        background: var(--primary-color);
        color: white;
        border: none;
        border-radius: 8px;
        font-size: 14px;
        font-weight: 500;
        cursor: pointer;
        transition: filter 0.2s, transform 0.1s;
        font-family: inherit;
      }
      .empty-chat button:hover {
        filter: brightness(1.1);
      }
      .empty-chat button:active {
        transform: scale(0.98);
      }
      .session-skeleton {
        padding: 12px;
        margin-bottom: 4px;
      }
      .skeleton-line {
        height: 14px;
        background: linear-gradient(90deg, var(--divider-color) 25%, var(--card-background-color) 50%, var(--divider-color) 75%);
        background-size: 200% 100%;
        animation: skeleton-shimmer 1.5s infinite;
        border-radius: 4px;
        margin-bottom: 8px;
      }
      .skeleton-line.short {
        width: 60%;
        height: 12px;
      }
      .skeleton-line.tiny {
        width: 40%;
        height: 10px;
      }
      @keyframes skeleton-shimmer {
        0% { background-position: 200% 0; }
        100% { background-position: -200% 0; }
      }
      @media (max-width: 768px) {
        .sidebar {
          position: fixed;
          left: 0;
          top: 0;
          bottom: 0;
          z-index: 100;
          transform: translateX(-100%);
          width: 280px;
        }
        .sidebar.open {
          transform: translateX(0);
          box-shadow: var(--shadow-elevation-4dp, 0 4px 5px 0 rgba(0,0,0,.14));
        }
        .sidebar.hidden {
          transform: translateX(-100%);
        }
        .menu-toggle {
          display: flex;
        }

        /* Issue 1: Header overlap fix - hide robot icon on mobile */
        .header > ha-icon {
          display: none;
        }

        /* Issue 2: Sidebar visibility - container overflow */
        .main-container {
          overflow: hidden;
        }

        /* Issue 3: New Chat button - icon only */
        .new-chat-btn {
          width: 44px;
          height: 44px;
          min-width: 44px;
          min-height: 44px;
          padding: 0;
          font-size: 0;
          gap: 0;
        }
        .new-chat-btn ha-icon {
          --mdc-icon-size: 24px;
        }

        /* Issue 4: Input footer - compact layout */
        .input-container {
          padding: 12px;
          padding-bottom: calc(12px + env(safe-area-inset-bottom, 0px));
        }
        .input-footer {
          gap: 8px;
        }
        .provider-selector {
          flex-shrink: 0;
        }
        .provider-label {
          display: none;
        }
        .provider-button {
          width: 44px;
          min-width: 44px;
          height: 44px;
          padding: 4px;
          font-size: 0;
          border-radius: 50%;
        }
        .thinking-toggle {
          display: none;
        }
        .send-button {
          min-width: 44px;
          --mdc-button-height: 44px;
        }
      }
    `;
  }

  constructor() {
    super();
    this._messages = [];
    this._isLoading = false;
    this._error = null;
    this._pendingAutomation = null;
    this._selectedProvider = null;
    this._availableProviders = [];
    this._showProviderDropdown = false;
    this._selectedModel = null;
    this._availableModels = [];
    this.providersLoaded = false;
    this._eventSubscriptionSetup = false;
    this._serviceCallTimeout = null;
    this._showThinking = false;
    this._thinkingExpanded = false;
    this._debugInfo = null;
    this._sessions = [];
    this._activeSessionId = null;
    this._sidebarOpen = window.innerWidth > 768;
    this._sessionsLoading = true;
    this._unsubscribeEvents = null;
    this._markdownCache = new Map();
    
    // Bind event handlers for proper cleanup
    this._handleDocumentClick = this._handleDocumentClick.bind(this);
    this._handleWindowResize = this._handleWindowResize.bind(this);
  }
  
  _renderMarkdown(text) {
    if (!text) return '';
    if (this._markdownCache.has(text)) {
      return this._markdownCache.get(text);
    }
    try {
      const rawHtml = marked.parse(text);
      const sanitized = DOMPurify.sanitize(rawHtml, {
        ALLOWED_TAGS: ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'br', 'ul', 'ol', 'li',
                       'strong', 'em', 'code', 'pre', 'blockquote', 'a', 'span', 'hr'],
        ALLOWED_ATTR: ['href', 'target', 'rel', 'class']
      });
      if (this._markdownCache.size > 100) {
        const firstKey = this._markdownCache.keys().next().value;
        this._markdownCache.delete(firstKey);
      }
      this._markdownCache.set(text, sanitized);
      return sanitized;
    } catch (e) {
      return text;
    }
  }
  
  _handleDocumentClick(e) {
    if (!this.shadowRoot.querySelector('.provider-selector')?.contains(e.target)) {
      this._showProviderDropdown = false;
    }
  }
  
  _handleWindowResize() {
    if (window.innerWidth > 768 && !this._sidebarOpen) {
      this._sidebarOpen = true;
    }
  }

  async _loadSessions() {
    if (!this.hass) return;
    this._sessionsLoading = true;
    try {
      const result = await this.hass.callWS({
        type: "ai_agent_ha/sessions/list"
      });
      this._sessions = result.sessions || [];
      if (this._sessions.length > 0 && !this._activeSessionId) {
        await this._selectSession(this._sessions[0].session_id);
      }
    } catch (error) {
      console.error("Failed to load sessions:", error);
      this._error = "Could not load conversations";
    } finally {
      this._sessionsLoading = false;
    }
  }

  async _selectSession(sessionId) {
    this._activeSessionId = sessionId;
    this._isLoading = true;
    this._error = null;
    try {
      const result = await this.hass.callWS({
        type: "ai_agent_ha/sessions/get",
        session_id: sessionId
      });
      this._messages = (result.messages || []).map(m => ({
        type: m.role === "user" ? "user" : "assistant",
        text: m.content,
        automation: m.metadata?.automation,
        dashboard: m.metadata?.dashboard,
        timestamp: m.timestamp,
        status: m.status,
        error_message: m.error_message
      }));
      if (window.innerWidth <= 768) {
        this._sidebarOpen = false;
      }
    } catch (error) {
      console.error("Failed to load session:", error);
      this._error = "Could not load conversation";
    } finally {
      this._isLoading = false;
    }
  }

  async _createNewSession() {
    if (!this._selectedProvider) {
      this._error = "Please select a provider first";
      return;
    }
    try {
      const result = await this.hass.callWS({
        type: "ai_agent_ha/sessions/create",
        provider: this._selectedProvider
      });
      this._sessions = [result, ...this._sessions];
      this._activeSessionId = result.session_id;
      this._messages = [];
      if (window.innerWidth <= 768) {
        this._sidebarOpen = false;
      }
    } catch (error) {
      console.error("Failed to create session:", error);
      this._error = "Could not create new conversation";
    }
  }

  async _deleteSession(e, sessionId) {
    e.stopPropagation();
    if (!confirm("Delete this conversation?")) return;
    
    try {
      await this.hass.callWS({
        type: "ai_agent_ha/sessions/delete",
        session_id: sessionId
      });
      this._sessions = this._sessions.filter(s => s.session_id !== sessionId);
      
      if (this._activeSessionId === sessionId) {
        if (this._sessions.length > 0) {
          await this._selectSession(this._sessions[0].session_id);
        } else {
          this._activeSessionId = null;
          this._messages = [];
        }
      }
    } catch (error) {
      console.error("Failed to delete session:", error);
      this._error = "Could not delete conversation";
    }
  }

  _toggleSidebar() {
    this._sidebarOpen = !this._sidebarOpen;
  }

  _formatSessionTime(timestamp) {
    if (!timestamp) return "";
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now - date;
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));
    
    if (diffDays === 0) {
      return date.toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit", hour12: true });
    } else if (diffDays === 1) {
      return "Yesterday";
    } else if (diffDays < 7) {
      return `${diffDays} days ago`;
    } else {
      return date.toLocaleDateString("en-US", { month: "short", day: "numeric" });
    }
  }

  _updateSessionInList(sessionId, preview, title) {
    this._sessions = this._sessions.map(s => {
      if (s.session_id === sessionId) {
        return {
          ...s,
          preview: preview ? preview.substring(0, 100) : s.preview,
          title: title || s.title,
          message_count: (s.message_count || 0) + 2,
          updated_at: new Date().toISOString()
        };
      }
      return s;
    });
    this._sessions = [...this._sessions].sort(
      (a, b) => new Date(b.updated_at) - new Date(a.updated_at)
    );
  }

  async connectedCallback() {
    super.connectedCallback();
    if (this.hass && !this._eventSubscriptionSetup) {
      this._eventSubscriptionSetup = true;
      this._unsubscribeEvents = await this.hass.connection.subscribeEvents(
        (event) => this._handleLlamaResponse(event),
        'ai_agent_ha_response'
      );
      await this._loadSessions();
    }

    document.addEventListener('click', this._handleDocumentClick);
    window.addEventListener('resize', this._handleWindowResize);
  }

  disconnectedCallback() {
    super.disconnectedCallback();
    if (this._unsubscribeEvents) {
      this._unsubscribeEvents();
      this._unsubscribeEvents = null;
      this._eventSubscriptionSetup = false;
    }
    document.removeEventListener('click', this._handleDocumentClick);
    window.removeEventListener('resize', this._handleWindowResize);
  }

  async updated(changedProps) {
    // Set up event subscription when hass becomes available
    if (changedProps.has('hass') && this.hass && !this._eventSubscriptionSetup) {
      this._eventSubscriptionSetup = true;
      this._unsubscribeEvents = await this.hass.connection.subscribeEvents(
        (event) => this._handleLlamaResponse(event),
        'ai_agent_ha_response'
      );
    }

    // Load providers when hass becomes available
    if (changedProps.has('hass') && this.hass && !this.providersLoaded) {
      this.providersLoaded = true;

      try {
        // Uses the WebSocket API to get all entries with their complete data
        const allEntries = await this.hass.callWS({ type: 'config_entries/get' });

        const aiAgentEntries = allEntries.filter(
          entry => entry.domain === 'ai_agent_ha'
        );

        if (aiAgentEntries.length > 0) {
          const providers = aiAgentEntries
            .map(entry => {
              const provider = this._resolveProviderFromEntry(entry);
              if (!provider) return null;

              return {
                value: provider,
                label: PROVIDERS[provider] || provider
              };
            })
            .filter(Boolean);

          this._availableProviders = providers;

          if (
            (!this._selectedProvider || !providers.find(p => p.value === this._selectedProvider)) &&
            providers.length > 0
          ) {
            this._selectedProvider = providers[0].value;
          }

          // Always fetch models for the selected provider
          if (this._selectedProvider) {
            this._fetchAvailableModels(this._selectedProvider);
          }
        } else {
          this._availableProviders = [];
        }
      } catch (error) {
        console.error("Error fetching config entries via WebSocket:", error);
        this._error = error.message || 'Failed to load AI provider configurations.';
        this._availableProviders = [];
      }
      this.requestUpdate();
    }

    // Load sessions when hass becomes available
    if (changedProps.has('hass') && this.hass && this._sessions.length === 0) {
      await this._loadSessions();
    }

    if (changedProps.has('_messages') || changedProps.has('_isLoading')) {
      this._scrollToBottom();
    }
  }

  _renderSidebar() {
    const sidebarClass = window.innerWidth <= 768 
      ? (this._sidebarOpen ? "sidebar open" : "sidebar hidden")
      : (this._sidebarOpen ? "sidebar" : "sidebar hidden");
    
    return html`
      <div 
        class="sidebar-overlay ${this._sidebarOpen && window.innerWidth <= 768 ? 'show' : ''}"
        @click=${this._toggleSidebar}
      ></div>
      <aside class="${sidebarClass}">
        <div class="sidebar-header">
          <div class="sidebar-title">
            <ha-icon icon="mdi:chat"></ha-icon>
            Conversations
          </div>
          <button class="new-chat-btn" @click=${this._createNewSession}>
            <ha-icon icon="mdi:plus"></ha-icon>
            New Chat
          </button>
        </div>
        <div class="session-list">
          ${this._sessionsLoading ? html`
            ${[1, 2, 3].map(() => html`
              <div class="session-skeleton">
                <div class="skeleton-line"></div>
                <div class="skeleton-line short"></div>
                <div class="skeleton-line tiny"></div>
              </div>
            `)}
          ` : this._sessions.length === 0 ? html`
            <div class="empty-sessions">
              <ha-icon icon="mdi:chat-outline"></ha-icon>
              <p>No conversations yet</p>
            </div>
          ` : this._sessions.map(session => html`
            <div 
              class="session-item ${session.session_id === this._activeSessionId ? 'active' : ''}"
              @click=${() => this._selectSession(session.session_id)}
            >
              <span class="session-title">${session.title || "New Conversation"}</span>
              <span class="session-preview">${session.preview || "Start typing..."}</span>
              <span class="session-time">${this._formatSessionTime(session.updated_at)}</span>
              <button 
                class="session-delete"
                @click=${(e) => this._deleteSession(e, session.session_id)}
              >
                <ha-icon icon="mdi:delete"></ha-icon>
              </button>
            </div>
          `)}
        </div>
      </aside>
    `;
  }

  render() {
    return html`
      <div class="header">
        <button class="menu-toggle" @click=${this._toggleSidebar}>
          <ha-icon icon="mdi:menu"></ha-icon>
        </button>
        <ha-icon icon="mdi:robot"></ha-icon>
        AI Agent HA
        <button
          class="clear-button"
          @click=${this._clearChat}
          ?disabled=${this._isLoading}
        >
          <ha-icon icon="mdi:delete-sweep"></ha-icon>
          <span>Clear Chat</span>
        </button>
      </div>
      <div class="main-container">
        ${this._renderSidebar()}
        <div class="content">
          <div class="chat-container">
          <div class="messages" id="messages">
            ${this._messages.length === 0 && !this._isLoading ? html`
              <div class="empty-chat">
                <ha-icon icon="mdi:chat-outline"></ha-icon>
                <h3>Start a conversation</h3>
                <p>Ask your AI assistant about your Home Assistant setup, automations, or devices.</p>
              </div>
            ` : ''}
            ${this._messages.map(msg => html`
              <div class="message ${msg.type}-message">
                <div class="message-content">${msg.type === 'assistant' ? unsafeHTML(this._renderMarkdown(msg.text)) : msg.text}</div>
                ${msg.automation ? html`
                  <div class="automation-suggestion">
                    <div class="automation-title">${msg.automation.alias}</div>
                    <div class="automation-description">${msg.automation.description}</div>
                    <div class="automation-details">
                      ${JSON.stringify(msg.automation, null, 2)}
                    </div>
                    <div class="automation-actions">
                      <ha-button
                        @click=${() => this._approveAutomation(msg.automation)}
                        .disabled=${this._isLoading}
                      >Approve</ha-button>
                      <ha-button
                        @click=${() => this._rejectAutomation()}
                        .disabled=${this._isLoading}
                      >Reject</ha-button>
                    </div>
                  </div>
                ` : ''}
                ${msg.dashboard ? html`
                  <div class="dashboard-suggestion">
                    <div class="dashboard-title">${msg.dashboard.title}</div>
                    <div class="dashboard-description">Dashboard with ${msg.dashboard.views ? msg.dashboard.views.length : 0} view(s)</div>
                    <div class="dashboard-details">
                      ${JSON.stringify(msg.dashboard, null, 2)}
                    </div>
                    <div class="dashboard-actions">
                      <ha-button
                        @click=${() => this._approveDashboard(msg.dashboard)}
                        .disabled=${this._isLoading}
                      >Create Dashboard</ha-button>
                      <ha-button
                        @click=${() => this._rejectDashboard()}
                        .disabled=${this._isLoading}
                      >Cancel</ha-button>
                    </div>
                  </div>
                ` : ''}
              </div>
            `)}
            ${this._isLoading ? html`
              <div class="loading">
                <span>AI Agent is thinking</span>
                <div class="loading-dots">
                  <div class="dot"></div>
                  <div class="dot"></div>
                  <div class="dot"></div>
                </div>
              </div>
            ` : ''}
            ${this._error ? html`
              <div class="error">
                <ha-icon icon="mdi:alert-circle"></ha-icon>
                <span class="error-message">${this._error}</span>
                <button class="error-dismiss" @click=${() => this._error = null}>
                  <ha-icon icon="mdi:close"></ha-icon>
                </button>
              </div>
            ` : ''}
            ${this._showThinking ? this._renderThinkingPanel() : ''}
          </div>
          <div class="input-container">
            <div class="input-main">
              <div class="input-wrapper">
                <textarea
                  id="prompt"
                  placeholder="Ask me anything about your Home Assistant..."
                  ?disabled=${this._isLoading}
                  @keydown=${this._handleKeyDown}
                  @input=${this._autoResize}
                ></textarea>
              </div>
            </div>

            <div class="input-footer">
              <div class="provider-selector">
                <span class="provider-label">Provider:</span>
                <select
                  class="provider-button"
                  @change=${(e) => this._selectProvider(e.target.value)}
                  .value=${this._selectedProvider || ''}
                >
                  ${this._availableProviders.map(provider => html`
                    <option
                      value=${provider.value}
                      ?selected=${provider.value === this._selectedProvider}
                    >
                      ${provider.label}
                    </option>
                  `)}
                </select>
              </div>
              ${this._availableModels.length > 0 ? html`
              <div class="provider-selector">
                <span class="provider-label">Model:</span>
                <select
                  class="provider-button"
                  @change=${(e) => this._selectModel(e.target.value)}
                  .value=${this._selectedModel || ''}
                >
                  ${this._availableModels.map(model => html`
                    <option
                      value=${model.id}
                      ?selected=${model.id === this._selectedModel}
                    >
                      ${model.name}
                    </option>
                  `)}
                </select>
              </div>
              ` : ''}
              <label class="thinking-toggle">
                <input
                  type="checkbox"
                  .checked=${this._showThinking}
                  @change=${(e) => this._toggleShowThinking(e)}
                />
                Show thinking
              </label>

              <ha-button
                class="send-button"
                @click=${this._sendMessage}
                .disabled=${this._isLoading || !this._hasProviders()}
              >
                <ha-icon icon="mdi:send"></ha-icon>
              </ha-button>
            </div>
          </div>
        </div>
        </div>
      </div>
    `;
  }

  _scrollToBottom() {
    const messages = this.shadowRoot.querySelector('#messages');
    if (messages) {
      messages.scrollTop = messages.scrollHeight;
    }
  }

  _autoResize(e) {
    const textarea = e.target;
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px';
  }

  _handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey && !this._isLoading) {
      e.preventDefault();
      this._sendMessage();
    }
  }

  _toggleProviderDropdown() {
    this._showProviderDropdown = !this._showProviderDropdown;
    this.requestUpdate();
  }

  async _selectProvider(provider) {
    console.log("[AI Agent] Provider selected:", provider);
    this._selectedProvider = provider;
    
    // Clear models immediately to hide old dropdown
    this._availableModels = [];
    this._selectedModel = null;
    this.requestUpdate();
    
    console.log("[AI Agent] Before fetch - availableModels count:", this._availableModels.length);
    await this._fetchAvailableModels(provider);
    console.log("[AI Agent] After fetch - availableModels count:", this._availableModels.length);
    console.log("[AI Agent] After fetch - availableModels:", this._availableModels);
    console.log("[AI Agent] requestUpdate() called");
  }

  async _fetchAvailableModels(provider) {
    console.log("[AI Agent] Fetching models for provider:", provider);
    console.log("[AI Agent] Request payload:", { type: "ai_agent_ha/models/list", provider: provider });
    
    // Clear models first to force re-render
    this._availableModels = [];
    this._selectedModel = null;
    this.requestUpdate();
    
    try {
      const result = await this.hass.callWS({
        type: "ai_agent_ha/models/list",
        provider: provider
      });
      console.log("[AI Agent] Models response:", result);
      console.log("[AI Agent] Models list:", result.models);
      
      // Create new array reference to trigger reactivity
      this._availableModels = [...(result.models || [])];
      
      // Select default model or first available
      const defaultModel = this._availableModels.find(m => m.default);
      this._selectedModel = defaultModel ? defaultModel.id : (this._availableModels[0]?.id || null);
      console.log("[AI Agent] Available models:", this._availableModels.length, "Selected:", this._selectedModel);
      this.requestUpdate();
    } catch (e) {
      console.warn("[AI Agent] Could not fetch available models:", e);
      this._availableModels = [];
      this._selectedModel = null;
      this.requestUpdate();
    }
  }

  _selectModel(modelId) {
    this._selectedModel = modelId;
    this.requestUpdate();
  }

  _getSelectedProviderLabel() {
    const provider = this._availableProviders.find(p => p.value === this._selectedProvider);
    return provider ? provider.label : 'Select Model';
  }

  async _sendMessage() {
    const promptEl = this.shadowRoot.querySelector('#prompt');
    const prompt = promptEl.value.trim();
    if (!prompt || this._isLoading) return;

    promptEl.value = '';
    promptEl.style.height = 'auto';
    this._isLoading = true;
    this._error = null;
    this._debugInfo = null;
    this._thinkingExpanded = false;

    if (this._serviceCallTimeout) {
      clearTimeout(this._serviceCallTimeout);
    }

    if (!this._activeSessionId) {
      try {
        await this._createNewSession();
      } catch (error) {
        console.error("Failed to create session:", error);
        this._clearLoadingState();
        this._error = "Could not create conversation";
        return;
      }
    }

    if (this._activeSessionId) {
      await this._sendMessageViaWebSocket(prompt);
    } else {
      await this._sendMessageViaService(prompt);
    }
  }

  async _sendMessageViaWebSocket(prompt) {
    this._messages = [...this._messages, { type: 'user', text: prompt }];
    
    try {
      const wsParams = {
        type: "ai_agent_ha/chat/send",
        session_id: this._activeSessionId,
        message: prompt,
        provider: this._selectedProvider,
        debug: this._showThinking
      };
      // Add model if selected (for providers that support model selection)
      if (this._selectedModel) {
        wsParams.model = this._selectedModel;
      }
      const result = await this.hass.callWS(wsParams);

      this._clearLoadingState();

      if (result.assistant_message) {
        let content = result.assistant_message.content || '';
        let automation = result.assistant_message.metadata?.automation;
        let dashboard = result.assistant_message.metadata?.dashboard;

        // Parse JSON response if content is pure JSON (starts with {)
        // Don't try to extract JSON from markdown - it may contain code blocks with JSON examples
        const trimmedContent = content.trim();
        if (trimmedContent.startsWith('{') && trimmedContent.endsWith('}')) {
          try {
            const parsed = JSON.parse(trimmedContent);
            if (parsed.request_type === 'automation_suggestion') {
              automation = parsed.automation;
              content = parsed.message || 'I found an automation that might help you.';
            } else if (parsed.request_type === 'dashboard_suggestion') {
              dashboard = parsed.dashboard;
              content = parsed.message || 'I created a dashboard configuration for you.';
            } else if (parsed.request_type === 'final_response') {
              content = parsed.response || parsed.message || content;
            } else if (parsed.message) {
              content = parsed.message;
            } else if (parsed.response) {
              content = parsed.response;
            }
          } catch (e) {
            // Not valid JSON, use content as-is
          }
        }

        const assistantMsg = {
          type: 'assistant',
          text: content,
          automation: automation,
          dashboard: dashboard,
          status: result.assistant_message.status,
          error_message: result.assistant_message.error_message
        };

        if (result.assistant_message.status === 'error') {
          this._error = result.assistant_message.error_message;
          assistantMsg.text = `Error: ${result.assistant_message.error_message}`;
        }

        this._messages = [...this._messages, assistantMsg];
        this._debugInfo = this._showThinking ? result.assistant_message.metadata?.debug : null;
        if (this._showThinking && this._debugInfo) {
          this._thinkingExpanded = true;
        }
      }

      const sessionTitle = this._sessions.find(s => s.session_id === this._activeSessionId)?.title;
      const isNewConversation = sessionTitle === "New Conversation";
      this._updateSessionInList(
        this._activeSessionId, 
        prompt,
        isNewConversation ? prompt.substring(0, 40) + (prompt.length > 40 ? "..." : "") : null
      );

      this._scrollToBottom();

    } catch (error) {
      console.error("WebSocket error, falling back to service:", error);
      this._messages = this._messages.slice(0, -1);
      await this._sendMessageViaService(prompt);
    }
  }

  async _sendMessageViaService(prompt) {
    this._messages = [...this._messages, { type: 'user', text: prompt }];

    this._serviceCallTimeout = setTimeout(() => {
      if (this._isLoading) {
        console.warn("Service call timeout - clearing loading state");
        this._isLoading = false;
        this._error = 'Request timed out. Please try again.';
        this._messages = [...this._messages, {
          type: 'assistant',
          text: 'Sorry, the request timed out. Please try again.'
        }];
        this.requestUpdate();
      }
    }, 60000);

    try {
      await this.hass.callService('ai_agent_ha', 'query', {
        prompt: prompt,
        provider: this._selectedProvider,
        debug: this._showThinking
      });
    } catch (error) {
      console.error("Error calling service:", error);
      this._clearLoadingState();
      this._error = error.message || 'An error occurred while processing your request';
      this._messages = [...this._messages, {
        type: 'assistant',
        text: `Error: ${this._error}`
      }];
    }
  }

  _clearLoadingState() {
    this._isLoading = false;
    if (this._serviceCallTimeout) {
      clearTimeout(this._serviceCallTimeout);
      this._serviceCallTimeout = null;
    }
  }

  _handleLlamaResponse(event) {
    try {
      this._clearLoadingState();
      this._debugInfo = this._showThinking ? (event.data.debug || null) : null;
      if (this._showThinking && this._debugInfo) {
        this._thinkingExpanded = true;
      }
    if (event.data.success) {
      if (!event.data.answer || event.data.answer.trim() === '') {
        this._messages = [
          ...this._messages,
          { type: 'assistant', text: 'I received your message but I\'m not sure how to respond. Could you please try rephrasing your question?' }
        ];
        return;
      }

      let message = { type: 'assistant', text: event.data.answer };

      const trimmedAnswer = (event.data.answer || '').trim();
      if (trimmedAnswer.startsWith('{') && trimmedAnswer.endsWith('}')) {
        try {
          const response = JSON.parse(trimmedAnswer);
          if (response.request_type === 'automation_suggestion') {
            message.automation = response.automation;
            message.text = response.message || 'I found an automation that might help you. Would you like me to create it?';
          } else if (response.request_type === 'dashboard_suggestion') {
            message.dashboard = response.dashboard;
            message.text = response.message || 'I created a dashboard configuration for you. Would you like me to create it?';
          } else if (response.request_type === 'final_response') {
            message.text = response.response || response.message || event.data.answer;
          } else if (response.message) {
            message.text = response.message;
          } else if (response.response) {
            message.text = response.response;
          }
        } catch (e) {
          // Not valid JSON, use as-is
        }
      }

      this._messages = [...this._messages, message];
    } else {
      this._error = event.data.error || 'An error occurred';
      this._messages = [
        ...this._messages,
        { type: 'assistant', text: `Error: ${this._error}` }
      ];
    }
    } catch (error) {
      console.error("Error in _handleLlamaResponse:", error);
      this._clearLoadingState();
      this._error = 'An error occurred while processing the response';
      this._messages = [...this._messages, {
        type: 'assistant',
        text: 'Sorry, an error occurred while processing the response. Please try again.'
      }];
      this.requestUpdate();
    }
  }

  async _approveAutomation(automation) {
    if (this._isLoading) return;
    this._isLoading = true;
    try {
      const result = await this.hass.callService('ai_agent_ha', 'create_automation', {
        automation: automation
      });

      if (result && result.message) {
        this._messages = [...this._messages, {
          type: 'assistant',
          text: result.message
        }];
      } else {
        // Fallback success message if no message is provided
        this._messages = [...this._messages, {
          type: 'assistant',
          text: `Automation "${automation.alias}" has been created successfully!`
        }];
      }
    } catch (error) {
      console.error("Error creating automation:", error);
      this._error = error.message || 'An error occurred while creating the automation';
      this._messages = [...this._messages, {
        type: 'assistant',
        text: `Error: ${this._error}`
      }];
    } finally {
      this._clearLoadingState();
    }
  }

  _rejectAutomation() {
    this._messages = [...this._messages, {
      type: 'assistant',
      text: 'Automation creation cancelled. Would you like to try something else?'
    }];
  }

  async _approveDashboard(dashboard) {
    if (this._isLoading) return;
    this._isLoading = true;
    try {
      const result = await this.hass.callService('ai_agent_ha', 'create_dashboard', {
        dashboard_config: dashboard
      });

      if (result && result.message) {
        this._messages = [...this._messages, {
          type: 'assistant',
          text: result.message
        }];
      } else {
        // Fallback success message if no message is provided
        this._messages = [...this._messages, {
          type: 'assistant',
          text: `Dashboard "${dashboard.title}" has been created successfully!`
        }];
      }
    } catch (error) {
      console.error("Error creating dashboard:", error);
      this._error = error.message || 'An error occurred while creating the dashboard';
      this._messages = [...this._messages, {
        type: 'assistant',
        text: `Error: ${this._error}`
      }];
    } finally {
      this._clearLoadingState();
    }
  }

  _rejectDashboard() {
    this._messages = [...this._messages, {
      type: 'assistant',
      text: 'Dashboard creation cancelled. Would you like me to create a different dashboard?'
    }];
  }

  shouldUpdate(changedProps) {
    return changedProps.has('_messages') ||
           changedProps.has('_isLoading') ||
           changedProps.has('_error') ||
           changedProps.has('_availableProviders') ||
           changedProps.has('_selectedProvider') ||
           changedProps.has('_showProviderDropdown') ||
           changedProps.has('_sessions') ||
           changedProps.has('_activeSessionId') ||
           changedProps.has('_sidebarOpen') ||
           changedProps.has('_sessionsLoading');
  }

  async _clearChat() {
    if (this._activeSessionId && this.hass) {
      try {
        await this.hass.callWS({
          type: "ai_agent_ha/sessions/delete",
          session_id: this._activeSessionId
        });
        this._sessions = this._sessions.filter(s => s.session_id !== this._activeSessionId);
      } catch (error) {
        console.error("Failed to delete session:", error);
      }
    }
    this._activeSessionId = null;
    this._messages = [];
    this._markdownCache.clear();
    this._clearLoadingState();
    this._error = null;
    this._pendingAutomation = null;
    this._debugInfo = null;
  }

  _resolveProviderFromEntry(entry) {
    if (!entry) return null;

    const providerFromData = entry.data?.ai_provider || entry.options?.ai_provider;
    console.log("[AI Agent] Resolving provider from entry:", entry.title, "ai_provider:", providerFromData);
    if (providerFromData && PROVIDERS[providerFromData]) {
      console.log("[AI Agent] Resolved provider:", providerFromData);
      return providerFromData;
    }

    const uniqueId = entry.unique_id || entry.uniqueId;
    if (uniqueId && uniqueId.startsWith("ai_agent_ha_")) {
      const fromUniqueId = uniqueId.replace("ai_agent_ha_", "");
      if (PROVIDERS[fromUniqueId]) {
        return fromUniqueId;
      }
    }

    const titleMap = {
      "ai agent ha (openrouter)": "openrouter",
      "ai agent ha (google gemini)": "gemini",
      "ai agent ha (openai)": "openai",
      "ai agent ha (llama)": "llama",
      "ai agent ha (anthropic (claude))": "anthropic",
      "ai agent ha (alter)": "alter",
      "ai agent ha (z.ai)": "zai",
      "ai agent ha (local model)": "local",
    };

    if (entry.title) {
      const lowerTitle = entry.title.toLowerCase();
      if (titleMap[lowerTitle]) {
        return titleMap[lowerTitle];
      }

      const match = entry.title.match(/\(([^)]+)\)/);
      if (match && match[1]) {
        const normalized = match[1].toLowerCase().replace(/[^a-z0-9]/g, "");
        const providerKey = Object.keys(PROVIDERS).find(
          key => key.replace(/[^a-z0-9]/g, "") === normalized
        );
        if (providerKey) {
          return providerKey;
        }
      }
    }

    return null;
  }

  _getProviderInfo(providerId) {
    return this._availableProviders.find(p => p.value === providerId);
  }

  _hasProviders() {
    return this._availableProviders && this._availableProviders.length > 0;
  }

  _toggleThinkingPanel() {
    this._thinkingExpanded = !this._thinkingExpanded;
  }

  _toggleShowThinking(e) {
    this._showThinking = e.target.checked;
    if (!this._showThinking) {
      this._thinkingExpanded = false;
    }
  }

  _renderThinkingPanel() {
    if (!this._debugInfo) {
      return '';
    }

    const subtitleParts = [];
    if (this._debugInfo.provider) subtitleParts.push(this._debugInfo.provider);
    if (this._debugInfo.model) subtitleParts.push(this._debugInfo.model);
    if (this._debugInfo.endpoint_type) subtitleParts.push(this._debugInfo.endpoint_type);
    const subtitle = subtitleParts.join(" · ");
    const conversation = this._debugInfo.conversation || [];

    return html`
      <div class="thinking-panel">
        <div class="thinking-header" @click=${() => this._toggleThinkingPanel()}>
          <div>
            <span class="thinking-title">Thinking trace</span>
            ${subtitle ? html`<span class="thinking-subtitle">${subtitle}</span>` : ''}
          </div>
          <ha-icon icon=${this._thinkingExpanded ? 'mdi:chevron-up' : 'mdi:chevron-down'}></ha-icon>
        </div>
        ${this._thinkingExpanded ? html`
          <div class="thinking-body">
            ${conversation.length === 0 ? html`
              <div class="thinking-empty">No trace captured.</div>
            ` : conversation.map((entry, index) => html`
              <div class="thinking-entry">
                <div class="badge">${entry.role || 'unknown'}</div>
                <pre>${entry.content || ''}</pre>
              </div>
            `)}
          </div>
        ` : ''}
      </div>
    `;
  }
}

customElements.define("ai_agent_ha-panel", AiAgentHaPanel);
