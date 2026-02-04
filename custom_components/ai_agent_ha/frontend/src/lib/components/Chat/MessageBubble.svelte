<script lang="ts">
  import type { Message } from '$lib/types';
  import { renderMarkdown } from '$lib/services/markdown.service';
  import { sessionState } from "$lib/stores/sessions"

  // Props
  let { message }: { message: Message } = $props();

  // Render markdown for assistant messages
  const renderedContent = $derived(
    message.type === 'assistant' 
      ? renderMarkdown(message.text, sessionState.activeSessionId || undefined)
      : message.text
  );
</script>

<div class="message" class:user={message.type === 'user'} class:assistant={message.type === 'assistant'}>
  <div class="message-content">
    {#if message.type === 'assistant'}
      {@html renderedContent}
    {:else}
      {message.text}
    {/if}
  </div>

  {#if message.automation}
    <slot name="automation" automation={message.automation} />
  {/if}

  {#if message.dashboard}
    <slot name="dashboard" dashboard={message.dashboard} />
  {/if}
</div>

<style>
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

  .user {
    background: var(--primary-color);
    color: var(--text-primary-color, white);
    margin-left: auto;
    border-bottom-right-radius: 4px;
  }

  .assistant {
    background: var(--secondary-background-color);
    margin-right: auto;
    border-bottom-left-radius: 4px;
    color: var(--primary-text-color);
  }

  .message-content {
    word-wrap: break-word;
    overflow-wrap: break-word;
  }

  /* Markdown styling for assistant messages */
  .assistant .message-content :global(h1),
  .assistant .message-content :global(h2),
  .assistant .message-content :global(h3),
  .assistant .message-content :global(h4),
  .assistant .message-content :global(h5),
  .assistant .message-content :global(h6) {
    margin: 0.5em 0 0.3em;
    line-height: 1.3;
    color: var(--primary-text-color);
  }

  .assistant .message-content :global(h1) { font-size: 1.4em; }
  .assistant .message-content :global(h2) { font-size: 1.2em; }
  .assistant .message-content :global(h3) { font-size: 1.1em; }

  .assistant .message-content :global(p) { margin: 0.5em 0; }
  .assistant .message-content :global(p:first-child) { margin-top: 0; }
  .assistant .message-content :global(p:last-child) { margin-bottom: 0; }

  .assistant .message-content :global(ul),
  .assistant .message-content :global(ol) {
    margin: 0.5em 0;
    padding-left: 1.5em;
  }

  .assistant .message-content :global(li) { margin: 0.25em 0; }
  .assistant .message-content :global(strong) { font-weight: 600; }

  .assistant .message-content :global(code) {
    background: rgba(0, 0, 0, 0.06);
    padding: 0.2em 0.4em;
    border-radius: 4px;
    font-family: 'SF Mono', Monaco, Consolas, monospace;
    font-size: 0.85em;
  }

  .assistant .message-content :global(pre) {
    background: var(--primary-background-color);
    border: 1px solid var(--divider-color);
    padding: 12px;
    border-radius: 8px;
    overflow-x: auto;
    margin: 0.5em 0;
  }

  .assistant .message-content :global(pre code) {
    background: none;
    padding: 0;
    font-size: 0.85em;
    line-height: 1.5;
  }

  .assistant .message-content :global(blockquote) {
    border-left: 3px solid var(--primary-color);
    margin: 0.5em 0;
    padding-left: 1em;
    color: var(--secondary-text-color);
  }

  .assistant .message-content :global(a) {
    color: var(--primary-color);
    text-decoration: none;
  }

  .assistant .message-content :global(a:hover) {
    text-decoration: underline;
  }

  .assistant .message-content :global(hr) {
    border: none;
    border-top: 1px solid var(--divider-color);
    margin: 1em 0;
  }

  /* Syntax highlighting */
  .assistant .message-content :global(.hljs) { background: transparent; }
  .assistant .message-content :global(.hljs-keyword) { color: #c678dd; }
  .assistant .message-content :global(.hljs-string) { color: #98c379; }
  .assistant .message-content :global(.hljs-attr) { color: #d19a66; }
  .assistant .message-content :global(.hljs-number) { color: #d19a66; }
  .assistant .message-content :global(.hljs-literal) { color: #56b6c2; }
  .assistant .message-content :global(.hljs-comment) { color: #5c6370; font-style: italic; }

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
</style>
