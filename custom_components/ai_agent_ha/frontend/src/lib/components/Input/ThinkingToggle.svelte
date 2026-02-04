<script lang="ts">
  import { appState } from "$lib/stores/appState"
import { get } from 'svelte/store';

  function toggleThinking(e: Event) {
    const target = e.target as HTMLInputElement;
    appState.showThinking = target.checked;
    if (!appState.showThinking) {
      appState.thinkingExpanded = false;
    }
  }
</script>

<label class="thinking-toggle">
  <input
    type="checkbox"
    checked={appState.showThinking}
    onchange={toggleThinking}
  />
  Show thinking
</label>

<style>
  .thinking-toggle {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    color: var(--secondary-text-color);
    cursor: pointer;
    user-select: none;
  }

  input {
    margin: 0;
    cursor: pointer;
  }

  @media (max-width: 768px) {
    .thinking-toggle {
      display: none;
    }
  }
</style>
