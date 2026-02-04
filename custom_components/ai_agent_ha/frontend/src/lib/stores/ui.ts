/**
 * UI state using Svelte 5 Runes
 * Controls sidebar, modals, dropdowns, etc.
 */
export const uiState = $state({
  sidebarOpen: typeof window !== 'undefined' ? window.innerWidth > 768 : false,
  showProviderDropdown: false,
});

/**
 * UI actions
 */
export function toggleSidebar() {
  uiState.sidebarOpen = !uiState.sidebarOpen;
}

export function closeSidebar() {
  uiState.sidebarOpen = false;
}

export function openSidebar() {
  uiState.sidebarOpen = true;
}

export function closeDropdowns() {
  uiState.showProviderDropdown = false;
}
