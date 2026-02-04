import { writable } from 'svelte/store';

/**
 * UI state
 */
export interface UIStateType {
  sidebarOpen: boolean;
  showProviderDropdown: boolean;
}

const initialState: UIStateType = {
  sidebarOpen: typeof window !== 'undefined' ? window.innerWidth > 768 : false,
  showProviderDropdown: false,
};

export const uiState = writable<UIStateType>(initialState);

/**
 * UI actions
 */
export function toggleSidebar() {
  uiState.update(state => ({ ...state, sidebarOpen: !state.sidebarOpen }));
}

export function closeSidebar() {
  uiState.update(state => ({ ...state, sidebarOpen: false }));
}

export function openSidebar() {
  uiState.update(state => ({ ...state, sidebarOpen: true }));
}

export function closeDropdowns() {
  uiState.update(state => ({ ...state, showProviderDropdown: false }));
}
