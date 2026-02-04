/**
 * AI Provider types
 */
export interface Provider {
  value: string;
  label: string;
}

export interface Model {
  id: string;
  name: string;
  default?: boolean;
}

export interface ProviderInfo {
  provider: string;
  models: Model[];
}

/**
 * Provider name mappings
 */
export const PROVIDERS: Record<string, string> = {
  openai: 'OpenAI',
  llama: 'Llama',
  gemini: 'Google Gemini',
  gemini_oauth: 'Gemini (OAuth)',
  openrouter: 'OpenRouter',
  anthropic: 'Anthropic',
  anthropic_oauth: 'Claude Pro/Max',
  alter: 'Alter',
  zai: 'z.ai',
  local: 'Local Model',
};
