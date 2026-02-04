/**
 * JSON parsing utilities
 */

/**
 * Safely parse JSON with fallback
 */
export function safeJSONParse<T = any>(text: string, fallback: T): T {
  try {
    return JSON.parse(text);
  } catch (e) {
    return fallback;
  }
}

/**
 * Check if string is valid JSON
 */
export function isValidJSON(text: string): boolean {
  try {
    JSON.parse(text);
    return true;
  } catch (e) {
    return false;
  }
}

/**
 * Parse JSON response handling both pure JSON and markdown with JSON
 */
export function parseJSONResponse(content: string): {
  isPureJSON: boolean;
  data: any | null;
} {
  const trimmed = content.trim();

  if (trimmed.startsWith('{') && trimmed.endsWith('}')) {
    try {
      return {
        isPureJSON: true,
        data: JSON.parse(trimmed),
      };
    } catch (e) {
      return {
        isPureJSON: false,
        data: null,
      };
    }
  }

  return {
    isPureJSON: false,
    data: null,
  };
}
