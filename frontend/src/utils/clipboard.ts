/**
 * Copy text to clipboard with fallback for older browsers.
 *
 * Uses the modern Clipboard API when available, falls back to
 * execCommand('copy') for older browsers (mainly older Safari).
 *
 * @param text - The text to copy to clipboard
 * @returns Promise that resolves to true on success, false on failure
 */
export async function copyToClipboard(text: string): Promise<boolean> {
  // Try modern Clipboard API first
  if (navigator.clipboard?.writeText) {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch {
      // Fall through to fallback
    }
  }

  // Fallback for older browsers
  return fallbackCopy(text);
}

/**
 * Fallback copy using execCommand and a temporary textarea.
 * This is needed for older Safari versions.
 */
function fallbackCopy(text: string): boolean {
  try {
    const textarea = document.createElement("textarea");
    textarea.value = text;

    // Make it invisible but still focusable
    textarea.style.position = "fixed";
    textarea.style.left = "-999999px";
    textarea.style.top = "-999999px";
    textarea.style.opacity = "0";

    document.body.appendChild(textarea);
    textarea.focus();
    textarea.select();

    const successful = document.execCommand("copy");
    document.body.removeChild(textarea);

    return successful;
  } catch {
    return false;
  }
}
