/**
 * Utilities for generating meaningful CSS selectors for elements
 */

/**
 * Generate a CSS selector for an element, prioritizing meaningful identifiers
 *
 * Priority order:
 * 1. data-testid attribute
 * 2. id attribute
 * 3. Tag + classes (up to 3 classes)
 * 4. Path-based selector (max 3 levels)
 */
export function generateSelector(element: HTMLElement): string {
  // Try data-testid first (most reliable for debugging)
  const testId = element.getAttribute('data-testid');
  if (testId) {
    return `[data-testid="${testId}"]`;
  }

  // Try id
  if (element.id) {
    return `#${CSS.escape(element.id)}`;
  }

  // Try tag + classes
  const classSelector = getClassSelector(element);
  if (classSelector) {
    // Check if this selector is unique enough
    const matches = document.querySelectorAll(classSelector);
    if (matches.length <= 3) {
      return classSelector;
    }
  }

  // Fall back to path-based selector
  return getPathSelector(element, 3);
}

/**
 * Generate a selector using tag name and class names
 */
function getClassSelector(element: HTMLElement): string | null {
  const tag = element.tagName.toLowerCase();
  const classes = Array.from(element.classList)
    .filter((cls) => !cls.startsWith('__')) // Skip internal classes
    .slice(0, 3); // Max 3 classes

  if (classes.length === 0) {
    return null;
  }

  return `${tag}.${classes.map((c) => CSS.escape(c)).join('.')}`;
}

/**
 * Generate a path-based selector (parent > child chain)
 */
function getPathSelector(element: HTMLElement, maxDepth: number): string {
  const parts: string[] = [];
  let current: HTMLElement | null = element;
  let depth = 0;

  while (current && depth < maxDepth) {
    const part = getElementPart(current);
    parts.unshift(part);

    // Stop if we hit a unique identifier
    if (current.id || current.getAttribute('data-testid')) {
      break;
    }

    current = current.parentElement;
    depth++;
  }

  return parts.join(' > ');
}

/**
 * Get the selector part for a single element
 */
function getElementPart(element: HTMLElement): string {
  // Prefer id
  if (element.id) {
    return `#${CSS.escape(element.id)}`;
  }

  // Prefer data-testid
  const testId = element.getAttribute('data-testid');
  if (testId) {
    return `[data-testid="${testId}"]`;
  }

  const tag = element.tagName.toLowerCase();
  const parent = element.parentElement;

  // If no parent, just return tag
  if (!parent) {
    return tag;
  }

  // Find position among siblings of same type
  const siblings = Array.from(parent.children).filter((child) => child.tagName === element.tagName);

  if (siblings.length === 1) {
    return tag;
  }

  const index = siblings.indexOf(element) + 1;
  return `${tag}:nth-of-type(${index})`;
}

/**
 * Extract first N characters of text content for preview
 */
export function getTextPreview(element: HTMLElement, maxLength = 50): string | undefined {
  const text = element.textContent?.trim();
  if (!text) return undefined;

  if (text.length <= maxLength) {
    return text;
  }

  return text.slice(0, maxLength) + '...';
}
