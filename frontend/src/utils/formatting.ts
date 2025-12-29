/**
 * Shared formatting utilities for display values
 */

/**
 * Format a number as hexadecimal with optional padding
 * @param value - The number to format (null/undefined returns "---")
 * @param digits - Minimum number of hex digits (default 3)
 * @param prefix - Whether to include "0x" prefix (default true)
 */
export function formatHex(
  value: number | null | undefined,
  digits: number = 3,
  prefix: boolean = true
): string {
  if (value === null || value === undefined) return "---";
  const hex = value.toString(16).toUpperCase().padStart(digits, "0");
  return prefix ? `0x${hex}` : hex;
}

/**
 * Format duration in seconds to MM:SS format
 */
export function formatDuration(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, "0")}`;
}

/**
 * Format a number with optional unit suffix
 */
export function formatWithUnit(value: number | null, unit: string, decimals: number = 1): string {
  if (value === null) return "---";
  return `${value.toFixed(decimals)} ${unit}`;
}
