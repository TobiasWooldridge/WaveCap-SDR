/**
 * Format frequency in Hz to MHz with proper precision
 */
export function formatFrequencyMHz(hz: number, decimals: number = 3): string {
  const mhz = hz / 1_000_000;
  return mhz.toFixed(decimals);
}

/**
 * Format frequency in Hz to MHz with unit suffix (e.g., "415.3750 MHz")
 */
export function formatFrequencyWithUnit(hz: number | null, decimals: number = 4): string {
  if (hz === null) return "---";
  return `${(hz / 1_000_000).toFixed(decimals)} MHz`;
}

/**
 * Format frequency in Hz to a human-readable string
 */
export function formatFrequency(hz: number): string {
  if (hz >= 1_000_000_000) {
    return `${(hz / 1_000_000_000).toFixed(3)} GHz`;
  } else if (hz >= 1_000_000) {
    return `${(hz / 1_000_000).toFixed(3)} MHz`;
  } else if (hz >= 1_000) {
    return `${(hz / 1_000).toFixed(1)} kHz`;
  }
  return `${hz} Hz`;
}

/**
 * Format sample rate to human-readable string
 */
export function formatSampleRate(hz: number): string {
  if (hz >= 1_000_000) {
    return `${(hz / 1_000_000).toFixed(2)} MHz`;
  } else if (hz >= 1_000) {
    return `${(hz / 1_000).toFixed(0)} kHz`;
  }
  return `${hz} Hz`;
}

/**
 * Format bandwidth to human-readable string
 */
export function formatBandwidth(hz: number): string {
  if (hz >= 1_000_000) {
    return `${(hz / 1_000_000).toFixed(1)} MHz`;
  } else if (hz >= 1_000) {
    return `${(hz / 1_000).toFixed(0)} kHz`;
  }
  return `${hz} Hz`;
}
