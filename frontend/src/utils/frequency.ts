/**
 * Format frequency in Hz to MHz with proper precision
 */
export function formatFrequencyMHz(hz: number): string {
  const mhz = hz / 1_000_000;
  return mhz.toFixed(3);
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
