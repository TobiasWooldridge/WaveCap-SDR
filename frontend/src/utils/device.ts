import type { Device } from '../types';

/**
 * Extract a readable device name from a device ID string.
 * Device IDs look like: "driver=sdrplay,label=SDRplay Dev0 RSPdx-R2 240309F070,serial=240309F070"
 */
export function getDeviceNameFromId(deviceId: string): string {
  // Try to extract driver from the ID
  const driverMatch = deviceId.match(/driver=([^,]+)/);
  const labelMatch = deviceId.match(/label=([^,]+)/);

  if (labelMatch) {
    return labelMatch[1];
  }
  if (driverMatch) {
    // Capitalize driver name
    const driver = driverMatch[1];
    if (driver === 'sdrplay') return 'SDRplay';
    if (driver === 'rtlsdr') return 'RTL-SDR';
    return driver.charAt(0).toUpperCase() + driver.slice(1);
  }
  return deviceId.substring(0, 30) + '...';
}

/**
 * Get the display name for a device.
 * Priority: nickname > shorthand > label
 */
export function getDeviceDisplayName(device: Device): string {
  if (device.nickname) {
    return device.nickname;
  }
  if (device.shorthand) {
    return device.shorthand;
  }
  // Fallback to label if neither nickname nor shorthand available
  return device.label;
}

/**
 * Get a full description for a device (for tooltips/detailed views).
 * Format: "Nickname (Shorthand)" or "Shorthand" or "Label"
 */
export function getDeviceFullDescription(device: Device): string {
  if (device.nickname && device.shorthand) {
    return `${device.nickname} (${device.shorthand})`;
  }
  return getDeviceDisplayName(device);
}
