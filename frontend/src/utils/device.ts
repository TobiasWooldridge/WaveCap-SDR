import type { Device } from '../types';

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
