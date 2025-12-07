import type { Capture, Device } from "../types";

/**
 * Extract a stable identifier from a SoapySDR device ID string.
 *
 * Device IDs from SoapySDR can contain volatile fields like 'tuner' that change
 * based on device availability/enumeration order. This function extracts
 * driver + serial (or label) to create a stable ID for matching captures to devices.
 *
 * For non-SoapySDR format IDs (e.g., "device0"), returns the original ID.
 *
 * @example
 * getStableDeviceId("driver=sdrplay,label=RSPdx,serial=12345,tuner=0")
 * // => "sdrplay:12345"
 *
 * getStableDeviceId("driver=rtlsdr,label=Generic RTL2832U")
 * // => "rtlsdr:Generic RTL2832U"
 */
export function getStableDeviceId(deviceId: string): string {
  // Check if this looks like a SoapySDR format ID (has key=value pairs)
  if (!deviceId.includes("=")) {
    return deviceId;
  }

  let driver = "";
  let serial = "";
  let label = "";

  for (const part of deviceId.split(",")) {
    const [key, value] = part.split("=");
    if (key === "driver") {
      driver = value || "";
    } else if (key === "serial") {
      serial = value || "";
    } else if (key === "label") {
      label = value || "";
    }
  }

  // If we couldn't extract useful fields, fall back to original ID
  if (!driver && !serial && !label) {
    return deviceId;
  }

  // Prefer serial over label (serial is more stable)
  return serial ? `${driver}:${serial}` : `${driver}:${label}`;
}

/**
 * Check if a device matches a capture's device ID.
 * Handles volatile SoapySDR device ID fields.
 */
export function matchDeviceToCapture(device: Device, capture: Capture): boolean {
  // Exact match first
  if (device.id === capture.deviceId) {
    return true;
  }

  // Fall back to stable ID comparison
  return getStableDeviceId(device.id) === getStableDeviceId(capture.deviceId);
}

/**
 * Find a device that matches a capture's device ID.
 */
export function findDeviceForCapture(
  devices: Device[] | undefined,
  capture: Capture
): Device | undefined {
  if (!devices) return undefined;
  return devices.find((device) => matchDeviceToCapture(device, capture));
}

/**
 * Group captures by their device, using stable device IDs for matching.
 */
export interface DeviceGroup {
  device: Device | null;
  stableId: string;
  captures: Capture[];
}

export function groupCapturesByDevice(
  captures: Capture[] | undefined,
  devices: Device[] | undefined
): DeviceGroup[] {
  if (!captures || captures.length === 0) {
    return [];
  }

  const groups = new Map<string, DeviceGroup>();

  for (const capture of captures) {
    const stableId = getStableDeviceId(capture.deviceId);

    if (!groups.has(stableId)) {
      const device = findDeviceForCapture(devices, capture) ?? null;
      groups.set(stableId, {
        device,
        stableId,
        captures: [],
      });
    }

    groups.get(stableId)!.captures.push(capture);
  }

  return Array.from(groups.values());
}

/**
 * Get a list of device IDs that are currently in use by running captures.
 */
export function getUsedDeviceIds(captures: Capture[] | undefined): Set<string> {
  if (!captures) return new Set();

  const usedIds = new Set<string>();
  for (const capture of captures) {
    if (capture.state === "running" || capture.state === "starting") {
      usedIds.add(getStableDeviceId(capture.deviceId));
    }
  }
  return usedIds;
}

/**
 * Get devices that are not currently in use by any running capture.
 */
export function getAvailableDevices(
  devices: Device[] | undefined,
  captures: Capture[] | undefined
): Device[] {
  if (!devices) return [];

  const usedIds = getUsedDeviceIds(captures);
  return devices.filter((device) => !usedIds.has(getStableDeviceId(device.id)));
}
