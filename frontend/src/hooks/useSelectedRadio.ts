import { useCallback, useEffect, useMemo, useState } from "react";
import { useCaptures } from "./useCaptures";
import { useDevices } from "./useDevices";
import { useTrunkingSystems } from "./useTrunking";
import { findDeviceForCapture, getStableDeviceId } from "../utils/deviceId";
import { getDeviceDisplayName } from "../utils/device";
import type { Capture, Device, DeviceTab, DeviceStatus, ControlChannelState, RadioTab, RadioTabType } from "../types";
import type { TrunkingSystem, ControlChannelState as TrunkingCCState } from "../types/trunking";

export type ViewMode = "radio" | "trunking" | "digital";

/**
 * Map trunking control channel state to our simplified type.
 * The trunking type includes "unlocked" which we treat as "searching".
 */
function mapControlChannelState(ccState: TrunkingCCState | undefined): ControlChannelState | undefined {
  if (!ccState) return undefined;
  switch (ccState) {
    case "locked":
      return "locked";
    case "searching":
    case "unlocked":
      return "searching";
    case "lost":
      return "lost";
    default:
      return "searching";
  }
}

/**
 * Compute the overall status for a device based on its capture and trunking states.
 */
function computeDeviceStatus(capture: Capture | null, trunking: TrunkingSystem | null): DeviceStatus {
  // Check for running state first (either capture or trunking running = device running)
  if (capture?.state === "running" || trunking?.state === "running") {
    return "running";
  }
  // Check for starting/syncing states
  if (capture?.state === "starting" || trunking?.state === "starting" || trunking?.state === "searching") {
    return "starting";
  }
  // Check for failed states
  if (capture?.state === "failed" || capture?.state === "error" || trunking?.state === "failed") {
    return "failed";
  }
  // Default to stopped
  return "stopped";
}

/**
 * Device-centric selection hook for unified radio/trunking UI.
 *
 * Architecture:
 * - Level 1 (DeviceTabBar): One tab per physical SDR device
 * - Level 2 (ModeTabBar): Radio | Trunking | Digital modes for the selected device
 *
 * URL format: ?device=<deviceId>&mode=<radio|trunking|digital>
 *
 * Features:
 * - Syncs device selection to URL query parameter
 * - Auto-selects first running device if none selected
 * - Returns DeviceTab[] for Level 1 tab bar
 * - Returns capture/trunking for the selected device
 */
export function useSelectedRadio() {
  const { data: captures, isLoading: capturesLoading } = useCaptures();
  const { data: devices, isLoading: devicesLoading } = useDevices();
  const { data: trunkingSystems, isLoading: trunkingLoading } = useTrunkingSystems();

  const isLoading = capturesLoading || devicesLoading || trunkingLoading;

  // =========================================================================
  // Build device tabs (Level 1)
  // =========================================================================

  const deviceTabs: DeviceTab[] = useMemo(() => {
    const deviceMap = new Map<string, DeviceTab>();

    // First pass: add devices from captures
    if (captures) {
      for (const capture of captures) {
        const stableDeviceId = getStableDeviceId(capture.deviceId);
        const device = findDeviceForCapture(devices, capture);
        const deviceName = device ? getDeviceDisplayName(device) : "Unknown Device";

        if (!deviceMap.has(stableDeviceId)) {
          deviceMap.set(stableDeviceId, {
            deviceId: stableDeviceId,
            deviceName,
            capture,
            trunkingSystem: null,
            status: "stopped",
            hasRadio: true,
            hasTrunking: false,
            frequencyHz: capture.centerHz,
          });
        } else {
          // Device already exists, add/update capture
          const existing = deviceMap.get(stableDeviceId)!;
          existing.capture = capture;
          existing.hasRadio = true;
          existing.frequencyHz = capture.centerHz;
        }
      }
    }

    // Second pass: add/update devices from trunking systems
    if (trunkingSystems) {
      for (const system of trunkingSystems) {
        if (!system.deviceId) continue;
        const stableDeviceId = getStableDeviceId(system.deviceId);

        // Map control channel state to our simplified type
        const ccState = mapControlChannelState(system.controlChannelState);

        if (!deviceMap.has(stableDeviceId)) {
          // Device only has trunking, no capture
          let deviceName = "Trunking";
          if (devices) {
            const device = devices.find((d) => getStableDeviceId(d.id) === stableDeviceId);
            if (device) deviceName = getDeviceDisplayName(device);
          }

          deviceMap.set(stableDeviceId, {
            deviceId: stableDeviceId,
            deviceName,
            capture: null,
            trunkingSystem: system,
            status: "stopped",
            hasRadio: false,
            hasTrunking: true,
            frequencyHz: system.controlChannelFreqHz ?? 0,
            // Trunking-specific status fields
            controlChannelState: ccState,
            activeCalls: system.activeCalls,
            isManuallyLocked: system.lockedFrequencyHz !== null,
          });
        } else {
          // Device already exists, add trunking
          const existing = deviceMap.get(stableDeviceId)!;
          existing.trunkingSystem = system;
          existing.hasTrunking = true;
          // Add trunking-specific status
          existing.controlChannelState = ccState;
          existing.activeCalls = system.activeCalls;
          existing.isManuallyLocked = system.lockedFrequencyHz !== null;
        }
      }
    }

    // Third pass: compute status for each device
    for (const tab of deviceMap.values()) {
      tab.status = computeDeviceStatus(tab.capture, tab.trunkingSystem);
    }

    // Sort: running devices first, then by device name
    return Array.from(deviceMap.values()).sort((a, b) => {
      if (a.status === "running" && b.status !== "running") return -1;
      if (a.status !== "running" && b.status === "running") return 1;
      return a.deviceName.localeCompare(b.deviceName);
    });
  }, [captures, devices, trunkingSystems]);

  // =========================================================================
  // Device Selection (Level 1)
  // =========================================================================

  // Parse device selection from URL
  const [urlDeviceId, setUrlDeviceId] = useState<string | null>(() => {
    const params = new URLSearchParams(window.location.search);
    // New format: ?device=xxx
    const deviceParam = params.get("device");
    if (deviceParam) return deviceParam;

    // Legacy format: ?radio=capture:c1 or ?radio=trunking:psern
    // Convert to device ID by looking up the capture/trunking
    const radioParam = params.get("radio");
    if (radioParam) {
      const [type, id] = radioParam.split(":");
      // We'll resolve this to a device ID after data loads
      return `legacy:${type}:${id}`;
    }

    // Very legacy: ?capture=c1
    const captureParam = params.get("capture");
    if (captureParam) {
      return `legacy:capture:${captureParam}`;
    }

    return null;
  });

  // Resolve legacy URL format to device ID
  const resolvedDeviceId = useMemo(() => {
    if (!urlDeviceId) return null;
    if (!urlDeviceId.startsWith("legacy:")) return urlDeviceId;

    // Parse legacy format
    const [, type, id] = urlDeviceId.split(":");

    if (type === "capture" && captures) {
      const capture = captures.find((c) => c.id === id);
      if (capture) return getStableDeviceId(capture.deviceId);
    } else if (type === "trunking" && trunkingSystems) {
      const system = trunkingSystems.find((s) => s.id === id);
      if (system?.deviceId) return getStableDeviceId(system.deviceId);
    }

    // Can't resolve yet (still loading) - return null to trigger auto-select
    if (isLoading) return urlDeviceId; // Keep legacy format while loading
    return null;
  }, [urlDeviceId, captures, trunkingSystems, isLoading]);

  // Determine selected device
  const selectedDeviceId = useMemo(() => {
    // If we have a resolved device ID from URL, use it
    if (resolvedDeviceId && !resolvedDeviceId.startsWith("legacy:")) {
      // Verify device exists
      const exists = deviceTabs.some((t) => t.deviceId === resolvedDeviceId);
      if (exists) return resolvedDeviceId;
      // If still loading, trust the URL
      if (isLoading) return resolvedDeviceId;
    }

    // Still loading - don't auto-select yet
    if (isLoading) return null;

    // Auto-select first running device
    const runningDevice = deviceTabs.find((t) => t.status === "running");
    if (runningDevice) return runningDevice.deviceId;

    // Or first device
    return deviceTabs[0]?.deviceId ?? null;
  }, [deviceTabs, resolvedDeviceId, isLoading]);

  // Track if user has explicitly selected
  const [userHasSelected, setUserHasSelected] = useState(() => {
    const params = new URLSearchParams(window.location.search);
    return params.has("device") || params.has("radio") || params.has("capture");
  });

  // Update URL when selection changes (for auto-selection)
  useEffect(() => {
    if (isLoading) return;
    if (userHasSelected && urlDeviceId) return;

    if (selectedDeviceId) {
      const url = new URL(window.location.href);
      // Remove legacy params
      url.searchParams.delete("radio");
      url.searchParams.delete("capture");
      // Set new format
      url.searchParams.set("device", selectedDeviceId);
      window.history.replaceState({}, "", url.toString());
      setUrlDeviceId(selectedDeviceId);
    }
  }, [selectedDeviceId, urlDeviceId, isLoading, userHasSelected]);

  // Get selected device tab
  const selectedDeviceTab = useMemo(() => {
    return deviceTabs.find((t) => t.deviceId === selectedDeviceId) ?? null;
  }, [deviceTabs, selectedDeviceId]);

  // Convenience accessors for selected device's capture and trunking
  const selectedCapture = selectedDeviceTab?.capture ?? null;
  const selectedTrunkingSystem = selectedDeviceTab?.trunkingSystem ?? null;

  // Get the Device object for the selected device
  const selectedDevice: Device | null = useMemo(() => {
    if (!selectedDeviceId || !devices) return null;
    return devices.find((d) => getStableDeviceId(d.id) === selectedDeviceId) ?? null;
  }, [selectedDeviceId, devices]);

  // Select a device
  const selectDevice = useCallback((deviceId: string) => {
    const url = new URL(window.location.href);
    url.searchParams.delete("radio");
    url.searchParams.delete("capture");
    url.searchParams.set("device", deviceId);
    window.history.replaceState({}, "", url.toString());
    setUrlDeviceId(deviceId);
    setUserHasSelected(true);
    // Dispatch custom event for sync
    window.dispatchEvent(new CustomEvent("deviceselectionchange", { detail: { deviceId } }));
  }, []);

  // Listen for browser back/forward navigation
  useEffect(() => {
    const handlePopState = () => {
      const params = new URLSearchParams(window.location.search);
      const deviceParam = params.get("device");
      if (deviceParam) {
        setUrlDeviceId(deviceParam);
        return;
      }
      // Handle legacy formats on back navigation
      const radioParam = params.get("radio");
      if (radioParam) {
        setUrlDeviceId(`legacy:${radioParam.replace(":", ":")}`);
        return;
      }
      setUrlDeviceId(null);
    };

    const handleDeviceChange = (event: Event) => {
      const { deviceId } = (event as CustomEvent).detail;
      if (deviceId) {
        setUrlDeviceId(deviceId);
        setUserHasSelected(true);
      }
    };

    window.addEventListener("popstate", handlePopState);
    window.addEventListener("deviceselectionchange", handleDeviceChange);
    return () => {
      window.removeEventListener("popstate", handlePopState);
      window.removeEventListener("deviceselectionchange", handleDeviceChange);
    };
  }, []);

  // =========================================================================
  // View Mode (Level 2 tabs: Radio | Trunking | Digital)
  // =========================================================================

  const [viewMode, setViewModeState] = useState<ViewMode>(() => {
    const params = new URLSearchParams(window.location.search);
    const modeParam = params.get("mode");
    if (modeParam === "radio" || modeParam === "trunking" || modeParam === "digital") {
      return modeParam;
    }
    return "radio";
  });

  // Set view mode and update URL
  const setViewMode = useCallback((mode: ViewMode) => {
    setViewModeState(mode);
    const url = new URL(window.location.href);
    url.searchParams.set("mode", mode);
    window.history.replaceState({}, "", url.toString());
  }, []);

  // Auto-switch mode based on what's available for selected device
  useEffect(() => {
    if (!selectedDeviceTab) return;

    // If viewing trunking but device doesn't have it, switch to radio
    if (viewMode === "trunking" && !selectedDeviceTab.hasTrunking) {
      setViewModeState("radio");
    }
    // If viewing radio but device doesn't have it, switch to trunking if available
    else if (viewMode === "radio" && !selectedDeviceTab.hasRadio && selectedDeviceTab.hasTrunking) {
      setViewModeState("trunking");
    }
  }, [selectedDeviceTab, viewMode]);

  // =========================================================================
  // Legacy compatibility: RadioTab[] and selectTab for old RadioTabBar
  // These can be removed once DeviceTabBar is fully implemented
  // =========================================================================

  const tabs: RadioTab[] = useMemo(() => {
    const result: RadioTab[] = [];

    if (captures) {
      for (const capture of captures) {
        const device = findDeviceForCapture(devices, capture);
        const stableDeviceId = getStableDeviceId(capture.deviceId);
        result.push({
          type: "capture",
          id: capture.id,
          name: capture.name || capture.autoName || formatCaptureId(capture.id),
          deviceId: stableDeviceId,
          deviceName: device ? getDeviceDisplayName(device) : "Unknown Device",
          state: capture.state,
          frequencyHz: capture.centerHz,
        });
      }
    }

    if (trunkingSystems) {
      for (const system of trunkingSystems) {
        const stableDeviceId = system.deviceId ? getStableDeviceId(system.deviceId) : "";
        let deviceName = "Trunking";
        if (system.deviceId && devices) {
          const device = devices.find((d) => getStableDeviceId(d.id) === stableDeviceId);
          if (device) deviceName = getDeviceDisplayName(device);
        }
        result.push({
          type: "trunking",
          id: system.id,
          name: system.name,
          deviceId: stableDeviceId,
          deviceName,
          state: system.state,
          frequencyHz: system.controlChannelFreqHz ?? 0,
        });
      }
    }

    return result;
  }, [captures, devices, trunkingSystems]);

  // Legacy: selectTab (converts to selectDevice + setViewMode)
  const selectTab = useCallback((type: RadioTabType, id: string) => {
    // Find the device for this tab
    let deviceId: string | null = null;

    if (type === "capture" && captures) {
      const capture = captures.find((c) => c.id === id);
      if (capture) deviceId = getStableDeviceId(capture.deviceId);
    } else if (type === "trunking" && trunkingSystems) {
      const system = trunkingSystems.find((s) => s.id === id);
      if (system?.deviceId) deviceId = getStableDeviceId(system.deviceId);
    }

    if (deviceId) {
      selectDevice(deviceId);
      setViewMode(type === "trunking" ? "trunking" : "radio");
    }
  }, [captures, trunkingSystems, selectDevice, setViewMode]);

  // Legacy: selectedType and selectedId (derived from device + mode)
  const selectedType: RadioTabType | null = useMemo(() => {
    if (!selectedDeviceTab) return null;
    return viewMode === "trunking" ? "trunking" : "capture";
  }, [selectedDeviceTab, viewMode]);

  const selectedId: string | null = useMemo(() => {
    if (!selectedDeviceTab) return null;
    if (viewMode === "trunking") {
      return selectedDeviceTab.trunkingSystem?.id ?? null;
    }
    return selectedDeviceTab.capture?.id ?? null;
  }, [selectedDeviceTab, viewMode]);

  return {
    // =========================================================================
    // New device-centric API
    // =========================================================================

    // Device tabs for Level 1
    deviceTabs,

    // Selected device
    selectedDeviceId,
    selectedDeviceTab,
    selectDevice,

    // =========================================================================
    // Common API (used by both old and new UI)
    // =========================================================================

    // For captures
    selectedCapture,
    selectedDevice,

    // For trunking
    selectedTrunkingSystem,

    // View mode (Level 2 tabs)
    viewMode,
    setViewMode,

    // Convenience: does selected device have trunking?
    hasTrunkingForDevice: selectedDeviceTab?.hasTrunking ?? false,
    trunkingSystemForDevice: selectedDeviceTab?.trunkingSystem ?? null,

    // All data
    captures: captures ?? [],
    trunkingSystems: trunkingSystems ?? [],
    devices: devices ?? [],
    isLoading,

    // =========================================================================
    // Legacy API (for backwards compatibility with RadioTabBar)
    // TODO: Remove once DeviceTabBar is fully implemented
    // =========================================================================

    selectedType,
    selectedId,
    selectTab,
    tabs,
  };
}

function formatCaptureId(id: string): string {
  const match = id.match(/^c(\d+)$/);
  return match ? `Radio ${match[1]}` : id;
}
