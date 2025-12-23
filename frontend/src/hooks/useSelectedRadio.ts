import { useCallback, useEffect, useMemo, useState } from "react";
import { useCaptures } from "./useCaptures";
import { useDevices } from "./useDevices";
import { useTrunkingSystems } from "./useTrunking";
import { findDeviceForCapture, getStableDeviceId } from "../utils/deviceId";
import { getDeviceDisplayName } from "../utils/device";
import type { Capture, Device, RadioTab, RadioTabType } from "../types";
import type { TrunkingSystem } from "../types/trunking";

/**
 * Unified selection hook for both captures and trunking systems.
 *
 * Features:
 * - Syncs selection to URL query parameter (?radio=capture:c1 or ?radio=trunking:psern)
 * - Auto-selects first running item if none selected
 * - Returns unified RadioTab[] array for the tab bar
 */
export function useSelectedRadio() {
  const { data: captures, isLoading: capturesLoading } = useCaptures();
  const { data: devices, isLoading: devicesLoading } = useDevices();
  const { data: trunkingSystems, isLoading: trunkingLoading } = useTrunkingSystems();

  // Parse selection from URL (format: "capture:c1" or "trunking:psern")
  const [urlSelection, setUrlSelection] = useState<{ type: RadioTabType; id: string } | null>(() => {
    const params = new URLSearchParams(window.location.search);
    const radioParam = params.get("radio");
    if (radioParam) {
      const [type, id] = radioParam.split(":");
      if ((type === "capture" || type === "trunking") && id) {
        return { type, id };
      }
    }
    // Fallback: check legacy ?capture= param
    const legacyCapture = params.get("capture");
    if (legacyCapture) {
      return { type: "capture", id: legacyCapture };
    }
    return null;
  });

  // Build unified tabs array
  const tabs: RadioTab[] = useMemo(() => {
    const result: RadioTab[] = [];

    // Add capture tabs
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

    // Add trunking system tabs
    if (trunkingSystems) {
      for (const system of trunkingSystems) {
        // Look up device name from deviceId
        const stableDeviceId = system.deviceId ? getStableDeviceId(system.deviceId) : "";
        let deviceName = "Trunking";
        if (system.deviceId && devices) {
          const device = devices.find(
            (d) => getStableDeviceId(d.id) === stableDeviceId
          );
          if (device) {
            deviceName = getDeviceDisplayName(device);
          }
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

  const isLoading = capturesLoading || devicesLoading || trunkingLoading;

  // Determine selected tab
  const selectedTab = useMemo(() => {
    // If URL has a valid selection, use it (even if tab not found yet - might still be loading)
    if (urlSelection) {
      const tab = tabs.find(
        (t) => t.type === urlSelection.type && t.id === urlSelection.id
      );
      if (tab) return { type: urlSelection.type, id: urlSelection.id };
      // If still loading, trust the URL selection - don't fall through to auto-select
      if (isLoading) return { type: urlSelection.type, id: urlSelection.id };
    }

    // Don't auto-select while still loading - wait for data
    if (isLoading) return null;

    // Auto-select first running capture
    const runningCapture = tabs.find(
      (t) => t.type === "capture" && t.state === "running"
    );
    if (runningCapture) {
      return { type: runningCapture.type, id: runningCapture.id };
    }

    // Or first running trunking system
    const runningSys = tabs.find(
      (t) => t.type === "trunking" && t.state === "running"
    );
    if (runningSys) {
      return { type: runningSys.type, id: runningSys.id };
    }

    // Or just the first tab
    const first = tabs[0];
    return first ? { type: first.type, id: first.id } : null;
  }, [tabs, urlSelection, isLoading]);

  // Track if user has explicitly selected (vs auto-selection)
  const [userHasSelected, setUserHasSelected] = useState(() => {
    // If URL has a selection on mount, user has "selected"
    const params = new URLSearchParams(window.location.search);
    return params.has("radio") || params.has("capture");
  });

  // Auto-update URL when selection changes, but only after initial load
  // and only when user hasn't explicitly selected something
  useEffect(() => {
    // Don't update URL while loading
    if (isLoading) return;
    // Don't update URL if user has an explicit selection from URL
    if (userHasSelected && urlSelection) return;

    if (selectedTab) {
      const urlKey = `${selectedTab.type}:${selectedTab.id}`;
      const currentParam = urlSelection
        ? `${urlSelection.type}:${urlSelection.id}`
        : null;
      if (urlKey !== currentParam) {
        const url = new URL(window.location.href);
        url.searchParams.delete("capture"); // Remove legacy param
        url.searchParams.set("radio", urlKey);
        window.history.replaceState({}, "", url.toString());
        setUrlSelection(selectedTab);
      }
    }
  }, [selectedTab, urlSelection, isLoading, userHasSelected]);

  // Get the selected capture (if type is capture)
  const selectedCapture: Capture | null = useMemo(() => {
    if (!selectedTab || selectedTab.type !== "capture" || !captures) return null;
    return captures.find((c) => c.id === selectedTab.id) ?? null;
  }, [selectedTab, captures]);

  // Get the device for the selected capture
  const selectedDevice: Device | null = useMemo(() => {
    if (!selectedCapture) return null;
    return findDeviceForCapture(devices, selectedCapture) ?? null;
  }, [selectedCapture, devices]);

  // Get the selected trunking system (if type is trunking)
  const selectedTrunkingSystem: TrunkingSystem | null = useMemo(() => {
    if (!selectedTab || selectedTab.type !== "trunking" || !trunkingSystems)
      return null;
    return trunkingSystems.find((s) => s.id === selectedTab.id) ?? null;
  }, [selectedTab, trunkingSystems]);

  // Select a tab by type and ID
  const selectTab = useCallback((type: RadioTabType, id: string) => {
    const urlKey = `${type}:${id}`;
    const url = new URL(window.location.href);
    url.searchParams.delete("capture"); // Remove legacy param
    url.searchParams.set("radio", urlKey);
    window.history.replaceState({}, "", url.toString());
    setUrlSelection({ type, id });
    setUserHasSelected(true);
    // Dispatch custom event so other hook instances can sync
    window.dispatchEvent(new CustomEvent("radioselectionchange", { detail: { type, id } }));
  }, []);

  // Listen for browser back/forward navigation and custom selection events
  useEffect(() => {
    const handlePopState = () => {
      const params = new URLSearchParams(window.location.search);
      const radioParam = params.get("radio");
      if (radioParam) {
        const [type, id] = radioParam.split(":");
        if ((type === "capture" || type === "trunking") && id) {
          setUrlSelection({ type, id });
          return;
        }
      }
      const legacyCapture = params.get("capture");
      if (legacyCapture) {
        setUrlSelection({ type: "capture", id: legacyCapture });
        return;
      }
      setUrlSelection(null);
    };

    // Listen for custom selection change events from other hook instances
    const handleSelectionChange = (event: Event) => {
      const { type, id } = (event as CustomEvent).detail;
      if ((type === "capture" || type === "trunking") && id) {
        setUrlSelection({ type, id });
        setUserHasSelected(true);
      }
    };

    window.addEventListener("popstate", handlePopState);
    window.addEventListener("radioselectionchange", handleSelectionChange);
    return () => {
      window.removeEventListener("popstate", handlePopState);
      window.removeEventListener("radioselectionchange", handleSelectionChange);
    };
  }, []);

  return {
    // Selection state
    selectedType: selectedTab?.type ?? null,
    selectedId: selectedTab?.id ?? null,

    // For captures
    selectedCapture,
    selectedDevice,

    // For trunking
    selectedTrunkingSystem,

    // Actions
    selectTab,

    // All data
    tabs,
    captures: captures ?? [],
    trunkingSystems: trunkingSystems ?? [],
    devices: devices ?? [],
    isLoading: capturesLoading || devicesLoading || trunkingLoading,
  };
}

function formatCaptureId(id: string): string {
  const match = id.match(/^c(\d+)$/);
  return match ? `Radio ${match[1]}` : id;
}
