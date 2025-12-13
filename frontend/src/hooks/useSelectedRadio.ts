import { useCallback, useEffect, useMemo, useState } from "react";
import { useCaptures } from "./useCaptures";
import { useDevices } from "./useDevices";
import { useTrunkingSystems } from "./useTrunking";
import { findDeviceForCapture } from "../utils/deviceId";
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
        result.push({
          type: "capture",
          id: capture.id,
          name: capture.name || capture.autoName || formatCaptureId(capture.id),
          deviceName: device ? getDeviceDisplayName(device) : "Unknown Device",
          state: capture.state,
          frequencyHz: capture.centerHz,
        });
      }
    }

    // Add trunking system tabs
    if (trunkingSystems) {
      for (const system of trunkingSystems) {
        result.push({
          type: "trunking",
          id: system.id,
          name: system.name,
          deviceName: "Trunking",
          state: system.state,
          frequencyHz: system.controlChannelFreqHz ?? 0,
        });
      }
    }

    return result;
  }, [captures, devices, trunkingSystems]);

  // Determine selected tab
  const selectedTab = useMemo(() => {
    // If URL has a valid selection, use it
    if (urlSelection) {
      const tab = tabs.find(
        (t) => t.type === urlSelection.type && t.id === urlSelection.id
      );
      if (tab) return { type: urlSelection.type, id: urlSelection.id };
    }

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
  }, [tabs, urlSelection]);

  // Auto-update URL when selection changes
  useEffect(() => {
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
      }
    }
  }, [selectedTab, urlSelection]);

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
  }, []);

  // Listen for browser back/forward navigation
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
    window.addEventListener("popstate", handlePopState);
    return () => window.removeEventListener("popstate", handlePopState);
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
