import { useCallback, useEffect, useMemo, useState } from "react";
import { useCaptures } from "./useCaptures";
import { useDevices } from "./useDevices";
import { findDeviceForCapture } from "../utils/deviceId";
import type { Capture, Device } from "../types";

/**
 * Hook for managing the currently selected capture with URL sync.
 *
 * Features:
 * - Syncs selection to URL query parameter (?capture=c1)
 * - Auto-selects first running capture if none selected
 * - Returns the selected capture with its associated device
 *
 * @example
 * const { selectedCapture, selectedDevice, selectCapture, captures } = useSelectedCapture();
 */
export function useSelectedCapture() {
  const { data: captures, isLoading: capturesLoading } = useCaptures();
  const { data: devices, isLoading: devicesLoading } = useDevices();

  // Track the URL capture ID in state to trigger re-renders
  const [urlCaptureId, setUrlCaptureId] = useState<string | null>(() => {
    const params = new URLSearchParams(window.location.search);
    return params.get("capture");
  });

  // Get the selected capture ID, with auto-selection fallback
  const selectedCaptureId = useMemo(() => {
    // If URL has a valid capture ID, use it
    if (urlCaptureId && captures?.some((c) => c.id === urlCaptureId)) {
      return urlCaptureId;
    }

    // Otherwise, select the first running capture
    const runningCapture = captures?.find((c) => c.state === "running");
    if (runningCapture) {
      return runningCapture.id;
    }

    // Or just the first capture
    return captures?.[0]?.id ?? null;
  }, [captures, urlCaptureId]);

  // Auto-update URL when selection changes (for auto-selection cases)
  useEffect(() => {
    if (selectedCaptureId && selectedCaptureId !== urlCaptureId) {
      const url = new URL(window.location.href);
      url.searchParams.set("capture", selectedCaptureId);
      window.history.replaceState({}, "", url.toString());
    }
  }, [selectedCaptureId, urlCaptureId]);

  // Get the selected capture object
  const selectedCapture = useMemo(() => {
    if (!selectedCaptureId || !captures) return null;
    return captures.find((c) => c.id === selectedCaptureId) ?? null;
  }, [selectedCaptureId, captures]);

  // Get the device for the selected capture
  const selectedDevice = useMemo(() => {
    if (!selectedCapture) return null;
    return findDeviceForCapture(devices, selectedCapture) ?? null;
  }, [selectedCapture, devices]);

  // Select a capture by ID
  const selectCapture = useCallback((captureId: string) => {
    // Update URL
    const url = new URL(window.location.href);
    url.searchParams.set("capture", captureId);
    window.history.replaceState({}, "", url.toString());
    // Update state to trigger re-render
    setUrlCaptureId(captureId);
  }, []);

  // Listen for browser back/forward navigation
  useEffect(() => {
    const handlePopState = () => {
      const params = new URLSearchParams(window.location.search);
      setUrlCaptureId(params.get("capture"));
    };
    window.addEventListener("popstate", handlePopState);
    return () => window.removeEventListener("popstate", handlePopState);
  }, []);

  return {
    selectedCaptureId,
    selectedCapture,
    selectedDevice,
    selectCapture,
    captures: captures ?? [],
    devices: devices ?? [],
    isLoading: capturesLoading || devicesLoading,
  };
}

/**
 * Get capture with device info.
 */
export interface CaptureWithDevice {
  capture: Capture;
  device: Device | null;
}

/**
 * Hook to get all captures with their associated devices.
 */
export function useCapturesWithDevices(): {
  captures: CaptureWithDevice[];
  isLoading: boolean;
} {
  const { data: captures, isLoading: capturesLoading } = useCaptures();
  const { data: devices, isLoading: devicesLoading } = useDevices();

  const capturesWithDevices = useMemo(() => {
    if (!captures) return [];
    return captures.map((capture) => ({
      capture,
      device: findDeviceForCapture(devices, capture) ?? null,
    }));
  }, [captures, devices]);

  return {
    captures: capturesWithDevices,
    isLoading: capturesLoading || devicesLoading,
  };
}
