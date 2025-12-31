import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import type { Device } from "../types";
import { useStateStreamStatus } from "./useStateWebSocket";

async function fetchDevices(): Promise<Device[]> {
  const response = await fetch("/api/v1/devices");
  if (!response.ok) {
    throw new Error("Failed to fetch devices");
  }
  return response.json();
}

interface RestartServiceResponse {
  status: string;
  message: string;
  stats?: {
    recovery_count: number;
    last_recovery_success: number | null;
  };
}

async function restartSDRplayService(): Promise<RestartServiceResponse> {
  const response = await fetch("/api/v1/devices/sdrplay/restart-service", {
    method: "POST",
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || "Failed to restart SDRplay service");
  }

  return response.json();
}

export function useDevices() {
  const isStateStreamConnected = useStateStreamStatus();

  return useQuery({
    queryKey: ["devices"],
    queryFn: fetchDevices,
    staleTime: 30_000, // Cache for 30 seconds
    refetchInterval: isStateStreamConnected ? 120_000 : 30_000,
  });
}

async function refreshDevices(): Promise<Device[]> {
  const response = await fetch("/api/v1/devices/refresh", {
    method: "POST",
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || "Failed to refresh devices");
  }

  return response.json();
}

/**
 * Hook to force re-enumeration of all SDR devices.
 * Invalidates device cache and performs fresh enumeration.
 * Use after USB power cycling or when devices aren't appearing.
 */
export function useRefreshDevices() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: refreshDevices,
    onSuccess: (devices) => {
      // Update the devices cache with fresh data
      queryClient.setQueryData(["devices"], devices);
      // Also invalidate captures in case device state changed
      queryClient.invalidateQueries({ queryKey: ["captures"] });
    },
  });
}

/**
 * Hook to restart the SDRplay API service.
 * Use when SDRplay captures are stuck in 'starting' state or device enumeration hangs.
 * Rate limited: max 5 restarts per hour with 60-second cooldown.
 */
export function useRestartSDRplayService() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: restartSDRplayService,
    onSuccess: () => {
      // Invalidate device and capture queries to refresh state after service restart
      queryClient.invalidateQueries({ queryKey: ["devices"] });
      queryClient.invalidateQueries({ queryKey: ["captures"] });
    },
  });
}

interface PowerCycleResponse {
  status: string;
  message: string;
  captureId: string;
  deviceId: string;
  wasRunning: boolean;
}

async function powerCycleDevice(
  captureId: string,
): Promise<PowerCycleResponse> {
  const response = await fetch(`/api/v1/devices/usb/power-cycle/${captureId}`, {
    method: "POST",
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || "Failed to power cycle device");
  }

  return response.json();
}

interface USBHub {
  location: string;
  vendorId: string;
  productId: string;
  description: string;
  ports: Array<{
    port: number;
    powered: boolean;
    connected: boolean;
    device: {
      vendorId: string;
      productId: string;
      description: string;
      serial: string | null;
    } | null;
  }>;
}

interface USBHubsResponse {
  available: boolean;
  hubs: USBHub[];
}

async function fetchUSBHubs(): Promise<USBHubsResponse> {
  const response = await fetch("/api/v1/devices/usb/hubs");
  if (!response.ok) {
    throw new Error("Failed to fetch USB hubs");
  }
  return response.json();
}

/**
 * Hook to get USB hub status.
 * Returns list of controllable USB hubs and their port status.
 */
export function useUSBHubs() {
  return useQuery({
    queryKey: ["usb-hubs"],
    queryFn: fetchUSBHubs,
    staleTime: 10_000, // Cache for 10 seconds
    refetchInterval: 30_000, // Refetch every 30 seconds
  });
}

/**
 * Hook to power cycle a capture's USB device.
 * Performs a hardware reset by cycling USB port power.
 * Requires uhubctl and device on controllable hub.
 */
export function usePowerCycleDevice() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: powerCycleDevice,
    onSuccess: () => {
      // Invalidate queries to refresh state after power cycle
      queryClient.invalidateQueries({ queryKey: ["devices"] });
      queryClient.invalidateQueries({ queryKey: ["captures"] });
      queryClient.invalidateQueries({ queryKey: ["usb-hubs"] });
    },
  });
}

interface PowerCycleAllResponse {
  status: string;
  message: string;
  portsCycled: number;
  stoppedCaptures: string[];
}

async function powerCycleAllUSB(): Promise<PowerCycleAllResponse> {
  const response = await fetch("/api/v1/devices/usb/power-cycle-all", {
    method: "POST",
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || "Failed to power cycle USB ports");
  }

  return response.json();
}

/**
 * Hook to power cycle all USB ports with connected devices.
 * Stops all running captures first, then cycles all ports.
 * Requires uhubctl and devices on controllable hubs.
 */
export function usePowerCycleAllUSB() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: powerCycleAllUSB,
    onSuccess: () => {
      // Invalidate queries to refresh state after power cycle
      queryClient.invalidateQueries({ queryKey: ["devices"] });
      queryClient.invalidateQueries({ queryKey: ["captures"] });
      queryClient.invalidateQueries({ queryKey: ["usb-hubs"] });
    },
  });
}
