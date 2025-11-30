import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import type { Device } from "../types";

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
  return useQuery({
    queryKey: ["devices"],
    queryFn: fetchDevices,
    staleTime: 30_000, // Cache for 30 seconds
    refetchInterval: 60_000, // Refetch every minute
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
