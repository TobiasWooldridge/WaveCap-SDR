import { useQuery } from "@tanstack/react-query";
import type { Device } from "../types";

async function fetchDevices(): Promise<Device[]> {
  const response = await fetch("/api/v1/devices");
  if (!response.ok) {
    throw new Error("Failed to fetch devices");
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
