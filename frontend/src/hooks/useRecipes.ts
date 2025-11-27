import { useQuery } from "@tanstack/react-query";
import { Recipe } from "../types";

const API_BASE = "/api/v1";

async function fetchRecipes(deviceId?: string): Promise<Recipe[]> {
  const url = deviceId
    ? `${API_BASE}/recipes?device_id=${encodeURIComponent(deviceId)}`
    : `${API_BASE}/recipes`;
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error("Failed to fetch recipes");
  }
  return response.json();
}

/**
 * Fetch recipes, optionally adjusted for a specific device's capabilities.
 * When deviceId is provided, sample rates, bandwidth, and gain values
 * are adjusted to fit the device's supported ranges.
 */
export function useRecipes(deviceId?: string) {
  return useQuery({
    queryKey: ["recipes", deviceId],
    queryFn: () => fetchRecipes(deviceId),
  });
}
