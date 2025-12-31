import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import type {
  Scanner,
  CreateScannerRequest,
  UpdateScannerRequest,
} from "../types";
import { useStateStreamStatus } from "./useStateWebSocket";

const API_BASE = "/api/v1";

async function parseErrorMessage(
  response: Response,
  fallback: string,
): Promise<string> {
  try {
    const error = (await response.json()) as { detail?: unknown };
    if (typeof error?.detail === "string") {
      return error.detail;
    }
  } catch {
    // Ignore JSON parse failures
  }

  try {
    const text = await response.text();
    if (text) return text;
  } catch {
    // Ignore text parse failures
  }

  return fallback;
}

// Fetch all scanners
async function fetchScanners(): Promise<Scanner[]> {
  const response = await fetch(`${API_BASE}/scanners`);
  if (!response.ok) {
    throw new Error(`Failed to fetch scanners: ${response.statusText}`);
  }
  return response.json();
}

// Fetch single scanner
async function fetchScanner(scannerId: string): Promise<Scanner> {
  const response = await fetch(`${API_BASE}/scanners/${scannerId}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch scanner: ${response.statusText}`);
  }
  return response.json();
}

// Create scanner
async function createScanner(request: CreateScannerRequest): Promise<Scanner> {
  const response = await fetch(`${API_BASE}/scanners`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
  if (!response.ok) {
    const message = await parseErrorMessage(
      response,
      `Failed to create scanner: ${response.statusText}`,
    );
    throw new Error(message);
  }
  return response.json();
}

// Update scanner
async function updateScanner(
  scannerId: string,
  request: UpdateScannerRequest,
): Promise<Scanner> {
  const response = await fetch(`${API_BASE}/scanners/${scannerId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
  if (!response.ok) {
    const message = await parseErrorMessage(
      response,
      `Failed to update scanner: ${response.statusText}`,
    );
    throw new Error(message);
  }
  return response.json();
}

// Delete scanner
async function deleteScanner(scannerId: string): Promise<void> {
  const response = await fetch(`${API_BASE}/scanners/${scannerId}`, {
    method: "DELETE",
  });
  if (!response.ok) {
    throw new Error(`Failed to delete scanner: ${response.statusText}`);
  }
}

// Scanner control actions
async function startScanner(scannerId: string): Promise<Scanner> {
  const response = await fetch(`${API_BASE}/scanners/${scannerId}/start`, {
    method: "POST",
  });
  if (!response.ok) {
    throw new Error(`Failed to start scanner: ${response.statusText}`);
  }
  return response.json();
}

async function stopScanner(scannerId: string): Promise<Scanner> {
  const response = await fetch(`${API_BASE}/scanners/${scannerId}/stop`, {
    method: "POST",
  });
  if (!response.ok) {
    throw new Error(`Failed to stop scanner: ${response.statusText}`);
  }
  return response.json();
}

async function pauseScanner(scannerId: string): Promise<Scanner> {
  const response = await fetch(`${API_BASE}/scanners/${scannerId}/pause`, {
    method: "POST",
  });
  if (!response.ok) {
    throw new Error(`Failed to pause scanner: ${response.statusText}`);
  }
  return response.json();
}

async function resumeScanner(scannerId: string): Promise<Scanner> {
  const response = await fetch(`${API_BASE}/scanners/${scannerId}/resume`, {
    method: "POST",
  });
  if (!response.ok) {
    throw new Error(`Failed to resume scanner: ${response.statusText}`);
  }
  return response.json();
}

async function lockScanner(scannerId: string): Promise<Scanner> {
  const response = await fetch(`${API_BASE}/scanners/${scannerId}/lock`, {
    method: "POST",
  });
  if (!response.ok) {
    throw new Error(`Failed to lock scanner: ${response.statusText}`);
  }
  return response.json();
}

async function unlockScanner(scannerId: string): Promise<Scanner> {
  const response = await fetch(`${API_BASE}/scanners/${scannerId}/unlock`, {
    method: "POST",
  });
  if (!response.ok) {
    throw new Error(`Failed to unlock scanner: ${response.statusText}`);
  }
  return response.json();
}

async function lockoutFrequency(scannerId: string): Promise<Scanner> {
  const response = await fetch(`${API_BASE}/scanners/${scannerId}/lockout`, {
    method: "POST",
  });
  if (!response.ok) {
    throw new Error(`Failed to lockout frequency: ${response.statusText}`);
  }
  return response.json();
}

async function clearLockout(
  scannerId: string,
  frequency: number,
): Promise<Scanner> {
  const response = await fetch(
    `${API_BASE}/scanners/${scannerId}/lockout/${frequency}`,
    {
      method: "DELETE",
    },
  );
  if (!response.ok) {
    throw new Error(`Failed to clear lockout: ${response.statusText}`);
  }
  return response.json();
}

async function clearAllLockouts(scannerId: string): Promise<Scanner> {
  const response = await fetch(`${API_BASE}/scanners/${scannerId}/lockouts`, {
    method: "DELETE",
  });
  if (!response.ok) {
    throw new Error(`Failed to clear all lockouts: ${response.statusText}`);
  }
  return response.json();
}

// Hooks
export function useScanners() {
  const isStateStreamConnected = useStateStreamStatus();

  return useQuery({
    queryKey: ["scanners"],
    queryFn: fetchScanners,
    // Fallback polling - WebSocket provides real-time updates
    // Polling is kept as backup for reconnection and stale data recovery
    refetchInterval: isStateStreamConnected ? false : 10_000,
  });
}

export function useScanner(scannerId: string | undefined) {
  const isStateStreamConnected = useStateStreamStatus();

  return useQuery({
    queryKey: ["scanners", scannerId],
    queryFn: () => fetchScanner(scannerId!),
    enabled: !!scannerId,
    // Active scanners can use faster polling for real-time updates
    // WebSocket handles major state changes, polling for detailed metrics
    refetchInterval: isStateStreamConnected ? false : 5_000,
  });
}

export function useCreateScanner() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: createScanner,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["scanners"] });
    },
  });
}

export function useUpdateScanner() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({
      scannerId,
      request,
    }: {
      scannerId: string;
      request: UpdateScannerRequest;
    }) => updateScanner(scannerId, request),
    onSuccess: (_data, variables) => {
      queryClient.invalidateQueries({ queryKey: ["scanners"] });
      queryClient.invalidateQueries({
        queryKey: ["scanners", variables.scannerId],
      });
    },
  });
}

export function useDeleteScanner() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: deleteScanner,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["scanners"] });
    },
  });
}

export function useStartScanner() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: startScanner,
    onSuccess: (_data, scannerId) => {
      queryClient.invalidateQueries({ queryKey: ["scanners"] });
      queryClient.invalidateQueries({ queryKey: ["scanners", scannerId] });
    },
  });
}

export function useStopScanner() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: stopScanner,
    onSuccess: (_data, scannerId) => {
      queryClient.invalidateQueries({ queryKey: ["scanners"] });
      queryClient.invalidateQueries({ queryKey: ["scanners", scannerId] });
    },
  });
}

export function usePauseScanner() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: pauseScanner,
    onSuccess: (_data, scannerId) => {
      queryClient.invalidateQueries({ queryKey: ["scanners"] });
      queryClient.invalidateQueries({ queryKey: ["scanners", scannerId] });
    },
  });
}

export function useResumeScanner() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: resumeScanner,
    onSuccess: (_data, scannerId) => {
      queryClient.invalidateQueries({ queryKey: ["scanners"] });
      queryClient.invalidateQueries({ queryKey: ["scanners", scannerId] });
    },
  });
}

export function useLockScanner() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: lockScanner,
    onSuccess: (_data, scannerId) => {
      queryClient.invalidateQueries({ queryKey: ["scanners"] });
      queryClient.invalidateQueries({ queryKey: ["scanners", scannerId] });
    },
  });
}

export function useUnlockScanner() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: unlockScanner,
    onSuccess: (_data, scannerId) => {
      queryClient.invalidateQueries({ queryKey: ["scanners"] });
      queryClient.invalidateQueries({ queryKey: ["scanners", scannerId] });
    },
  });
}

export function useLockoutFrequency() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: lockoutFrequency,
    onSuccess: (_data, scannerId) => {
      queryClient.invalidateQueries({ queryKey: ["scanners"] });
      queryClient.invalidateQueries({ queryKey: ["scanners", scannerId] });
    },
  });
}

export function useClearLockout() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({
      scannerId,
      frequency,
    }: {
      scannerId: string;
      frequency: number;
    }) => clearLockout(scannerId, frequency),
    onSuccess: (_data, variables) => {
      queryClient.invalidateQueries({ queryKey: ["scanners"] });
      queryClient.invalidateQueries({
        queryKey: ["scanners", variables.scannerId],
      });
    },
  });
}

export function useClearAllLockouts() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: clearAllLockouts,
    onSuccess: (_data, scannerId) => {
      queryClient.invalidateQueries({ queryKey: ["scanners"] });
      queryClient.invalidateQueries({ queryKey: ["scanners", scannerId] });
    },
  });
}
