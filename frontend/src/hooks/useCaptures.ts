import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import type { Capture, UpdateCaptureRequest, CreateCaptureRequest } from "../types";

async function fetchCaptures(): Promise<Capture[]> {
  const response = await fetch("/api/v1/captures");
  if (!response.ok) {
    throw new Error("Failed to fetch captures");
  }
  return response.json();
}

async function updateCapture(
  captureId: string,
  request: UpdateCaptureRequest,
): Promise<Capture> {
  const response = await fetch(`/api/v1/captures/${captureId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    // Try to parse JSON error, but handle plain text responses too
    let errorMessage = "Failed to update capture";
    try {
      const error = await response.json();
      errorMessage = error.detail || errorMessage;
    } catch {
      // Response wasn't JSON, try to get text
      try {
        const text = await response.text();
        if (text) errorMessage = text;
      } catch {
        // Ignore
      }
    }
    throw new Error(errorMessage);
  }

  return response.json();
}

async function createCapture(request: CreateCaptureRequest): Promise<Capture> {
  const response = await fetch("/api/v1/captures", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Failed to create capture");
  }

  return response.json();
}

async function startCapture(captureId: string): Promise<Capture> {
  const response = await fetch(`/api/v1/captures/${captureId}/start`, {
    method: "POST",
  });

  if (!response.ok) {
    throw new Error("Failed to start capture");
  }

  return response.json();
}

async function stopCapture(captureId: string): Promise<Capture> {
  const response = await fetch(`/api/v1/captures/${captureId}/stop`, {
    method: "POST",
  });

  if (!response.ok) {
    throw new Error("Failed to stop capture");
  }

  return response.json();
}

async function restartCapture(captureId: string): Promise<Capture> {
  const response = await fetch(`/api/v1/captures/${captureId}/restart`, {
    method: "POST",
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || "Failed to restart capture");
  }

  return response.json();
}

async function deleteCapture(captureId: string): Promise<void> {
  const response = await fetch(`/api/v1/captures/${captureId}`, {
    method: "DELETE",
  });

  if (!response.ok) {
    throw new Error("Failed to delete capture");
  }
}

export function useCaptures() {
  return useQuery({
    queryKey: ["captures"],
    queryFn: fetchCaptures,
    // Fallback polling - WebSocket provides real-time updates
    // Polling is kept as backup for reconnection and stale data recovery
    refetchInterval: 10_000,
  });
}

export function useUpdateCapture() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ captureId, request }: { captureId: string; request: UpdateCaptureRequest }) =>
      updateCapture(captureId, request),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["captures"] });
    },
    onError: () => {
      // Refetch captures to reset UI to server state
      // This ensures local state gets reset via the useEffect in RadioTuner
      queryClient.invalidateQueries({ queryKey: ["captures"] });
    },
  });
}

export function useCreateCapture() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (request: CreateCaptureRequest) => createCapture(request),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["captures"] });
    },
  });
}

export function useStartCapture() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (captureId: string) => startCapture(captureId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["captures"] });
    },
  });
}

export function useStopCapture() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (captureId: string) => stopCapture(captureId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["captures"] });
    },
  });
}

export function useRestartCapture() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (captureId: string) => restartCapture(captureId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["captures"] });
    },
  });
}

export function useDeleteCapture() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (captureId: string) => deleteCapture(captureId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["captures"] });
      queryClient.invalidateQueries({ queryKey: ["channels"] });
    },
  });
}
