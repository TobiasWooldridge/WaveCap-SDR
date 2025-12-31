import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import type {
  Channel,
  CreateChannelRequest,
  UpdateChannelRequest,
} from "../types";
import { useStateStreamStatus } from "./useStateWebSocket";

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

async function fetchChannels(captureId: string): Promise<Channel[]> {
  const response = await fetch(`/api/v1/captures/${captureId}/channels`);
  if (!response.ok) {
    throw new Error("Failed to fetch channels");
  }
  return response.json();
}

async function createChannel(
  captureId: string,
  request: CreateChannelRequest,
): Promise<Channel> {
  const response = await fetch(`/api/v1/captures/${captureId}/channels`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const message = await parseErrorMessage(
      response,
      "Failed to create channel",
    );
    throw new Error(message);
  }

  return response.json();
}

async function deleteChannel(channelId: string): Promise<void> {
  const response = await fetch(`/api/v1/channels/${channelId}`, {
    method: "DELETE",
  });

  if (!response.ok) {
    throw new Error("Failed to delete channel");
  }
}

async function updateChannel(
  channelId: string,
  request: UpdateChannelRequest,
): Promise<Channel> {
  const response = await fetch(`/api/v1/channels/${channelId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const message = await parseErrorMessage(
      response,
      "Failed to update channel",
    );
    throw new Error(message);
  }

  return response.json();
}

async function startChannel(channelId: string): Promise<Channel> {
  const response = await fetch(`/api/v1/channels/${channelId}/start`, {
    method: "POST",
  });
  if (!response.ok) {
    const message = await parseErrorMessage(
      response,
      "Failed to start channel",
    );
    throw new Error(message);
  }
  return response.json();
}

async function stopChannel(channelId: string): Promise<Channel> {
  const response = await fetch(`/api/v1/channels/${channelId}/stop`, {
    method: "POST",
  });
  if (!response.ok) {
    const message = await parseErrorMessage(response, "Failed to stop channel");
    throw new Error(message);
  }
  return response.json();
}

export function useChannels(captureId: string | undefined) {
  const isStateStreamConnected = useStateStreamStatus();

  return useQuery({
    queryKey: ["channels", captureId],
    queryFn: () => fetchChannels(captureId!),
    enabled: !!captureId,
    // Fallback polling - WebSocket provides real-time updates
    // Polling is kept as backup for reconnection and stale data recovery
    refetchInterval: isStateStreamConnected ? false : 10_000,
  });
}

export function useCreateChannel() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      captureId,
      request,
    }: {
      captureId: string;
      request: CreateChannelRequest;
    }) => createChannel(captureId, request),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({
        queryKey: ["channels", variables.captureId],
      });
    },
  });
}

export function useDeleteChannel() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (channelId: string) => deleteChannel(channelId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["channels"] });
    },
  });
}

export function useUpdateChannel(captureId: string) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      channelId,
      request,
    }: {
      channelId: string;
      request: UpdateChannelRequest;
    }) => updateChannel(channelId, request),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["channels", captureId] });
    },
  });
}

export function useStartChannel(captureId: string) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (channelId: string) => startChannel(channelId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["channels", captureId] });
    },
  });
}

export function useStopChannel(captureId: string) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (channelId: string) => stopChannel(channelId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["channels", captureId] });
    },
  });
}
