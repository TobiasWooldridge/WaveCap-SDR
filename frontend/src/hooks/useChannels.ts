import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import type { Channel, CreateChannelRequest } from "../types";

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
    const error = await response.json();
    throw new Error(error.detail || "Failed to create channel");
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

export function useChannels(captureId: string | undefined) {
  return useQuery({
    queryKey: ["channels", captureId],
    queryFn: () => fetchChannels(captureId!),
    enabled: !!captureId,
    refetchInterval: 2_000, // Poll every 2 seconds
  });
}

export function useCreateChannel() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ captureId, request }: { captureId: string; request: CreateChannelRequest }) =>
      createChannel(captureId, request),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: ["channels", variables.captureId] });
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
