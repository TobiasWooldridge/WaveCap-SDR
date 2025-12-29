import { useQueries, useQuery } from "@tanstack/react-query";
import type { Channel, POCSAGMessage } from "../types";

interface AggregatedPOCSAGMessage extends POCSAGMessage {
  channelId: string;
  channelName: string;
}

async function fetchPOCSAGMessages(
  channelId: string,
  limit: number = 50,
): Promise<POCSAGMessage[]> {
  const params = new URLSearchParams({ limit: limit.toString() });
  const response = await fetch(
    `/api/v1/channels/${channelId}/decode/pocsag?${params}`,
  );
  if (!response.ok) {
    if (response.status === 404) {
      return []; // Channel not found or no POCSAG enabled
    }
    throw new Error("Failed to fetch POCSAG messages");
  }
  return response.json();
}

/**
 * Hook to fetch all POCSAG-enabled channels for a given capture.
 */
export function usePOCSAGChannels(captureId: string | undefined) {
  return useQuery({
    queryKey: ["channels", captureId],
    queryFn: async () => {
      if (!captureId) return [];
      const response = await fetch(`/api/v1/captures/${captureId}/channels`);
      if (!response.ok) {
        throw new Error("Failed to fetch channels");
      }
      const channels: Channel[] = await response.json();
      // Filter to only POCSAG-enabled channels
      return channels.filter((ch) => ch.enablePocsag);
    },
    enabled: !!captureId,
    refetchInterval: 5000, // Refresh channel list every 5s
  });
}

/**
 * Hook to aggregate POCSAG messages from all POCSAG-enabled channels on a capture.
 *
 * @param captureId - The capture ID to fetch messages from
 * @param options - Configuration options
 * @param options.enabled - Whether to enable fetching (default: true)
 * @param options.limit - Maximum messages per channel (default: 50)
 * @param options.refetchInterval - Polling interval in ms (default: 2000)
 */
export function useDigitalMessages(
  captureId: string | undefined,
  options?: {
    enabled?: boolean;
    limit?: number;
    refetchInterval?: number;
  },
) {
  const { enabled = true, limit = 50, refetchInterval = 2000 } = options ?? {};

  const { data: pocsagChannels = [] } = usePOCSAGChannels(captureId);

  // Create a query for each POCSAG-enabled channel
  const messageQueries = useQueries({
    queries: pocsagChannels.map((channel) => ({
      queryKey: ["pocsag-messages", channel.id, limit],
      queryFn: async (): Promise<AggregatedPOCSAGMessage[]> => {
        const messages = await fetchPOCSAGMessages(channel.id, limit);
        return messages.map((msg) => ({
          ...msg,
          channelId: channel.id,
          channelName:
            channel.name ||
            channel.autoName ||
            `Channel ${channel.id.slice(0, 8)}`,
        }));
      },
      enabled: enabled && !!channel.id,
      refetchInterval,
    })),
  });

  // Aggregate all messages from all channels
  const allMessages = messageQueries.flatMap((q) => q.data ?? []);

  // Sort by timestamp descending (newest first)
  const sortedMessages = [...allMessages].sort(
    (a, b) => b.timestamp - a.timestamp,
  );

  // Dedupe by timestamp + address (in case of overlap)
  const seen = new Set<string>();
  const uniqueMessages = sortedMessages.filter((msg) => {
    const key = `${msg.timestamp}-${msg.address}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });

  const isLoading = messageQueries.some((q) => q.isLoading);
  const isError = messageQueries.some((q) => q.isError);
  const error = messageQueries.find((q) => q.error)?.error;

  return {
    messages: uniqueMessages,
    channels: pocsagChannels,
    isLoading,
    isError,
    error,
  };
}

export type { AggregatedPOCSAGMessage };
