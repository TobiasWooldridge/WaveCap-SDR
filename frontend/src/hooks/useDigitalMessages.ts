import { useMemo } from "react";
import { useQueries, useQuery } from "@tanstack/react-query";
import type { Channel, FlexMessage, POCSAGMessage } from "../types";

type PagerProtocol = "pocsag" | "flex";

interface AggregatedPagerMessage {
  protocol: PagerProtocol;
  channelId: string;
  channelName: string;
  address: number;
  messageType: string;
  message: string;
  timestamp: number;
  alias?: string | null;
}

interface PagerChannelSummary {
  id: string;
  name: string | null;
  autoName: string | null;
  protocols: PagerProtocol[];
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

async function fetchFlexMessages(
  channelId: string,
  limit: number = 50,
): Promise<FlexMessage[]> {
  const params = new URLSearchParams({ limit: limit.toString() });
  const response = await fetch(
    `/api/v1/channels/${channelId}/decode/flex?${params}`,
  );
  if (!response.ok) {
    if (response.status === 404) {
      return [];
    }
    throw new Error("Failed to fetch FLEX messages");
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
 * Hook to fetch all FLEX-enabled channels for a given capture.
 */
export function useFlexChannels(captureId: string | undefined) {
  return useQuery({
    queryKey: ["flex-channels", captureId],
    queryFn: async () => {
      if (!captureId) return [];
      const response = await fetch(`/api/v1/captures/${captureId}/channels`);
      if (!response.ok) {
        throw new Error("Failed to fetch channels");
      }
      const channels: Channel[] = await response.json();
      return channels.filter((ch) => ch.enableFlex);
    },
    enabled: !!captureId,
    refetchInterval: 5000,
  });
}

/**
 * Hook to aggregate pager messages from all POCSAG/FLEX-enabled channels on a capture.
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
  const { data: flexChannels = [] } = useFlexChannels(captureId);

  const pagerChannels = useMemo(
    () => [
      ...pocsagChannels.map((channel) => ({
        channel,
        protocol: "pocsag" as const,
      })),
      ...flexChannels.map((channel) => ({
        channel,
        protocol: "flex" as const,
      })),
    ],
    [pocsagChannels, flexChannels],
  );

  const channelSummaries = useMemo(() => {
    const channelMap = new Map<string, PagerChannelSummary>();
    for (const entry of pagerChannels) {
      const existing = channelMap.get(entry.channel.id);
      if (!existing) {
        channelMap.set(entry.channel.id, {
          id: entry.channel.id,
          name: entry.channel.name,
          autoName: entry.channel.autoName,
          protocols: [entry.protocol],
        });
      } else if (!existing.protocols.includes(entry.protocol)) {
        existing.protocols = [...existing.protocols, entry.protocol];
      }
    }
    return Array.from(channelMap.values());
  }, [pagerChannels]);

  // Create a query for each POCSAG-enabled channel
  const messageQueries = useQueries({
    queries: pagerChannels.map(({ channel, protocol }) => ({
      queryKey: [`${protocol}-messages`, channel.id, limit],
      queryFn: async (): Promise<AggregatedPagerMessage[]> => {
        if (protocol === "pocsag") {
          const messages = await fetchPOCSAGMessages(channel.id, limit);
          return messages.map((msg) => ({
            protocol,
            channelId: channel.id,
            channelName:
              channel.name ||
              channel.autoName ||
              `Channel ${channel.id.slice(0, 8)}`,
            address: msg.address,
            messageType: msg.messageType,
            message: msg.message,
            timestamp: msg.timestamp,
            alias: msg.alias ?? null,
          }));
        }

        const messages = await fetchFlexMessages(channel.id, limit);
        return messages.map((msg) => ({
          protocol,
          channelId: channel.id,
          channelName:
            channel.name ||
            channel.autoName ||
            `Channel ${channel.id.slice(0, 8)}`,
          address: msg.capcode,
          messageType: msg.messageType,
          message: msg.message,
          timestamp: msg.timestamp,
          alias: msg.alias ?? null,
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
    const key = `${msg.protocol}-${msg.timestamp}-${msg.address}-${msg.message}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });

  const isLoading = messageQueries.some((q) => q.isLoading);
  const isError = messageQueries.some((q) => q.isError);
  const error = messageQueries.find((q) => q.error)?.error;

  return {
    messages: uniqueMessages,
    channels: channelSummaries,
    isLoading,
    isError,
    error,
  };
}

export type { AggregatedPagerMessage, PagerChannelSummary, PagerProtocol };
