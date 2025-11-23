import { useQuery } from "@tanstack/react-query";
import type { POCSAGMessage } from "../types";

async function fetchPOCSAGMessages(
  channelId: string,
  limit: number = 50,
  since?: number,
): Promise<POCSAGMessage[]> {
  const params = new URLSearchParams({ limit: limit.toString() });
  if (since !== undefined) {
    params.append("since", since.toString());
  }

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
 * Hook to fetch POCSAG messages from a channel with POCSAG decoding enabled.
 *
 * @param channelId - The channel ID to fetch messages from
 * @param options - Configuration options
 * @param options.enabled - Whether to enable fetching (default: true)
 * @param options.limit - Maximum number of messages to fetch (default: 50)
 * @param options.refetchInterval - Polling interval in ms (default: 2000)
 */
export function usePOCSAGMessages(
  channelId: string | undefined,
  options?: {
    enabled?: boolean;
    limit?: number;
    refetchInterval?: number;
  },
) {
  const { enabled = true, limit = 50, refetchInterval = 2000 } = options ?? {};

  return useQuery({
    queryKey: ["pocsag-messages", channelId, limit],
    queryFn: () => fetchPOCSAGMessages(channelId!, limit),
    enabled: enabled && !!channelId,
    refetchInterval,
  });
}
