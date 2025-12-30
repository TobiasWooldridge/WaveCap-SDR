import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import type {
  TrunkingSystem,
  TrunkingRecipe,
  Talkgroup,
  ActiveCall,
  VocoderStatus,
  CreateSystemRequest,
  AddTalkgroupRequest,
  HuntMode,
  ControlChannel,
} from "../types/trunking";
import { formatFrequencyMHz } from "../utils/frequency";

const API_BASE = "/api/v1/trunking";

// Query keys
export const trunkingKeys = {
  all: ["trunking"] as const,
  recipes: () => [...trunkingKeys.all, "recipes"] as const,
  systems: () => [...trunkingKeys.all, "systems"] as const,
  system: (id: string) => [...trunkingKeys.systems(), id] as const,
  talkgroups: (systemId: string) =>
    [...trunkingKeys.system(systemId), "talkgroups"] as const,
  calls: (systemId: string) =>
    [...trunkingKeys.system(systemId), "calls"] as const,
  allCalls: () => [...trunkingKeys.all, "calls"] as const,
  vocoders: () => [...trunkingKeys.all, "vocoders"] as const,
};

// ============================================================================
// Recipes (presets/templates)
// ============================================================================

export function useTrunkingRecipes() {
  return useQuery<TrunkingRecipe[]>({
    queryKey: trunkingKeys.recipes(),
    queryFn: async () => {
      const response = await fetch(`${API_BASE}/recipes`);
      if (!response.ok) {
        throw new Error(`Failed to fetch trunking recipes: ${response.status}`);
      }
      return response.json();
    },
    staleTime: 60000, // Consider fresh for 1 minute (rarely changes)
  });
}

// ============================================================================
// Systems
// ============================================================================

export function useTrunkingSystems() {
  return useQuery<TrunkingSystem[]>({
    queryKey: trunkingKeys.systems(),
    queryFn: async () => {
      const response = await fetch(`${API_BASE}/systems`);
      if (!response.ok) {
        throw new Error(`Failed to fetch trunking systems: ${response.status}`);
      }
      return response.json();
    },
    refetchInterval: 5000, // Poll every 5 seconds for state updates
  });
}

export function useTrunkingSystem(systemId: string | null) {
  return useQuery<TrunkingSystem>({
    queryKey: trunkingKeys.system(systemId ?? ""),
    queryFn: async () => {
      const response = await fetch(`${API_BASE}/systems/${systemId}`);
      if (!response.ok) {
        throw new Error(`Failed to fetch trunking system: ${response.status}`);
      }
      return response.json();
    },
    enabled: !!systemId,
    refetchInterval: 2000, // Poll every 2 seconds for active system
  });
}

export function useCreateTrunkingSystem() {
  const queryClient = useQueryClient();

  return useMutation<TrunkingSystem, Error, CreateSystemRequest>({
    mutationFn: async (request) => {
      const response = await fetch(`${API_BASE}/systems`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          id: request.id,
          name: request.name,
          protocol: request.protocol,
          control_channels: request.controlChannels,
          center_hz: request.centerHz,
          sample_rate: request.sampleRate ?? 8_000_000,
          device_id: request.deviceId,
          max_voice_recorders: request.maxVoiceRecorders ?? 4,
          recording_path: request.recordingPath,
          record_unknown: request.recordUnknown ?? false,
          talkgroups: request.talkgroups,
        }),
      });
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Failed to create system");
      }
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: trunkingKeys.systems() });
    },
  });
}

export function useDeleteTrunkingSystem() {
  const queryClient = useQueryClient();

  return useMutation<void, Error, string>({
    mutationFn: async (systemId) => {
      const response = await fetch(`${API_BASE}/systems/${systemId}`, {
        method: "DELETE",
      });
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Failed to delete system");
      }
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: trunkingKeys.systems() });
    },
  });
}

export function useStartTrunkingSystem() {
  const queryClient = useQueryClient();

  return useMutation<void, Error, string>({
    mutationFn: async (systemId) => {
      const response = await fetch(`${API_BASE}/systems/${systemId}/start`, {
        method: "POST",
      });
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Failed to start system");
      }
    },
    onSuccess: (_, systemId) => {
      queryClient.invalidateQueries({ queryKey: trunkingKeys.system(systemId) });
      queryClient.invalidateQueries({ queryKey: trunkingKeys.systems() });
    },
  });
}

export function useStopTrunkingSystem() {
  const queryClient = useQueryClient();

  return useMutation<void, Error, string>({
    mutationFn: async (systemId) => {
      const response = await fetch(`${API_BASE}/systems/${systemId}/stop`, {
        method: "POST",
      });
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Failed to stop system");
      }
    },
    onSuccess: (_, systemId) => {
      queryClient.invalidateQueries({ queryKey: trunkingKeys.system(systemId) });
      queryClient.invalidateQueries({ queryKey: trunkingKeys.systems() });
    },
  });
}

// ============================================================================
// Talkgroups
// ============================================================================

export function useTalkgroups(systemId: string | null) {
  return useQuery<Talkgroup[]>({
    queryKey: trunkingKeys.talkgroups(systemId ?? ""),
    queryFn: async () => {
      const response = await fetch(
        `${API_BASE}/systems/${systemId}/talkgroups`
      );
      if (!response.ok) {
        throw new Error(`Failed to fetch talkgroups: ${response.status}`);
      }
      return response.json();
    },
    enabled: !!systemId,
  });
}

export function useAddTalkgroups() {
  const queryClient = useQueryClient();

  return useMutation<
    { added: number; updated: number; total: number },
    Error,
    { systemId: string; talkgroups: AddTalkgroupRequest[] }
  >({
    mutationFn: async ({ systemId, talkgroups }) => {
      const response = await fetch(
        `${API_BASE}/systems/${systemId}/talkgroups`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(talkgroups),
        }
      );
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Failed to add talkgroups");
      }
      return response.json();
    },
    onSuccess: (_, { systemId }) => {
      queryClient.invalidateQueries({
        queryKey: trunkingKeys.talkgroups(systemId),
      });
    },
  });
}

// ============================================================================
// Calls
// ============================================================================

export function useActiveCalls(systemId: string | null) {
  return useQuery<ActiveCall[]>({
    queryKey: trunkingKeys.calls(systemId ?? ""),
    queryFn: async () => {
      const response = await fetch(
        `${API_BASE}/systems/${systemId}/calls/active`
      );
      if (!response.ok) {
        throw new Error(`Failed to fetch active calls: ${response.status}`);
      }
      return response.json();
    },
    enabled: !!systemId,
    refetchInterval: 1000, // Poll every second for active calls
  });
}

export function useAllActiveCalls() {
  return useQuery<ActiveCall[]>({
    queryKey: trunkingKeys.allCalls(),
    queryFn: async () => {
      const response = await fetch(`${API_BASE}/calls`);
      if (!response.ok) {
        throw new Error(`Failed to fetch active calls: ${response.status}`);
      }
      return response.json();
    },
    refetchInterval: 1000, // Poll every second
  });
}

// ============================================================================
// Vocoders
// ============================================================================

export function useVocoderStatus() {
  return useQuery<VocoderStatus>({
    queryKey: trunkingKeys.vocoders(),
    queryFn: async () => {
      const response = await fetch(`${API_BASE}/vocoders`);
      if (!response.ok) {
        throw new Error(`Failed to fetch vocoder status: ${response.status}`);
      }
      return response.json();
    },
    staleTime: 60000, // Consider fresh for 1 minute
  });
}

// ============================================================================
// Hunt Mode Control
// ============================================================================

export function useControlChannels(systemId: string | null) {
  return useQuery<ControlChannel[]>({
    queryKey: [...trunkingKeys.system(systemId ?? ""), "channels"],
    queryFn: async () => {
      const response = await fetch(`${API_BASE}/systems/${systemId}/channels`);
      if (!response.ok) {
        throw new Error(`Failed to fetch control channels: ${response.status}`);
      }
      return response.json();
    },
    enabled: !!systemId,
    refetchInterval: 2000, // Poll every 2 seconds to get signal measurements
  });
}

export function useSetHuntMode() {
  const queryClient = useQueryClient();

  return useMutation<
    { status: string; mode: string; lockedFrequencyHz: number | null },
    Error,
    { systemId: string; mode: HuntMode; lockedFrequency?: number }
  >({
    mutationFn: async ({ systemId, mode, lockedFrequency }) => {
      const response = await fetch(`${API_BASE}/systems/${systemId}/hunt-mode`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode, lockedFrequency }),
      });
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Failed to set hunt mode");
      }
      return response.json();
    },
    onSuccess: (_, { systemId }) => {
      queryClient.invalidateQueries({ queryKey: trunkingKeys.system(systemId) });
      queryClient.invalidateQueries({ queryKey: trunkingKeys.systems() });
    },
  });
}

export function useSetChannelEnabled() {
  const queryClient = useQueryClient();

  return useMutation<
    { status: string; frequencyHz: number; enabled: boolean },
    Error,
    { systemId: string; frequencyHz: number; enabled: boolean }
  >({
    mutationFn: async ({ systemId, frequencyHz, enabled }) => {
      const freqMhz = formatFrequencyMHz(frequencyHz, 6);
      const response = await fetch(
        `${API_BASE}/systems/${systemId}/channels/${freqMhz}/enabled`,
        {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ enabled }),
        }
      );
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Failed to set channel enabled");
      }
      return response.json();
    },
    onSuccess: (_, { systemId }) => {
      queryClient.invalidateQueries({ queryKey: trunkingKeys.system(systemId) });
    },
  });
}

export function useLockToChannel() {
  const queryClient = useQueryClient();

  return useMutation<
    { status: string; mode: string; lockedFrequencyHz: number },
    Error,
    { systemId: string; frequencyHz: number }
  >({
    mutationFn: async ({ systemId, frequencyHz }) => {
      const freqMhz = formatFrequencyMHz(frequencyHz, 6);
      const response = await fetch(
        `${API_BASE}/systems/${systemId}/channels/${freqMhz}/lock`,
        { method: "POST" }
      );
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Failed to lock to channel");
      }
      return response.json();
    },
    onSuccess: (_, { systemId }) => {
      queryClient.invalidateQueries({ queryKey: trunkingKeys.system(systemId) });
      queryClient.invalidateQueries({ queryKey: trunkingKeys.systems() });
    },
  });
}

export function useTriggerScan() {
  const queryClient = useQueryClient();

  return useMutation<
    { status: string; measurements: Record<string, unknown>; bestChannelHz: number | null },
    Error,
    string
  >({
    mutationFn: async (systemId) => {
      const response = await fetch(`${API_BASE}/systems/${systemId}/scan`, {
        method: "POST",
      });
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Failed to trigger scan");
      }
      return response.json();
    },
    onSuccess: (_, systemId) => {
      queryClient.invalidateQueries({ queryKey: trunkingKeys.system(systemId) });
    },
  });
}
