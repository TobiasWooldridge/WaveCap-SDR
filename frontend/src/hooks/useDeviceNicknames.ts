import { useMutation, useQueryClient } from "@tanstack/react-query";

interface DeviceNickname {
  nickname: string | null;
}

async function getDeviceNickname(deviceId: string): Promise<DeviceNickname> {
  const response = await fetch(`/api/v1/devices/${deviceId}/name`);
  if (!response.ok) {
    throw new Error("Failed to fetch device nickname");
  }
  return response.json();
}

async function updateDeviceNickname(
  deviceId: string,
  nickname: string | null,
): Promise<DeviceNickname> {
  const response = await fetch(`/api/v1/devices/${deviceId}/name`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ nickname }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Failed to update device nickname");
  }

  return response.json();
}

export function useUpdateDeviceNickname() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ deviceId, nickname }: { deviceId: string; nickname: string | null }) =>
      updateDeviceNickname(deviceId, nickname),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["devices"] });
    },
  });
}

export { getDeviceNickname };
