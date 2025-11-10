import { useQuery } from "@tanstack/react-query";

interface FrequencyLabelProps {
  frequencyHz: number;
  autoName?: string | null;  // If provided, use this instead of fetching
}

const API_BASE = "/api/v1";

async function fetchFrequencyName(frequencyHz: number): Promise<string | null> {
  const response = await fetch(
    `${API_BASE}/frequency/identify?frequency_hz=${frequencyHz}`
  );

  if (!response.ok) {
    return null;
  }

  const data = await response.json();
  return data?.name || null;
}

export const FrequencyLabel = ({ frequencyHz, autoName }: FrequencyLabelProps) => {
  // If autoName is provided, use it directly
  const { data: fetchedName } = useQuery({
    queryKey: ["frequencyName", frequencyHz],
    queryFn: () => fetchFrequencyName(frequencyHz),
    enabled: !autoName, // Only fetch if autoName not provided
    staleTime: Infinity, // Frequency names don't change
  });

  const displayName = autoName || fetchedName;

  if (!displayName) {
    return null;
  }

  return <span>{displayName}</span>;
};
