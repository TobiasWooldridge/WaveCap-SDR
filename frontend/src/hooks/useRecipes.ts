import { useQuery } from "@tanstack/react-query";
import { Recipe } from "../types";

const API_BASE = "/api/v1";

async function fetchRecipes(): Promise<Recipe[]> {
  const response = await fetch(`${API_BASE}/recipes`);
  if (!response.ok) {
    throw new Error("Failed to fetch recipes");
  }
  return response.json();
}

export function useRecipes() {
  return useQuery({
    queryKey: ["recipes"],
    queryFn: fetchRecipes,
  });
}
