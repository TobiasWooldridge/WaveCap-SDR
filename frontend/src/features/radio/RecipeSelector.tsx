import { useState } from "react";
import { Loader2 } from "lucide-react";
import type { Capture, Recipe } from "../../types";
import { useRecipes } from "../../hooks/useRecipes";
import { useUpdateCapture } from "../../hooks/useCaptures";
import { useCreateChannel, useStartChannel, useDeleteChannel, useChannels } from "../../hooks/useChannels";
import { useToast } from "../../hooks/useToast";
import Button from "../../components/primitives/Button.react";
import Flex from "../../components/primitives/Flex.react";

interface RecipeSelectorProps {
  capture: Capture;
}

// Content-only version for use in accordions
export function RecipeSelectorContent({ capture }: RecipeSelectorProps) {
  const [isApplying, setIsApplying] = useState(false);
  const [customFrequency, setCustomFrequency] = useState<number | null>(null);

  const { data: recipes, isLoading } = useRecipes(capture.deviceId);
  const { data: existingChannels } = useChannels(capture.id);
  const updateCapture = useUpdateCapture();
  const createChannel = useCreateChannel();
  const startChannel = useStartChannel(capture.id);
  const deleteChannel = useDeleteChannel();
  const toast = useToast();

  // Group recipes by category
  const recipesByCategory = recipes?.reduce((acc, recipe) => {
    if (!acc[recipe.category]) {
      acc[recipe.category] = [];
    }
    acc[recipe.category].push(recipe);
    return acc;
  }, {} as Record<string, Recipe[]>) || {};

  const handleApplyRecipe = async (recipe: Recipe) => {
    if (isApplying) return;

    const confirmMsg = existingChannels && existingChannels.length > 0
      ? `Apply "${recipe.name}"? This will delete ${existingChannels.length} existing channel(s) and update capture settings.`
      : `Apply "${recipe.name}"? This will update capture settings and create ${recipe.channels.length} channel(s).`;

    if (!confirm(confirmMsg)) return;

    setIsApplying(true);

    try {
      // Determine center frequency
      const centerHz = recipe.allowFrequencyInput && customFrequency
        ? customFrequency * 1_000_000
        : recipe.centerHz;

      // Delete existing channels
      if (existingChannels) {
        for (const ch of existingChannels) {
          await deleteChannel.mutateAsync(ch.id);
        }
      }

      // Update capture settings
      await updateCapture.mutateAsync({
        captureId: capture.id,
        request: {
          centerHz,
          sampleRate: recipe.sampleRate,
          gain: recipe.gain ?? undefined,
          bandwidth: recipe.bandwidth ?? undefined,
        },
      });

      // Create channels from recipe
      for (const channelDef of recipe.channels) {
        const result = await createChannel.mutateAsync({
          captureId: capture.id,
          request: {
            mode: channelDef.mode as "wbfm" | "nbfm" | "am" | "ssb" | "raw" | "p25" | "dmr",
            offsetHz: channelDef.offsetHz,
            audioRate: 48000,
            squelchDb: channelDef.squelchDb,
            name: channelDef.name,
          },
        });
        await startChannel.mutateAsync(result.id);
      }

      toast.success(`Applied recipe: ${recipe.name}`);
      setCustomFrequency(null);
    } catch (error) {
      console.error("Failed to apply recipe:", error);
      toast.error(error instanceof Error ? error.message : "Failed to apply recipe");
    } finally {
      setIsApplying(false);
    }
  };

  if (isLoading) {
    return (
      <div className="text-center py-2">
        <Loader2 size={16} className="animate-spin" />
        <small className="text-muted d-block">Loading recipes...</small>
      </div>
    );
  }

  if (Object.keys(recipesByCategory).length === 0) {
    return <small className="text-muted">No recipes available for this device.</small>;
  }

  return (
    <Flex direction="column" gap={2}>
      {Object.entries(recipesByCategory).map(([category, categoryRecipes]) => (
        <div key={category}>
          <small className="text-muted fw-semibold text-uppercase">{category}</small>
          <div className="list-group list-group-flush mt-1">
            {categoryRecipes.map((recipe) => (
              <RecipeItem
                key={recipe.id}
                recipe={recipe}
                isApplying={isApplying}
                customFrequency={customFrequency}
                onCustomFrequencyChange={setCustomFrequency}
                onApply={() => handleApplyRecipe(recipe)}
              />
            ))}
          </div>
        </div>
      ))}
    </Flex>
  );
}

// Legacy wrapper for backwards compatibility
export function RecipeSelector({ capture }: RecipeSelectorProps) {
  return <RecipeSelectorContent capture={capture} />;
}

interface RecipeItemProps {
  recipe: Recipe;
  isApplying: boolean;
  customFrequency: number | null;
  onCustomFrequencyChange: (freq: number | null) => void;
  onApply: () => void;
}

function RecipeItem({ recipe, isApplying, customFrequency, onCustomFrequencyChange, onApply }: RecipeItemProps) {
  const [showFreqInput, setShowFreqInput] = useState(false);

  const handleClick = () => {
    if (recipe.allowFrequencyInput) {
      setShowFreqInput(!showFreqInput);
      if (!showFreqInput) {
        onCustomFrequencyChange(recipe.centerHz / 1_000_000);
      }
    } else {
      onApply();
    }
  };

  return (
    <div className="list-group-item p-1">
      <Flex direction="column" gap={1}>
        <Flex justify="between" align="center">
          <div style={{ flex: 1, minWidth: 0 }}>
            <div className="small fw-semibold text-truncate">{recipe.name}</div>
            <div className="text-muted" style={{ fontSize: "0.7rem" }}>
              {recipe.description}
            </div>
            {recipe.channels.length > 0 && (
              <div className="text-muted" style={{ fontSize: "0.65rem" }}>
                {recipe.channels.length} channel{recipe.channels.length !== 1 ? "s" : ""}
              </div>
            )}
          </div>
          <Button
            use="primary"
            size="sm"
            onClick={handleClick}
            disabled={isApplying}
            className="ms-2"
          >
            {isApplying ? <Loader2 size={12} className="animate-spin" /> : "Apply"}
          </Button>
        </Flex>

        {showFreqInput && recipe.allowFrequencyInput && (
          <Flex gap={1} align="center" className="mt-1">
            <input
              type="number"
              className="form-control form-control-sm"
              style={{ width: "100px" }}
              value={customFrequency ?? recipe.centerHz / 1_000_000}
              onChange={(e) => onCustomFrequencyChange(parseFloat(e.target.value))}
              step="0.001"
              placeholder="MHz"
            />
            <small className="text-muted">MHz</small>
            <Button use="success" size="sm" onClick={onApply} disabled={isApplying}>
              {isApplying ? <Loader2 size={12} className="animate-spin" /> : "Go"}
            </Button>
          </Flex>
        )}
      </Flex>
    </div>
  );
}
