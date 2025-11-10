import { useState } from "react";
import { Wand2, Radio } from "lucide-react";
import { useRecipes } from "../hooks/useRecipes";
import { useDevices } from "../hooks/useDevices";
import { useCreateCapture } from "../hooks/useCaptures";
import { useCreateChannel } from "../hooks/useChannels";
import type { Recipe } from "../types";
import Flex from "./primitives/Flex.react";
import Button from "./primitives/Button.react";
import Spinner from "./primitives/Spinner.react";

interface CreateCaptureWizardProps {
  onClose: () => void;
  onSuccess: (captureId: string) => void;
}

export function CreateCaptureWizard({ onClose, onSuccess }: CreateCaptureWizardProps) {
  const { data: recipes, isLoading: recipesLoading } = useRecipes();
  const { data: devices, isLoading: devicesLoading } = useDevices();
  const createCapture = useCreateCapture();
  const createChannel = useCreateChannel();

  const [step, setStep] = useState<"select-recipe" | "configure">("select-recipe");
  const [selectedRecipe, setSelectedRecipe] = useState<Recipe | null>(null);
  const [customFrequency, setCustomFrequency] = useState<number>(100);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>("");

  // Group recipes by category
  const recipesByCategory = recipes?.reduce((acc, recipe) => {
    if (!acc[recipe.category]) {
      acc[recipe.category] = [];
    }
    acc[recipe.category].push(recipe);
    return acc;
  }, {} as Record<string, Recipe[]>) || {};

  const handleSelectRecipe = (recipe: Recipe) => {
    setSelectedRecipe(recipe);
    setCustomFrequency(recipe.centerHz / 1_000_000);
    setStep("configure");
  };

  const handleCreate = async () => {
    if (!selectedRecipe) return;

    const centerHz = selectedRecipe.allowFrequencyInput
      ? customFrequency * 1_000_000
      : selectedRecipe.centerHz;

    try {
      // Create the capture without default channel since we'll create recipe-specific channels
      const newCapture = await createCapture.mutateAsync({
        deviceId: selectedDeviceId || undefined,
        centerHz,
        sampleRate: selectedRecipe.sampleRate,
        gain: selectedRecipe.gain || undefined,
        bandwidth: selectedRecipe.bandwidth || undefined,
        createDefaultChannel: false,
      });

      // Create channels from recipe
      for (const channelDef of selectedRecipe.channels) {
        await createChannel.mutateAsync({
          captureId: newCapture.id,
          request: {
            mode: channelDef.mode as any,
            offsetHz: channelDef.offsetHz,
            squelchDb: channelDef.squelchDb,
          },
        });
      }

      onSuccess(newCapture.id);
      onClose();
    } catch (error) {
      console.error("Failed to create capture from recipe:", error);
    }
  };

  const isLoading = recipesLoading || devicesLoading;

  return (
    <div className="modal show d-block" style={{ backgroundColor: "rgba(0,0,0,0.5)" }}>
      <div className="modal-dialog modal-lg modal-dialog-centered">
        <div className="modal-content">
          {/* Header */}
          <div className="modal-header">
            <Flex align="center" gap={2}>
              <Wand2 size={24} />
              <h5 className="modal-title">
                {step === "select-recipe" ? "Choose a Recipe" : `Configure ${selectedRecipe?.name}`}
              </h5>
            </Flex>
            <button type="button" className="btn-close" onClick={onClose} />
          </div>

          {/* Body */}
          <div className="modal-body" style={{ maxHeight: "70vh", overflowY: "auto" }}>
            {isLoading ? (
              <Flex justify="center" align="center" className="py-5">
                <Spinner />
              </Flex>
            ) : step === "select-recipe" ? (
              /* Recipe Selection */
              <Flex direction="column" gap={4}>
                {Object.keys(recipesByCategory).length === 0 ? (
                  <div className="text-center text-muted py-4">
                    No recipes available. Check your configuration.
                  </div>
                ) : (
                  Object.entries(recipesByCategory).map(([category, categoryRecipes]) => (
                    <div key={category}>
                      <h6 className="text-muted mb-3">{category}</h6>
                      <div className="row g-3">
                        {categoryRecipes.map((recipe) => (
                          <div key={recipe.id} className="col-md-6">
                            <div
                              className="card h-100 cursor-pointer"
                              style={{ cursor: "pointer" }}
                              onClick={() => handleSelectRecipe(recipe)}
                            >
                              <div className="card-body">
                                <Flex direction="column" gap={2}>
                                  <Flex align="center" gap={2}>
                                    <Radio size={18} className="text-primary" />
                                    <h6 className="card-title mb-0">{recipe.name}</h6>
                                  </Flex>
                                  <p className="card-text small text-muted mb-0">
                                    {recipe.description}
                                  </p>
                                  {recipe.channels.length > 0 && (
                                    <div className="small">
                                      <strong>{recipe.channels.length}</strong> channel
                                      {recipe.channels.length !== 1 ? "s" : ""}
                                    </div>
                                  )}
                                </Flex>
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  ))
                )}
              </Flex>
            ) : (
              /* Configuration */
              <Flex direction="column" gap={4}>
                <div className="alert alert-info">
                  <strong>{selectedRecipe?.name}</strong>
                  <p className="mb-0 small">{selectedRecipe?.description}</p>
                </div>

                {/* Device Selection */}
                <Flex direction="column" gap={2}>
                  <label className="form-label fw-semibold">SDR Device</label>
                  <select
                    className="form-select"
                    value={selectedDeviceId}
                    onChange={(e) => setSelectedDeviceId(e.target.value)}
                  >
                    <option value="">Auto-select device</option>
                    {devices?.map((device) => (
                      <option key={device.id} value={device.id}>
                        {device.driver.toUpperCase()} - {device.label.substring(0, 50)}
                      </option>
                    ))}
                  </select>
                </Flex>

                {/* Frequency Input (if allowed) */}
                {selectedRecipe?.allowFrequencyInput && (
                  <Flex direction="column" gap={2}>
                    <label className="form-label fw-semibold">
                      {selectedRecipe.frequencyLabel || "Frequency (MHz)"}
                    </label>
                    <input
                      type="number"
                      className="form-control"
                      value={customFrequency}
                      onChange={(e) => setCustomFrequency(parseFloat(e.target.value))}
                      step="0.1"
                      min="1"
                      max="6000"
                    />
                  </Flex>
                )}

                {/* Channels Preview */}
                {selectedRecipe && selectedRecipe.channels.length > 0 && (
                  <Flex direction="column" gap={2}>
                    <label className="form-label fw-semibold">Channels to Create</label>
                    <div className="list-group">
                      {selectedRecipe.channels.map((channel, idx) => (
                        <div key={idx} className="list-group-item">
                          <Flex justify="between" align="center">
                            <div>
                              <strong>{channel.name}</strong>
                              <div className="small text-muted">
                                {channel.mode.toUpperCase()} • Offset:{" "}
                                {(channel.offsetHz / 1000).toFixed(0)} kHz
                              </div>
                            </div>
                            <span className="badge bg-secondary">{channel.mode}</span>
                          </Flex>
                        </div>
                      ))}
                    </div>
                  </Flex>
                )}
              </Flex>
            )}
          </div>

          {/* Footer */}
          <div className="modal-footer">
            {step === "configure" && (
              <Button use="secondary" size="sm" onClick={() => setStep("select-recipe")}>
                Back
              </Button>
            )}
            <Button use="secondary" size="sm" onClick={onClose}>
              Cancel
            </Button>
            {step === "configure" && (
              <Button
                use="success"
                size="sm"
                onClick={handleCreate}
                disabled={createCapture.isPending || createChannel.isPending}
              >
                {createCapture.isPending || createChannel.isPending ? (
                  <>
                    <Spinner size="sm" /> Creating...
                  </>
                ) : (
                  "Create Capture"
                )}
              </Button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
