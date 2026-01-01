import { useState, useEffect, useMemo, useRef } from "react";
import { Wand2, Radio } from "lucide-react";
import { useRecipes } from "../hooks/useRecipes";
import { useDevices } from "../hooks/useDevices";
import { useCreateCapture, useCaptures } from "../hooks/useCaptures";
import { useCreateChannel } from "../hooks/useChannels";
import { useToast } from "../hooks/useToast";
import type { Recipe } from "../types";
import { getDeviceDisplayName } from "../utils/device";
import { formatFrequencyMHz, formatFrequencyWithUnit } from "../utils/frequency";
import Flex from "./primitives/Flex.react";
import Button from "./primitives/Button.react";
import Spinner from "./primitives/Spinner.react";

interface CreateCaptureWizardProps {
  onClose: () => void;
  onSuccess: (captureId: string) => void;
  initialDeviceId?: string;
}

export function CreateCaptureWizard({
  onClose,
  onSuccess,
  initialDeviceId,
}: CreateCaptureWizardProps) {
  const { data: devices, isLoading: devicesLoading } = useDevices();
  const { data: captures } = useCaptures();
  const createCapture = useCreateCapture();
  const createChannel = useCreateChannel();
  const toast = useToast();

  const [step, setStep] = useState<"select-device" | "select-recipe" | "configure">(
    "select-device",
  );
  const [selectedRecipe, setSelectedRecipe] = useState<Recipe | null>(null);
  const [customFrequency, setCustomFrequency] = useState<number>(100);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>("");
  const autoAdvancedRef = useRef(false);

  // Get set of device IDs currently in use by running/starting captures
  const usedDeviceIds = useMemo(() => {
    if (!captures) return new Set<string>();
    return new Set(
      captures
        .filter((c) => c.state === "running" || c.state === "starting")
        .map((c) => c.deviceId)
    );
  }, [captures]);

  const lockedDevice = useMemo(() => {
    if (!devices || !initialDeviceId) return null;
    return devices.find((device) => device.id === initialDeviceId) ?? null;
  }, [devices, initialDeviceId]);

  const lockedDeviceAvailable = useMemo(() => {
    if (!lockedDevice) return false;
    return !usedDeviceIds.has(lockedDevice.id);
  }, [lockedDevice, usedDeviceIds]);

  // Filter to only available (not in use) devices
  const availableDevices = useMemo(() => {
    if (!devices) return [];
    return devices.filter((d) => !usedDeviceIds.has(d.id));
  }, [devices, usedDeviceIds]);

  // Default to first available device when devices become available
  useEffect(() => {
    if (initialDeviceId) {
      if (selectedDeviceId !== initialDeviceId) {
        setSelectedDeviceId(initialDeviceId);
      }
      if (lockedDeviceAvailable && !autoAdvancedRef.current) {
        setStep("select-recipe");
        autoAdvancedRef.current = true;
      }
      return;
    }

    if (!availableDevices.length) return;

    if (!selectedDeviceId) {
      setSelectedDeviceId(availableDevices[0].id);
      return;
    }
    if (!availableDevices.find((d) => d.id === selectedDeviceId)) {
      setSelectedDeviceId(availableDevices[0].id);
    }
  }, [
    availableDevices,
    selectedDeviceId,
    initialDeviceId,
    lockedDeviceAvailable,
  ]);

  // Fetch recipes adjusted for selected device's capabilities
  const { data: recipes, isLoading: recipesLoading } = useRecipes(selectedDeviceId || undefined);

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
    setCustomFrequency(parseFloat(formatFrequencyMHz(recipe.centerHz, 3)));
    setStep("configure");
  };

  const handleCreate = async () => {
    if (!selectedRecipe || !selectedDeviceId) return;

    const centerHz = selectedRecipe.allowFrequencyInput
      ? customFrequency * 1_000_000
      : selectedRecipe.centerHz;

    try {
      // Create the capture without default channel since we'll create recipe-specific channels
      const newCapture = await createCapture.mutateAsync({
        deviceId: selectedDeviceId,
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
            mode: channelDef.mode,
            offsetHz: channelDef.offsetHz,
            squelchDb: channelDef.squelchDb,
          },
        });
      }

      onSuccess(newCapture.id);
      onClose();
    } catch (error) {
      console.error("Failed to create capture from recipe:", error);
      toast.error(`Failed to create capture: ${error instanceof Error ? error.message : "Unknown error"}`);
    }
  };

  const isLoading = recipesLoading || devicesLoading;
  const canProceedFromDeviceStep = initialDeviceId
    ? lockedDeviceAvailable
    : availableDevices.length > 0 && Boolean(selectedDeviceId);

  return (
    <div className="modal show d-block" style={{ backgroundColor: "rgba(0,0,0,0.5)" }} onClick={onClose}>
      <div className="modal-dialog modal-lg modal-dialog-centered" onClick={(e) => e.stopPropagation()}>
        <div className="modal-content">
          {/* Header */}
          <div className="modal-header">
            <Flex align="center" gap={2}>
              <Wand2 size={24} />
              <h5 className="modal-title">
                {step === "select-device" &&
                  (initialDeviceId ? "Selected SDR Device" : "Select SDR Device")}
                {step === "select-recipe" && "Choose a Recipe"}
                {step === "configure" && `Configure ${selectedRecipe?.name}`}
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
            ) : step === "select-device" ? (
              /* Step 1: Device Selection */
              <Flex direction="column" gap={3}>
                {initialDeviceId ? (
                  <>
                    <p className="text-muted mb-2">
                      This capture will use the selected device.
                    </p>
                    {lockedDevice ? (
                      <div className="list-group">
                        <div className="list-group-item d-flex justify-content-between align-items-center">
                          <div>
                            <div className="fw-semibold">{getDeviceDisplayName(lockedDevice)}</div>
                            <small className="text-muted">
                              {lockedDevice.driver} •{" "}
                              {formatFrequencyWithUnit(lockedDevice.freqMinHz, 0)}-
                              {formatFrequencyWithUnit(lockedDevice.freqMaxHz, 0)}
                            </small>
                          </div>
                          <span className="badge bg-primary">Locked</span>
                        </div>
                      </div>
                    ) : (
                      <div className="alert alert-warning mb-0">
                        <strong>Device not available.</strong>
                        <span className="ms-1">Reconnect the SDR and try again.</span>
                      </div>
                    )}
                    {!lockedDeviceAvailable && lockedDevice && (
                      <div className="alert alert-warning mb-0">
                        <strong>Device in use.</strong>
                        <span className="ms-1">
                          Stop the existing capture to create a new one.
                        </span>
                      </div>
                    )}
                  </>
                ) : availableDevices.length === 0 ? (
                  <div className="alert alert-warning mb-0">
                    <strong>No devices available.</strong>
                    {devices && devices.length > 0 ? (
                      <span> All SDR devices are currently in use by other captures.</span>
                    ) : (
                      <span> No SDR devices detected.</span>
                    )}
                  </div>
                ) : (
                  <>
                    <p className="text-muted mb-2">
                      Select which SDR radio to use for this capture.
                      {usedDeviceIds.size > 0 && (
                        <span className="ms-1">({usedDeviceIds.size} device{usedDeviceIds.size > 1 ? "s" : ""} already in use)</span>
                      )}
                    </p>
                    <div className="list-group">
                      {availableDevices.map((device) => (
                        <button
                          key={device.id}
                          type="button"
                          className={`list-group-item list-group-item-action d-flex justify-content-between align-items-center ${
                            selectedDeviceId === device.id ? "active" : ""
                          }`}
                          onClick={() => setSelectedDeviceId(device.id)}
                        >
                          <div>
                            <div className="fw-semibold">{getDeviceDisplayName(device)}</div>
                            <small className={selectedDeviceId === device.id ? "text-white-50" : "text-muted"}>
                              {device.driver} • {formatFrequencyWithUnit(device.freqMinHz, 0)}-{formatFrequencyWithUnit(device.freqMaxHz, 0)}
                            </small>
                          </div>
                          {selectedDeviceId === device.id && (
                            <span className="badge bg-light text-primary">Selected</span>
                          )}
                        </button>
                      ))}
                    </div>
                  </>
                )}
              </Flex>
            ) : step === "select-recipe" ? (
              /* Step 2: Recipe Selection */
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
                  <div className="small mt-1">
                    <strong>Device:</strong> {devices?.find(d => d.id === selectedDeviceId)?.label || selectedDeviceId}
                  </div>
                </div>

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
            {/* Back button */}
            {step === "select-recipe" && !initialDeviceId && (
              <Button use="secondary" size="sm" onClick={() => setStep("select-device")}>
                Back
              </Button>
            )}
            {step === "configure" && (
              <Button use="secondary" size="sm" onClick={() => setStep("select-recipe")}>
                Back
              </Button>
            )}

            <Button use="secondary" size="sm" onClick={onClose}>
              Cancel
            </Button>

            {/* Next/Create button */}
            {step === "select-device" && (
              <Button
                use="primary"
                size="sm"
                onClick={() => setStep("select-recipe")}
                disabled={!canProceedFromDeviceStep}
              >
                Next
              </Button>
            )}
            {step === "configure" && (
              <Button
                use="success"
                size="sm"
                onClick={handleCreate}
                disabled={createCapture.isPending || createChannel.isPending || !selectedDeviceId}
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
