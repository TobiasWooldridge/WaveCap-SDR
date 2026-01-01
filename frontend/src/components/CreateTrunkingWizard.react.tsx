import { useState, useEffect, useMemo } from "react";
import { Antenna, Plus, X } from "lucide-react";
import { useDevices } from "../hooks/useDevices";
import { useCaptures } from "../hooks/useCaptures";
import { useTrunkingRecipes, useCreateTrunkingSystem, useTrunkingSystems } from "../hooks/useTrunking";
import type { TrunkingRecipe, TrunkingProtocol } from "../types/trunking";
import { getDeviceDisplayName } from "../utils/device";
import { formatFrequencyMHz, formatFrequencyWithUnit } from "../utils/frequency";
import Flex from "./primitives/Flex.react";
import Button from "./primitives/Button.react";
import Spinner from "./primitives/Spinner.react";

interface CreateTrunkingWizardProps {
  onClose: () => void;
  onSuccess: (systemId: string) => void;
}

type WizardStep = "select-device" | "select-preset" | "configure";

export function CreateTrunkingWizard({ onClose, onSuccess }: CreateTrunkingWizardProps) {
  const { data: devices, isLoading: devicesLoading } = useDevices();
  const { data: captures } = useCaptures();
  const { data: trunkingSystems } = useTrunkingSystems();
  const { data: recipes, isLoading: recipesLoading } = useTrunkingRecipes();
  const createSystem = useCreateTrunkingSystem();

  const [step, setStep] = useState<WizardStep>("select-device");
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>("");
  const [selectedRecipe, setSelectedRecipe] = useState<TrunkingRecipe | null>(null);
  const [isCustom, setIsCustom] = useState(false);

  // Configuration state
  const [systemName, setSystemName] = useState("");
  const [systemId, setSystemId] = useState("");
  const [protocol, setProtocol] = useState<TrunkingProtocol>("p25_phase2");
  const [controlChannels, setControlChannels] = useState<number[]>([]);
  const [sampleRate, setSampleRate] = useState(8_000_000);
  const [error, setError] = useState<string | null>(null);

  // Get set of device IDs currently in use by running captures or trunking systems
  const usedDeviceIds = useMemo(() => {
    const used = new Set<string>();
    if (captures) {
      for (const c of captures) {
        if (c.state === "running" || c.state === "starting") {
          used.add(c.deviceId);
        }
      }
    }
    if (trunkingSystems) {
      for (const s of trunkingSystems) {
        if (s.deviceId && (s.state === "running" || s.state === "starting" || s.state === "searching" || s.state === "synced")) {
          used.add(s.deviceId);
        }
      }
    }
    return used;
  }, [captures, trunkingSystems]);

  // Filter to only available devices
  const availableDevices = useMemo(() => {
    if (!devices) return [];
    return devices.filter((d) => !usedDeviceIds.has(d.id));
  }, [devices, usedDeviceIds]);

  // Auto-select first available device
  useEffect(() => {
    if (availableDevices.length && !selectedDeviceId) {
      setSelectedDeviceId(availableDevices[0].id);
    }
    if (selectedDeviceId && availableDevices.length && !availableDevices.find(d => d.id === selectedDeviceId)) {
      setSelectedDeviceId(availableDevices[0].id);
    }
  }, [availableDevices, selectedDeviceId]);

  // Generate a unique system ID
  const generateSystemId = (name: string): string => {
    return name.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
  };

  // When selecting a recipe, populate config
  const handleSelectRecipe = (recipe: TrunkingRecipe) => {
    setSelectedRecipe(recipe);
    setIsCustom(false);
    setSystemName(recipe.name);
    setSystemId(generateSystemId(recipe.name));
    setProtocol(recipe.protocol);
    setControlChannels([...recipe.controlChannels]);
    setSampleRate(recipe.sampleRate);
    setStep("configure");
  };

  // Handle custom system selection
  const handleSelectCustom = () => {
    setSelectedRecipe(null);
    setIsCustom(true);
    setSystemName("Custom P25 System");
    setSystemId(generateSystemId("custom-p25"));
    setProtocol("p25_phase2");
    setControlChannels([851_000_000]); // Default to UHF P25 range
    setSampleRate(8_000_000);
    setStep("configure");
  };

  // Add a control channel
  const handleAddFrequency = () => {
    const last = controlChannels[controlChannels.length - 1] || 851_000_000;
    setControlChannels([...controlChannels, last]);
  };

  // Remove a control channel
  const handleRemoveFrequency = (index: number) => {
    if (controlChannels.length <= 1) return;
    setControlChannels(controlChannels.filter((_, i) => i !== index));
  };

  // Update a control channel frequency
  const handleUpdateFrequency = (index: number, valueMHz: number) => {
    const newChannels = [...controlChannels];
    newChannels[index] = valueMHz * 1_000_000;
    setControlChannels(newChannels);
  };

  // Create the trunking system
  const handleCreate = async () => {
    if (controlChannels.length === 0) {
      setError("At least one control channel frequency is required");
      return;
    }

    // Calculate center frequency from control channels
    const minFreq = Math.min(...controlChannels);
    const maxFreq = Math.max(...controlChannels);
    const centerHz = (minFreq + maxFreq) / 2;

    try {
      setError(null);
      const result = await createSystem.mutateAsync({
        id: systemId,
        name: systemName,
        protocol,
        controlChannels,
        centerHz,
        sampleRate,
        deviceId: selectedDeviceId || undefined,
      });
      onSuccess(result.id);
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create trunking system");
    }
  };

  const isLoading = devicesLoading || recipesLoading;

  return (
    <div className="modal show d-block" style={{ backgroundColor: "rgba(0,0,0,0.5)" }}>
      <div className="modal-dialog modal-lg modal-dialog-centered">
        <div className="modal-content">
          {/* Header */}
          <div className="modal-header">
            <Flex align="center" gap={2}>
              <Antenna size={24} />
              <h5 className="modal-title">
                {step === "select-device" && "Select SDR Device"}
                {step === "select-preset" && "Choose Trunking System"}
                {step === "configure" && `Configure ${systemName}`}
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
                {availableDevices.length === 0 ? (
                  <div className="alert alert-warning mb-0">
                    <strong>No devices available.</strong>
                    {devices && devices.length > 0 ? (
                      <span> All SDR devices are currently in use.</span>
                    ) : (
                      <span> No SDR devices detected.</span>
                    )}
                  </div>
                ) : (
                  <>
                    <p className="text-muted mb-2">
                      Select which SDR radio to use for trunking.
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
                              {device.driver} - {formatFrequencyWithUnit(device.freqMinHz, 0)}-{formatFrequencyWithUnit(device.freqMaxHz, 0)}
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
            ) : step === "select-preset" ? (
              /* Step 2: Preset/Custom Selection */
              <Flex direction="column" gap={4}>
                <p className="text-muted mb-0">
                  Select a preset trunking system or create a custom configuration.
                </p>

                {/* Custom option */}
                <div
                  className="card cursor-pointer border-primary"
                  style={{ cursor: "pointer" }}
                  onClick={handleSelectCustom}
                >
                  <div className="card-body">
                    <Flex align="center" gap={2}>
                      <Plus size={18} className="text-primary" />
                      <div>
                        <h6 className="card-title mb-0">Custom P25 System</h6>
                        <small className="text-muted">
                          Manually enter control channel frequencies
                        </small>
                      </div>
                    </Flex>
                  </div>
                </div>

                {/* Recipe presets */}
                {recipes && recipes.length > 0 && (
                  <>
                    <h6 className="text-muted mb-0 mt-2">Available Presets</h6>
                    <div className="row g-3">
                      {recipes.map((recipe) => (
                        <div key={recipe.id} className="col-md-6">
                          <div
                            className="card h-100"
                            style={{ cursor: "pointer" }}
                            onClick={() => handleSelectRecipe(recipe)}
                          >
                            <div className="card-body">
                              <Flex direction="column" gap={2}>
                                <Flex align="center" gap={2}>
                                  <Antenna size={18} className="text-success" />
                                  <h6 className="card-title mb-0">{recipe.name}</h6>
                                </Flex>
                                {recipe.description && (
                                  <p className="card-text small text-muted mb-0">
                                    {recipe.description}
                                  </p>
                                )}
                                <div className="small">
                                  <span className="badge bg-secondary me-2">
                                    {recipe.protocol.replace("_", " ").toUpperCase()}
                                  </span>
                                  <span className="text-muted">
                                    {recipe.controlChannels.length} control channel{recipe.controlChannels.length !== 1 ? "s" : ""}
                                  </span>
                                  {recipe.talkgroupCount > 0 && (
                                    <span className="text-muted ms-2">
                                      - {recipe.talkgroupCount} talkgroups
                                    </span>
                                  )}
                                </div>
                              </Flex>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </>
                )}
              </Flex>
            ) : (
              /* Step 3: Configuration */
              <Flex direction="column" gap={4}>
                {error && (
                  <div className="alert alert-danger">{error}</div>
                )}

                {/* System info */}
                <div className="alert alert-info mb-0">
                  <strong>{isCustom ? "Custom P25 System" : selectedRecipe?.name}</strong>
                  {selectedRecipe?.description && (
                    <p className="mb-0 small">{selectedRecipe.description}</p>
                  )}
                  <div className="small mt-1">
                    <strong>Device:</strong> {devices?.find(d => d.id === selectedDeviceId)?.label || selectedDeviceId}
                  </div>
                </div>

                {/* System Name */}
                <div>
                  <label className="form-label fw-semibold">System Name</label>
                  <input
                    type="text"
                    className="form-control"
                    value={systemName}
                    onChange={(e) => {
                      setSystemName(e.target.value);
                      setSystemId(generateSystemId(e.target.value));
                    }}
                  />
                  <small className="text-muted">ID: {systemId}</small>
                </div>

                {/* Protocol */}
                <div>
                  <label className="form-label fw-semibold">Protocol</label>
                  <select
                    className="form-select"
                    value={protocol}
                    onChange={(e) => setProtocol(e.target.value as TrunkingProtocol)}
                  >
                    <option value="p25_phase1">P25 Phase I (FDMA)</option>
                    <option value="p25_phase2">P25 Phase II (TDMA)</option>
                  </select>
                </div>

                {/* Control Channel Frequencies */}
                <div>
                  <label className="form-label fw-semibold">
                    Control Channel Frequencies (MHz)
                  </label>
                  <div className="border rounded p-2 bg-light">
                    {controlChannels.map((freq, index) => (
                      <Flex key={index} align="center" gap={2} className="mb-2">
                        <input
                          type="number"
                          className="form-control"
                          value={formatFrequencyMHz(freq, 5)}
                          onChange={(e) => handleUpdateFrequency(index, parseFloat(e.target.value) || 0)}
                          step="0.00625"
                          min="1"
                          max="3000"
                          style={{ flex: 1 }}
                        />
                        <button
                          type="button"
                          className="btn btn-outline-danger btn-sm"
                          onClick={() => handleRemoveFrequency(index)}
                          disabled={controlChannels.length <= 1}
                          title="Remove frequency"
                        >
                          <X size={16} />
                        </button>
                      </Flex>
                    ))}
                    <Button
                      use="secondary"
                      size="sm"
                      appearance="outline"
                      onClick={handleAddFrequency}
                    >
                      <Plus size={14} className="me-1" /> Add Frequency
                    </Button>
                  </div>
                  <small className="text-muted">
                    Enter all control channel frequencies for the trunking system.
                    The system will hunt across these channels to maintain lock.
                  </small>
                </div>

                {/* Sample Rate */}
                <div>
                  <label className="form-label fw-semibold">Sample Rate</label>
                  <select
                    className="form-select"
                    value={sampleRate}
                    onChange={(e) => setSampleRate(parseInt(e.target.value))}
                  >
                    <option value={2_000_000}>2 MHz</option>
                    <option value={4_000_000}>4 MHz</option>
                    <option value={6_000_000}>6 MHz</option>
                    <option value={8_000_000}>8 MHz (Recommended)</option>
                    <option value={10_000_000}>10 MHz</option>
                  </select>
                  <small className="text-muted">
                    Higher sample rates capture more bandwidth but use more CPU.
                  </small>
                </div>
              </Flex>
            )}
          </div>

          {/* Footer */}
          <div className="modal-footer">
            {/* Back button */}
            {step === "select-preset" && (
              <Button use="secondary" size="sm" onClick={() => setStep("select-device")}>
                Back
              </Button>
            )}
            {step === "configure" && (
              <Button use="secondary" size="sm" onClick={() => setStep("select-preset")}>
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
                onClick={() => setStep("select-preset")}
                disabled={!selectedDeviceId || availableDevices.length === 0}
              >
                Next
              </Button>
            )}
            {step === "configure" && (
              <Button
                use="success"
                size="sm"
                onClick={handleCreate}
                disabled={createSystem.isPending || !systemId || controlChannels.length === 0}
              >
                {createSystem.isPending ? (
                  <>
                    <Spinner size="sm" /> Creating...
                  </>
                ) : (
                  "Create Trunking System"
                )}
              </Button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
