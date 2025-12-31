import { useState } from "react";
import { ChevronDown, ChevronUp } from "lucide-react";
import type { Capture, Channel, UpdateChannelRequest } from "../../types";
import { useUpdateChannel } from "../../hooks/useChannels";
import { useToast } from "../../hooks/useToast";
import FrequencySelector from "../../components/primitives/FrequencySelector.react";
import Slider from "../../components/primitives/Slider.react";
import Flex from "../../components/primitives/Flex.react";

interface ChannelSettingsProps {
  channel: Channel;
  capture: Capture;
}

export function ChannelSettings({ channel, capture }: ChannelSettingsProps) {
  const [showDsp, setShowDsp] = useState(false);
  const [showAgc, setShowAgc] = useState(false);
  const [showPager, setShowPager] = useState(false);
  const [newNotchFreq, setNewNotchFreq] = useState("");

  const updateChannel = useUpdateChannel(capture.id);
  const toast = useToast();

  const updateSetting = (request: UpdateChannelRequest) => {
    updateChannel.mutate(
      { channelId: channel.id, request },
      { onError: (error: Error) => toast.error(error.message) },
    );
  };

  const channelFrequency = capture.centerHz + channel.offsetHz;
  const spectrumMin = capture.centerHz - capture.sampleRate / 2;
  const spectrumMax = capture.centerHz + capture.sampleRate / 2;

  const handleAddNotch = () => {
    const freq = parseFloat(newNotchFreq);
    if (isNaN(freq) || freq <= 0 || freq > 20000) {
      toast.error("Notch frequency must be between 0 and 20000 Hz");
      return;
    }
    const currentNotches = channel.notchFrequencies || [];
    if (currentNotches.includes(freq)) {
      toast.error("Frequency already in notch list");
      return;
    }
    if (currentNotches.length >= 10) {
      toast.error("Maximum 10 notch filters");
      return;
    }
    updateSetting({ notchFrequencies: [...currentNotches, freq] });
    setNewNotchFreq("");
    toast.success(`Added notch at ${freq} Hz`);
  };

  const handleRemoveNotch = (freq: number) => {
    const currentNotches = channel.notchFrequencies || [];
    updateSetting({
      notchFrequencies: currentNotches.filter((f) => f !== freq),
    });
    toast.success(`Removed notch at ${freq} Hz`);
  };

  return (
    <div className="border-top pt-2 mt-1">
      <Flex direction="column" gap={2}>
        {/* Mode */}
        <Flex direction="column" gap={1}>
          <label className="form-label small mb-0">Mode</label>
          <select
            className="form-select form-select-sm"
            value={channel.mode}
            onChange={(e) =>
              updateSetting({ mode: e.target.value as Channel["mode"] })
            }
          >
            <option value="wbfm">WBFM</option>
            <option value="nbfm">NBFM</option>
            <option value="am">AM</option>
            <option value="ssb">SSB</option>
            <option value="raw">Raw IQ</option>
            <option value="p25">P25</option>
            <option value="dmr">DMR</option>
          </select>
        </Flex>

        {/* SSB Mode */}
        {channel.mode === "ssb" && (
          <Flex direction="column" gap={1}>
            <label className="form-label small mb-0">SSB Mode</label>
            <select
              className="form-select form-select-sm"
              value={channel.ssbMode || "usb"}
              onChange={(e) =>
                updateSetting({ ssbMode: e.target.value as "usb" | "lsb" })
              }
            >
              <option value="usb">USB (Upper Sideband)</option>
              <option value="lsb">LSB (Lower Sideband)</option>
            </select>
          </Flex>
        )}

        {/* Squelch */}
        <Slider
          label="Squelch"
          value={channel.squelchDb ?? -60}
          min={-80}
          max={0}
          step={1}
          coarseStep={10}
          unit="dB"
          formatValue={(val) => `${val.toFixed(0)} dB`}
          onChange={(val) => updateSetting({ squelchDb: val })}
        />

        {/* Audio Rate */}
        <Flex direction="column" gap={1}>
          <label className="form-label small mb-0">Audio Rate</label>
          <select
            className="form-select form-select-sm"
            value={channel.audioRate}
            onChange={(e) =>
              updateSetting({ audioRate: parseInt(e.target.value) })
            }
          >
            <option value={8000}>8 kHz</option>
            <option value={16000}>16 kHz</option>
            <option value={24000}>24 kHz</option>
            <option value={48000}>48 kHz</option>
          </select>
        </Flex>

        {/* Frequency */}
        <FrequencySelector
          label="Frequency"
          value={channelFrequency}
          min={spectrumMin}
          max={spectrumMax}
          step={1000}
          onChange={(hz) => updateSetting({ offsetHz: hz - capture.centerHz })}
        />

        {/* DSP Filters Section */}
        <CollapsibleSection
          title="DSP Filters"
          isOpen={showDsp}
          onToggle={() => setShowDsp(!showDsp)}
        >
          {/* FM Filters */}
          {(channel.mode === "wbfm" || channel.mode === "nbfm") && (
            <>
              <div className="form-check">
                <input
                  className="form-check-input"
                  type="checkbox"
                  id="enableDeemphasis"
                  checked={channel.enableDeemphasis ?? true}
                  onChange={(e) =>
                    updateSetting({ enableDeemphasis: e.target.checked })
                  }
                />
                <label
                  className="form-check-label small"
                  htmlFor="enableDeemphasis"
                >
                  FM De-emphasis
                </label>
              </div>
              <div className="form-check">
                <input
                  className="form-check-input"
                  type="checkbox"
                  id="enableMpxFilter"
                  checked={channel.enableMpxFilter ?? false}
                  onChange={(e) =>
                    updateSetting({ enableMpxFilter: e.target.checked })
                  }
                />
                <label
                  className="form-check-label small"
                  htmlFor="enableMpxFilter"
                >
                  MPX Filter (15kHz stereo cutoff)
                </label>
              </div>
              <Slider
                label="FM Highpass"
                value={channel.fmHighpassHz ?? 0}
                min={0}
                max={2000}
                step={50}
                unit="Hz"
                onChange={(val) => updateSetting({ fmHighpassHz: val })}
              />
              <Slider
                label="FM Lowpass"
                value={channel.fmLowpassHz ?? 0}
                min={0}
                max={20000}
                step={500}
                unit="Hz"
                onChange={(val) => updateSetting({ fmLowpassHz: val })}
              />
            </>
          )}

          {/* AM/SSB Filters */}
          {(channel.mode === "am" || channel.mode === "ssb") && (
            <>
              <Slider
                label="AM Highpass"
                value={channel.amHighpassHz ?? 0}
                min={0}
                max={2000}
                step={50}
                unit="Hz"
                onChange={(val) => updateSetting({ amHighpassHz: val })}
              />
              <Slider
                label="AM Lowpass"
                value={channel.amLowpassHz ?? 0}
                min={0}
                max={20000}
                step={500}
                unit="Hz"
                onChange={(val) => updateSetting({ amLowpassHz: val })}
              />
            </>
          )}
        </CollapsibleSection>

        {/* AGC Section */}
        <CollapsibleSection
          title="AGC Settings"
          isOpen={showAgc}
          onToggle={() => setShowAgc(!showAgc)}
        >
          <div className="form-check">
            <input
              className="form-check-input"
              type="checkbox"
              id="enableAgc"
              checked={channel.enableAgc ?? true}
              onChange={(e) => updateSetting({ enableAgc: e.target.checked })}
            />
            <label className="form-check-label small" htmlFor="enableAgc">
              Enable AGC
            </label>
          </div>

          {channel.enableAgc !== false && (
            <>
              <Slider
                label="Target Level"
                value={channel.agcTargetDb ?? -20}
                min={-40}
                max={0}
                step={1}
                unit="dB"
                onChange={(val) => updateSetting({ agcTargetDb: val })}
              />
              <Slider
                label="Attack Time"
                value={channel.agcAttackMs ?? 10}
                min={1}
                max={100}
                step={1}
                unit="ms"
                onChange={(val) => updateSetting({ agcAttackMs: val })}
              />
              <Slider
                label="Release Time"
                value={channel.agcReleaseMs ?? 500}
                min={50}
                max={2000}
                step={50}
                unit="ms"
                onChange={(val) => updateSetting({ agcReleaseMs: val })}
              />
            </>
          )}
        </CollapsibleSection>

        {/* Pager Decoding */}
        {channel.mode === "nbfm" && (
          <CollapsibleSection
            title="Pager Decoding"
            isOpen={showPager}
            onToggle={() => setShowPager(!showPager)}
          >
            <div className="form-check">
              <input
                className="form-check-input"
                type="checkbox"
                id="enablePocsag"
                checked={channel.enablePocsag ?? false}
                onChange={(e) =>
                  updateSetting({ enablePocsag: e.target.checked })
                }
              />
              <label className="form-check-label small" htmlFor="enablePocsag">
                Enable POCSAG
              </label>
            </div>

            {channel.enablePocsag && (
              <Flex direction="column" gap={1} className="mt-2">
                <label className="form-label small mb-0">POCSAG Baud</label>
                <select
                  className="form-select form-select-sm"
                  value={channel.pocsagBaud ?? 1200}
                  onChange={(e) =>
                    updateSetting({ pocsagBaud: parseInt(e.target.value) })
                  }
                >
                  <option value={512}>512</option>
                  <option value={1200}>1200</option>
                  <option value={2400}>2400</option>
                </select>
              </Flex>
            )}

            <div className="form-check mt-2">
              <input
                className="form-check-input"
                type="checkbox"
                id="enableFlex"
                checked={channel.enableFlex ?? false}
                onChange={(e) =>
                  updateSetting({ enableFlex: e.target.checked })
                }
              />
              <label className="form-check-label small" htmlFor="enableFlex">
                Enable FLEX (multimon-ng)
              </label>
            </div>
            <div className="small text-muted mt-1">
              FLEX decoding uses multimon-ng and targets 22.05 kHz audio
              (resampled automatically).
            </div>
          </CollapsibleSection>
        )}

        {/* Notch Filters */}
        <CollapsibleSection
          title={`Notch Filters (${channel.notchFrequencies?.length ?? 0})`}
          isOpen={false}
          onToggle={() => {}}
        >
          <Flex direction="column" gap={1}>
            <Flex gap={1}>
              <input
                type="number"
                className="form-control form-control-sm"
                placeholder="Frequency (Hz)"
                value={newNotchFreq}
                onChange={(e) => setNewNotchFreq(e.target.value)}
                style={{ width: "100px" }}
              />
              <button
                className="btn btn-sm btn-primary"
                onClick={handleAddNotch}
              >
                Add
              </button>
            </Flex>
            {(channel.notchFrequencies || []).map((freq) => (
              <Flex
                key={freq}
                justify="between"
                align="center"
                className="small"
              >
                <span>{freq} Hz</span>
                <button
                  className="btn btn-sm btn-outline-danger p-0 px-1"
                  onClick={() => handleRemoveNotch(freq)}
                >
                  Remove
                </button>
              </Flex>
            ))}
          </Flex>
        </CollapsibleSection>
      </Flex>
    </div>
  );
}

interface CollapsibleSectionProps {
  title: string;
  isOpen: boolean;
  onToggle: () => void;
  children: React.ReactNode;
}

function CollapsibleSection({
  title,
  isOpen,
  onToggle,
  children,
}: CollapsibleSectionProps) {
  return (
    <div className="border rounded">
      <div
        className="d-flex justify-content-between align-items-center p-2 bg-light"
        style={{ cursor: "pointer" }}
        onClick={onToggle}
      >
        <span className="small fw-semibold">{title}</span>
        {isOpen ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
      </div>
      {isOpen && <div className="p-2">{children}</div>}
    </div>
  );
}
