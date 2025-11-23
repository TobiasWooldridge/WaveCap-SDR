import { useState, useRef, useEffect } from "react";
import { Volume2, VolumeX, Trash2, Copy, CheckCircle, Settings, ChevronUp, ChevronDown, Edit2 } from "lucide-react";
import type { Capture, Channel } from "../types";
import { useUpdateChannel, useDeleteChannel } from "../hooks/useChannels";
import { useToast } from "../hooks/useToast";
import { formatFrequencyMHz } from "../utils/frequency";
import Button from "./primitives/Button.react";
import Flex from "./primitives/Flex.react";
import Slider from "./primitives/Slider.react";
import FrequencySelector from "./primitives/FrequencySelector.react";
import SMeter from "./primitives/SMeter.react";
import AudioWaveform from "./primitives/AudioWaveform.react";
import { FrequencyLabel } from "./FrequencyLabel.react";
import { POCSAGFeed } from "./POCSAGFeed.react";

interface CompactChannelCardProps {
  channel: Channel;
  capture: Capture;
  isPlaying: boolean;
  onTogglePlay: () => void;
  onCopyUrl: (url: string) => void;
  copiedUrl: string | null;
}

// Format channel ID for display (e.g., "ch1" -> "Ch 1")
function formatChannelId(id: string): string {
  const match = id.match(/^ch(\d+)$/);
  return match ? `Ch ${match[1]}` : id;
}

export const CompactChannelCard = ({
  channel,
  capture,
  isPlaying,
  onTogglePlay,
  onCopyUrl,
  copiedUrl,
}: CompactChannelCardProps) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [showStreamDropdown, setShowStreamDropdown] = useState(false);
  const [isEditingName, setIsEditingName] = useState(false);
  const [editNameValue, setEditNameValue] = useState("");
  const [newNotchFreq, setNewNotchFreq] = useState("");
  const [showDspFilters, setShowDspFilters] = useState(false);
  const [showAgcSettings, setShowAgcSettings] = useState(false);
  const [showNoiseBlanker, setShowNoiseBlanker] = useState(false);
  const [showNoiseReduction, setShowNoiseReduction] = useState(false);
  const nameInputRef = useRef<HTMLInputElement>(null);
  const updateChannel = useUpdateChannel(capture.id);
  const deleteChannel = useDeleteChannel();
  const toast = useToast();

  const getChannelFrequency = () => capture.centerHz + channel.offsetHz;
  const displayName = channel.name || channel.autoName || formatChannelId(channel.id);

  useEffect(() => {
    if (isEditingName && nameInputRef.current) {
      nameInputRef.current.focus();
      nameInputRef.current.select();
    }
  }, [isEditingName]);

  const handleStartEditName = () => {
    setEditNameValue(channel.name || channel.autoName || "");
    setIsEditingName(true);
  };

  const handleSaveEditName = () => {
    const trimmedValue = editNameValue.trim();
    updateChannel.mutate({
      channelId: channel.id,
      request: { name: trimmedValue || null },
    }, {
      onSuccess: () => {
        toast.success("Channel name updated");
      },
      onError: (error: any) => {
        toast.error(error?.message || "Failed to update channel name");
      },
    });
    setIsEditingName(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      handleSaveEditName();
    } else if (e.key === "Escape") {
      setIsEditingName(false);
    }
  };

  const handleDelete = () => {
    if (confirm("Delete this channel?")) {
      deleteChannel.mutate(channel.id, {
        onSuccess: () => {
          toast.success("Channel deleted successfully");
        },
        onError: (error: any) => {
          toast.error(error?.message || "Failed to delete channel");
        },
      });
    }
  };

  // Helper function to update channel with toast notifications
  const updateChannelWithToast = (request: any) => {
    updateChannel.mutate({
      channelId: channel.id,
      request,
    }, {
      onError: (error: any) => {
        toast.error(error?.message || "Failed to update channel");
      },
    });
  };

  const handleAddNotch = () => {
    const freq = parseFloat(newNotchFreq);
    if (isNaN(freq) || freq <= 0 || freq > 20000) {
      toast.error("Notch frequency must be between 0 and 20000 Hz");
      return;
    }
    const currentNotches = channel.notchFrequencies || [];
    if (currentNotches.includes(freq)) {
      toast.error("This frequency is already in the notch list");
      return;
    }
    if (currentNotches.length >= 10) {
      toast.error("Maximum 10 notch filters allowed");
      return;
    }
    updateChannelWithToast({ notchFrequencies: [...currentNotches, freq] });
    setNewNotchFreq("");
    toast.success(`Added notch filter at ${freq} Hz`);
  };

  const handleRemoveNotch = (freq: number) => {
    const currentNotches = channel.notchFrequencies || [];
    updateChannelWithToast({
      notchFrequencies: currentNotches.filter(f => f !== freq)
    });
    toast.success(`Removed notch filter at ${freq} Hz`);
  };

  const streamFormats = [
    { format: 'PCM', ext: '.pcm', label: 'Raw PCM' },
    { format: 'MP3', ext: '.mp3', label: 'MP3 (128k)' },
    { format: 'Opus', ext: '.opus', label: 'Opus' },
    { format: 'AAC', ext: '.aac', label: 'AAC' },
  ];

  return (
    <div className="card shadow-sm h-100">
      {/* Compact Header */}
      <div className="card-header bg-body-tertiary p-2">
        <Flex justify="between" align="center">
          <div className="small fw-semibold text-truncate" style={{ flex: 1, minWidth: 0 }}>
            {formatChannelId(channel.id)} • {formatFrequencyMHz(getChannelFrequency())} MHz
          </div>
          <Flex gap={1}>
            <Button
              use={isPlaying ? "warning" : "success"}
              size="sm"
              onClick={onTogglePlay}
              title={isPlaying ? "Stop Listening" : "Listen Now"}
              className="px-2 py-1"
            >
              <Flex align="center" gap={1}>
                {isPlaying ? <VolumeX size={14} /> : <Volume2 size={14} />}
                <span style={{ fontSize: "11px" }}>{isPlaying ? "Stop" : "Listen"}</span>
              </Flex>
            </Button>
            <Button
              use="secondary"
              size="sm"
              onClick={() => setIsExpanded(!isExpanded)}
              title={isExpanded ? "Collapse" : "Settings"}
              className="p-1"
            >
              {isExpanded ? <ChevronUp size={14} /> : <Settings size={14} />}
            </Button>
            <Button
              use="danger"
              size="sm"
              appearance="outline"
              onClick={handleDelete}
              title="Delete"
              className="p-1"
            >
              <Trash2 size={14} />
            </Button>
          </Flex>
        </Flex>
      </div>

      {/* Compact Body - Always Visible */}
      <div className="card-body p-2">
        <Flex direction="column" gap={2}>
          {/* Channel Info */}
          <div>
            <Flex justify="between" align="center">
              <div className="small text-muted">{channel.mode.toUpperCase()}</div>
              <button
                className="btn btn-sm p-0"
                style={{ width: "16px", height: "16px" }}
                onClick={handleStartEditName}
                title="Edit name"
              >
                <Edit2 size={12} />
              </button>
            </Flex>
            {isEditingName ? (
              <input
                ref={nameInputRef}
                type="text"
                className="form-control form-control-sm mt-1"
                value={editNameValue}
                onChange={(e) => setEditNameValue(e.target.value)}
                onBlur={handleSaveEditName}
                onKeyDown={handleKeyDown}
                placeholder="Channel name"
              />
            ) : (
              <div>
                <div className="fw-semibold" title={channel.name && channel.autoName ? `Auto: ${channel.autoName}` : undefined}>
                  {displayName}
                </div>
                <FrequencyLabel frequencyHz={getChannelFrequency()} autoName={channel.name ? channel.autoName : null} />
                {/* Warning for channels outside spectrum range */}
                {(() => {
                  const channelFreq = getChannelFrequency();
                  const spectrumMin = capture.centerHz - (capture.sampleRate / 2);
                  const spectrumMax = capture.centerHz + (capture.sampleRate / 2);
                  const isOutOfRange = channelFreq < spectrumMin || channelFreq > spectrumMax;

                  if (isOutOfRange) {
                    return (
                      <div className="alert alert-warning py-1 px-2 mt-1 mb-0" style={{ fontSize: "0.7rem" }}>
                        <strong>Warning:</strong> Channel frequency ({formatFrequencyMHz(channelFreq)} MHz) is outside observable spectrum ({formatFrequencyMHz(spectrumMin)} - {formatFrequencyMHz(spectrumMax)} MHz)
                      </div>
                    );
                  }
                  return null;
                })()}
              </div>
            )}
          </div>

          {/* S-Meter Display */}
          <div className="border rounded p-2 bg-light">
            <Flex direction="column" gap={1}>
              <Flex align="center" gap={1}>
                <span className="badge bg-success text-white" style={{fontSize: "8px", width: "32px"}}>LIVE</span>
                <div style={{ flex: 1 }}>
                  <SMeter rssiDbFs={channel.rssiDb} frequencyHz={getChannelFrequency()} width={200} height={24} />
                </div>
              </Flex>
              <Flex direction="row" gap={1} align="center" style={{fontSize: "9px"}}>
                <span className="text-muted" style={{ width: "40px" }}>RSSI:</span>
                <span className="fw-semibold">{channel.rssiDb?.toFixed(1) ?? 'N/A'} dBFS</span>
                <span className="text-muted ms-2" style={{ width: "35px" }}>SNR:</span>
                <span className="fw-semibold">{channel.snrDb?.toFixed(1) ?? 'N/A'} dB</span>
              </Flex>
            </Flex>
          </div>

          {/* RDS Display (WBFM only) */}
          {channel.mode === "wbfm" && channel.rdsData && (channel.rdsData.psName || channel.rdsData.radioText) && (
            <div className="border rounded p-2 bg-dark text-light" style={{ fontSize: "10px" }}>
              <Flex direction="column" gap={1}>
                {/* Station Name (PS) */}
                {channel.rdsData.psName && (
                  <Flex align="center" gap={1}>
                    <span className="badge bg-info text-dark" style={{ fontSize: "8px", width: "28px" }}>RDS</span>
                    <span className="fw-bold font-monospace" style={{ fontSize: "14px", letterSpacing: "1px" }}>
                      {channel.rdsData.psName}
                    </span>
                    {channel.rdsData.ptyName && channel.rdsData.ptyName !== "None" && (
                      <span className="badge bg-secondary ms-auto" style={{ fontSize: "8px" }}>
                        {channel.rdsData.ptyName}
                      </span>
                    )}
                  </Flex>
                )}
                {/* Radio Text (RT) - scrolling marquee style */}
                {channel.rdsData.radioText && (
                  <div className="text-truncate font-monospace text-muted" title={channel.rdsData.radioText}>
                    {channel.rdsData.radioText}
                  </div>
                )}
                {/* Flags row */}
                {(channel.rdsData.ta || channel.rdsData.tp || channel.rdsData.piCode) && (
                  <Flex gap={1} align="center">
                    {channel.rdsData.piCode && (
                      <span className="text-muted" style={{ fontSize: "8px" }}>PI:{channel.rdsData.piCode}</span>
                    )}
                    {channel.rdsData.tp && (
                      <span className="badge bg-warning text-dark" style={{ fontSize: "7px" }}>TP</span>
                    )}
                    {channel.rdsData.ta && (
                      <span className="badge bg-danger" style={{ fontSize: "7px" }}>TA</span>
                    )}
                    {!channel.rdsData.ms && (
                      <span className="badge bg-primary" style={{ fontSize: "7px" }}>Speech</span>
                    )}
                  </Flex>
                )}
              </Flex>
            </div>
          )}

          {/* POCSAG Feed (NBFM only) */}
          {channel.mode === "nbfm" && (
            <POCSAGFeed channelId={channel.id} enabled={channel.state === "running"} />
          )}

          {/* Audio Waveform Display */}
          {isPlaying && (
            <div className="border rounded p-2 bg-dark">
              <AudioWaveform channelId={channel.id} isPlaying={isPlaying} width={200} height={40} />
            </div>
          )}

          {/* Stream URL Dropdown */}
          <div className="dropdown" style={{ position: 'relative' }}>
            <Button
              use="secondary"
              size="sm"
              onClick={() => setShowStreamDropdown(!showStreamDropdown)}
              className="w-100 d-flex justify-content-between align-items-center"
            >
              <span className="small">Copy Stream URL</span>
              <ChevronDown size={12} />
            </Button>
            {showStreamDropdown && (
              <div
                className="dropdown-menu show w-100"
                style={{ position: 'absolute', top: '100%', zIndex: 1000 }}
              >
                {streamFormats.map(({ format, ext, label }) => {
                  const url = `${window.location.origin}/api/v1/stream/channels/${channel.id}${ext}`;
                  const isCopied = copiedUrl === url;
                  return (
                    <button
                      key={format}
                      className="dropdown-item d-flex justify-content-between align-items-center"
                      onClick={() => {
                        onCopyUrl(url);
                        setShowStreamDropdown(false);
                      }}
                    >
                      <span className="small">{label}</span>
                      {isCopied ? <CheckCircle size={12} className="text-success" /> : <Copy size={12} />}
                    </button>
                  );
                })}
              </div>
            )}
          </div>

          {/* Expanded Settings */}
          {isExpanded && (
            <div className="border-top pt-2 mt-1">
              <Flex direction="column" gap={2}>
                {/* Mode Selector */}
                <Flex direction="column" gap={1}>
                  <label className="form-label small mb-0">Mode</label>
                  <select
                    className="form-select form-select-sm"
                    value={channel.mode}
                    onChange={(e) =>
                      updateChannelWithToast({ mode: e.target.value as any })
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

                {/* SSB Mode Selector (SSB only) */}
                {channel.mode === "ssb" && (
                  <Flex direction="column" gap={1}>
                    <label className="form-label small mb-0">SSB Mode</label>
                    <select
                      className="form-select form-select-sm"
                      value={channel.ssbMode}
                      onChange={(e) =>
                        updateChannelWithToast({ ssbMode: e.target.value as any })
                      }
                    >
                      <option value="usb">USB (Upper Sideband)</option>
                      <option value="lsb">LSB (Lower Sideband)</option>
                    </select>
                    <small className="text-muted">
                      USB: Amateur radio above 10 MHz. LSB: Below 10 MHz
                    </small>
                  </Flex>
                )}

                {/* Squelch Slider */}
                <Slider
                  label="Squelch"
                  value={channel.squelchDb ?? -60}
                  min={-80}
                  max={0}
                  step={1}
                  coarseStep={10}
                  unit="dB"
                  formatValue={(val) => `${val.toFixed(0)} dB`}
                  onChange={(val) =>
                    updateChannelWithToast({ squelchDb: val })
                  }
                />

                {/* Audio Rate */}
                <Flex direction="column" gap={1}>
                  <label className="form-label small mb-0">Audio Rate</label>
                  <select
                    className="form-select form-select-sm"
                    value={channel.audioRate}
                    onChange={(e) =>
                      updateChannelWithToast({ audioRate: parseInt(e.target.value) })
                    }
                  >
                    <option value={8000}>8 kHz</option>
                    <option value={16000}>16 kHz</option>
                    <option value={24000}>24 kHz</option>
                    <option value={48000}>48 kHz</option>
                  </select>
                </Flex>

                {/* Frequency */}
                <Flex direction="column" gap={1}>
                  <FrequencySelector
                    label="Frequency"
                    value={getChannelFrequency()}
                    min={capture.centerHz - (capture.sampleRate / 2)}
                    max={capture.centerHz + (capture.sampleRate / 2)}
                    step={1000}
                    onChange={(hz) =>
                      updateChannelWithToast({ offsetHz: hz - capture.centerHz })
                    }
                  />
                  <small className="text-muted">
                    Offset: {(channel.offsetHz / 1000).toFixed(0)} kHz
                  </small>
                </Flex>

                {/* DSP Filters Section */}
                <div className="border rounded">
                  <button
                    className="btn btn-sm w-100 text-start d-flex justify-content-between align-items-center p-2"
                    onClick={() => setShowDspFilters(!showDspFilters)}
                    style={{ background: "transparent", border: "none" }}
                  >
                    <span className="fw-semibold small">DSP Filters</span>
                    {showDspFilters ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                  </button>
                  {showDspFilters && (
                    <div className="p-2 border-top">
                      <Flex direction="column" gap={2}>
                        {/* Info card */}
                        <div className="alert alert-info py-1 px-2 mb-0" style={{ fontSize: "0.7rem" }}>
                          <strong>DSP Filters</strong> shape audio frequency response for optimal clarity
                        </div>

                        {/* FM Mode Filters */}
                        {(channel.mode === "wbfm" || channel.mode === "nbfm") && (
                          <>
                            {/* Deemphasis */}
                            <Flex direction="column" gap={1}>
                              <Flex justify="between" align="center">
                                <label className="form-label small mb-0">Deemphasis</label>
                                <input
                                  type="checkbox"
                                  checked={channel.enableDeemphasis}
                                  onChange={(e) =>
                                    updateChannelWithToast({ enableDeemphasis: e.target.checked })
                                  }
                                  style={{ width: "16px", height: "16px" }}
                                />
                              </Flex>
                              {channel.enableDeemphasis && (
                                <select
                                  className="form-select form-select-sm"
                                  value={channel.deemphasisTauUs}
                                  onChange={(e) =>
                                    updateChannelWithToast({ deemphasisTauUs: parseFloat(e.target.value) })
                                  }
                                >
                                  <option value={50}>50 µs (Europe)</option>
                                  <option value={75}>75 µs (USA)</option>
                                </select>
                              )}
                              <small className="text-muted">
                                Compensates for FM pre-emphasis (boosts treble)
                              </small>
                            </Flex>

                            {/* MPX Filter (WBFM only) */}
                            {channel.mode === "wbfm" && (
                              <Flex direction="column" gap={1}>
                                <Flex justify="between" align="center">
                                  <label className="form-label small mb-0">MPX Filter (19 kHz Pilot Removal)</label>
                                  <input
                                    type="checkbox"
                                    checked={channel.enableMpxFilter}
                                    onChange={(e) =>
                                      updateChannelWithToast({ enableMpxFilter: e.target.checked })
                                    }
                                    style={{ width: "16px", height: "16px" }}
                                  />
                                </Flex>
                                {channel.enableMpxFilter && (
                                  <Slider
                                    label=""
                                    value={channel.mpxCutoffHz}
                                    min={10000}
                                    max={18000}
                                    step={500}
                                    unit="Hz"
                                    formatValue={(val) => `${(val / 1000).toFixed(1)} kHz`}
                                    onChange={(val) =>
                                      updateChannelWithToast({ mpxCutoffHz: val })
                                    }
                                  />
                                )}
                                <small className="text-muted">
                                  Removes stereo pilot tone and subcarriers (eliminates high-pitch whine)
                                </small>
                              </Flex>
                            )}

                            {/* FM Highpass */}
                            <Flex direction="column" gap={1}>
                              <Flex justify="between" align="center">
                                <label className="form-label small mb-0">Highpass Filter</label>
                                <input
                                  type="checkbox"
                                  checked={channel.enableFmHighpass}
                                  onChange={(e) =>
                                    updateChannelWithToast({ enableFmHighpass: e.target.checked })
                                  }
                                  style={{ width: "16px", height: "16px" }}
                                />
                              </Flex>
                              {channel.enableFmHighpass && (
                                <Slider
                                  label=""
                                  value={channel.fmHighpassHz}
                                  min={50}
                                  max={500}
                                  step={10}
                                  unit="Hz"
                                  formatValue={(val) => `${val.toFixed(0)} Hz`}
                                  onChange={(val) =>
                                    updateChannelWithToast({ fmHighpassHz: val })
                                  }
                                />
                              )}
                              <small className="text-muted">
                                Removes DC offset and rumble
                              </small>
                            </Flex>

                            {/* FM Lowpass (NBFM only) */}
                            {channel.mode === "nbfm" && (
                              <Flex direction="column" gap={1}>
                                <Flex justify="between" align="center">
                                  <label className="form-label small mb-0">Lowpass Filter (Voice BW)</label>
                                  <input
                                    type="checkbox"
                                    checked={channel.enableFmLowpass}
                                    onChange={(e) =>
                                      updateChannelWithToast({ enableFmLowpass: e.target.checked })
                                    }
                                    style={{ width: "16px", height: "16px" }}
                                  />
                                </Flex>
                                {channel.enableFmLowpass && (
                                  <Slider
                                    label=""
                                    value={channel.fmLowpassHz}
                                    min={2000}
                                    max={5000}
                                    step={100}
                                    unit="Hz"
                                    formatValue={(val) => `${val.toFixed(0)} Hz`}
                                    onChange={(val) =>
                                      updateChannelWithToast({ fmLowpassHz: val })
                                    }
                                  />
                                )}
                                <small className="text-muted">
                                  Limits voice bandwidth (3000 Hz typical)
                                </small>
                              </Flex>
                            )}
                          </>
                        )}

                        {/* AM Mode Filters */}
                        {channel.mode === "am" && (
                          <>
                            {/* AM Highpass */}
                            <Flex direction="column" gap={1}>
                              <Flex justify="between" align="center">
                                <label className="form-label small mb-0">Highpass Filter (DC Removal)</label>
                                <input
                                  type="checkbox"
                                  checked={channel.enableAmHighpass}
                                  onChange={(e) =>
                                    updateChannelWithToast({ enableAmHighpass: e.target.checked })
                                  }
                                  style={{ width: "16px", height: "16px" }}
                                />
                              </Flex>
                              {channel.enableAmHighpass && (
                                <Slider
                                  label=""
                                  value={channel.amHighpassHz}
                                  min={50}
                                  max={500}
                                  step={10}
                                  unit="Hz"
                                  formatValue={(val) => `${val.toFixed(0)} Hz`}
                                  onChange={(val) =>
                                    updateChannelWithToast({ amHighpassHz: val })
                                  }
                                />
                              )}
                              <small className="text-muted">
                                Removes AM carrier offset and rumble
                              </small>
                            </Flex>

                            {/* AM Lowpass */}
                            <Flex direction="column" gap={1}>
                              <Flex justify="between" align="center">
                                <label className="form-label small mb-0">Lowpass Filter (Bandwidth)</label>
                                <input
                                  type="checkbox"
                                  checked={channel.enableAmLowpass}
                                  onChange={(e) =>
                                    updateChannelWithToast({ enableAmLowpass: e.target.checked })
                                  }
                                  style={{ width: "16px", height: "16px" }}
                                />
                              </Flex>
                              {channel.enableAmLowpass && (
                                <Slider
                                  label=""
                                  value={channel.amLowpassHz}
                                  min={3000}
                                  max={10000}
                                  step={100}
                                  unit="Hz"
                                  formatValue={(val) => `${val.toFixed(0)} Hz`}
                                  onChange={(val) =>
                                    updateChannelWithToast({ amLowpassHz: val })
                                  }
                                />
                              )}
                              <small className="text-muted">
                                AM broadcast: 5000 Hz, Aviation: 3000 Hz
                              </small>
                            </Flex>
                          </>
                        )}

                        {/* SSB Mode Filters */}
                        {channel.mode === "ssb" && (
                          <Flex direction="column" gap={1}>
                            <Flex justify="between" align="center">
                              <label className="form-label small mb-0">Bandpass Filter (Voice)</label>
                              <input
                                type="checkbox"
                                checked={channel.enableSsbBandpass}
                                onChange={(e) =>
                                  updateChannelWithToast({ enableSsbBandpass: e.target.checked })
                                }
                                style={{ width: "16px", height: "16px" }}
                              />
                            </Flex>
                            {channel.enableSsbBandpass && (
                              <Flex direction="column" gap={1}>
                                <Slider
                                  label="Low Cutoff"
                                  value={channel.ssbBandpassLowHz}
                                  min={100}
                                  max={1000}
                                  step={50}
                                  unit="Hz"
                                  formatValue={(val) => `${val.toFixed(0)} Hz`}
                                  onChange={(val) =>
                                    updateChannelWithToast({ ssbBandpassLowHz: val })
                                  }
                                />
                                <Slider
                                  label="High Cutoff"
                                  value={channel.ssbBandpassHighHz}
                                  min={2000}
                                  max={4000}
                                  step={100}
                                  unit="Hz"
                                  formatValue={(val) => `${val.toFixed(0)} Hz`}
                                  onChange={(val) =>
                                    updateChannelWithToast({ ssbBandpassHighHz: val })
                                  }
                                />
                              </Flex>
                            )}
                            <small className="text-muted">
                              Typical voice: 300-3000 Hz. Narrow: 500-2500 Hz
                            </small>
                          </Flex>
                        )}
                      </Flex>
                    </div>
                  )}
                </div>

                {/* AGC Settings Section */}
                {(channel.mode === "am" || channel.mode === "ssb") && (
                  <div className="border rounded">
                    <button
                      className="btn btn-sm w-100 text-start d-flex justify-content-between align-items-center p-2"
                      onClick={() => setShowAgcSettings(!showAgcSettings)}
                      style={{ background: "transparent", border: "none" }}
                    >
                      <Flex align="center" gap={1}>
                        <span className="fw-semibold small">AGC (Auto Gain Control)</span>
                        <span className={`badge ${channel.enableAgc ? "bg-success" : "bg-secondary"}`} style={{ fontSize: "8px" }}>
                          {channel.enableAgc ? "ON" : "OFF"}
                        </span>
                      </Flex>
                      {showAgcSettings ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                    </button>
                    {showAgcSettings && (
                      <div className="p-2 border-top">
                        <Flex direction="column" gap={2}>
                          {/* Info card */}
                          <div className="alert alert-info py-1 px-2 mb-0" style={{ fontSize: "0.7rem" }}>
                            <strong>AGC</strong> automatically adjusts gain to maintain consistent audio levels. Essential for AM/SSB signals with varying strength.
                          </div>

                          {/* Enable AGC */}
                          <Flex justify="between" align="center">
                            <label className="form-label small mb-0">Enable AGC</label>
                            <input
                              type="checkbox"
                              checked={channel.enableAgc}
                              onChange={(e) =>
                                updateChannelWithToast({ enableAgc: e.target.checked })
                              }
                              style={{ width: "16px", height: "16px" }}
                            />
                          </Flex>

                          {channel.enableAgc && (
                            <>
                              <Slider
                                label="Target Level"
                                value={channel.agcTargetDb}
                                min={-60}
                                max={-10}
                                step={1}
                                unit="dB"
                                formatValue={(val) => `${val.toFixed(0)} dB`}
                                onChange={(val) =>
                                  updateChannelWithToast({ agcTargetDb: val })
                                }
                              />
                              <Slider
                                label="Attack Time"
                                value={channel.agcAttackMs}
                                min={1}
                                max={100}
                                step={1}
                                unit="ms"
                                formatValue={(val) => `${val.toFixed(0)} ms`}
                                onChange={(val) =>
                                  updateChannelWithToast({ agcAttackMs: val })
                                }
                              />
                              <Slider
                                label="Release Time"
                                value={channel.agcReleaseMs}
                                min={10}
                                max={500}
                                step={10}
                                unit="ms"
                                formatValue={(val) => `${val.toFixed(0)} ms`}
                                onChange={(val) =>
                                  updateChannelWithToast({ agcReleaseMs: val })
                                }
                              />
                              <small className="text-muted">
                                Attack: how quickly gain increases. Release: how quickly gain decreases.
                              </small>
                            </>
                          )}
                        </Flex>
                      </div>
                    )}
                  </div>
                )}

                {/* Noise Blanker Section */}
                <div className="border rounded">
                  <button
                    className="btn btn-sm w-100 text-start d-flex justify-content-between align-items-center p-2"
                    onClick={() => setShowNoiseBlanker(!showNoiseBlanker)}
                    style={{ background: "transparent", border: "none" }}
                  >
                    <Flex align="center" gap={1}>
                      <span className="fw-semibold small">Noise Blanker</span>
                      <span className={`badge ${channel.enableNoiseBlanker ? "bg-success" : "bg-secondary"}`} style={{ fontSize: "8px" }}>
                        {channel.enableNoiseBlanker ? "ON" : "OFF"}
                      </span>
                    </Flex>
                    {showNoiseBlanker ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                  </button>
                  {showNoiseBlanker && (
                    <div className="p-2 border-top">
                      <Flex direction="column" gap={2}>
                        {/* Info card */}
                        <div className="alert alert-warning py-1 px-2 mb-0" style={{ fontSize: "0.7rem" }}>
                          <strong>Noise Blanker</strong> suppresses impulse noise from lightning, ignition, power lines. Use when experiencing static pops/clicks.
                        </div>

                        {/* Enable Noise Blanker */}
                        <Flex justify="between" align="center">
                          <label className="form-label small mb-0">Enable Noise Blanker</label>
                          <input
                            type="checkbox"
                            checked={channel.enableNoiseBlanker}
                            onChange={(e) =>
                              updateChannelWithToast({ enableNoiseBlanker: e.target.checked })
                            }
                            style={{ width: "16px", height: "16px" }}
                          />
                        </Flex>

                        {channel.enableNoiseBlanker && (
                          <>
                            <Slider
                              label="Threshold"
                              value={channel.noiseBlankerThresholdDb}
                              min={3}
                              max={30}
                              step={1}
                              unit="dB"
                              formatValue={(val) => `${val.toFixed(0)} dB`}
                              onChange={(val) =>
                                updateChannelWithToast({ noiseBlankerThresholdDb: val })
                              }
                            />
                            <small className="text-muted">
                              Lower = more aggressive (may remove weak signals). Higher = less aggressive. Start at 10 dB.
                            </small>
                          </>
                        )}
                      </Flex>
                    </div>
                  )}
                </div>

                {/* Noise Reduction Section */}
                <div className="border rounded">
                  <button
                    className="btn btn-sm w-100 text-start d-flex justify-content-between align-items-center p-2"
                    onClick={() => setShowNoiseReduction(!showNoiseReduction)}
                    style={{ background: "transparent", border: "none" }}
                  >
                    <Flex align="center" gap={1}>
                      <span className="fw-semibold small">Noise Reduction</span>
                      <span className={`badge ${channel.enableNoiseReduction ? "bg-success" : "bg-secondary"}`} style={{ fontSize: "8px" }}>
                        {channel.enableNoiseReduction ? "ON" : "OFF"}
                      </span>
                    </Flex>
                    {showNoiseReduction ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                  </button>
                  {showNoiseReduction && (
                    <div className="p-2 border-top">
                      <Flex direction="column" gap={2}>
                        {/* Info card */}
                        <div className="alert alert-info py-1 px-2 mb-0" style={{ fontSize: "0.7rem" }}>
                          <strong>Noise Reduction</strong> suppresses background hiss and static using spectral processing. Use when hearing constant noise floor.
                        </div>

                        {/* Enable Noise Reduction */}
                        <Flex justify="between" align="center">
                          <label className="form-label small mb-0">Enable Noise Reduction</label>
                          <input
                            type="checkbox"
                            checked={channel.enableNoiseReduction}
                            onChange={(e) =>
                              updateChannelWithToast({ enableNoiseReduction: e.target.checked })
                            }
                            style={{ width: "16px", height: "16px" }}
                          />
                        </Flex>

                        {channel.enableNoiseReduction && (
                          <>
                            <Slider
                              label="Reduction Strength"
                              value={channel.noiseReductionDb}
                              min={3}
                              max={30}
                              step={1}
                              unit="dB"
                              formatValue={(val) => `${val.toFixed(0)} dB`}
                              onChange={(val) =>
                                updateChannelWithToast({ noiseReductionDb: val })
                              }
                            />
                            <small className="text-muted">
                              Lower = subtle noise reduction. Higher = aggressive (may affect audio quality). Start at 12 dB.
                            </small>
                          </>
                        )}
                      </Flex>
                    </div>
                  )}
                </div>

                {/* Notch Filters */}
                <Flex direction="column" gap={1}>
                  <label className="form-label small mb-0">Notch Filters (Interference Rejection)</label>
                  {channel.notchFrequencies && channel.notchFrequencies.length > 0 ? (
                    <Flex direction="column" gap={1}>
                      {channel.notchFrequencies.map((freq) => (
                        <Flex key={freq} justify="between" align="center" className="border rounded p-1 bg-light">
                          <span className="small fw-semibold">{freq} Hz</span>
                          <button
                            className="btn btn-sm btn-danger p-0"
                            style={{ width: "20px", height: "20px", fontSize: "12px" }}
                            onClick={() => handleRemoveNotch(freq)}
                            title="Remove notch"
                          >
                            ×
                          </button>
                        </Flex>
                      ))}
                    </Flex>
                  ) : (
                    <small className="text-muted">No notch filters active</small>
                  )}
                  <Flex gap={1}>
                    <input
                      type="number"
                      className="form-control form-control-sm"
                      placeholder="Frequency (Hz)"
                      value={newNotchFreq}
                      onChange={(e) => setNewNotchFreq(e.target.value)}
                      onKeyDown={(e) => e.key === "Enter" && handleAddNotch()}
                      min={0}
                      max={20000}
                      step={10}
                    />
                    <Button
                      use="primary"
                      size="sm"
                      onClick={handleAddNotch}
                      disabled={!newNotchFreq}
                    >
                      Add
                    </Button>
                  </Flex>
                  <small className="text-muted">
                    Remove interfering tones (power line hum, carriers, etc.). Common: 60 Hz, 120 Hz
                  </small>
                </Flex>
              </Flex>
            </div>
          )}
        </Flex>
      </div>

      {/* Status Badge Footer */}
      <div className="card-footer p-1 bg-body-tertiary">
        <Flex justify="between" align="center">
          <span className={`badge bg-${channel.state === "running" ? "success" : "secondary"}`} style={{fontSize: "9px"}}>
            {channel.state}
          </span>
          <small className="text-muted" style={{fontSize: "9px"}}>
            {(channel.offsetHz / 1000).toFixed(0)} kHz • {channel.audioRate / 1000} kHz
          </small>
        </Flex>
      </div>
    </div>
  );
};
