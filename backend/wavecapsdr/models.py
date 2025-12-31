from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Mode = Literal["wbfm", "nbfm", "am", "sam", "ssb", "raw", "p25", "dmr", "nxdn", "dstar", "ysf"]
StreamFormat = Literal["iq16", "f32", "pcm16"]
Transport = Literal["ws", "http"]


class DeviceModel(BaseModel):
    id: str
    driver: str
    label: str
    freqMinHz: float = Field(..., alias="freq_min_hz")
    freqMaxHz: float = Field(..., alias="freq_max_hz")
    sampleRates: list[int] = Field(..., alias="sample_rates")
    gains: list[str]
    gainMin: float | None = Field(None, alias="gain_min")
    gainMax: float | None = Field(None, alias="gain_max")
    bandwidthMin: float | None = Field(None, alias="bandwidth_min")
    bandwidthMax: float | None = Field(None, alias="bandwidth_max")
    ppmMin: float | None = Field(None, alias="ppm_min")
    ppmMax: float | None = Field(None, alias="ppm_max")
    antennas: list[str]
    nickname: str | None = None  # Custom user-provided nickname
    shorthand: str | None = None  # Auto-generated shorthand name (e.g., "RTL-SDR", "SDRplay RSPdx")
    model_config = ConfigDict(populate_by_name=True)


class CreateCaptureRequest(BaseModel):
    deviceId: str | None = Field(None, max_length=500)  # Device IDs can be long SoapySDR strings
    centerHz: float = Field(..., gt=0, lt=10e9)  # 0 to 10 GHz
    sampleRate: int = Field(..., gt=0, le=100_000_000)  # Max 100 MHz
    gain: float | None = Field(None, ge=-100, le=100)  # -100 to 100 dB
    bandwidth: float | None = Field(None, gt=0, le=100_000_000)  # Max 100 MHz
    ppm: float | None = Field(None, ge=-1000, le=1000)  # -1000 to 1000 PPM
    antenna: str | None = Field(None, max_length=100)
    deviceSettings: dict[str, str] | None = None
    elementGains: dict[str, float] | None = None
    streamFormat: str | None = Field(None, max_length=20)
    dcOffsetAuto: bool = True
    iqBalanceAuto: bool = True
    createDefaultChannel: bool = True
    name: str | None = Field(None, max_length=200)  # User-provided name (optional)
    # FFT/Spectrum settings
    fftFps: int | None = Field(None, ge=1, le=60)  # Target FPS (1-60)
    fftMaxFps: int | None = Field(None, ge=1, le=120)  # Max FPS cap (1-120)
    fftSize: int | None = Field(None)  # 512, 1024, 2048, 4096
    fftAccelerator: str | None = Field(None, pattern=r"^(auto|scipy|fftw|mlx|cuda)$")  # FFT backend

    @field_validator("fftSize")
    @classmethod
    def validate_fft_size(cls, v: int | None) -> int | None:
        if v is not None and v not in [512, 1024, 2048, 4096]:
            raise ValueError("fftSize must be 512, 1024, 2048, or 4096")
        return v

    @field_validator("name", "antenna", "deviceId", "streamFormat")
    @classmethod
    def sanitize_string(cls, v: str | None) -> str | None:
        if v is None:
            return None
        # Remove control characters and trim whitespace
        v = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", v)
        v = v.strip()
        # Return None if empty after sanitization
        return v if v else None

    @model_validator(mode="after")
    def validate_bandwidth_with_sample_rate(self) -> CreateCaptureRequest:
        if self.bandwidth is not None and self.sampleRate is not None:
            if self.bandwidth > self.sampleRate:
                raise ValueError("Bandwidth cannot exceed sample rate")
        return self


class UpdateCaptureRequest(BaseModel):
    deviceId: str | None = Field(None, max_length=500)  # Device IDs can be long SoapySDR strings
    centerHz: float | None = Field(None, gt=0, lt=10e9)  # 0 to 10 GHz
    sampleRate: int | None = Field(None, gt=0, le=100_000_000)  # Max 100 MHz
    gain: float | None = Field(None, ge=-100, le=100)  # -100 to 100 dB
    bandwidth: float | None = Field(None, gt=0, le=100_000_000)  # Max 100 MHz
    ppm: float | None = Field(None, ge=-1000, le=1000)  # -1000 to 1000 PPM
    antenna: str | None = Field(None, max_length=100)
    deviceSettings: dict[str, str] | None = None
    elementGains: dict[str, float] | None = None
    streamFormat: str | None = Field(None, max_length=20)
    dcOffsetAuto: bool | None = None
    iqBalanceAuto: bool | None = None
    name: str | None = Field(None, max_length=200)  # User-provided name (optional)
    # FFT/Spectrum settings
    fftFps: int | None = Field(None, ge=1, le=60)  # Target FPS (1-60)
    fftMaxFps: int | None = Field(None, ge=1, le=120)  # Max FPS cap (1-120)
    fftSize: int | None = Field(None)  # 512, 1024, 2048, 4096
    fftAccelerator: str | None = Field(None, pattern=r"^(auto|scipy|fftw|mlx|cuda)$")  # FFT backend

    @field_validator("fftSize")
    @classmethod
    def validate_fft_size(cls, v: int | None) -> int | None:
        if v is not None and v not in [512, 1024, 2048, 4096]:
            raise ValueError("fftSize must be 512, 1024, 2048, or 4096")
        return v

    @field_validator("name", "antenna", "deviceId", "streamFormat")
    @classmethod
    def sanitize_string(cls, v: str | None) -> str | None:
        if v is None:
            return None
        # Remove control characters and trim whitespace
        v = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", v)
        v = v.strip()
        # Return None if empty after sanitization
        return v if v else None


class ConfigWarning(BaseModel):
    """A configuration warning for display in the UI."""

    code: str  # Machine-readable code (e.g., "rtl_unstable_sample_rate")
    severity: Literal["warning", "info"] = "warning"
    message: str  # Human-readable message


class CaptureModel(BaseModel):
    id: str
    deviceId: str
    state: Literal["created", "starting", "running", "stopping", "stopped", "failed"]
    centerHz: float
    sampleRate: int
    gain: float | None = None
    bandwidth: float | None = None
    ppm: float | None = None
    antenna: str | None = None
    deviceSettings: dict[str, str] | None = None
    elementGains: dict[str, float] | None = None
    streamFormat: str | None = None
    dcOffsetAuto: bool | None = None
    iqBalanceAuto: bool | None = None
    errorMessage: str | None = None
    name: str | None = None  # User-provided name (optional)
    autoName: str | None = None  # Auto-generated name (e.g., "FM 90.3 - RTL-SDR")
    # FFT/Spectrum settings
    fftFps: int = 15  # Target FFT frames per second
    fftMaxFps: int = 60  # Maximum FFT frames per second (hard cap)
    fftSize: int = 2048  # FFT bin count
    fftAccelerator: str = "auto"  # FFT backend: auto, scipy, fftw, mlx, cuda
    # Error indicators for UI
    iqOverflowCount: int = 0
    iqOverflowRate: float = 0.0  # Overflows per second
    retryAttempt: int | None = None  # Current retry attempt (null if not retrying)
    retryMaxAttempts: int | None = None
    retryDelay: float | None = None  # Delay in seconds before next retry
    # Configuration warnings
    configWarnings: list[ConfigWarning] = []  # Lint warnings about configuration
    # Trunking system ownership
    trunkingSystemId: str | None = None  # Set if this capture is managed by a trunking system


class CreateChannelRequest(BaseModel):
    mode: Mode = "wbfm"
    offsetHz: float | None = Field(0.0, ge=-50_000_000, le=50_000_000)  # +/- 50 MHz offset
    audioRate: int | None = Field(None, ge=8000, le=192_000)  # 8 kHz to 192 kHz
    squelchDb: float | None = Field(None, ge=-120, le=0)  # -120 to 0 dB
    name: str | None = Field(None, max_length=200)  # User-provided name (optional)

    # Filter configuration (optional, mode-specific defaults applied if None)
    # FM filters
    enableDeemphasis: bool | None = None
    deemphasisTauUs: float | None = Field(None, ge=1, le=200)  # 1-200 µs
    enableMpxFilter: bool | None = None
    mpxCutoffHz: float | None = Field(None, gt=0, le=20_000)
    enableFmHighpass: bool | None = None
    fmHighpassHz: float | None = Field(None, gt=0, le=1_000)
    enableFmLowpass: bool | None = None
    fmLowpassHz: float | None = Field(None, gt=0, le=20_000)

    # AM/SSB filters
    enableAmHighpass: bool | None = None
    amHighpassHz: float | None = Field(None, gt=0, le=1_000)
    enableAmLowpass: bool | None = None
    amLowpassHz: float | None = Field(None, gt=0, le=20_000)
    enableSsbBandpass: bool | None = None
    ssbBandpassLowHz: float | None = Field(None, gt=0, le=10_000)
    ssbBandpassHighHz: float | None = Field(None, gt=0, le=10_000)
    ssbMode: Literal["usb", "lsb"] | None = None
    ssbBfoOffsetHz: float | None = Field(None, gt=0, le=5_000)  # BFO offset for centering voice

    # SAM (Synchronous AM) settings
    samSideband: Literal["dsb", "usb", "lsb"] | None = None  # Sideband selection
    samPllBandwidthHz: float | None = Field(None, ge=10, le=200)  # PLL bandwidth (10-200 Hz)

    # AGC
    enableAgc: bool | None = None
    agcTargetDb: float | None = Field(None, ge=-60, le=0)
    agcAttackMs: float | None = Field(None, gt=0, le=1000)
    agcReleaseMs: float | None = Field(None, gt=0, le=5000)

    # Notch filters (interference rejection)
    notchFrequencies: list[float] | None = Field(None, max_length=10)  # Max 10 notches

    # Spectral noise reduction (hiss/static suppression)
    enableNoiseReduction: bool | None = None
    noiseReductionDb: float | None = Field(None, ge=3, le=30)  # 3-30 dB reduction

    @field_validator("name")
    @classmethod
    def sanitize_name(cls, v: str | None) -> str | None:
        if v is None:
            return None
        # Remove control characters and trim whitespace
        v = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", v)
        v = v.strip()
        # Return None if empty after sanitization
        return v if v else None

    @field_validator("notchFrequencies")
    @classmethod
    def validate_notch_frequencies(cls, v: list[float] | None) -> list[float] | None:
        if v is None:
            return None
        # Validate each frequency is positive and reasonable (0-20kHz for audio)
        for freq in v:
            if freq <= 0 or freq > 20000:
                raise ValueError(f"Notch frequency must be between 0 and 20000 Hz, got {freq}")
        return v


class UpdateChannelRequest(BaseModel):
    mode: Mode | None = None
    offsetHz: float | None = Field(None, ge=-50_000_000, le=50_000_000)  # +/- 50 MHz offset
    audioRate: int | None = Field(None, ge=8000, le=192_000)  # 8 kHz to 192 kHz
    squelchDb: float | None = Field(None, ge=-120, le=0)  # -120 to 0 dB
    name: str | None = Field(None, max_length=200)  # User-provided name (optional)

    # Filter configuration (optional)
    # FM filters
    enableDeemphasis: bool | None = None
    deemphasisTauUs: float | None = Field(None, ge=1, le=200)  # 1-200 µs
    enableMpxFilter: bool | None = None
    mpxCutoffHz: float | None = Field(None, gt=0, le=20_000)
    enableFmHighpass: bool | None = None
    fmHighpassHz: float | None = Field(None, gt=0, le=1_000)
    enableFmLowpass: bool | None = None
    fmLowpassHz: float | None = Field(None, gt=0, le=20_000)

    # AM/SSB filters
    enableAmHighpass: bool | None = None
    amHighpassHz: float | None = Field(None, gt=0, le=1_000)
    enableAmLowpass: bool | None = None
    amLowpassHz: float | None = Field(None, gt=0, le=20_000)
    enableSsbBandpass: bool | None = None
    ssbBandpassLowHz: float | None = Field(None, gt=0, le=10_000)
    ssbBandpassHighHz: float | None = Field(None, gt=0, le=10_000)
    ssbMode: Literal["usb", "lsb"] | None = None
    ssbBfoOffsetHz: float | None = Field(None, gt=0, le=5_000)  # BFO offset for centering voice

    # SAM (Synchronous AM) settings
    samSideband: Literal["dsb", "usb", "lsb"] | None = None  # Sideband selection
    samPllBandwidthHz: float | None = Field(None, ge=10, le=200)  # PLL bandwidth (10-200 Hz)

    # AGC
    enableAgc: bool | None = None
    agcTargetDb: float | None = Field(None, ge=-60, le=0)
    agcAttackMs: float | None = Field(None, gt=0, le=1000)
    agcReleaseMs: float | None = Field(None, gt=0, le=5000)

    # Noise blanker (impulse noise suppression)
    enableNoiseBlanker: bool | None = None
    noiseBlankerThresholdDb: float | None = Field(None, ge=3, le=30)  # 3-30 dB above median

    # Notch filters (interference rejection)
    notchFrequencies: list[float] | None = Field(None, max_length=10)  # Max 10 notches

    # Spectral noise reduction (hiss/static suppression)
    enableNoiseReduction: bool | None = None
    noiseReductionDb: float | None = Field(None, ge=3, le=30)  # 3-30 dB reduction

    @field_validator("name")
    @classmethod
    def sanitize_name(cls, v: str | None) -> str | None:
        if v is None:
            return None
        # Remove control characters and trim whitespace
        v = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", v)
        v = v.strip()
        # Return None if empty after sanitization
        return v if v else None

    @field_validator("notchFrequencies")
    @classmethod
    def validate_notch_frequencies(cls, v: list[float] | None) -> list[float] | None:
        if v is None:
            return None
        # Validate each frequency is positive and reasonable (0-20kHz for audio)
        for freq in v:
            if freq <= 0 or freq > 20000:
                raise ValueError(f"Notch frequency must be between 0 and 20000 Hz, got {freq}")
        return v


class ChannelModel(BaseModel):
    id: str
    captureId: str
    mode: Mode
    state: Literal["created", "running", "stopped"]
    offsetHz: float
    audioRate: int
    squelchDb: float | None = None
    name: str | None = None  # User-provided name
    autoName: str | None = None  # Auto-generated contextual name (e.g., "Marine Ch 16")
    signalPowerDb: float | None = None
    rssiDb: float | None = None  # Server-side RSSI from IQ samples
    snrDb: float | None = None  # Server-side SNR estimate
    # Audio output level metering
    audioRmsDb: float | None = None  # Output audio RMS level in dB
    audioPeakDb: float | None = None  # Output audio peak level in dB
    audioClippingCount: int = 0  # Number of samples near clipping
    # Error indicators for UI
    audioDropCount: int = 0  # Total dropped audio packets
    audioDropRate: float = 0.0  # Drops per second

    # Filter configuration
    # FM filters
    enableDeemphasis: bool
    deemphasisTauUs: float
    enableMpxFilter: bool
    mpxCutoffHz: float
    enableFmHighpass: bool
    fmHighpassHz: float
    enableFmLowpass: bool
    fmLowpassHz: float

    # AM/SSB filters
    enableAmHighpass: bool
    amHighpassHz: float
    enableAmLowpass: bool
    amLowpassHz: float
    enableSsbBandpass: bool
    ssbBandpassLowHz: float
    ssbBandpassHighHz: float
    ssbMode: str
    ssbBfoOffsetHz: float

    # AGC
    enableAgc: bool
    agcTargetDb: float
    agcAttackMs: float
    agcReleaseMs: float

    # Noise blanker
    enableNoiseBlanker: bool
    noiseBlankerThresholdDb: float

    # Notch filters
    notchFrequencies: list[float] = []

    # Spectral noise reduction
    enableNoiseReduction: bool
    noiseReductionDb: float

    # RDS data (WBFM only)
    rdsData: RDSDataModel | None = None


class RecipeChannelModel(BaseModel):
    offsetHz: float
    name: str
    mode: str = "wbfm"
    squelchDb: float = -60
    # POCSAG decoding settings (NBFM only)
    enablePocsag: bool = False
    pocsagBaud: int = 1200


class RecipeModel(BaseModel):
    id: str
    name: str
    description: str
    category: str
    centerHz: float
    sampleRate: int
    gain: float | None = None
    bandwidth: float | None = None
    channels: list[RecipeChannelModel] = []
    allowFrequencyInput: bool = False
    frequencyLabel: str | None = None


# Scanner models
class ScanMode(str):
    SEQUENTIAL = "sequential"
    PRIORITY = "priority"
    ACTIVITY = "activity"


class ScanState(str):
    STOPPED = "stopped"
    SCANNING = "scanning"
    PAUSED = "paused"
    LOCKED = "locked"


class CreateScannerRequest(BaseModel):
    captureId: str
    scanList: list[float]  # Frequencies in Hz
    mode: Literal["sequential", "priority", "activity"] = "sequential"
    dwellTimeMs: int = Field(500, ge=100, le=10000)  # 100ms to 10s
    priorityFrequencies: list[float] = []
    priorityIntervalS: int = Field(5, ge=1, le=60)
    squelchThresholdDb: float = Field(-60.0, ge=-120.0, le=0.0)
    lockoutFrequencies: list[float] = []
    pauseDurationMs: int = Field(3000, ge=500, le=30000)  # 0.5s to 30s


class UpdateScannerRequest(BaseModel):
    scanList: list[float] | None = None
    mode: Literal["sequential", "priority", "activity"] | None = None
    dwellTimeMs: int | None = Field(None, ge=100, le=10000)
    priorityFrequencies: list[float] | None = None
    priorityIntervalS: int | None = Field(None, ge=1, le=60)
    squelchThresholdDb: float | None = Field(None, ge=-120.0, le=0.0)
    lockoutFrequencies: list[float] | None = None
    pauseDurationMs: int | None = Field(None, ge=500, le=30000)


class ScanHitModel(BaseModel):
    frequencyHz: float
    timestamp: float


class ScannerModel(BaseModel):
    id: str
    captureId: str
    state: Literal["stopped", "scanning", "paused", "locked"]
    currentFrequency: float
    currentIndex: int
    scanList: list[float]
    mode: Literal["sequential", "priority", "activity"]
    dwellTimeMs: int
    priorityFrequencies: list[float]
    priorityIntervalS: int
    squelchThresholdDb: float
    lockoutList: list[float]
    pauseDurationMs: int
    hits: list[ScanHitModel]


# RDS (Radio Data System) data for FM broadcast
class RDSDataModel(BaseModel):
    """RDS data decoded from FM broadcast."""

    piCode: str | None = None  # Program Identification (hex string like "A1B2")
    psName: str | None = None  # Program Service name (8 chars, station name)
    radioText: str | None = None  # Radio Text (up to 64 chars)
    pty: int = 0  # Program Type code
    ptyName: str = "None"  # Program Type name (e.g., "Rock", "News")
    ta: bool = False  # Traffic Announcement flag
    tp: bool = False  # Traffic Program flag
    ms: bool = True  # Music/Speech switch (True = Music)


# POCSAG pager message
class POCSAGMessageModel(BaseModel):
    """A decoded POCSAG pager message."""

    address: int  # 21-bit address (capcode)
    function: int  # 2-bit function code (0-3)
    messageType: str  # "numeric", "alpha", "alert_only", or "alpha_2"
    message: str  # Decoded message content
    timestamp: float  # Unix timestamp
    baudRate: int = 1200  # 512, 1200, or 2400
    alias: str | None = None  # Human-readable name from config (e.g., "CFS Dispatch")


# Signal monitoring models
class SpectrumSnapshotModel(BaseModel):
    """Single FFT spectrum snapshot (no WebSocket required)."""

    power: list[float]  # Power spectrum in dB
    freqs: list[float]  # Frequency bins in Hz
    centerHz: float
    sampleRate: int
    timestamp: float


class ClassifiedChannelModel(BaseModel):
    """A channel classified by power variance analysis."""

    freqHz: float
    powerDb: float
    stdDevDb: float
    channelType: str  # "control", "voice", "variable", "unknown"


class ClassifiedChannelsResponse(BaseModel):
    """Response for channel classification endpoint."""

    channels: list[ClassifiedChannelModel]
    status: dict[str, Any]  # elapsed_seconds, sample_count, is_ready, remaining_seconds


class ExtendedMetricsModel(BaseModel):
    """Extended signal metrics for tuning and monitoring."""

    channelId: str
    # Core signal metrics
    rssiDb: float | None = None  # Received signal strength (dB)
    snrDb: float | None = None  # Signal-to-noise ratio (dB)
    signalPowerDb: float | None = None  # Audio signal power (dB)
    # Derived metrics
    sUnits: str | None = None  # S-meter reading (S0-S9, S9+10, etc.)
    squelchOpen: bool = False  # Whether squelch is currently open
    # Stream health
    streamSubscribers: int = 0
    streamDropsPerSec: float = 0.0
    # Capture-level info
    captureState: str = "unknown"
    timestamp: float = 0.0


class MetricsHistoryPoint(BaseModel):
    """Single point in metrics history."""

    timestamp: float
    rssiDb: float | None = None
    snrDb: float | None = None
    signalPowerDb: float | None = None


class MetricsHistoryModel(BaseModel):
    """Time-series of signal metrics."""

    channelId: str
    points: list[MetricsHistoryPoint]
    durationSeconds: float
