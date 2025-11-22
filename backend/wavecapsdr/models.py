from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional
import re

from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator


Mode = Literal["wbfm", "nbfm", "am", "ssb", "raw", "p25", "dmr"]
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
    gainMin: Optional[float] = Field(None, alias="gain_min")
    gainMax: Optional[float] = Field(None, alias="gain_max")
    bandwidthMin: Optional[float] = Field(None, alias="bandwidth_min")
    bandwidthMax: Optional[float] = Field(None, alias="bandwidth_max")
    ppmMin: Optional[float] = Field(None, alias="ppm_min")
    ppmMax: Optional[float] = Field(None, alias="ppm_max")
    antennas: list[str]
    nickname: Optional[str] = None  # Custom user-provided nickname
    shorthand: Optional[str] = None  # Auto-generated shorthand name (e.g., "RTL-SDR", "SDRplay RSPdx")
    model_config = ConfigDict(populate_by_name=True)


class CreateCaptureRequest(BaseModel):
    deviceId: Optional[str] = Field(None, max_length=500)  # Device IDs can be long SoapySDR strings
    centerHz: float = Field(..., gt=0, lt=10e9)  # 0 to 10 GHz
    sampleRate: int = Field(..., gt=0, le=100_000_000)  # Max 100 MHz
    gain: Optional[float] = Field(None, ge=-100, le=100)  # -100 to 100 dB
    bandwidth: Optional[float] = Field(None, gt=0, le=100_000_000)  # Max 100 MHz
    ppm: Optional[float] = Field(None, ge=-1000, le=1000)  # -1000 to 1000 PPM
    antenna: Optional[str] = Field(None, max_length=100)
    deviceSettings: Optional[dict[str, str]] = None
    elementGains: Optional[dict[str, float]] = None
    streamFormat: Optional[str] = Field(None, max_length=20)
    dcOffsetAuto: bool = True
    iqBalanceAuto: bool = True
    createDefaultChannel: bool = True
    name: Optional[str] = Field(None, max_length=200)  # User-provided name (optional)

    @field_validator('name', 'antenna', 'deviceId', 'streamFormat')
    @classmethod
    def sanitize_string(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        # Remove control characters and trim whitespace
        v = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', v)
        v = v.strip()
        # Return None if empty after sanitization
        return v if v else None

    @model_validator(mode='after')
    def validate_bandwidth_with_sample_rate(self):
        if self.bandwidth is not None and self.sampleRate is not None:
            if self.bandwidth > self.sampleRate:
                raise ValueError('Bandwidth cannot exceed sample rate')
        return self


class UpdateCaptureRequest(BaseModel):
    deviceId: Optional[str] = Field(None, max_length=500)  # Device IDs can be long SoapySDR strings
    centerHz: Optional[float] = Field(None, gt=0, lt=10e9)  # 0 to 10 GHz
    sampleRate: Optional[int] = Field(None, gt=0, le=100_000_000)  # Max 100 MHz
    gain: Optional[float] = Field(None, ge=-100, le=100)  # -100 to 100 dB
    bandwidth: Optional[float] = Field(None, gt=0, le=100_000_000)  # Max 100 MHz
    ppm: Optional[float] = Field(None, ge=-1000, le=1000)  # -1000 to 1000 PPM
    antenna: Optional[str] = Field(None, max_length=100)
    deviceSettings: Optional[dict[str, str]] = None
    elementGains: Optional[dict[str, float]] = None
    streamFormat: Optional[str] = Field(None, max_length=20)
    dcOffsetAuto: Optional[bool] = None
    iqBalanceAuto: Optional[bool] = None
    name: Optional[str] = Field(None, max_length=200)  # User-provided name (optional)

    @field_validator('name', 'antenna', 'deviceId', 'streamFormat')
    @classmethod
    def sanitize_string(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        # Remove control characters and trim whitespace
        v = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', v)
        v = v.strip()
        # Return None if empty after sanitization
        return v if v else None


class CaptureModel(BaseModel):
    id: str
    deviceId: str
    state: Literal["created", "starting", "running", "stopping", "stopped", "failed"]
    centerHz: float
    sampleRate: int
    gain: Optional[float] = None
    bandwidth: Optional[float] = None
    ppm: Optional[float] = None
    antenna: Optional[str] = None
    deviceSettings: Optional[dict[str, str]] = None
    elementGains: Optional[dict[str, float]] = None
    streamFormat: Optional[str] = None
    dcOffsetAuto: Optional[bool] = None
    iqBalanceAuto: Optional[bool] = None
    errorMessage: Optional[str] = None
    name: Optional[str] = None  # User-provided name (optional)
    autoName: Optional[str] = None  # Auto-generated name (e.g., "FM 90.3 - RTL-SDR")


class CreateChannelRequest(BaseModel):
    mode: Mode = "wbfm"
    offsetHz: Optional[float] = Field(0.0, ge=-50_000_000, le=50_000_000)  # +/- 50 MHz offset
    audioRate: Optional[int] = Field(None, ge=8000, le=192_000)  # 8 kHz to 192 kHz
    squelchDb: Optional[float] = Field(None, ge=-120, le=0)  # -120 to 0 dB
    name: Optional[str] = Field(None, max_length=200)  # User-provided name (optional)

    # Filter configuration (optional, mode-specific defaults applied if None)
    # FM filters
    enableDeemphasis: Optional[bool] = None
    deemphasisTauUs: Optional[float] = Field(None, ge=1, le=200)  # 1-200 µs
    enableMpxFilter: Optional[bool] = None
    mpxCutoffHz: Optional[float] = Field(None, gt=0, le=20_000)
    enableFmHighpass: Optional[bool] = None
    fmHighpassHz: Optional[float] = Field(None, gt=0, le=1_000)
    enableFmLowpass: Optional[bool] = None
    fmLowpassHz: Optional[float] = Field(None, gt=0, le=20_000)

    # AM/SSB filters
    enableAmHighpass: Optional[bool] = None
    amHighpassHz: Optional[float] = Field(None, gt=0, le=1_000)
    enableAmLowpass: Optional[bool] = None
    amLowpassHz: Optional[float] = Field(None, gt=0, le=20_000)
    enableSsbBandpass: Optional[bool] = None
    ssbBandpassLowHz: Optional[float] = Field(None, gt=0, le=10_000)
    ssbBandpassHighHz: Optional[float] = Field(None, gt=0, le=10_000)
    ssbMode: Optional[Literal["usb", "lsb"]] = None

    # AGC
    enableAgc: Optional[bool] = None
    agcTargetDb: Optional[float] = Field(None, ge=-60, le=0)
    agcAttackMs: Optional[float] = Field(None, gt=0, le=1000)
    agcReleaseMs: Optional[float] = Field(None, gt=0, le=5000)

    # Notch filters (interference rejection)
    notchFrequencies: Optional[list[float]] = Field(None, max_length=10)  # Max 10 notches

    @field_validator('name')
    @classmethod
    def sanitize_name(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        # Remove control characters and trim whitespace
        v = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', v)
        v = v.strip()
        # Return None if empty after sanitization
        return v if v else None

    @field_validator('notchFrequencies')
    @classmethod
    def validate_notch_frequencies(cls, v: Optional[list[float]]) -> Optional[list[float]]:
        if v is None:
            return None
        # Validate each frequency is positive and reasonable (0-20kHz for audio)
        for freq in v:
            if freq <= 0 or freq > 20000:
                raise ValueError(f'Notch frequency must be between 0 and 20000 Hz, got {freq}')
        return v


class UpdateChannelRequest(BaseModel):
    mode: Optional[Mode] = None
    offsetHz: Optional[float] = Field(None, ge=-50_000_000, le=50_000_000)  # +/- 50 MHz offset
    audioRate: Optional[int] = Field(None, ge=8000, le=192_000)  # 8 kHz to 192 kHz
    squelchDb: Optional[float] = Field(None, ge=-120, le=0)  # -120 to 0 dB
    name: Optional[str] = Field(None, max_length=200)  # User-provided name (optional)

    # Filter configuration (optional)
    # FM filters
    enableDeemphasis: Optional[bool] = None
    deemphasisTauUs: Optional[float] = Field(None, ge=1, le=200)  # 1-200 µs
    enableMpxFilter: Optional[bool] = None
    mpxCutoffHz: Optional[float] = Field(None, gt=0, le=20_000)
    enableFmHighpass: Optional[bool] = None
    fmHighpassHz: Optional[float] = Field(None, gt=0, le=1_000)
    enableFmLowpass: Optional[bool] = None
    fmLowpassHz: Optional[float] = Field(None, gt=0, le=20_000)

    # AM/SSB filters
    enableAmHighpass: Optional[bool] = None
    amHighpassHz: Optional[float] = Field(None, gt=0, le=1_000)
    enableAmLowpass: Optional[bool] = None
    amLowpassHz: Optional[float] = Field(None, gt=0, le=20_000)
    enableSsbBandpass: Optional[bool] = None
    ssbBandpassLowHz: Optional[float] = Field(None, gt=0, le=10_000)
    ssbBandpassHighHz: Optional[float] = Field(None, gt=0, le=10_000)
    ssbMode: Optional[Literal["usb", "lsb"]] = None

    # AGC
    enableAgc: Optional[bool] = None
    agcTargetDb: Optional[float] = Field(None, ge=-60, le=0)
    agcAttackMs: Optional[float] = Field(None, gt=0, le=1000)
    agcReleaseMs: Optional[float] = Field(None, gt=0, le=5000)

    # Noise blanker (impulse noise suppression)
    enableNoiseBlanker: Optional[bool] = None
    noiseBlankerThresholdDb: Optional[float] = Field(None, ge=3, le=30)  # 3-30 dB above median

    # Notch filters (interference rejection)
    notchFrequencies: Optional[list[float]] = Field(None, max_length=10)  # Max 10 notches

    @field_validator('name')
    @classmethod
    def sanitize_name(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        # Remove control characters and trim whitespace
        v = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', v)
        v = v.strip()
        # Return None if empty after sanitization
        return v if v else None

    @field_validator('notchFrequencies')
    @classmethod
    def validate_notch_frequencies(cls, v: Optional[list[float]]) -> Optional[list[float]]:
        if v is None:
            return None
        # Validate each frequency is positive and reasonable (0-20kHz for audio)
        for freq in v:
            if freq <= 0 or freq > 20000:
                raise ValueError(f'Notch frequency must be between 0 and 20000 Hz, got {freq}')
        return v


class ChannelModel(BaseModel):
    id: str
    captureId: str
    mode: Mode
    state: Literal["created", "running", "stopped"]
    offsetHz: float
    audioRate: int
    squelchDb: Optional[float] = None
    name: Optional[str] = None  # User-provided name
    autoName: Optional[str] = None  # Auto-generated contextual name (e.g., "Marine Ch 16")
    signalPowerDb: Optional[float] = None
    rssiDb: Optional[float] = None  # Server-side RSSI from IQ samples
    snrDb: Optional[float] = None  # Server-side SNR estimate

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


class RecipeChannelModel(BaseModel):
    offsetHz: float
    name: str
    mode: str = "wbfm"
    squelchDb: float = -60


class RecipeModel(BaseModel):
    id: str
    name: str
    description: str
    category: str
    centerHz: float
    sampleRate: int
    gain: Optional[float] = None
    bandwidth: Optional[float] = None
    channels: list[RecipeChannelModel] = []
    allowFrequencyInput: bool = False
    frequencyLabel: Optional[str] = None

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
    scanList: Optional[list[float]] = None
    mode: Optional[Literal["sequential", "priority", "activity"]] = None
    dwellTimeMs: Optional[int] = Field(None, ge=100, le=10000)
    priorityFrequencies: Optional[list[float]] = None
    priorityIntervalS: Optional[int] = Field(None, ge=1, le=60)
    squelchThresholdDb: Optional[float] = Field(None, ge=-120.0, le=0.0)
    lockoutFrequencies: Optional[list[float]] = None
    pauseDurationMs: Optional[int] = Field(None, ge=500, le=30000)


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
