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
    deviceId: Optional[str] = Field(None, max_length=100)
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
    deviceId: Optional[str] = Field(None, max_length=100)
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


class UpdateChannelRequest(BaseModel):
    mode: Optional[Mode] = None
    offsetHz: Optional[float] = Field(None, ge=-50_000_000, le=50_000_000)  # +/- 50 MHz offset
    audioRate: Optional[int] = Field(None, ge=8000, le=192_000)  # 8 kHz to 192 kHz
    squelchDb: Optional[float] = Field(None, ge=-120, le=0)  # -120 to 0 dB
    name: Optional[str] = Field(None, max_length=200)  # User-provided name (optional)

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
