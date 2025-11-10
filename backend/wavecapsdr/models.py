from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from pydantic import BaseModel, Field, ConfigDict


Mode = Literal["wbfm", "am", "ssb"]
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
    model_config = ConfigDict(populate_by_name=True)


class CreateCaptureRequest(BaseModel):
    deviceId: Optional[str] = None
    centerHz: float
    sampleRate: int
    gain: Optional[float] = None
    bandwidth: Optional[float] = None
    ppm: Optional[float] = None
    antenna: Optional[str] = None
    deviceSettings: Optional[dict[str, str]] = None
    elementGains: Optional[dict[str, float]] = None
    streamFormat: Optional[str] = None
    dcOffsetAuto: bool = True
    iqBalanceAuto: bool = True
    createDefaultChannel: bool = True


class UpdateCaptureRequest(BaseModel):
    centerHz: Optional[float] = None
    sampleRate: Optional[int] = None
    gain: Optional[float] = None
    bandwidth: Optional[float] = None
    ppm: Optional[float] = None
    antenna: Optional[str] = None
    deviceSettings: Optional[dict[str, str]] = None
    elementGains: Optional[dict[str, float]] = None
    streamFormat: Optional[str] = None
    dcOffsetAuto: Optional[bool] = None
    iqBalanceAuto: Optional[bool] = None


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


class CreateChannelRequest(BaseModel):
    mode: Mode = "wbfm"
    offsetHz: Optional[float] = 0.0
    audioRate: Optional[int] = None
    squelchDb: Optional[float] = None


class UpdateChannelRequest(BaseModel):
    mode: Optional[Mode] = None
    offsetHz: Optional[float] = None
    audioRate: Optional[int] = None
    squelchDb: Optional[float] = None


class ChannelModel(BaseModel):
    id: str
    captureId: str
    mode: Mode
    state: Literal["created", "running", "stopped"]
    offsetHz: float
    audioRate: int
    squelchDb: Optional[float] = None


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
