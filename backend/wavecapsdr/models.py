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
    model_config = ConfigDict(populate_by_name=True)


class CreateCaptureRequest(BaseModel):
    deviceId: Optional[str] = None
    centerHz: float
    sampleRate: int
    gain: Optional[float] = None
    bandwidth: Optional[float] = None
    ppm: Optional[float] = None


class CaptureModel(BaseModel):
    id: str
    deviceId: str
    state: Literal["created", "running", "stopped"]
    centerHz: float
    sampleRate: int


class CreateChannelRequest(BaseModel):
    mode: Literal["wbfm"] = "wbfm"
    offsetHz: Optional[float] = 0.0
    audioRate: Optional[int] = None
    squelchDb: Optional[float] = None


class ChannelModel(BaseModel):
    id: str
    captureId: str
    mode: Literal["wbfm"]
    state: Literal["created", "running", "stopped"]
    offsetHz: float
    audioRate: int
