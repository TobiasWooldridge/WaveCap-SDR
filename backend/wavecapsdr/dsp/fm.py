from __future__ import annotations

import numpy as np


def quadrature_demod(iq: np.ndarray) -> np.ndarray:
    if iq.size == 0:
        return np.empty(0, dtype=np.float32)
    # y[n] = angle(x[n] * conj(x[n-1]))
    x = iq.astype(np.complex64, copy=False)
    prod = x[1:] * np.conj(x[:-1])
    out = np.angle(prod).astype(np.float32)
    # Prepend zero to keep alignment
    return np.concatenate([np.zeros(1, dtype=np.float32), out])


def deemphasis_filter(x: np.ndarray, sample_rate: int, tau: float = 75e-6) -> np.ndarray:
    # Single-pole IIR: y[n] = y[n-1] + alpha * (x[n] - y[n-1])
    alpha = 1.0 / (1.0 + (1.0 / (2.0 * np.pi * tau * sample_rate)))
    y = np.empty_like(x, dtype=np.float32)
    acc = np.float32(0.0)
    for i in range(x.shape[0]):
        acc = acc + np.float32(alpha) * (np.float32(x[i]) - acc)
        y[i] = acc
    return y


def resample_linear(x: np.ndarray, in_rate: int, out_rate: int) -> np.ndarray:
    if x.size == 0 or in_rate == out_rate:
        return x.astype(np.float32, copy=False)
    t_in = np.arange(x.shape[0], dtype=np.float64) / float(in_rate)
    duration = t_in[-1] if x.shape[0] > 0 else 0.0
    n_out = max(1, int(round(duration * out_rate)))
    t_out = np.arange(n_out, dtype=np.float64) / float(out_rate)
    y = np.interp(t_out, t_in, x.astype(np.float64))
    return y.astype(np.float32)


def wbfm_demod(iq: np.ndarray, sample_rate: int, audio_rate: int = 48_000) -> np.ndarray:
    fm = quadrature_demod(iq)
    # Simple deemphasis
    fm = deemphasis_filter(fm, sample_rate)
    # Normalize roughly
    if fm.size:
        fm = fm / max(1e-6, np.max(np.abs(fm)))
    audio = resample_linear(fm, sample_rate, audio_rate)
    # Hard clip to [-1,1]
    np.clip(audio, -1.0, 1.0, out=audio)
    return audio

