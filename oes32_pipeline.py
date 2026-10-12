# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Jean-François Brisson, Spark AI NLP
"""GUI-free signal path used by the OES-32 dashboard.

Everything here is synthetic: ``synthetic_sample`` generates a 10 Hz test tone
per channel plus a common 50 Hz tone, in arbitrary units. No sensor, hardware,
or biological signal is involved.
"""

from __future__ import annotations

import math

import numpy as np
import scipy.signal as signal


def synthetic_sample(t: float, num_channels: int = 32) -> np.ndarray:
  """Return one synthetic frame at time ``t`` (seconds), arbitrary units."""
  tone = [math.sin(2 * math.pi * 10.0 * t + ch * 0.1) * 25.0 for ch in range(num_channels)]
  mains = math.sin(2 * math.pi * 50.0 * t) * 15.0
  return np.asarray(tone, dtype=np.float64) + mains


class OES32FilterPipeline:
  """Stateful 50 Hz notch (Q=30) followed by a 4th-order 0.5-45 Hz bandpass."""

  def __init__(self, num_channels: int = 32, fs: float = 250.0) -> None:
    if num_channels < 1 or not fs > 100.0:
      raise ValueError("need at least one channel and fs above 100 Hz")
    self.num_channels = num_channels
    self.fs = fs
    b_notch, a_notch = signal.iirnotch(50.0, 30.0, fs=fs)
    self.sos_notch = signal.tf2sos(b_notch, a_notch)
    self.zi_notch = np.repeat(
        signal.sosfilt_zi(self.sos_notch)[:, :, np.newaxis], num_channels, axis=2
    )
    self.sos_bp = signal.butter(4, [0.5, 45.0], btype="bandpass", fs=fs, output="sos")
    self.zi_bp = np.repeat(
        signal.sosfilt_zi(self.sos_bp)[:, :, np.newaxis], num_channels, axis=2
    )

  def process(self, sample: np.ndarray) -> np.ndarray:
    """Filter one frame of shape ``(num_channels,)`` and return the result."""
    x = np.asarray(sample, dtype=np.float64)
    if x.shape != (self.num_channels,):
      raise ValueError(f"sample must have shape ({self.num_channels},)")
    y, self.zi_notch = signal.sosfilt(self.sos_notch, x[np.newaxis, :], axis=0, zi=self.zi_notch)
    y, self.zi_bp = signal.sosfilt(self.sos_bp, y, axis=0, zi=self.zi_bp)
    return y.squeeze(axis=0)


def mean_welch_psd(buffer: np.ndarray, fs: float = 250.0, nperseg: int = 256):
  """Welch PSD of each column of ``buffer``, averaged across channels."""
  freqs, psd = signal.welch(buffer, fs=fs, nperseg=nperseg, axis=0)
  return freqs, np.mean(psd, axis=1)
