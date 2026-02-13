# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2026 ESRF - the European Synchrotron

"""
Signal processing primitives.
"""

from __future__ import annotations

# FFT / PSD
from .fft import (
    fft1d,
    fft2d,
    freq_axis1d,
    freq_axes2d,
    ifft1d,
    ifft2d,
    psd1d,
    psd2d,
)

# Correlation
from .corr import (
    autocorr1d,
    autocorr2d,
    xcorr1d,
    xcorr2d,
)

__all__ = [
    # fft.py
    "fft1d",
    "ifft1d",
    "psd1d",
    "freq_axis1d",
    "fft2d",
    "ifft2d",
    "psd2d",
    "freq_axes2d",
    # corr.py
    "xcorr1d",
    "autocorr1d",
    "xcorr2d",
    "autocorr2d",
]
