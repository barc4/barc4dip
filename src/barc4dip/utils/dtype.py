# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2026 ESRF - the European Synchrotron

""" 
dtype conversions
"""

from __future__ import annotations

import numpy as np
from .range import filtered_minmax_range

def to_uint16(data: np.ndarray, *, median_size: int = 3, counts_threshold: float = 10.0,
              scaling: float = 1/np.sqrt(2)) -> np.ndarray:
    """
    Convert 2D image or 3D stack to uint16.

    Parameters:
            data (np.ndarray): 2D image (H, W) or 3D stack (N, H, W).
            median_size (int): Spatial median window size used in filtered_minmax_range.
            counts_threshold (float): If float data has values above this, it is treated as counts.
            scaling (float): Target mean grey value in [0, 1] used to scale contrast (default 1/sqrt(2)).
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("to_uint16 expects a numpy.ndarray")
    if data.dtype == np.uint16:
        return data
    if data.ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D array, got ndim={data.ndim}")

    m = float(np.nanmean(data))

    if m > counts_threshold:
        if np.issubdtype(data.dtype, np.floating):
            x = np.clip(data, 0.0, 65535.0)
        else:
            x = np.clip(data, 0, 65535)
        return x.astype(np.uint16, copy=False)

    vmin, vmax = filtered_minmax_range(data, size=median_size)
    vmin *= 0.95
    vmax /= 0.95

    inv = 65535 * scaling / (vmax - vmin)

    x = data.astype(np.float32, copy=True)
    np.subtract(x, vmin, out=x)
    np.multiply(x, inv, out=x)
    np.clip(x, 0.0, 65535.0, out=x)

    return x.astype(np.uint16, copy=False)

