# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2026 ESRF - the European Synchrotron

""" 
range (min max) calculations for numpy arrays
"""

from __future__ import annotations
import numpy as np
from scipy.ndimage import median_filter

def filtered_minmax_range(image: np.ndarray, size: int = 3) -> tuple[float, float]:
    """
    Estimate scaling bounds from a median-filtered copy (salt & pepper suppression).
    Uses nan-aware min/max.

    Parameters:
        image (np.ndarray): 2D image (H, W) or 3D stack (N, H, W).
        size (int): Spatial median window size (applied in y/x only for stacks).

    Returns:
        tuple[float, float]: (vmin, vmax) from the filtered data.

    Raises:
        ValueError: If image is not 2D/3D or the computed range is invalid.
    """
    if image.ndim == 2:
        mf_size = (size, size)
    elif image.ndim == 3:
        mf_size = (1, size, size)
    else:
        raise ValueError(f"Expected 2D or 3D array, got ndim={image.ndim}")

    ref = median_filter(image.astype(np.float32, copy=False), size=mf_size)

    vmin = float(np.nanmin(ref))
    vmax = float(np.nanmax(ref))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        raise ValueError(f"Invalid range after filtering: vmin={vmin}, vmax={vmax}")
    return vmin, vmax

def percentile_minmax_range(image: np.ndarray,
                            p_low: float = 0.05,
                            p_high: float = 99.95) -> tuple[float, float]:
    """
    image: iterable of ndarrays (or a big stacked array).
    Computes global percentile range across all pixels seen.
    """
    arr = np.asarray(image)
    vmin = np.nanpercentile(arr, p_low)
    vmax = np.nanpercentile(arr, p_high)
    return float(vmin), float(vmax)

def filtered_minmax_range_streaming(image: np.ndarray, size: int = 3) -> tuple[float, float]:
    """
    Salt & pepper robust bounds using per-frame 2D median filtering.

    Parameters:
        image (np.ndarray): 2D (H,W) or 3D (N,H,W).
        size (int): Median filter window size in (y,x).

    Returns:
        tuple[float, float]: (vmin, vmax) global over frames.

    Raises:
        ValueError: If ndim is not 2/3 or range is invalid.
    """
    if image.ndim == 2:
        ref = median_filter(image.astype(np.float32, copy=False), size=(size, size))
        vmin = float(np.nanmin(ref))
        vmax = float(np.nanmax(ref))
    elif image.ndim == 3:
        vmin = np.inf
        vmax = -np.inf
        for i in range(image.shape[0]):
            ref = median_filter(image[i].astype(np.float32, copy=False), size=(size, size))
            vmin = min(vmin, float(np.nanmin(ref)))
            vmax = max(vmax, float(np.nanmax(ref)))
    else:
        raise ValueError(f"Expected 2D or 3D array, got ndim={image.ndim}")

    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        raise ValueError(f"Invalid range after filtering: vmin={vmin}, vmax={vmax}")
    return float(vmin), float(vmax)