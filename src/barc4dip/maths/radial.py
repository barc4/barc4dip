# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2026 ESRF - the European Synchrotron

"""
Radial reductions (pixel domain).

This module provides two radial-average estimators:

- radial_mean_binned:
    Fast, interpolation-free annular binning on the pixel grid.
    Preferred for metrics and rolling-window computations.

- radial_mean_interpolated:
    High-definition polar sampling with interpolation.
    Preferred for smooth diagnostic curves and theory comparisons.

Conventions:
- Inputs are plain NumPy arrays.
- All distances are returned in pixels.
- The origin is the array center as defined by pixel-center coordinates:
    x = arange(nx) - nx//2
    y = arange(ny) - ny//2
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator


def _pixel_axes(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    ny, nx = shape
    x = np.arange(nx, dtype=float) - (nx // 2)
    y = np.arange(ny, dtype=float) - (ny // 2)
    return x, y


def radial_mean_binned(
    signal_2d: np.ndarray,
    *,
    r_max: float | None = None,
    bin_size: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Radial mean of a 2D signal using annular binning (no interpolation).

    Parameters:
        signal_2d (np.ndarray):
            2D signal to radially average.
        r_max (float | None):
            Maximum radius in pixels. If None, uses the inscribed circle radius in pixels.
        bin_size (float):
            Radial bin size in pixels (default: 1.0).

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - radial (np.ndarray): radial mean values per bin.
            - r (np.ndarray): radius values at bin centers (pixels).

    Raises:
        ValueError
    """
    z = np.asarray(signal_2d, dtype=float)
    if z.ndim != 2:
        raise ValueError("signal_2d must be a 2D array.")
    if not np.isfinite(z).all():
        raise ValueError("signal_2d contains non-finite values.")
    if bin_size <= 0:
        raise ValueError("bin_size must be > 0.")

    ny, nx = z.shape
    x, y = _pixel_axes((ny, nx))

    if r_max is None:
        # Inscribed circle radius around the origin in pixel-center coordinates.
        r_max = min(float(np.max(np.abs(x))), float(np.max(np.abs(y))))
    if r_max <= 0:
        raise ValueError("r_max must be > 0 (or leave it as None with valid shape).")

    Y, X = np.meshgrid(y, x, indexing="ij")
    R = np.sqrt(X * X + Y * Y)

    nbins = int(np.floor(r_max / bin_size)) + 1
    bin_idx = np.floor(R / bin_size).astype(np.int64)

    mask = bin_idx < nbins
    idx = bin_idx[mask].ravel()
    val = z[mask].ravel()

    sums = np.bincount(idx, weights=val, minlength=nbins).astype(float)
    counts = np.bincount(idx, minlength=nbins).astype(float)

    radial = np.full(nbins, np.nan, dtype=float)
    valid = counts > 0
    radial[valid] = sums[valid] / counts[valid]

    r = (np.arange(nbins, dtype=float) + 0.5) * float(bin_size)
    return radial, r


def radial_mean_interpolated(
    signal_2d: np.ndarray,
    *,
    r_max: float | None = None,
    nr: int | None = None,
    ntheta: int | None = None,
    fill_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Radial mean of a 2D signal via polar sampling + interpolation (high-definition).

    Parameters:
        signal_2d (np.ndarray):
            2D signal to radially average.
        r_max (float | None):
            Maximum radius in pixels. If None, uses the inscribed circle radius in pixels.
        nr (int | None):
            Number of radial samples. If None, uses floor(r_max) + 1.
        ntheta (int | None):
            Number of angular samples. If None, uses ~1 degree sampling.
        fill_value (float):
            Value used outside interpolation bounds (default: 0.0).

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - radial (np.ndarray): radially averaged values.
            - r (np.ndarray): radial distances (pixels).

    Raises:
        ValueError
    """
    z = np.asarray(signal_2d, dtype=float)
    if z.ndim != 2:
        raise ValueError("signal_2d must be a 2D array.")
    if not np.isfinite(z).all():
        raise ValueError("signal_2d contains non-finite values.")

    ny, nx = z.shape
    x, y = _pixel_axes((ny, nx))

    if r_max is None:
        r_max = min(float(np.max(np.abs(x))), float(np.max(np.abs(y))))
    if r_max <= 0:
        raise ValueError("r_max must be > 0 (or leave it as None with valid shape).")

    if nr is None:
        nr = int(np.floor(r_max)) + 1
    if ntheta is None:
        ntheta = int(2.0 * np.pi * 180.0)  # ~1 degree sampling

    if nr <= 1:
        raise ValueError("nr must be > 1.")
    if ntheta <= 3:
        raise ValueError("ntheta must be > 3.")

    r = np.linspace(0.0, r_max, nr)
    theta = np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False)

    R_grid, THETA_grid = np.meshgrid(r, theta, indexing="ij")
    X_grid = R_grid * np.cos(THETA_grid)
    Y_grid = R_grid * np.sin(THETA_grid)

    interp = RegularGridInterpolator((y, x), z, bounds_error=False, fill_value=fill_value)

    points = np.column_stack([Y_grid.ravel(), X_grid.ravel()])
    values = interp(points).reshape(R_grid.shape)

    radial = np.mean(values, axis=1)
    return radial, r
