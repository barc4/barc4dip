# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2026 ESRF - the European Synchrotron

"""
Radial/azimuthal reductions.

This module provides two radial-average estimators:

- radial_mean_binned:
    Fast, interpolation-free annular binning on the pixel grid.
    Preferred for metrics and rolling-window computations.

- radial_mean_interpolated:
    High-definition polar sampling with interpolation.
    Preferred for smooth diagnostic curves and theory comparisons.

Both functions assume:
    - x and y are 1D coordinate axes (pixel centers or physical units),
    - x and y are centered at 0,
    - the radial mean is computed around the origin (0, 0).
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator


def radial_mean_binned(
    signal_2d: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    r_max: float | None = None,
    bin_size: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Radial mean of a 2D signal using annular binning (no interpolation).

    Parameters:
        signal_2d (np.ndarray):
            2D signal to radially average with shape (ny, nx).
        x (np.ndarray):
            1D x-axis coordinates (length nx), centered at 0.
        y (np.ndarray):
            1D y-axis coordinates (length ny), centered at 0.
        r_max (float | None):
            Maximum radius. If None, uses the inscribed circle:
            min(max(|x|), max(|y|)).
        bin_size (float | None):
            Radial bin size. If None, uses min(|dx|, |dy|) inferred from x/y.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - radial (np.ndarray): radial mean values per bin.
            - r (np.ndarray): radius values at bin centers (same units as x/y).

    Raises:
        ValueError
    """
    z = np.asarray(signal_2d, dtype=float)
    if z.ndim != 2:
        raise ValueError("signal_2d must be a 2D array.")

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1D arrays.")
    if z.shape != (y.size, x.size):
        raise ValueError(
            f"signal_2d shape must match (y.size, x.size) = {(y.size, x.size)}, got {z.shape}."
        )

    if r_max is None:
        r_max = min(float(np.max(np.abs(x))), float(np.max(np.abs(y))))
    if r_max <= 0:
        raise ValueError("r_max must be > 0 (or leave it as None with valid x/y).")

    if bin_size is None:
        dx = float(np.median(np.abs(np.diff(x)))) if x.size > 1 else 1.0
        dy = float(np.median(np.abs(np.diff(y)))) if y.size > 1 else 1.0
        bin_size = float(min(dx, dy))
    if bin_size <= 0:
        raise ValueError("bin_size must be > 0.")

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

    r = (np.arange(nbins, dtype=float) + 0.5) * bin_size
    return radial, r


def radial_mean_interpolated(
    signal_2d: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
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
            2D signal to radially average with shape (ny, nx).
        x (np.ndarray):
            1D x-axis coordinates (length nx), centered at 0.
        y (np.ndarray):
            1D y-axis coordinates (length ny), centered at 0.
        r_max (float | None):
            Maximum radius. If None, uses the inscribed circle:
            min(max(|x|), max(|y|)).
        nr (int | None):
            Number of radial samples. If None, uses nx//2.
        ntheta (int | None):
            Number of angular samples. If None, uses ~1 degree sampling.
        fill_value (float):
            Value used outside interpolation bounds (default: 0.0).

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - radial (np.ndarray): radially averaged values.
            - r (np.ndarray): radial distances (same units as x/y).

    Raises:
        ValueError
    """
    z = np.asarray(signal_2d, dtype=float)
    if z.ndim != 2:
        raise ValueError("signal_2d must be a 2D array.")

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1D arrays.")
    if z.shape != (y.size, x.size):
        raise ValueError(
            f"signal_2d shape must match (y.size, x.size) = {(y.size, x.size)}, got {z.shape}."
        )

    if r_max is None:
        r_max = min(float(np.max(np.abs(x))), float(np.max(np.abs(y))))
    if r_max <= 0:
        raise ValueError("r_max must be > 0 (or leave it as None with valid x/y).")

    ny, nx = z.shape
    if nr is None:
        nr = int(nx * 0.5)
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
