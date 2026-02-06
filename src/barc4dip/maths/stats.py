# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2026 ESRF - the European Synchrotron

from __future__ import annotations

import numpy as np


def width_at_fraction(
    profile: np.ndarray,
    *,
    fraction: float = 1.0 / np.e,
    center_index: int = None,
) -> tuple[float, bool]:
    """
    Compute the full width (left+right) of a 1D peak at a given fraction of its peak value.

    The function finds the peak (or uses `center_index`), then walks left and right
    to locate the first samples below the threshold and linearly interpolates the
    crossing positions.

    Parameters
    ----------
    profile : np.ndarray
        1D array containing a single dominant peak.
    fraction : float, optional
        Fraction of the peak value to define the width (default: 1/e).
    center_index : int, optional
        Peak index. If None, uses argmax(profile).

    Returns
    -------
    width_px : float
        Width in pixels (can be non-integer due to linear interpolation).
        If the threshold is not reached on either side, returns `profile.size`.
    hit_edge : bool
        True if the threshold was not found on at least one side (width clipped
        by array extent); in that case width is `profile.size`.

    Raises
    ------
    ValueError
        If `profile` is not 1D or is empty, or if `fraction` is not in (0, 1).
    """
    p = np.asarray(profile, dtype=float)
    if p.ndim != 1 or p.size == 0:
        raise ValueError("profile must be a non-empty 1D array.")
    if not (0.0 < fraction < 1.0):
        raise ValueError("fraction must be in (0, 1).")

    c = int(np.argmax(p) if center_index is None else center_index)
    c = max(0, min(c, p.size - 1))

    peak = p[c]
    thr = peak * fraction

    i_left = None
    for i in range(c, -1, -1):
        if p[i] < thr:
            i_left = i
            break

    i_right = None
    for i in range(c, p.size):
        if p[i] < thr:
            i_right = i
            break

    if i_left is None or i_right is None:
        return float(p.size), True

    i0 = i_left
    i1 = i_left + 1
    y0, y1 = p[i0], p[i1]
    if y1 == y0:
        x_left = float(i0)
    else:
        x_left = i0 + (thr - y0) / (y1 - y0)

    j1 = i_right
    j0 = i_right - 1
    y0, y1 = p[j0], p[j1]
    if y1 == y0:
        x_right = float(j1)
    else:
        x_right = j0 + (thr - y0) / (y1 - y0)

    width = float(x_right - x_left)
    return width, False


def distance_at_fraction_from_peak(
    profile: np.ndarray,
    *,
    fraction: float = 1.0 / np.e,
    peak_index: int = 0,
) -> tuple[float, bool]:
    """
    Compute the one-sided distance from a peak index to the first crossing below
    `fraction * peak_value` going toward increasing index.

    This is intended for radial profiles (r >= 0), where you want a radius at 1/e
    rather than a full width.

    Parameters
    ----------
    profile : np.ndarray
        1D array (e.g., radial mean of autocorrelation).
    fraction : float, optional
        Fraction of peak value (default: 1/e).
    peak_index : int, optional
        Index of the peak (default: 0 for radial profiles).

    Returns
    -------
    dist_px : float
        Distance in pixels (can be non-integer due to interpolation).
        If the threshold is not reached, returns `profile.size`.
    hit_edge : bool
        True if the threshold was not found before the end of the array.

    Raises
    ------
    ValueError
        If input is invalid.
    """
    p = np.asarray(profile, dtype=float)
    if p.ndim != 1 or p.size == 0:
        raise ValueError("profile must be a non-empty 1D array.")
    if not (0.0 < fraction < 1.0):
        raise ValueError("fraction must be in (0, 1).")

    k0 = max(0, min(int(peak_index), p.size - 1))
    peak = p[k0]
    thr = peak * fraction

    i_cross = None
    for i in range(k0, p.size):
        if p[i] < thr:
            i_cross = i
            break

    if i_cross is None:
        return float(p.size), True
    if i_cross == k0:
        return 0.0, False

    i1 = i_cross
    i0 = i_cross - 1
    y0, y1 = p[i0], p[i1]
    if y1 == y0:
        x_cross = float(i1)
    else:
        x_cross = i0 + (thr - y0) / (y1 - y0)

    return float(x_cross - k0), False