# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2026 ESRF - the European Synchrotron

from __future__ import annotations

from typing import Literal

import numpy as np

from barc4dip.math.radial import radial_mean_binned, radial_mean_interpolated
from barc4dip.math.stats import distance_at_fraction_from_peak, width_at_fraction
from barc4dip.signal.corr import autocorr2d


def _uniform_step(axis: np.ndarray, name: str) -> float:
    a = np.asarray(axis, dtype=float)
    if a.ndim != 1 or a.size < 2:
        raise ValueError(f"{name} must be a 1D array with at least 2 samples.")

    d = np.diff(a)
    if not np.all(np.isfinite(d)):
        raise ValueError(f"{name} contains non-finite values.")

    if not (np.all(d > 0) or np.all(d < 0)):
        raise ValueError(f"{name} must be strictly monotonic (uniform sampling assumed).")

    d_abs = np.abs(d)
    step = float(np.median(d_abs))
    if step <= 0:
        raise ValueError(f"{name} has non-positive sampling step.")

    rel = float(np.max(np.abs(d_abs - step)) / step)
    if rel > 1e-6:
        raise ValueError(
            f"{name} appears non-uniform (max relative deviation {rel:.2e}). "
            "Provide uniformly sampled axes."
        )

    return step


def speckle_grain_metrics(
    image: np.ndarray,
    *,
    x: np.ndarray,
    y: np.ndarray,
    remove_mean: bool = True,
    standardize: bool = False,
    fraction: float = 1.0 / np.e,
    radial_method: Literal["binned", "interpolated"] = "binned",
    return_autocorr: bool = False,
    verbose: bool = False,
) -> dict[str, float] | tuple[dict[str, float], dict[str, np.ndarray]]:
    """
    Compute speckle grain metrics from the autocorrelation peak.

    Parameters:
        image (np.ndarray):
            2D speckle intensity image with shape (y.size, x.size). Crop the ROI before calling.
        x (np.ndarray):
            1D x-axis coordinates (length nx), uniformly sampled, typically centered at 0.
        y (np.ndarray):
            1D y-axis coordinates (length ny), uniformly sampled, typically centered at 0.
        remove_mean (bool):
            If True, subtracts the patch mean before autocorrelation (default: True).
        standardize (bool):
            If True, computes correlation on (I - mean) / std (default: False).
        fraction (float):
            Threshold fraction for 1D widths and radial radius (default: 1/e).
        radial_method (Literal["binned", "interpolated"]):
            Radial averaging method for leq (default: "binned").
        return_autocorr (bool):
            If True, also returns an autocorrelation dictionary with axes (default: False).
        verbose (bool):
            If True, prints a short summary of the computed metrics (default: False).

    Returns:
        dict[str, float]:
            Dictionary with keys:
                - lx: 1/e full width along x cut (same units as x).
                - ly: 1/e full width along y cut (same units as y).
                - leq: 1/e radius of the radially averaged autocorrelation (same units as x/y).
                - r: anisotropy ratio lx / ly.
        tuple[dict[str, float], dict[str, np.ndarray]]:
            If return_autocorr is True, returns (metrics, ac_dict) where ac_dict contains:
                - ac: 2D autocorrelation map (peak-normalized, shifted).
                - xlag: 1D lag axis for x (same units as x).
                - ylag: 1D lag axis for y (same units as y).

    Raises:
        ValueError
    """
    img = np.asarray(image, dtype=float)
    if img.ndim != 2:
        raise ValueError("image must be a 2D array.")

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1D arrays.")
    if img.shape != (y.size, x.size):
        raise ValueError("image shape must match (y.size, x.size).")
    if min(img.shape) < 8:
        raise ValueError("image too small for speckle grain metrics (min dimension < 8).")

    dx = _uniform_step(x, "x")
    dy = _uniform_step(y, "y")

    ac, xlag, ylag = autocorr2d(
        img,
        x=x,
        y=y,
        remove_mean=remove_mean,
        standardize=standardize,
        normalize="peak",
    )

    ac = np.asarray(ac)
    if ac.ndim != 2:
        raise ValueError("autocorr2d returned a non-2D array (unexpected).")

    iy, ix = np.unravel_index(int(np.argmax(ac)), ac.shape)
    y_cut = ac[:, ix]
    x_cut = ac[iy, :]

    wy_idx, _wy_hit_edge = width_at_fraction(y_cut, fraction=fraction, center_index=iy)
    wx_idx, _wx_hit_edge = width_at_fraction(x_cut, fraction=fraction, center_index=ix)

    ly = float(wy_idx) * dy
    lx = float(wx_idx) * dx

    if radial_method == "binned":
        rad, r = radial_mean_binned(ac, x, y)
    elif radial_method == "interpolated":
        rad, r = radial_mean_interpolated(ac, x, y)
    else:
        raise ValueError("radial_method must be 'binned' or 'interpolated'.")

    rad = np.asarray(rad, dtype=float)
    r = np.asarray(r, dtype=float)
    if rad.size < 2 or r.size < 2:
        raise ValueError("Radial profile is too short to estimate leq.")

    dr = float(r[1] - r[0])
    if dr <= 0:
        raise ValueError("Invalid radial sampling (non-positive dr).")

    dist_bins, _leq_hit_edge = distance_at_fraction_from_peak(rad, fraction=fraction, peak_index=0)
    leq = float(dist_bins) * dr

    r_aniso = float(lx / ly) if ly != 0 else float("inf")

    metrics = {
        "lx": float(lx),
        "ly": float(ly),
        "leq": float(leq),
        "r": float(r_aniso),
    }

    if verbose:
        print(
            ">> Speckle grain metrics\n"
            f"   - lx (H):   {metrics['lx']:.6g}\n"
            f"   - ly (V):   {metrics['ly']:.6g}\n"
            f"   - leq:      {metrics['leq']:.6g}\n"
            f"   - r=lx/ly:  {metrics['r']:.6g}"
        )

    if return_autocorr:
        ac_dict = {
            "ac": np.asarray(ac),
            "xlag": np.asarray(xlag, dtype=float),
            "ylag": np.asarray(ylag, dtype=float),
        }
        return metrics, ac_dict

    return metrics
