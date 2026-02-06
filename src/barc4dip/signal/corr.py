# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2026

"""
FFT-based correlation helpers.

Conventions:
- 2D arrays use NumPy shape (ny, nx) and axes (y, x).
- Correlation is computed via FFT and is therefore circular (wrap-around).
- Returned correlations are shifted (zero-lag at the center) via fftshift().
- Lag axes (xlag, ylag) are returned in the same units as x/y.

Normalization:
- normalize="none": raw circular correlation (shifted).
- normalize="peak": divides by the maximum absolute correlation value (shifted), so peak is 1.

Notes:
- remove_mean=True is a good default for texture/speckle work (avoids DC pedestal).
- standardize=True computes correlation on (signal - mean)/std.
"""

from __future__ import annotations

from typing import Literal

import numpy as np


def _uniform_step(axis: np.ndarray, name: str) -> float:
    a = np.asarray(axis, dtype=float)
    if a.ndim != 1 or a.size < 2:
        raise ValueError(f"{name} must be a 1D array with at least 2 samples.")

    d = np.diff(a)
    if not np.all(np.isfinite(d)):
        raise ValueError(f"{name} contains non-finite values.")

    # Strict monotonicity (either increasing or decreasing).
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


def _lag_axis(axis: np.ndarray, name: str) -> np.ndarray:
    step = _uniform_step(axis, name)
    n = int(np.asarray(axis).size)
    return (np.arange(n, dtype=float) - (n // 2)) * step


def _as_real_if_close(z: np.ndarray) -> np.ndarray:
    return np.real_if_close(z, tol=1000)


def xcorr1d(
    a: np.ndarray,
    b: np.ndarray,
    *,
    x: np.ndarray,
    remove_mean: bool = True,
    standardize: bool = False,
    normalize: Literal["none", "peak"] = "peak",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Circular cross-correlation of two 1D signals using FFT.

    Parameters:
        a (np.ndarray):
            1D signal A (length n).
        b (np.ndarray):
            1D signal B (length n).
        x (np.ndarray):
            1D coordinate axis (length n), uniformly sampled.
        remove_mean (bool):
            If True, subtracts mean from each signal before correlation.
        standardize (bool):
            If True, divides each (optionally de-meaned) signal by its standard deviation.
        normalize (str):
            "none" or "peak".

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - corr (np.ndarray): shifted circular cross-correlation (length n).
            - xlag (np.ndarray): lag axis (length n), same units as x.

    Raises:
        ValueError
    """
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    if aa.ndim != 1 or bb.ndim != 1:
        raise ValueError("a and b must be 1D arrays.")
    if aa.size != bb.size:
        raise ValueError("a and b must have the same length.")
    if aa.size != np.asarray(x).size:
        raise ValueError("a/b length must match x.size.")

    xlag = _lag_axis(x, "x")

    if remove_mean:
        aa = aa - float(np.mean(aa))
        bb = bb - float(np.mean(bb))

    if standardize:
        sa = float(np.std(aa))
        sb = float(np.std(bb))
        if sa > 0:
            aa = aa / sa
        if sb > 0:
            bb = bb / sb

    Fa = np.fft.fft(aa)
    Fb = np.fft.fft(bb)

    corr = np.fft.ifft(Fa * np.conjugate(Fb))
    corr = np.fft.fftshift(corr)
    corr = _as_real_if_close(corr)

    if normalize == "none":
        return corr, xlag

    if normalize == "peak":
        m = float(np.max(np.abs(corr)))
        if m > 0:
            corr = corr / m
        return corr, xlag

    raise ValueError(f"Invalid normalize='{normalize}'. Use 'none' or 'peak'.")


def autocorr1d(
    a: np.ndarray,
    *,
    x: np.ndarray,
    remove_mean: bool = True,
    standardize: bool = False,
    normalize: Literal["none", "peak"] = "peak",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Circular auto-correlation of a 1D signal using FFT.

    Parameters:
        a (np.ndarray):
            1D signal (length n).
        x (np.ndarray):
            1D coordinate axis (length n), uniformly sampled.
        remove_mean (bool):
            If True, subtracts mean before correlation.
        standardize (bool):
            If True, divides (optionally de-meaned) signal by its standard deviation.
        normalize (str):
            "none" or "peak".

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - corr (np.ndarray): shifted circular auto-correlation (length n).
            - xlag (np.ndarray): lag axis (length n), same units as x.

    Raises:
        ValueError
    """
    return xcorr1d(
        a,
        a,
        x=x,
        remove_mean=remove_mean,
        standardize=standardize,
        normalize=normalize,
    )


def xcorr2d(
    a: np.ndarray,
    b: np.ndarray,
    *,
    x: np.ndarray,
    y: np.ndarray,
    remove_mean: bool = True,
    standardize: bool = False,
    normalize: Literal["none", "peak"] = "peak",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Circular cross-correlation of two 2D signals using FFT.

    Parameters:
        a (np.ndarray):
            2D signal A with shape (ny, nx).
        b (np.ndarray):
            2D signal B with shape (ny, nx).
        x (np.ndarray):
            1D x-axis coordinates (length nx), uniformly sampled.
        y (np.ndarray):
            1D y-axis coordinates (length ny), uniformly sampled.
        remove_mean (bool):
            If True, subtracts mean from each signal before correlation.
        standardize (bool):
            If True, divides each (optionally de-meaned) signal by its standard deviation.
        normalize (str):
            "none" or "peak".

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            - corr (np.ndarray): shifted circular cross-correlation with shape (ny, nx).
            - xlag (np.ndarray): lag axis (length nx), same units as x.
            - ylag (np.ndarray): lag axis (length ny), same units as y.

    Raises:
        ValueError
    """
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    if aa.ndim != 2 or bb.ndim != 2:
        raise ValueError("a and b must be 2D arrays.")
    if aa.shape != bb.shape:
        raise ValueError("a and b must have the same shape.")

    ny, nx = aa.shape
    if nx != np.asarray(x).size or ny != np.asarray(y).size:
        raise ValueError("a/b shape must match (y.size, x.size).")

    xlag = _lag_axis(x, "x")
    ylag = _lag_axis(y, "y")

    if remove_mean:
        aa = aa - float(np.mean(aa))
        bb = bb - float(np.mean(bb))

    if standardize:
        sa = float(np.std(aa))
        sb = float(np.std(bb))
        if sa > 0:
            aa = aa / sa
        if sb > 0:
            bb = bb / sb

    Fa = np.fft.fft2(aa)
    Fb = np.fft.fft2(bb)

    corr = np.fft.ifft2(Fa * np.conjugate(Fb))
    corr = np.fft.fftshift(corr)
    corr = _as_real_if_close(corr)

    if normalize == "none":
        return corr, xlag, ylag

    if normalize == "peak":
        m = float(np.max(np.abs(corr)))
        if m > 0:
            corr = corr / m
        return corr, xlag, ylag

    raise ValueError(f"Invalid normalize='{normalize}'. Use 'none' or 'peak'.")


def autocorr2d(
    a: np.ndarray,
    *,
    x: np.ndarray,
    y: np.ndarray,
    remove_mean: bool = True,
    standardize: bool = False,
    normalize: Literal["none", "peak"] = "peak",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Circular auto-correlation of a 2D signal using FFT.

    Parameters:
        a (np.ndarray):
            2D signal with shape (ny, nx).
        x (np.ndarray):
            1D x-axis coordinates (length nx), uniformly sampled.
        y (np.ndarray):
            1D y-axis coordinates (length ny), uniformly sampled.
        remove_mean (bool):
            If True, subtracts mean before correlation.
        standardize (bool):
            If True, divides (optionally de-meaned) signal by its standard deviation.
        normalize (str):
            "none" or "peak".

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            - corr (np.ndarray): shifted circular auto-correlation with shape (ny, nx).
            - xlag (np.ndarray): lag axis (length nx), same units as x.
            - ylag (np.ndarray): lag axis (length ny), same units as y.

    Raises:
        ValueError
    """
    return xcorr2d(
        a,
        a,
        x=x,
        y=y,
        remove_mean=remove_mean,
        standardize=standardize,
        normalize=normalize,
    )
