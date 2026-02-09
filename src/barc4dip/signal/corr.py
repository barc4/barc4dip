# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2026 ESRF - the European Synchrotron

"""
FFT-based correlation helpers (pixel-first).

Conventions:
- 2D arrays use NumPy shape (ny, nx) and axes (y, x).
- Correlation is computed via FFT and is therefore circular (wrap-around).
- Returned correlations are shifted (zero-lag at the center) via fftshift().

Units:
- If called with no calibration, dx=dy=1 and lag axes are in pixels.
- If dx/dy are provided, lag axes are in those physical units.
- If x/y are provided, dx/dy are inferred from the axes (uniform sampling required).

Calibration rules:
- Provide either:
    - dx (and dy for 2D), OR
    - x (and y for 2D).
- If x is provided, dx must remain default (1.0). Same for y/dy.

Normalization:
- normalize="none": raw circular correlation (shifted).
- normalize="peak": divides by the maximum absolute value (shifted), so peak is 1.

Notes:
- remove_mean=True is a good default for texture/speckle work (avoids DC pedestal).
- standardize=True computes correlation on (signal - mean)/std.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from .common import _lag_axis_from_step, _resolve_step_1d, _resolve_steps_2d


def _as_real_if_close(z: np.ndarray) -> np.ndarray:
    return np.real_if_close(z, tol=1000)


def xcorr1d(
    a: np.ndarray,
    b: np.ndarray,
    *,
    x: np.ndarray | None = None,
    dx: float = 1.0,
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
        x (np.ndarray | None):
            Optional 1D coordinate axis (length n), uniformly sampled.
        dx (float):
            Sample spacing (default: 1.0). Used only if x is None.
        remove_mean (bool):
            If True, subtracts mean from each signal before correlation (default: True).
        standardize (bool):
            If True, divides each (optionally de-meaned) signal by its standard deviation (default: False).
        normalize (str):
            "none" or "peak" (default: "peak").

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - corr (np.ndarray): shifted circular cross-correlation (length n).
            - xlag (np.ndarray): lag axis (length n), in pixels or physical units.

    Raises:
        ValueError
    """
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    if aa.ndim != 1 or bb.ndim != 1:
        raise ValueError("a and b must be 1D arrays.")
    if aa.size != bb.size:
        raise ValueError("a and b must have the same length.")

    n = int(aa.size)
    step = _resolve_step_1d(n=n, x=x, dx=dx, name="x")
    xlag = _lag_axis_from_step(n, step)

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
    x: np.ndarray | None = None,
    dx: float = 1.0,
    remove_mean: bool = True,
    standardize: bool = False,
    normalize: Literal["none", "peak"] = "peak",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Circular auto-correlation of a 1D signal using FFT.

    Parameters:
        a (np.ndarray):
            1D signal (length n).
        x (np.ndarray | None):
            Optional 1D coordinate axis (length n), uniformly sampled.
        dx (float):
            Sample spacing (default: 1.0). Used only if x is None.
        remove_mean (bool):
            If True, subtracts mean before correlation (default: True).
        standardize (bool):
            If True, divides (optionally de-meaned) signal by its standard deviation (default: False).
        normalize (str):
            "none" or "peak" (default: "peak").

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - corr (np.ndarray): shifted circular auto-correlation (length n).
            - xlag (np.ndarray): lag axis (length n), in pixels or physical units.

    Raises:
        ValueError
    """
    return xcorr1d(
        a,
        a,
        x=x,
        dx=dx,
        remove_mean=remove_mean,
        standardize=standardize,
        normalize=normalize,
    )


def xcorr2d(
    a: np.ndarray,
    b: np.ndarray,
    *,
    x: np.ndarray | None = None,
    y: np.ndarray | None = None,
    dx: float = 1.0,
    dy: float = 1.0,
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
        x (np.ndarray | None):
            Optional 1D x-axis (length nx), uniformly sampled.
        y (np.ndarray | None):
            Optional 1D y-axis (length ny), uniformly sampled.
        dx (float):
            Pixel size in x (default: 1.0). Used only if x/y are None.
        dy (float):
            Pixel size in y (default: 1.0). Used only if x/y are None.
        remove_mean (bool):
            If True, subtracts mean from each signal before correlation (default: True).
        standardize (bool):
            If True, divides each (optionally de-meaned) signal by its standard deviation (default: False).
        normalize (str):
            "none" or "peak" (default: "peak").

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            - corr (np.ndarray): shifted circular cross-correlation with shape (ny, nx).
            - xlag (np.ndarray): lag axis (length nx), in pixels or physical units.
            - ylag (np.ndarray): lag axis (length ny), in pixels or physical units.

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
    step_x, step_y = _resolve_steps_2d(shape=(ny, nx), x=x, y=y, dx=dx, dy=dy)
    xlag = _lag_axis_from_step(nx, step_x)
    ylag = _lag_axis_from_step(ny, step_y)

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
    x: np.ndarray | None = None,
    y: np.ndarray | None = None,
    dx: float = 1.0,
    dy: float = 1.0,
    remove_mean: bool = True,
    standardize: bool = False,
    normalize: Literal["none", "peak"] = "peak",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Circular auto-correlation of a 2D signal using FFT.

    Parameters:
        a (np.ndarray):
            2D signal with shape (ny, nx).
        x (np.ndarray | None):
            Optional 1D x-axis (length nx), uniformly sampled.
        y (np.ndarray | None):
            Optional 1D y-axis (length ny), uniformly sampled.
        dx (float):
            Pixel size in x (default: 1.0). Used only if x/y are None.
        dy (float):
            Pixel size in y (default: 1.0). Used only if x/y are None.
        remove_mean (bool):
            If True, subtracts mean before correlation (default: True).
        standardize (bool):
            If True, divides (optionally de-meaned) signal by its standard deviation (default: False).
        normalize (str):
            "none" or "peak" (default: "peak").

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            - corr (np.ndarray): shifted circular auto-correlation with shape (ny, nx).
            - xlag (np.ndarray): lag axis (length nx), in pixels or physical units.
            - ylag (np.ndarray): lag axis (length ny), in pixels or physical units.

    Raises:
        ValueError
    """
    return xcorr2d(
        a,
        a,
        x=x,
        y=y,
        dx=dx,
        dy=dy,
        remove_mean=remove_mean,
        standardize=standardize,
        normalize=normalize,
    )
