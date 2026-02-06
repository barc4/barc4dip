# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2026 ESRF - the European Synchrotron

"""
FFT and power spectral density helpers.

Conventions:
- 2D images use NumPy shape (ny, nx) and axes (y, x).
- 1D/2D FFT outputs are always shifted (DC centered) via fftshift().
- Frequency axes (fx, fy) are returned shifted to match the shifted FFT indexing.
- Axes x and y are assumed to be 1D coordinate vectors (pixel centers or physical units),
  typically centered at 0. They must be uniformly sampled.

Notes:
- Inputs are expected to be float32/float64 in typical workflows, but any numeric dtype
  is accepted and internally promoted as needed.
"""

from __future__ import annotations

import numpy as np


def freq_axis1d(*, x: np.ndarray) -> np.ndarray:
    """
    Build the shifted 1D frequency axis corresponding to a sampled coordinate axis.

    Parameters:
        x (np.ndarray):
            1D coordinate axis (length n), uniformly sampled.

    Returns:
        np.ndarray:
            Shifted frequency axis fx (length n), in cycles per unit of x.

    Raises:
        ValueError
    """
    dx = _uniform_step(x, "x")
    n = int(np.asarray(x).size)
    fx = np.fft.fftshift(np.fft.fftfreq(n, d=dx))
    return fx


def freq_axes2d(*, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the shifted 2D frequency axes corresponding to sampled coordinate axes.

    Parameters:
        x (np.ndarray):
            1D x-axis coordinates (length nx), uniformly sampled.
        y (np.ndarray):
            1D y-axis coordinates (length ny), uniformly sampled.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - fx (np.ndarray): shifted frequency axis for x (length nx).
            - fy (np.ndarray): shifted frequency axis for y (length ny).

    Raises:
        ValueError
    """
    dx = _uniform_step(x, "x")
    dy = _uniform_step(y, "y")
    nx = int(np.asarray(x).size)
    ny = int(np.asarray(y).size)
    fx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    fy = np.fft.fftshift(np.fft.fftfreq(ny, d=dy))
    return fx, fy


def fft1d(signal: np.ndarray, *, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the shifted 1D FFT of a signal and its shifted frequency axis.

    Parameters:
        signal (np.ndarray):
            1D input array (length n).
        x (np.ndarray):
            1D coordinate axis (length n), uniformly sampled.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - F (np.ndarray): shifted complex FFT (length n).
            - fx (np.ndarray): shifted frequency axis (length n).

    Raises:
        ValueError
    """
    s = np.asarray(signal)
    if s.ndim != 1:
        raise ValueError("signal must be a 1D array.")
    if s.size != np.asarray(x).size:
        raise ValueError("signal length must match x.size.")

    fx = freq_axis1d(x=x)
    F = np.fft.fftshift(np.fft.fft(s))
    return F, fx


def ifft1d(F: np.ndarray, *, x: np.ndarray) -> np.ndarray:
    """
    Compute the 1D inverse FFT from a shifted spectrum.

    Parameters:
        F (np.ndarray):
            Shifted complex spectrum (length n), as returned by fft1d().
        x (np.ndarray):
            1D coordinate axis (length n), uniformly sampled.

    Returns:
        np.ndarray:
            Reconstructed complex signal (length n).

    Raises:
        ValueError
    """
    F = np.asarray(F)
    if F.ndim != 1:
        raise ValueError("F must be a 1D array.")
    if F.size != np.asarray(x).size:
        raise ValueError("F length must match x.size.")

    # Validate axis (uniformity), even though we don't use dx directly here.
    _ = _uniform_step(x, "x")

    s = np.fft.ifft(np.fft.ifftshift(F))
    return s


def psd1d(signal: np.ndarray, *, x: np.ndarray, scale: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the shifted 1D power spectral density (PSD) of a signal.

    Parameters:
        signal (np.ndarray):
            1D input array (length n).
        x (np.ndarray):
            1D coordinate axis (length n), uniformly sampled.
        scale (bool):
            If True, applies scaling: PSD *= dx / n.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - P (np.ndarray): shifted PSD (length n).
            - fx (np.ndarray): shifted frequency axis (length n).

    Raises:
        ValueError
    """
    dx = _uniform_step(x, "x")
    F, fx = fft1d(signal, x=x)
    P = np.abs(F) ** 2

    if scale:
        n = float(P.size)
        P = P * (dx / n)

    return P, fx


def fft2d(image: np.ndarray, *, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the shifted 2D FFT of an image and shifted frequency axes.

    Parameters:
        image (np.ndarray):
            2D input array with shape (ny, nx).
        x (np.ndarray):
            1D x-axis coordinates (length nx), uniformly sampled.
        y (np.ndarray):
            1D y-axis coordinates (length ny), uniformly sampled.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            - F (np.ndarray): shifted complex FFT with shape (ny, nx).
            - fx (np.ndarray): shifted frequency axis for x (length nx).
            - fy (np.ndarray): shifted frequency axis for y (length ny).

    Raises:
        ValueError
    """
    img = np.asarray(image)
    if img.ndim != 2:
        raise ValueError("image must be a 2D array.")
    ny, nx = img.shape
    if nx != np.asarray(x).size or ny != np.asarray(y).size:
        raise ValueError("image shape must match (y.size, x.size).")

    fx, fy = freq_axes2d(x=x, y=y)
    F = np.fft.fftshift(np.fft.fft2d(img))
    return F, fx, fy


def ifft2d(F: np.ndarray, *, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute the 2D inverse FFT from a shifted spectrum.

    Parameters:
        F (np.ndarray):
            Shifted complex spectrum with shape (ny, nx), as returned by fft2d().
        x (np.ndarray):
            1D x-axis coordinates (length nx), uniformly sampled.
        y (np.ndarray):
            1D y-axis coordinates (length ny), uniformly sampled.

    Returns:
        np.ndarray:
            Reconstructed complex image with shape (ny, nx).

    Raises:
        ValueError
    """
    F = np.asarray(F)
    if F.ndim != 2:
        raise ValueError("F must be a 2D array.")
    ny, nx = F.shape
    if nx != np.asarray(x).size or ny != np.asarray(y).size:
        raise ValueError("F shape must match (y.size, x.size).")

    _ = _uniform_step(x, "x")
    _ = _uniform_step(y, "y")

    img = np.fft.ifft2d(np.fft.ifftshift(F))
    return img


def psd2d(
    image: np.ndarray,
    *,
    x: np.ndarray,
    y: np.ndarray,
    scale: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the shifted 2D power spectral density (PSD) of an image.

    Parameters:
        image (np.ndarray):
            2D input array with shape (ny, nx).
        x (np.ndarray):
            1D x-axis coordinates (length nx), uniformly sampled.
        y (np.ndarray):
            1D y-axis coordinates (length ny), uniformly sampled.
        scale (bool):
            If True, applies scaling: PSD *= (dx * dy) / (nx * ny).

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            - P (np.ndarray): shifted PSD with shape (ny, nx).
            - fx (np.ndarray): shifted frequency axis for x (length nx).
            - fy (np.ndarray): shifted frequency axis for y (length ny).

    Raises:
        ValueError
    """
    dx = _uniform_step(x, "x")
    dy = _uniform_step(y, "y")

    F, fx, fy = fft2d(image, x=x, y=y)
    P = np.abs(F) ** 2

    if scale:
        ny, nx = P.shape
        P = P * ((dx * dy) / (float(nx) * float(ny)))

    return P, fx, fy


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

    rel = np.max(np.abs(d_abs - step)) / step
    if rel > 1e-6:
        raise ValueError(
            f"{name} appears non-uniform (max relative deviation {rel:.2e}). "
            "Provide uniformly sampled axes."
        )
    return step