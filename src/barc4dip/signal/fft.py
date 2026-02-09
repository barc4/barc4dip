# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2026 ESRF - the European Synchrotron

"""
FFT and power spectral density helpers.

Conventions:
- 2D arrays use NumPy shape (ny, nx) and axes (y, x).
- FFT outputs are always shifted (DC centered) via fftshift().
- Frequency axes (fx, fy) are returned shifted to match the shifted FFT indexing.

Units:
- If called with no calibration, dx=dy=1 and frequencies are in cycles/pixel.
- If dx/dy are provided, frequencies are in cycles/unit.
- If x/y are provided, dx/dy are inferred from the axes (uniform sampling required).

Calibration rules:
- Provide either:
    - dx (and dy for 2D), OR
    - x (and y for 2D).
- If x is provided, dx must remain default (1.0). Same for y/dy.
"""

from __future__ import annotations

import numpy as np

from .common import _resolve_step_1d, _resolve_steps_2d


def freq_axis1d(*, n: int, x: np.ndarray | None = None, dx: float = 1.0) -> np.ndarray:
    """
    Build the shifted 1D frequency axis.

    Parameters:
        n (int):
            Number of samples.
        x (np.ndarray | None):
            Optional 1D coordinate axis (length n), uniformly sampled.
        dx (float):
            Sample spacing (default: 1.0). Used only if x is None.

    Returns:
        np.ndarray:
            Shifted frequency axis fx (length n), in cycles per unit.

    Raises:
        ValueError
    """
    if n < 1:
        raise ValueError("n must be >= 1.")

    step = _resolve_step_1d(n=n, x=x, dx=dx, name="x")
    fx = np.fft.fftshift(np.fft.fftfreq(int(n), d=step))
    return fx


def freq_axes2d(
    *,
    shape: tuple[int, int],
    x: np.ndarray | None = None,
    y: np.ndarray | None = None,
    dx: float = 1.0,
    dy: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the shifted 2D frequency axes.

    Parameters:
        shape (tuple[int, int]):
            Image shape (ny, nx).
        x (np.ndarray | None):
            Optional 1D x-axis (length nx), uniformly sampled.
        y (np.ndarray | None):
            Optional 1D y-axis (length ny), uniformly sampled.
        dx (float):
            Pixel size in x (default: 1.0). Used only if x/y are None.
        dy (float):
            Pixel size in y (default: 1.0). Used only if x/y are None.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - fx (np.ndarray): shifted frequency axis for x (length nx).
            - fy (np.ndarray): shifted frequency axis for y (length ny).

    Raises:
        ValueError
    """
    ny, nx = shape
    if ny < 1 or nx < 1:
        raise ValueError("shape must contain positive integers.")

    step_x, step_y = _resolve_steps_2d(shape=shape, x=x, y=y, dx=dx, dy=dy)
    fx = np.fft.fftshift(np.fft.fftfreq(int(nx), d=step_x))
    fy = np.fft.fftshift(np.fft.fftfreq(int(ny), d=step_y))
    return fx, fy


def fft1d(
    signal: np.ndarray,
    *,
    x: np.ndarray | None = None,
    dx: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the shifted 1D FFT of a signal and its shifted frequency axis.

    Parameters:
        signal (np.ndarray):
            1D input array (length n).
        x (np.ndarray | None):
            Optional 1D coordinate axis (length n), uniformly sampled.
        dx (float):
            Sample spacing (default: 1.0). Used only if x is None.

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
    n = int(s.size)

    fx = freq_axis1d(n=n, x=x, dx=dx)
    F = np.fft.fftshift(np.fft.fft(s))
    return F, fx


def ifft1d(F: np.ndarray) -> np.ndarray:
    """
    Compute the 1D inverse FFT from a shifted spectrum.

    Parameters:
        F (np.ndarray):
            Shifted complex spectrum (length n), as returned by fft1d().

    Returns:
        np.ndarray:
            Reconstructed complex signal (length n).

    Raises:
        ValueError
    """
    F = np.asarray(F)
    if F.ndim != 1:
        raise ValueError("F must be a 1D array.")
    return np.fft.ifft(np.fft.ifftshift(F))


def psd1d(
    signal: np.ndarray,
    *,
    x: np.ndarray | None = None,
    dx: float = 1.0,
    scale: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the shifted 1D power spectral density (PSD) of a signal.

    Parameters:
        signal (np.ndarray):
            1D input array (length n).
        x (np.ndarray | None):
            Optional 1D coordinate axis (length n), uniformly sampled.
        dx (float):
            Sample spacing (default: 1.0). Used only if x is None.
        scale (bool):
            If True, applies scaling: PSD *= dx / n.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - P (np.ndarray): shifted PSD (length n).
            - fx (np.ndarray): shifted frequency axis (length n).

    Raises:
        ValueError
    """
    s = np.asarray(signal)
    if s.ndim != 1:
        raise ValueError("signal must be a 1D array.")
    n = int(s.size)

    step = _resolve_step_1d(n=n, x=x, dx=dx, name="x")
    F, fx = fft1d(s, x=x, dx=dx)
    P = np.abs(F) ** 2

    if scale:
        P = P * (step / float(n))

    return P, fx


def fft2d(
    image: np.ndarray,
    *,
    x: np.ndarray | None = None,
    y: np.ndarray | None = None,
    dx: float = 1.0,
    dy: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the shifted 2D FFT of an image and shifted frequency axes.

    Parameters:
        image (np.ndarray):
            2D input array with shape (ny, nx).
        x (np.ndarray | None):
            Optional 1D x-axis (length nx), uniformly sampled.
        y (np.ndarray | None):
            Optional 1D y-axis (length ny), uniformly sampled.
        dx (float):
            Pixel size in x (default: 1.0). Used only if x/y are None.
        dy (float):
            Pixel size in y (default: 1.0). Used only if x/y are None.

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
    fx, fy = freq_axes2d(shape=(ny, nx), x=x, y=y, dx=dx, dy=dy)
    F = np.fft.fftshift(np.fft.fft2(img))
    return F, fx, fy


def ifft2d(F: np.ndarray) -> np.ndarray:
    """
    Compute the 2D inverse FFT from a shifted spectrum.

    Parameters:
        F (np.ndarray):
            Shifted complex spectrum with shape (ny, nx), as returned by fft2d().

    Returns:
        np.ndarray:
            Reconstructed complex image with shape (ny, nx).

    Raises:
        ValueError
    """
    F = np.asarray(F)
    if F.ndim != 2:
        raise ValueError("F must be a 2D array.")
    return np.fft.ifft2(np.fft.ifftshift(F))


def psd2d(
    image: np.ndarray,
    *,
    x: np.ndarray | None = None,
    y: np.ndarray | None = None,
    dx: float = 1.0,
    dy: float = 1.0,
    scale: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the shifted 2D power spectral density (PSD) of an image.

    Parameters:
        image (np.ndarray):
            2D input array with shape (ny, nx).
        x (np.ndarray | None):
            Optional 1D x-axis (length nx), uniformly sampled.
        y (np.ndarray | None):
            Optional 1D y-axis (length ny), uniformly sampled.
        dx (float):
            Pixel size in x (default: 1.0). Used only if x/y are None.
        dy (float):
            Pixel size in y (default: 1.0). Used only if x/y are None.
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
    img = np.asarray(image)
    if img.ndim != 2:
        raise ValueError("image must be a 2D array.")

    ny, nx = img.shape
    step_x, step_y = _resolve_steps_2d(shape=(ny, nx), x=x, y=y, dx=dx, dy=dy)

    F, fx, fy = fft2d(img, x=x, y=y, dx=dx, dy=dy)
    P = np.abs(F) ** 2

    if scale:
        P = P * ((step_x * step_y) / (float(nx) * float(ny)))

    return P, fx, fy
