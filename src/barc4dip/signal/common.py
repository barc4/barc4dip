# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2026 ESRF - the European Synchrotron

"""
common utilities to fft.py and corr.py module
"""

from __future__ import annotations

import numpy as np

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


def _resolve_step_1d(*, n: int, x: np.ndarray | None, dx: float, name: str) -> float:
    if x is not None and dx != 1.0:
        raise ValueError(f"Provide either {name} or d{name}, not both.")

    if x is None:
        if dx <= 0:
            raise ValueError(f"d{name} must be > 0.")
        return float(dx)

    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"{name} must be a 1D array.")
    if x.size != n:
        raise ValueError(f"{name}.size must match the signal length ({n}).")

    return _uniform_step(x, name)


def _resolve_steps_2d(
    *,
    shape: tuple[int, int],
    x: np.ndarray | None,
    y: np.ndarray | None,
    dx: float,
    dy: float,
) -> tuple[float, float]:
    ny, nx = shape

    if (x is None) ^ (y is None):
        raise ValueError("Provide both x and y axes, or neither.")

    if (x is not None and dx != 1.0) or (y is not None and dy != 1.0):
        raise ValueError("Provide either (x, y) or (dx, dy), not both.")

    if x is None and y is None:
        if dx <= 0 or dy <= 0:
            raise ValueError("dx and dy must be > 0.")
        return float(dx), float(dy)

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1D arrays.")
    if x.size != nx or y.size != ny:
        raise ValueError("x/y sizes must match (nx, ny) of the image.")

    return _uniform_step(x, "x"), _uniform_step(y, "y")


def _lag_axis_from_step(n: int, step: float) -> np.ndarray:
    return (np.arange(n, dtype=float) - (n // 2)) * float(step)