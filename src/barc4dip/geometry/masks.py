# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2026 ESRF - the European Synchrotron

from __future__ import annotations

import numpy as np

from .roi import embed_roi


def pad_to_square(
    image: np.ndarray,
    *,
    fill_value: float = 0.0,
    dtype: np.dtype | None = None,
) -> np.ndarray:
    """
    Symmetrically zero-pad a 2D array to a square shape.

    Parameters
    ----------
    image : np.ndarray
        Input 2D array (H, W).
    fill_value : float, optional
        Padding value (default: 0.0).
    dtype : np.dtype | None, optional
        Output dtype. If None, uses image.dtype.

    Returns
    -------
    np.ndarray
        Square array of shape (N, N), with the input centered.

    Raises
    ------
    ValueError
        If input is not 2D.
    """
    if image.ndim != 2:
        raise ValueError("Input must be a 2D array.")

    H, W = image.shape
    N = max(H, W)

    y0 = (N - H) // 2
    x0 = (N - W) // 2

    sy = slice(y0, y0 + H)
    sx = slice(x0, x0 + W)

    return embed_roi(
        image,
        out_shape=(N, N),
        slices_yx=(sy, sx),
        fill_value=fill_value,
        dtype=dtype,
    )