# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2026 ESRF - the European Synchrotron

from __future__ import annotations

import numpy as np


def odd_size(n: float | int, *, min_size: int = 3) -> int:
    """
    Return the smallest odd integer >= ceil(n), constrained by a minimum size.

    Parameters
    ----------
    n : float | int
        Desired size (may be non-integer).
    min_size : int, optional
        Minimum allowed size (default: 3). Must be >= 1.

    Returns
    -------
    int
        Odd integer >= max(ceil(n), min_size).

    Raises
    ------
    ValueError
        If n is not finite or min_size < 1.
    """
    if not np.isfinite(n):
        raise ValueError("n must be finite.")
    if min_size < 1:
        raise ValueError("min_size must be >= 1.")

    size = int(np.ceil(n))
    size = max(size, min_size)

    if size % 2 == 0:
        size += 1

    return size


def roi_slices(
    image_shape: tuple[int, int],
    size_yx: tuple[int, int],
    *,
    center_yx: tuple[int, int] | None = None,
    clip: bool = False,
) -> tuple[slice, slice]:
    """
    Compute ROI slices from an image shape, odd ROI size, and optional center.

    Parameters
    ----------
    image_shape : tuple[int, int]
        Image shape (H, W).
    size_yx : tuple[int, int]
        ROI size (height, width). Must be positive odd integers.
    center_yx : tuple[int, int] | None, optional
        Center position (y, x). If None, uses image center (H//2, W//2).
    clip : bool, optional
        If False (default), raises ValueError if ROI exceeds bounds.
        If True, ROI is clipped to image bounds (ROI size may shrink).

    Returns
    -------
    tuple[slice, slice]
        (slice_y, slice_x)

    Raises
    ------
    ValueError
        If ROI size is invalid or exceeds bounds when clip=False.
    """
    H, W = image_shape
    size_y, size_x = size_yx

    if size_y <= 0 or size_x <= 0:
        raise ValueError("ROI sizes must be positive.")
    if size_y % 2 == 0 or size_x % 2 == 0:
        raise ValueError("ROI sizes must be odd for symmetry.")

    if center_yx is None:
        cy, cx = (H // 2, W // 2)
    else:
        cy, cx = center_yx

    hy = size_y // 2
    hx = size_x // 2

    y0 = int(cy) - hy
    y1 = int(cy) + hy + 1
    x0 = int(cx) - hx
    x1 = int(cx) + hx + 1

    if not clip:
        if y0 < 0 or y1 > H or x0 < 0 or x1 > W:
            raise ValueError("ROI exceeds image bounds.")
    else:
        y0 = max(0, y0)
        y1 = min(H, y1)
        x0 = max(0, x0)
        x1 = min(W, x1)

    return slice(y0, y1), slice(x0, x1)


def roi_grid_3x3(
    image_shape: tuple[int, int],
    roi_size_yx: tuple[int, int],
    step_yx: tuple[int, int],
    *,
    center_yx: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a 3x3 grid of ROIs centered around a given position.

    Parameters
    ----------
    image_shape : tuple[int, int]
        Image shape (H, W).
    roi_size_yx : tuple[int, int]
        ROI size (height, width). Must be odd.
    step_yx : tuple[int, int]
        Center-to-center offset between adjacent ROIs (delta_y, delta_x).
    center_yx : tuple[int, int] | None, optional
        Grid center (y, x). If None, uses image center (H//2, W//2).

    Returns
    -------
    slices : np.ndarray
        Array of shape (3, 3) containing (slice_y, slice_x) objects.
        Order is row-major: (NW, N, NE; W, C, E; SW, S, SE).
    labels : np.ndarray
        Array of shape (3, 3) with ROI labels:
        [["NW", "N", "NE"],
         ["W",  "C", "E"],
         ["SW", "S", "SE"]]

    Raises
    ------
    ValueError
        If any ROI in the grid exceeds image bounds.
    """
    H, W = image_shape
    if center_yx is None:
        center_yx = (H // 2, W // 2)

    step_y, step_x = step_yx
    offsets_y = (-step_y, 0, step_y)
    offsets_x = (-step_x, 0, step_x)

    slices = np.empty((3, 3), dtype=object)

    for i, dy in enumerate(offsets_y):
        for j, dx in enumerate(offsets_x):
            cy = center_yx[0] + dy
            cx = center_yx[1] + dx
            slices[i, j] = roi_slices(
                image_shape,
                roi_size_yx,
                center_yx=(int(cy), int(cx)),
                clip=False,
            )

    labels = np.array(
        [["NW", "N", "NE"], ["W", "C", "E"], ["SW", "S", "SE"]],
        dtype=object,
    )

    return slices, labels


def embed_roi(
    roi: np.ndarray,
    *,
    out_shape: tuple[int, int],
    slices_yx: tuple[slice, slice],
    fill_value: float = 0.0,
    dtype: np.dtype | None = None,
) -> np.ndarray:
    """
    Embed a 2D ROI into a full-size array at the specified location.

    Parameters
    ----------
    roi : np.ndarray
        2D array to embed.
    out_shape : tuple[int, int]
        Output shape (H, W).
    slices_yx : tuple[slice, slice]
        Location (slice_y, slice_x) where ROI will be placed.
    fill_value : float, optional
        Fill value for the background (default: 0.0).
    dtype : np.dtype | None, optional
        Output dtype. If None, uses roi.dtype.

    Returns
    -------
    np.ndarray
        Full-size array with ROI embedded.

    Raises
    ------
    ValueError
        If ROI shape does not match slice extents.
    """
    H, W = out_shape
    sy, sx = slices_yx

    if dtype is None:
        dtype = roi.dtype

    out = np.full((H, W), fill_value, dtype=dtype)

    expected_shape = (sy.stop - sy.start, sx.stop - sx.start)
    if roi.shape != expected_shape:
        raise ValueError("ROI shape does not match target slice dimensions.")

    out[sy, sx] = roi
    return out