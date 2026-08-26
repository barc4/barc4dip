# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2026 ESRF - the European Synchrotron

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    binary_erosion,
    binary_opening,
)

from .roi import embed_roi


def threshold_mask(image: np.ndarray, threshold: float) -> np.ndarray:
    """
    Create a validity mask by applying a lower intensity threshold.

    Pixels with values greater than or equal to ``threshold`` are valid
    (``True``). Pixels below the threshold, as well as NaN values, are invalid
    (``False``).

    Parameters
    ----------
    image : np.ndarray
        Input image or image stack.
    threshold : float
        Minimum valid intensity value.

    Returns
    -------
    np.ndarray
        Boolean validity mask with the same shape as ``image``.
    """
    return np.asarray(image) >= threshold


def refine_mask(
    mask: np.ndarray,
    operations: str | Sequence[str],
    *,
    radius: int = 1,
) -> np.ndarray:
    """
    Refine a 2D validity mask with binary morphological operations.

    The mask follows the convention that ``True`` represents a valid pixel and
    ``False`` represents an invalid pixel. Consequently:

    - ``erosion`` shrinks valid regions and enlarges invalid regions.
    - ``dilation`` expands valid regions.
    - ``opening`` applies erosion then dilation and removes small valid islands.
    - ``closing`` applies dilation then erosion and fills small invalid holes.

    One operation or an ordered sequence of operations may be supplied. Each
    operation is applied once, in the given order. The structuring element is a
    disk with the requested radius in pixels. For ``radius=1``, this is the
    cross-shaped footprint::

        0 1 0
        1 1 1
        0 1 0

    Pixels outside the image are considered invalid (``False``).

    Parameters
    ----------
    mask : np.ndarray
        Two-dimensional boolean validity mask.
    operations : str | Sequence[str]
        One operation or an ordered sequence containing ``"erosion"``,
        ``"dilation"``, ``"opening"``, or ``"closing"``.
    radius : int, optional
        Radius of the disk-shaped structuring element in pixels (default: 1).
        Must be a positive integer.

    Returns
    -------
    np.ndarray
        Refined boolean validity mask with the same shape as ``mask``.

    Raises
    ------
    TypeError
        If ``mask`` is not boolean or ``radius`` is not an integer.
    ValueError
        If ``mask`` is not 2D, ``radius`` is less than 1, the operation sequence
        is empty, or an operation is unknown.
    """
    mask = np.asarray(mask)
    if mask.dtype != np.bool_:
        raise TypeError("mask must have boolean dtype.")
    if mask.ndim != 2:
        raise ValueError("mask must be a 2D array.")

    if isinstance(radius, (bool, np.bool_)) or not isinstance(
        radius, (int, np.integer)
    ):
        raise TypeError("radius must be an integer.")
    if radius < 1:
        raise ValueError("radius must be >= 1.")

    if isinstance(operations, str):
        operation_names = [operations]
    else:
        operation_names = list(operations)
    if not operation_names:
        raise ValueError("operations must not be empty.")

    morphology = {
        "erosion": binary_erosion,
        "dilation": binary_dilation,
        "opening": binary_opening,
        "closing": binary_closing,
    }
    unknown = [name for name in operation_names if name not in morphology]
    if unknown:
        valid = ", ".join(morphology)
        raise ValueError(f"Unknown operation {unknown[0]!r}; expected one of: {valid}.")

    coordinates = np.arange(-radius, radius + 1)
    yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
    structure = xx**2 + yy**2 <= radius**2

    result = mask.copy()
    for name in operation_names:
        result = morphology[name](result, structure=structure, border_value=0)

    return result


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
