# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2025

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter


def flat_field_correction(
    images: np.ndarray,
    *,
    flats: np.ndarray | None = None,
    darks: np.ndarray | None = None,
    scale: str = "flat_median",
    bad_pixel_removal: bool = False,
    eps: float | None = None,
) -> np.ndarray:
    """
    Apply flat-field (gain) correction to one or multiple images.

    The correction follows:
        (I - D) / (F - D) * scale_factor

    where I is the image, D the dark field, and F the flat field.

    Parameters:
        images (np.ndarray):
            Input image or image stack.
            - 2D array (H, W)
            - 3D array (N, H, W)

        flats (np.ndarray, optional):
            Flat-field image or stack.
            If 3D, the mean is taken along axis 0.

        darks (np.ndarray, optional):
            Dark-field image or stack.
            If 3D, the mean is taken along axis 0.

        scale (str, optional):
            Scaling applied after gain correction.
            One of:
                - "none"
                - "flat_mean"
                - "flat_median"
            Default is "flat_median".

        bad_pixel_removal (bool, optional):
            If True, apply a local 3x3 median filter only on pixels
            where (flat - dark) <= eps, after correction.
            Default is False.

        eps (float, optional):
            Threshold below which (flat - dark) is considered invalid.
            If None, a relative threshold based on the median denominator
            is used.

    Returns:
        np.ndarray:
            Gain-corrected image(s) as float32, with the same shape
            as the input images.

    Raises:
        ValueError:
            If scale is not a valid option.
        ValueError:
            If input shapes are incompatible.
    """
    if scale not in {"none", "flat_mean", "flat_median"}:
        raise ValueError(f"Invalid scale option: {scale}")

    img = images.astype(np.float32, copy=False)

    is_stack = img.ndim == 3
    if img.ndim not in {2, 3}:
        raise ValueError("images must be 2D or 3D")

    def _reduce_stack(arr: np.ndarray | None) -> np.ndarray | None:
        if arr is None:
            return None
        if arr.ndim == 3:
            return arr.astype(np.float32).mean(axis=0)
        if arr.ndim == 2:
            return arr.astype(np.float32)
        raise ValueError("flats/darks must be 2D or 3D")

    flat2d = _reduce_stack(flats)
    dark2d = _reduce_stack(darks)

    if flat2d is None and dark2d is None:
        return img.copy()

    if dark2d is None:
        dark2d = np.zeros_like(flat2d)

    if flat2d is None:
        return img - dark2d if not is_stack else img - dark2d[None, ...]

    den = flat2d - dark2d

    if eps is None:
        med = np.median(den)
        eps = 1e-6 * med if med > 0 else 1e-6

    bad = den <= eps

    num = img - dark2d if not is_stack else img - dark2d[None, ...]
    den_safe = den.copy()
    den_safe[bad] = 1.0

    out = num / den_safe if not is_stack else num / den_safe[None, ...]

    if scale != "none":
        valid = ~bad
        if scale == "flat_mean":
            s = np.mean(den[valid])
        else:
            s = np.median(den[valid])
        out *= s

    if not is_stack:
        out[bad] = 0.0
    else:
        out[:, bad] = 0.0

    if bad_pixel_removal:

        repaired = median_filter(out, size=(1, 3, 3) if is_stack else (3, 3))
        if not is_stack:
            out[bad] = repaired[bad]
        else:
            out[:, bad] = repaired[:, bad]

    return out.astype(np.float32, copy=False)
