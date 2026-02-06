# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2026 ESRF - the European Synchrotron

""" 
Read and of edf images - legacy format
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .uti_EdfFile import EdfFile

def read_edf(image_path: str | Sequence[str], *, index: int = 0, 
                   dtype: np.dtype | str = np.float32,) -> np.ndarray:
    """
    Reads one or multiple EDF images from disk.

    Parameters:
        image_path (str | Sequence[str]):
            Path to an EDF file, or a sequence of paths to EDF files.
            If a sequence is provided, images are stacked along axis 0.
        index (int):
            Frame index inside the EDF file (EDF may contain multiple frames).
            Default is 0.
        dtype (np.dtype | str):
            Output dtype to cast the data to (default: np.float32).

    Returns:
        np.ndarray:
            - 2D array if a single image path is provided.
            - 3D array (N, H, W) if a sequence of image paths is provided.

    Raises:
        TypeError:
            If image_path is not a str or a sequence of str.
        ValueError:
            If the sequence is empty, if image shapes are inconsistent,
            or if index is negative.
        FileNotFoundError:
            If any provided path does not exist.
        OSError:
            If reading the EDF file fails.
    """

    if index < 0:
        raise ValueError("index must be >= 0")

    def _read_one(p: str) -> np.ndarray:
        if not isinstance(p, str):
            raise TypeError("All elements of image_path must be strings")

        fp = Path(p)
        if not fp.exists():
            raise FileNotFoundError(f"EDF file not found: '{p}'")

        arr = EdfFile(str(fp)).GetData(index)
        return np.asarray(arr, dtype=dtype)

    if isinstance(image_path, str):
        return _read_one(image_path)

    if isinstance(image_path, Sequence):
        if len(image_path) == 0:
            raise ValueError("image_path sequence is empty")

        frames: list[np.ndarray] = []
        ref_shape: tuple[int, ...] | None = None

        for p in image_path:
            arr = _read_one(p)

            if arr.ndim != 2:
                raise ValueError(f"Expected a 2D EDF image, got shape {arr.shape} for '{p}'")

            if ref_shape is None:
                ref_shape = arr.shape
            elif arr.shape != ref_shape:
                raise ValueError(
                    f"Inconsistent image shapes in stack: expected {ref_shape}, "
                    f"got {arr.shape} for '{p}'"
                )

            frames.append(arr)

        return np.stack(frames, axis=0)

    raise TypeError("image_path must be a str or a sequence of str")

