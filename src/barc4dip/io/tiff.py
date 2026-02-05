# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2026 ESRF - the European Synchrotron

""" 
Read and write of tiff images
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image
from barc4dip.utils.dtype import to_uint16

def read_tiff_image(image_path: str | Sequence[str]) -> np.ndarray:
    """
    Reads one or multiple TIFF images from disk.

    Parameters:
        image_path (str | Sequence[str]):
            Path to a TIFF file, or a sequence of paths to TIFF files.
            If a sequence is provided, images are stacked along axis 0.

    Returns:
        np.ndarray:
            - 2D array if a single image path is provided.
            - 3D array (N, H, W) if a sequence of image paths is provided.

    Raises:
        TypeError:
            If image_path is not a str or a sequence of str.
        ValueError:
            If the sequence is empty or if image shapes are inconsistent.
    """

    if isinstance(image_path, str):
        with Image.open(image_path) as img:
            return np.array(img)

    if isinstance(image_path, Sequence):
        if len(image_path) == 0:
            raise ValueError("image_path sequence is empty")

        frames: list[np.ndarray] = []
        ref_shape: tuple[int, ...] | None = None

        for path in image_path:
            if not isinstance(path, str):
                raise TypeError("All elements of image_path must be strings")

            with Image.open(path) as img:
                arr = np.array(img)

            if ref_shape is None:
                ref_shape = arr.shape
            elif arr.shape != ref_shape:
                raise ValueError(
                    f"Inconsistent image shapes in stack: "
                    f"expected {ref_shape}, got {arr.shape} for '{path}'"
                )

            frames.append(arr)

        return np.stack(frames, axis=0)

    raise TypeError("image_path must be a str or a sequence of str")
        

def save_tiff_image(data: np.ndarray, output_path: str | Path) -> None:
    """
    Saves a 2D image or a 3D image stack as TIFF(s).

    Parameters:
        data (np.ndarray):
            Image data to save.
            - If 2D: saved as a single TIFF.
            - If 3D: saved as one TIFF per slice along axis 0.
        output_path (str | Path):
            Output file path.
            - For 2D input: written exactly to this path.
            - For 3D input: used as a base name; files are saved as
              "<stem>_0000.tif", "<stem>_0001.tif", ... in the same folder.

    Returns:
        None

    Raises:
        TypeError:
            If data is not a numpy array.
        ValueError:
            If data is not 2D or 3D, or if output_path is invalid.
        OSError:
            If the destination path does not exist or is not writable.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("data must be a numpy.ndarray")
    if data.ndim not in (2, 3):
        raise ValueError(f"data must be 2D or 3D, got ndim={data.ndim}")

    out = Path(output_path)

    if out.name == "":
        raise ValueError("output_path must include a filename")

    if not out.parent.exists():
        raise OSError(f"Invalid path: directory does not exist: {out.parent}")

    if not out.parent.is_dir():
        raise OSError(f"Invalid path: not a directory: {out.parent}")

    suffix = out.suffix.lower()
    if suffix not in {".tif", ".tiff"}:
        suffix = ".tif"

    img_u16 = to_uint16(data)

    if data.ndim == 2:
        try:
            Image.fromarray(img_u16).save(out.with_suffix(suffix))
        except OSError as e:
            raise OSError(f"Failed to write TIFF file: {out}") from e
        return

    base = out.with_suffix("")
    for i in range(data.shape[0]):
        frame_path = base.parent / f"{base.name}_{i:04d}{suffix}"
        try:
            Image.fromarray(img_u16).save(frame_path)
        except OSError as e:
            raise OSError(f"Failed to write TIFF file: {frame_path}") from e