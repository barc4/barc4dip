# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2026 ESRF - the European Synchrotron

"""
Public image I/O dispatchers.

This module provides:
    - read_image(): reads TIFF / EDF / HDF5 based on file extension.
    - write_image(): writes TIFF / HDF5 based on file extension.

EDF writing is intentionally not supported (legacy read-only format).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ..utils import elapsed_time, now
from .edf import read_edf
from .h5 import read_h5, save_h5
from .tiff import read_tiff, save_tiff

_READ_EXTS = {
    "tif": "tiff",
    "tiff": "tiff",
    "edf": "edf",
    "h5": "h5",
    "hdf5": "h5",
}

_WRITE_EXTS = {
    "tif": "tiff",
    "tiff": "tiff",
    "h5": "h5",
    "hdf5": "h5",
    "edf": "edf",  # explicitly blocked
}


def _normalize_extension(ext: str) -> str:
    ext = ext.lower().lstrip(".")
    return ext


def _infer_extension_from_path(path: str) -> str:
    suffix = Path(path).suffix
    if suffix == "":
        raise ValueError(
            "Cannot infer file extension from path (no suffix). "
            "Provide file_extension explicitly."
        )
    return _normalize_extension(suffix)


def _infer_extension_from_paths(paths: Sequence[str]) -> str:
    exts = [_infer_extension_from_path(p) for p in paths]
    first = exts[0]
    if any(e != first for e in exts):
        raise ValueError(f"Mixed file extensions in image_path sequence: {sorted(set(exts))}")
    return first


def read_image(
    image_path: str | Sequence[str],
    *,
    file_extension: str | None = None,
    image_number: int | None = None,
    mean: bool = False,
    verbose: bool = False,
) -> np.ndarray:
    """
    Reads one image or a stack of images from disk.

    Parameters:
        image_path (str | Sequence[str]):
            Path to a file, or a sequence of file paths.
        file_extension (str | None):
            Optional extension override ("tif", "edf", "h5", ...).
        image_number (int | None):
            Frame index to load when reading a single HDF5 file containing a 3D dataset
            (N, H, W). If None (default), the full dataset is returned.
            This option is not supported for TIFF/EDF in this project.
        mean : bool
            If True and the loaded data is a 3D stack (N, H, W), return the mean
            image computed along axis 0. Default is False.
        verbose (bool):
            If True, prints basic information about the loaded data.

    Returns:
        np.ndarray:
            2D or 3D image array.

    Raises:
        TypeError, ValueError
    """
    t0 = now()
    if isinstance(image_path, str):
        ext = _normalize_extension(file_extension) if file_extension else _infer_extension_from_path(image_path)
    elif isinstance(image_path, Sequence):
        if len(image_path) == 0:
            raise ValueError("image_path sequence is empty")
        ext = _normalize_extension(file_extension) if file_extension else _infer_extension_from_paths(image_path)
    else:
        raise TypeError("image_path must be a str or a sequence of str")

    if not isinstance(image_path, str) and image_number is not None:
        raise ValueError("image_number is only supported when image_path is a single file (str)")

    kind = _READ_EXTS.get(ext)
    if kind is None:
        raise ValueError(f"Unsupported read extension: '{ext}'")

    if kind == "tiff":
        if image_number is not None:
            raise ValueError("image_number is only supported for HDF5 stacks (single-file .h5/.hdf5).")
        data = read_tiff(image_path)
    elif kind == "edf":
        if image_number is not None:
            raise ValueError("image_number is only supported for HDF5 stacks (single-file .h5/.hdf5).")
        data = read_edf(image_path)
    elif kind == "h5":
        data = read_h5(image_path, image_number=image_number)
    else:
        raise RuntimeError(f"Unhandled reader kind: {kind}")

    if mean and data.ndim == 3:
        data = data.mean(axis=0)
        if verbose:
            print("Collapsed 3D stack to mean image along axis 0.")

    if verbose:
        if data.ndim == 2:
            n_img = 1
            h, w = data.shape
        else:
            n_img, h, w = data.shape

        mem_gb = data.nbytes / (1024 ** 3)
        print(f"> {n_img} image(s) found ({h} x {w}), {mem_gb:.2f} Gb in memory")
        elapsed_time(t0)

    return data


def write_image(data: np.ndarray, output_path: str | Path, *, 
                file_extension: str | None = None, verbose: bool = False, ) -> None:
    """
    Writes an image or image stack to disk.

    Parameters:
        data (np.ndarray):
            Image data to save (2D or 3D).
        output_path (str | Path):
            Output file path.
        file_extension (str | None):
            Optional extension override ("tif", "h5", ...).
        verbose (bool):
            If True, prints a confirmation message after writing.

    Returns:
        None

    Raises:
        TypeError, ValueError, OSError
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("data must be a numpy.ndarray")

    out = Path(output_path)
    ext = _normalize_extension(file_extension) if file_extension else _infer_extension_from_path(str(out))
    kind = _WRITE_EXTS.get(ext)

    if kind is None:
        raise ValueError(f"Unsupported write extension: '{ext}'")

    if kind == "edf":
        raise ValueError("Writing EDF is not supported (legacy read-only format).")

    if kind == "tiff":
        save_tiff(data, out)
    elif kind == "h5":
        save_h5(data, out)
    else:
        raise RuntimeError(f"Unhandled writer kind: {kind}")

    if verbose:
        print(f"Data written successfully to '{out}'")