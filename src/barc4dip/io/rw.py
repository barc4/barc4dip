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

from .edf import read_edf_image
from .h5 import read_h5_image, save_h5_image
from .tiff import read_tiff_image, save_tiff_image


_READ_EXTS: dict[str, str] = {
    "tiff": "tiff",
    "tif": "tiff",
    "edf": "edf",
    "h5": "h5",
    "hdf5": "h5",
}

_WRITE_EXTS: dict[str, str] = {
    "tiff": "tiff",
    "tif": "tiff",
    "h5": "h5",
    "hdf5": "h5",
    "edf": "edf",
}

def _normalize_extension(ext: str) -> str:
    e = ext.strip().lower()
    if e.startswith("."):
        e = e[1:]
    return e


def _infer_extension_from_path(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix == "":
        raise ValueError(
            "Cannot infer file extension from path (no suffix). "
            "Provide file_extension explicitly."
        )
    return _normalize_extension(suffix)


def _infer_extension_from_paths(paths: Sequence[str]) -> str:
    exts = []
    for p in paths:
        if not isinstance(p, str):
            raise TypeError("All elements of image_path must be strings")
        exts.append(_infer_extension_from_path(p))

    first = exts[0]
    if any(e != first for e in exts):
        raise ValueError(f"Mixed file extensions in image_path sequence: {sorted(set(exts))}")
    return first


def read_image(image_path: str | Sequence[str], *, file_extension: str | None = None) -> np.ndarray:
    """
    Reads one image or a stack of images from disk.

    The reader is selected based on the file extension, either:
        - explicitly via file_extension, or
        - inferred from the path suffix.

    Supported read formats:
        - TIFF: .tif, .tiff
        - EDF:  .edf
        - HDF5: .h5, .hdf5

    Parameters:
        image_path (str | Sequence[str]):
            Path to a file, or a sequence of file paths.
            If a sequence is provided, the underlying format reader stacks or
            concatenates frames along axis 0.
        file_extension (str | None):
            Optional override for the file extension (e.g. "tif", "edf", "h5").
            If None, the extension is inferred from image_path.

    Returns:
        np.ndarray:
            2D or 3D array depending on input and file content.

    Raises:
        TypeError:
            If image_path is not a str or a sequence of str.
        ValueError:
            If the sequence is empty, if extensions are mixed, if the extension
            cannot be inferred, or if the extension is not supported.
    """

    if isinstance(image_path, str):
        ext = _normalize_extension(file_extension) if file_extension is not None else _infer_extension_from_path(image_path)
    elif isinstance(image_path, Sequence):
        if len(image_path) == 0:
            raise ValueError("image_path sequence is empty")
        ext = _normalize_extension(file_extension) if file_extension is not None else _infer_extension_from_paths(image_path)
    else:
        raise TypeError("image_path must be a str or a sequence of str")

    kind = _READ_EXTS.get(ext)
    if kind is None:
        raise ValueError(f"Unsupported read extension: '{ext}'")

    if kind == "tiff":
        return read_tiff_image(image_path)
    if kind == "edf":
        return read_edf_image(image_path)
    if kind == "h5":
        return read_h5_image(image_path)

    # Should never happen
    raise RuntimeError(f"Unhandled reader kind: {kind}")


def write_image(data: np.ndarray, output_path: str | Path, *, 
                file_extension: str | None = None, ) -> None:
    """
    Writes an image or image stack to disk.

    The writer is selected based on the file extension, either:
        - explicitly via file_extension, or
        - inferred from output_path suffix.

    Supported write formats:
        - TIFF: .tif, .tiff
        - HDF5: .h5, .hdf5

    EDF writing is not supported and raises an error.

    Parameters:
        data (np.ndarray):
            Image data to save (2D or 3D).
        output_path (str | Path):
            Output file path.
        file_extension (str | None):
            Optional override for the file extension (e.g. "tif", "h5").
            If None, the extension is inferred from output_path.

    Returns:
        None

    Raises:
        TypeError:
            If data is not a numpy array.
        ValueError:
            If the extension cannot be inferred, if the extension is not supported,
            or if an EDF write is requested.
        OSError:
            If the destination path does not exist, is not writable,
            or the underlying writer fails.
    """

    if not isinstance(data, np.ndarray):
        raise TypeError("data must be a numpy.ndarray")

    out = Path(output_path)

    ext = _normalize_extension(file_extension) if file_extension is not None else _infer_extension_from_path(str(out))
    kind = _WRITE_EXTS.get(ext)
    if kind is None:
        raise ValueError(f"Unsupported write extension: '{ext}'")

    if kind == "edf":
        raise ValueError("Writing EDF is not supported (legacy read-only format).")

    if kind == "tiff":
        save_tiff_image(data, out)
        return

    if kind == "h5":
        save_h5_image(data, out)
        return

    raise RuntimeError(f"Unhandled writer kind: {kind}")
