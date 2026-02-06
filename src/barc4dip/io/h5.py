# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2026 ESRF - the European Synchrotron

""" 
Read and write of h5 ESRF-style images
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import h5py


def read_h5_image(image_path: str | Sequence[str]) -> np.ndarray:
    """
    Reads one or multiple HDF5 images from disk.

    Assumes the ESRF-style dataset location:
        /entry_0000/measurement/data

    Parameters:
        image_path (str | Sequence[str]):
            Path to an HDF5 file, or a sequence of paths to HDF5 files.

    Returns:
        np.ndarray:
            - 2D array if a single file is provided and the dataset is 2D.
            - 3D array (N, H, W) if:
                * a single file contains a 3D dataset, OR
                * a sequence of files is provided and each file contains a 2D dataset,
                in which case one frame per file is stacked along axis 0.
            - If a sequence of files is provided and each file contains a 3D dataset,
            datasets are concatenated along axis 0.

    Raises:
        TypeError:
            If image_path is not a str or a sequence of str.
        ValueError:
            If the sequence is empty, if dataset dimensionality is not 2D/3D,
            or if shapes are inconsistent across files.
        FileNotFoundError:
            If any provided file does not exist.
        KeyError:
            If the expected dataset path is missing.
        OSError:
            If reading the HDF5 file fails.
    """

    dataset_path = "entry_0000/measurement/data"

    def _read_one(p: str) -> np.ndarray:
        if not isinstance(p, str):
            raise TypeError("All elements of image_path must be strings")

        fp = Path(p)
        if not fp.exists():
            raise FileNotFoundError(f"HDF5 file not found: '{p}'")

        try:
            with h5py.File(fp, "r") as f:
                if dataset_path not in f:
                    raise KeyError(f"Dataset not found: '{dataset_path}' in '{p}'")
                arr = f[dataset_path][()]
        except OSError as e:
            raise OSError(f"Failed to read HDF5 file: '{p}'") from e

        arr = np.asarray(arr)
        if arr.ndim not in (2, 3):
            raise ValueError(
                f"Expected 2D or 3D dataset at '{dataset_path}', got shape {arr.shape} in '{p}'"
            )
        return arr

    if isinstance(image_path, str):
        return _read_one(image_path)

    if isinstance(image_path, Sequence):
        if len(image_path) == 0:
            raise ValueError("image_path sequence is empty")

        arrays = [_read_one(p) for p in image_path]
        ndims = {a.ndim for a in arrays}

        if ndims == {2}:
            ref_shape = arrays[0].shape
            for p, a in zip(image_path, arrays, strict=False):
                if a.shape != ref_shape:
                    raise ValueError(
                        f"Inconsistent image shapes in stack: expected {ref_shape}, "
                        f"got {a.shape} for '{p}'"
                    )
            return np.stack(arrays, axis=0)

        if ndims == {3}:
            ref_hw = arrays[0].shape[1:]
            for p, a in zip(image_path, arrays, strict=False):
                if a.shape[1:] != ref_hw:
                    raise ValueError(
                        f"Inconsistent stack shapes: expected (*, {ref_hw}), "
                        f"got {a.shape} for '{p}'"
                    )
            return np.concatenate(arrays, axis=0)

        raise ValueError(f"Mixed dataset dimensionality across files: ndims={sorted(ndims)}")

    raise TypeError("image_path must be a str or a sequence of str")

def save_h5_image(data: np.ndarray, output_path: str | Path) -> None:
    """
    Saves a 2D image or a 3D image stack to a single HDF5 file.

    Writes dataset to:
        /entry_0000/measurement/data

    Parameters:
        data (np.ndarray):
            Image data to save.
            - If 2D: saved as (H, W).
            - If 3D: saved as (N, H, W).
        output_path (str | Path):
            Output file path.
            If suffix is not .h5/.hdf5, .h5 is used.

    Returns:
        None

    Raises:
        TypeError:
            If data is not a numpy array.
        ValueError:
            If data is not 2D or 3D, or if output_path is invalid.
        OSError:
            If the destination path does not exist or is not writable,
            or if the output file already exists.
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

    if out.suffix.lower() not in {".h5", ".hdf5"}:
        out = out.with_suffix(".h5")

    if out.exists():
        raise OSError(f"Refusing to overwrite existing file: {out}")

    dataset_path = "entry_0000/measurement/data"

    try:
        with h5py.File(out, "x") as f:
            entry = f.require_group("entry_0000")
            meas = entry.require_group("measurement")

            entry.attrs.setdefault("NX_class", "NXentry")
            meas.attrs.setdefault("NX_class", "NXcollection")

            meas.create_dataset(
                "data",
                data=data,
                compression="gzip",
                compression_opts=4,
                chunks=True,
            )
    except OSError as e:
        raise OSError(f"Failed to write HDF5 file: {out}") from e