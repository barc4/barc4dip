# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2026 ESRF - the European Synchrotron

from __future__ import annotations

import warnings
from typing import Callable, Literal

import numpy as np

TILE_GRID_SHAPE_3X3: tuple[int, int] = (3, 3)
TILE_ORDER: str = "row-major"
TILE_LABELS_3X3: np.ndarray = np.array(
    [["NW", "N", "NE"], ["W", "C", "E"], ["SW", "S", "SE"]],
    dtype=object,
)


def normalize_display_origin(display_origin: str) -> Literal["upper", "lower"]:
    """
    Normalize the display origin convention.

    Parameters:
        display_origin (str):
            Display convention for the vertical axis. Supported values are:
            - "upper": row 0 is at the top (no vertical flip)
            - "lower": row 0 is at the bottom (vertical flip)

    Returns:
        Literal["upper", "lower"]:
            Normalized origin.

    Raises:
        ValueError:
            If display_origin is not one of {"upper", "lower"}.
    """
    origin = str(display_origin).strip().lower()
    if origin not in ("upper", "lower"):
        raise ValueError("display_origin must be 'upper' or 'lower'.")
    return origin


def apply_display_origin(image: np.ndarray, *, display_origin: str) -> np.ndarray:
    """
    Apply the display origin convention to a 2D image.

    This is an ergonomics helper to ensure that tiling labels (NW/N/NE, ...)
    correspond to what is displayed to the user.

    Parameters:
        image (np.ndarray):
            2D image.
        display_origin (str):
            Display origin convention ("upper" or "lower").

    Returns:
        np.ndarray:
            View of the image with the requested origin convention applied.

    Raises:
        ValueError:
            If image is not 2D.
    """
    img = np.asarray(image)
    if img.ndim != 2:
        raise ValueError(f"apply_display_origin expects a 2D array, got ndim={img.ndim}")

    origin = normalize_display_origin(display_origin)
    if origin == "lower":
        return img[::-1, :]
    return img


def split_edges(length: int, n_parts: int) -> list[tuple[int, int]]:
    """
    Split an integer interval [0, length) into n_parts contiguous slices.

    Parameters:
        length (int):
            Total length to split.
        n_parts (int):
            Number of contiguous parts.

    Returns:
        list[tuple[int, int]]:
            List of (start, stop) pairs (stop exclusive).

    Raises:
        ValueError:
            If length < 1 or n_parts < 1.
    """
    if length < 1:
        raise ValueError("length must be >= 1.")
    if n_parts < 1:
        raise ValueError("n_parts must be >= 1.")

    edges = np.linspace(0, length, n_parts + 1)
    out: list[tuple[int, int]] = []
    for i in range(n_parts):
        a = int(round(float(edges[i])))
        b = int(round(float(edges[i + 1])))
        b = max(b, a + 1)
        out.append((a, b))
    out[-1] = (out[-1][0], length) 
    return out


def choose_tiling_mode(
    h: int,
    w: int,
    *,
    tiles: bool = False,
    min_tile_px: int = 128,
) -> tuple[Literal["off", "tiles_3x3", "subtiles_9x9"], tuple[int, int] | None]:
    """
    Choose the tiling evaluation mode and the effective evaluation tile shape.

    Policy:
        - If tiles is False: tiling is disabled ("off").
        - If tiles is True: try "subtiles_9x9" first. If subtiles are too small,
          fall back to "tiles_3x3". If tiles are still too small, disable tiling
          and emit a warning ("Image too small for tiling").

    The `min_tile_px` threshold applies to the *evaluation tiles*:
        - subtiles_9x9 uses tiles of size (h//9, w//9)
        - tiles_3x3 uses tiles of size (h//3, w//3)

    Parameters:
        h (int):
            Image height in pixels.
        w (int):
            Image width in pixels.
        tiles (bool):
            If True, enable tiling mode selection. If False, tiling is disabled.
        min_tile_px (int):
            Minimum acceptable evaluation tile size in pixels (default: 128).

    Returns:
        (tile_mode, tile_shape_px)
            tile_mode:
                One of {"off", "tiles_3x3", "subtiles_9x9"}.
            tile_shape_px:
                (tile_h, tile_w) for the evaluation tiles used in the chosen mode,
                or None if tile_mode is "off".

    Raises:
        ValueError:
            If h or w are invalid or min_tile_px is invalid.
    """
    if h < 1 or w < 1:
        raise ValueError("Invalid image shape (h and w must be >= 1).")
    if min_tile_px < 1:
        raise ValueError("min_tile_px must be >= 1.")

    if not bool(tiles):
        return "off", None

    if (h // 9) >= min_tile_px and (w // 9) >= min_tile_px:
        return "subtiles_9x9", (h // 9, w // 9)

    if (h // 3) >= min_tile_px and (w // 3) >= min_tile_px:
        return "tiles_3x3", (h // 3, w // 3)

    warnings.warn(
        f"Image too small for tiling: shape=({h}, {w}), min_tile_px={min_tile_px}.",
        RuntimeWarning,
        stacklevel=2,
    )
    return "off", None


def tiles_meta(
    h: int,
    w: int,
    *,
    tile_mode: Literal["off", "tiles_3x3", "subtiles_9x9"],
    tile_shape_px: tuple[int, int] | None = None,
) -> dict:
    """
    Standardized metadata fields related to tiling.

    Parameters:
        h (int):
            Image height (pixels).
        w (int):
            Image width (pixels).
        tile_mode (Literal["off", "tiles_3x3", "subtiles_9x9"]):
            Tiling mode used.
        tile_shape_px (tuple[int, int] | None):
            Evaluation tile shape (pixels). Must be provided when tile_mode != "off".

    Returns:
        dict:
            Metadata fields to be merged into out["meta"].

    Raises:
        ValueError:
            If tile_mode is not "off" and tile_shape_px is None.
    """
    meta: dict = {"tile_mode": tile_mode}
    if tile_mode == "off":
        return meta

    if tile_shape_px is None:
        raise ValueError("tile_shape_px must be provided when tile_mode is not 'off'.")

    meta.update(
        {
            "tile_grid_shape": TILE_GRID_SHAPE_3X3,
            "tile_labels": TILE_LABELS_3X3,
            "tile_order": TILE_ORDER,
            "tile_shape_px": (int(tile_shape_px[0]), int(tile_shape_px[1])),
            "used_subtiles": bool(tile_mode == "subtiles_9x9"),
        }
    )
    return meta


def nan_std_grid_3x3() -> np.ndarray:
    """
    Create a 3x3 float grid filled with NaNs (std placeholder).

    Returns:
        np.ndarray:
            Array of shape (3, 3) with NaN values.
    """
    return np.full((3, 3), np.nan, dtype=float)


def pack_mean_std(mean: np.ndarray, std: np.ndarray) -> dict:
    """
    Pack mean/std tile grids into the standard schema.

    Parameters:
        mean (np.ndarray):
            Mean grid, shape (3, 3).
        std (np.ndarray):
            Std grid, shape (3, 3).

    Returns:
        dict:
            {"mean": mean, "std": std}
    """
    return {"mean": np.asarray(mean, dtype=float), "std": np.asarray(std, dtype=float)}


def aggregate_subtiles_9x9_to_3x3(sub: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Aggregate a 9x9 grid of subtile values into a 3x3 grid of mean and std.

    Parameters:
        sub (np.ndarray):
            Array of shape (9, 9) containing per-subtile scalar values.

    Returns:
        (mean, std):
            mean and std grids, each of shape (3, 3).

    Raises:
        ValueError:
            If input does not have shape (9, 9).
    """
    arr = np.asarray(sub, dtype=float)
    if arr.shape != (9, 9):
        raise ValueError("Expected subtiles grid of shape (9, 9).")

    mean = np.empty((3, 3), dtype=float)
    std = np.empty((3, 3), dtype=float)
    for r in range(3):
        for c in range(3):
            block = arr[3 * r : 3 * r + 3, 3 * c : 3 * c + 3]
            mean[r, c] = float(np.mean(block))
            std[r, c] = float(np.std(block, ddof=0))
    return mean, std


def tiled_scalar_fields(
    image: np.ndarray,
    *,
    tile_mode: Literal["tiles_3x3", "subtiles_9x9"],
    compute_fn: Callable[[np.ndarray], dict[str, float]],
) -> dict[str, dict[str, np.ndarray]]:
    """
    Compute per-tile scalar fields in the standard tiles schema.

    This is a generic tiling executor used by metric aggregators. It preserves
    the public return schema of the aggregators by returning per-field entries:

        tiles[field] = {"mean": grid3x3, "std": grid3x3}

    Two evaluation modes are supported:
        - tiles_3x3: compute directly on 3x3 tiles; std grid is NaNs.
        - subtiles_9x9: compute on 9x9 subtiles; aggregate to 3x3 mean/std.

    Parameters:
        image (np.ndarray):
            2D image (already oriented via apply_display_origin).
        tile_mode (Literal["tiles_3x3", "subtiles_9x9"]):
            Evaluation mode.
        compute_fn (Callable[[np.ndarray], dict[str, float]]):
            Function called on each tile (2D) returning a dict of scalar fields.
            Keys must be consistent across calls.

    Returns:
        dict[str, dict[str, np.ndarray]]:
            Mapping from scalar field name to {"mean": grid3x3, "std": grid3x3}.

    Raises:
        ValueError:
            If image is not 2D or too small for the requested tile_mode.
    """
    img = np.asarray(image)
    if img.ndim != 2:
        raise ValueError(f"tiled_scalar_fields expects a 2D array, got ndim={img.ndim}")
    h, w = img.shape

    out: dict[str, dict[str, np.ndarray]] = {}

    if tile_mode == "tiles_3x3":
        y_edges = split_edges(h, 3)
        x_edges = split_edges(w, 3)

        y0, y1 = y_edges[0]
        x0, x1 = x_edges[0]
        first = compute_fn(img[y0:y1, x0:x1])
        if not first:
            raise ValueError("compute_fn returned an empty dict for the first tile.")

        grids: dict[str, np.ndarray] = {k: np.empty((3, 3), dtype=float) for k in first.keys()}
        for k, v in first.items():
            grids[k][0, 0] = float(v)

        for r in range(3):
            y0, y1 = y_edges[r]
            for c in range(3):
                if r == 0 and c == 0:
                    continue
                x0, x1 = x_edges[c]
                vals = compute_fn(img[y0:y1, x0:x1])
                for k in grids.keys():
                    grids[k][r, c] = float(vals[k])

        nan_std = nan_std_grid_3x3()
        for k, grid in grids.items():
            out[k] = pack_mean_std(grid, nan_std)
        return out

    if tile_mode == "subtiles_9x9":
        y_edges = split_edges(h, 9)
        x_edges = split_edges(w, 9)

        y0, y1 = y_edges[0]
        x0, x1 = x_edges[0]
        first = compute_fn(img[y0:y1, x0:x1])
        if not first:
            raise ValueError("compute_fn returned an empty dict for the first subtile.")

        subgrids: dict[str, np.ndarray] = {k: np.empty((9, 9), dtype=float) for k in first.keys()}
        for k, v in first.items():
            subgrids[k][0, 0] = float(v)

        for r in range(9):
            y0, y1 = y_edges[r]
            for c in range(9):
                if r == 0 and c == 0:
                    continue
                x0, x1 = x_edges[c]
                vals = compute_fn(img[y0:y1, x0:x1])
                for k in subgrids.keys():
                    subgrids[k][r, c] = float(vals[k])

        for k, sub in subgrids.items():
            mean3, std3 = aggregate_subtiles_9x9_to_3x3(sub)
            out[k] = pack_mean_std(mean3, std3)
        return out

    raise ValueError("tile_mode must be 'tiles_3x3' or 'subtiles_9x9'.")
