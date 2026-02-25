# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2026 ESRF - the European Synchrotron

"""
speckle field metrics
"""
from __future__ import annotations

import logging
import warnings
from typing import Literal, Sequence

import numpy as np

from ..maths.radial import radial_mean_binned, radial_mean_interpolated
from ..maths.stats import distance_at_fraction_from_peak, width_at_fraction
from ..signal.corr import autocorr2d
from ..signal.fft import psd2d
from ..utils import elapsed_time, now
from ..utils.range import percentile_minmax_range

from ..geometry.roi import odd_size, roi_grid_3x3, roi_slices
from ..signal.tracking import track_translation
from .common import (
    apply_display_origin,
    choose_tiling_mode,
    tiled_scalar_fields,
    tiles_meta,
)
from .statistics import distribution_moments

logger = logging.getLogger(__name__)


def speckle_stats(
    image: np.ndarray,
    *,
    metrics: str | Sequence[str] = "all",
    tiles: bool = True,
    display_origin: Literal["upper", "lower"] = "lower",
    saturation_value: float | None = 65535.0,
    eps: float = 1e-6,
    verbose: bool = True,
) -> dict:
    """
    Compute speckle metrics on a single 2D image.

    Parameters:
        image (np.ndarray):
            2D speckle intensity image (H, W).
        metrics (str | Sequence[str]):
            Metric group(s) to compute:
                - "amplitude"
                - "grain"
                - "bandwidth"
                - "stats"
                - "all" (default)
        tiles (bool):
            If True, also compute metrics on a 3x3 grid of tiles, where each
            tile is either:
                - aggregated from 9x9 sub-tiles (mean and std per 3x3 cell), or
                - computed directly on 3x3 tiles (std returned as NaNs).
            Tiles are only computed if the implied tile size meets MIN_TILE_PX.
        display_origin (Literal["upper", "lower"]):
            Defines the vertical origin convention used for analysis.
                - "lower" (default): detector-aligned convention.
                - "upper": NumPy convention. Index (0, 0) is the
                top-left pixel
        saturation_value (float | None):
            Passed to distribution_moments (default: 65535.0).
        eps (float):
            Passed to distribution_moments (default: 1e-6).
        verbose (bool):
            Verbose output for full-frame metrics only. Tile computations force
            verbose=True.

    Returns:
        dict:
            Dictionary with:
                - "full": dict of requested full-frame blocks.
                - "tiles": dict of requested tile blocks (only if feasible).

            Tile metrics are stored as:
                {"mean": grid3x3, "std": grid3x3}

            For direct 3x3 tiling, std_grid_3x3 is a 3x3 array of NaNs.

    Raises:
        TypeError:
            If image is not a NumPy array.
        ValueError:
            If image is not 2D.
    """
    t0 = now()

    if not isinstance(image, np.ndarray):
        raise TypeError("speckle_stats expects a numpy.ndarray")
    if image.ndim != 2:
        raise ValueError(f"Expected 2D array, got ndim={image.ndim}")

    image = apply_display_origin(image, display_origin=display_origin)

    h, w = image.shape
    groups = _normalize_metric_groups(metrics)

    if verbose:
        logger.info("\nspeckle stats for a (h x w: %.0f x %.0f) image:", h, w)

    out: dict = {
        "meta": {
            "kind": "speckles",
            "display_origin": display_origin,
            "input_shape": (int(h), int(w)),
            "requested_groups": sorted(groups),
        },
        "full": {},
    }

    if "amplitude" in groups:
        out["full"]["amplitude"] = amplitude(image, verbose=verbose)

    if "grain" in groups:
        out["full"]["grain"] = grain(image, verbose=verbose)

    if "stats" in groups:
        try:
            out["full"]["stats"] = distribution_moments(
                image,
                saturation_value=saturation_value,
                eps=eps,
                verbose=verbose,
            )
        except TypeError:
            out["full"]["stats"] = distribution_moments(
                image,
                saturation_value=saturation_value,
                eps=eps,
            )

    if "bandwidth" in groups:
        out["full"]["bandwidth"] = bandwidth(image, verbose=verbose)

    MIN_TILE_PX = 128
    mode, tile_shape_px = choose_tiling_mode(h, w, tiles=tiles, min_tile_px=MIN_TILE_PX)
    if mode == "off":
        if verbose:
            elapsed_time(t0)
        return out

    out["meta"].update(tiles_meta(h, w, tile_mode=mode, tile_shape_px=tile_shape_px))

    tiles_out: dict = {}

    if "amplitude" in groups:
        def _amp_tile(tile: np.ndarray) -> dict[str, float]:
            a = amplitude(tile, verbose=False)
            return {
                "visibility": float(a["visibility"]),
                "contrast": float(a["contrast"]),
            }

        tiles_out["amplitude"] = tiled_scalar_fields(image, tile_mode=mode, compute_fn=_amp_tile)

    if "grain" in groups:
        def _grain_tile(tile: np.ndarray) -> dict[str, float]:
            g = grain(tile, verbose=False)
            return {
                "lx": float(g["lx"]),
                "ly": float(g["ly"]),
                "leq": float(g["leq"]),
                "r": float(g["r"]),
            }

        tiles_out["grain"] = tiled_scalar_fields(image, tile_mode=mode, compute_fn=_grain_tile)

    if "stats" in groups:
        def _stats_tile(tile: np.ndarray) -> dict[str, float]:
            try:
                d = distribution_moments(tile, saturation_value=saturation_value, eps=eps, verbose=False)
            except TypeError:
                d = distribution_moments(tile, saturation_value=saturation_value, eps=eps)
            return {k: float(v) for k, v in d.items()}

        tiles_out["stats"] = tiled_scalar_fields(image, tile_mode=mode, compute_fn=_stats_tile)

    if "bandwidth" in groups:
        def _bw_tile(tile: np.ndarray) -> dict[str, float]:
            b = bandwidth(tile, verbose=False)
            return {
                "spr": float(b["spr"]),
                "feq": float(b["feq"]),
                "f95": float(b["f95"]),
                "sig_fx": float(b["sig_fx"]),
                "sig_fy": float(b["sig_fy"]),
                "rf": float(b["rf"]),
            }

        tiles_out["bandwidth"] = tiled_scalar_fields(image, tile_mode=mode, compute_fn=_bw_tile)

    if tiles_out:
        out["tiles"] = tiles_out

    if verbose:
        elapsed_time(t0)

    return out


def _normalize_metric_groups(metrics: str | Sequence[str]) -> set[str]:
    if isinstance(metrics, str):
        m = metrics.strip().lower()
        if m == "all":
            return {"amplitude", "grain", "bandwidth", "stats"}
        return {m}

    groups: set[str] = set()
    for item in metrics:
        if not isinstance(item, str):
            raise TypeError("metrics must be a str or a sequence of str")
        m = item.strip().lower()
        if m == "all":
            groups.update({"amplitude", "grain", "bandwidth", "stats"})
        else:
            groups.add(m)
    return groups


def speckle_stack_stats(
    stack: np.ndarray,
    *,
    metrics: str | Sequence[str] = "all",
    tiles: bool = True,
    display_origin: Literal["upper", "lower"] = "lower",
    roi_grain_factor: float = 3.0,
    roi_step_factor: float = 0.5,
    tracking_method: str = "phase",
    tracking_backend: Literal["internal", "skimage"] = "internal",
    subpixel: bool = True,
    saturation_value: float | None = 65535.0,
    eps: float = 1e-6,
    verbose: bool = True,
) -> dict:
    """Compute speckle metrics over time for a 3D stack, plus translation tracking.

    This function is the time-stack analogue of :func:`speckle_stats`:
    it computes the same per-frame metric blocks and stacks them along a
    leading time axis ``T``. In addition, it computes translation estimates
    (absolute and incremental) from a central 3x3 ROI grid using the tracking
    primitives in :mod:`barc4dip.signal.tracking`.

    Parameters
    ----------
    stack : np.ndarray
        3D image stack with shape (T, H, W).
    metrics : str | Sequence[str]
        Metric group(s) to compute per frame (passed to :func:`speckle_stats`).
    tiles : bool
        If True, also compute per-frame tile metrics (passed to :func:`speckle_stats`).
    display_origin : Literal["upper", "lower"]
        Analysis origin convention (passed to :func:`speckle_stats`). Temporal tracking
        is computed in NumPy convention and exported in detector convention by default
        when ``display_origin="lower"``.
    roi_grain_factor : float
        ROI size factor relative to grain size. The tracking ROI size is
        ``odd_size(roi_grain_factor * max(lx, ly, leq))``.
    roi_step_factor : float
        Step factor for the 3×3 ROI grid relative to ROI size. A value of 0.5 gives
        roughly ROI/2 overlap.
    tracking_method : str
        Tracking method identifier passed to :func:`track_translation` (default: "phase").
    tracking_backend : {"internal", "skimage"}
        Backend for the selected tracking method (default: "internal").
    subpixel : bool
        Enable subpixel refinement in the tracker (default: True).
    saturation_value : float | None
        Passed to :func:`speckle_stats` (stats block).
    eps : float
        Numerical stability constant passed to :func:`speckle_stats`.
    verbose : bool
        If True, emit a concise per-frame timing summary via the logger.

    Returns
    -------
    dict
        Dictionary with keys:
        - ``"meta"``: stack + tracking metadata
        - ``"full"``: per-frame full-frame metrics stacked on time axis
        - ``"tiles"``: per-frame tile metrics stacked on time axis (if requested/feasible)
        - ``"temporal"``: translation tracking summary (abs/inc) aggregated across a 3×3 ROI grid

    Raises
    ------
    TypeError
        If stack is not a NumPy array.
    ValueError
        If stack is not 3D or has too few frames.
    """
    t0 = now()

    if not isinstance(stack, np.ndarray):
        raise TypeError("speckle_stack_stats expects a numpy.ndarray")
    if stack.ndim != 3:
        raise ValueError(f"stack must be a 3D array with shape (T, H, W); got ndim={stack.ndim}")
    T, H, W = (int(stack.shape[0]), int(stack.shape[1]), int(stack.shape[2]))
    if T < 1:
        raise ValueError("stack must contain at least one frame.")

    per_frame: list[dict] = []
    last_bucket = -1
    for t in range(T):
        if verbose:
            bucket = int((10 * t) // max(1, T - 1))
            if bucket != last_bucket:
                last_bucket = bucket
                progress = 10 * bucket
                num_hashes = bucket
                bar = "#" * num_hashes + "-" * (10 - num_hashes)
                print(f"\rSpeckle stats loop: [{bar}] {progress:3d}%", end="", flush=True)
        frame = stack[t, :, :]
        stats_t = speckle_stats(
            frame,
            metrics=metrics,
            tiles=tiles,
            display_origin=display_origin,
            saturation_value=saturation_value,
            eps=eps,
            verbose=False,
        )
        per_frame.append(stats_t)
    out_full = _stack_time_series([d["full"] for d in per_frame])
    out_tiles = None
    if tiles and all(isinstance(d.get("tiles"), dict) for d in per_frame):
        out_tiles = _stack_time_series([d["tiles"] for d in per_frame])
    if verbose:
        print("\rSpeckle stats loop: [##########] 100%", flush=True)

    frame0 = stack[0, :, :]
    grain0 = grain(frame0, verbose=False)

    l = float(np.nanmax([grain0.get("lx", np.nan), grain0.get("ly", np.nan), grain0.get("leq", np.nan)]))
    if not np.isfinite(l) or l <= 0:
        raise ValueError("Could not infer a valid grain size from frame 0 (lx/ly/leq).")

    roi_side = odd_size(int(np.ceil(roi_grain_factor * l)))
    roi_size_yx = (roi_side, roi_side)

    step = int(max(1, round(roi_step_factor * roi_side)))
    step_yx = (step, step)

    grid_slices, grid_labels = roi_grid_3x3((H, W), roi_size_yx, step_yx, center_yx=None)

    dx_abs_tiles = np.empty((T, 3, 3), dtype=np.float32)
    dy_abs_tiles = np.empty((T, 3, 3), dtype=np.float32)
    dx_inc_tiles = np.empty((T, 3, 3), dtype=np.float32)
    dy_inc_tiles = np.empty((T, 3, 3), dtype=np.float32)

    last_bucket = -1
    for t in range(T):
        if verbose:
            bucket = int((10 * t) // max(1, T - 1))
            if bucket != last_bucket:
                last_bucket = bucket
                progress = 10 * bucket
                num_hashes = bucket
                bar = "#" * num_hashes + "-" * (10 - num_hashes)
                print(f"\rSpeckle stability loop: [{bar}] {progress:3d}%", end="", flush=True)
        img_t = stack[t, :, :]

        img_prev = stack[t - 1, :, :] if t > 0 else stack[0, :, :]

        for iy in range(3):
            for ix in range(3):
                sy, sx = grid_slices[iy, ix]

                tpl_abs = frame0[sy, sx]
                dy_a, dx_a, _, _ = track_translation(
                    tpl_abs,
                    img_t,
                    slices_yx=(sy, sx),
                    method=tracking_method,
                    backend=tracking_backend,
                    subpixel=subpixel,
                    eps=1e-9,
                )
                dy_abs_tiles[t, iy, ix] = dy_a
                dx_abs_tiles[t, iy, ix] = dx_a

                tpl_inc = img_prev[sy, sx]
                dy_i, dx_i, _, _ = track_translation(
                    tpl_inc,
                    img_t,
                    slices_yx=(sy, sx),
                    method=tracking_method,
                    backend=tracking_backend,
                    subpixel=subpixel,
                    eps=1e-9,
                )
                dy_inc_tiles[t, iy, ix] = dy_i
                dx_inc_tiles[t, iy, ix] = dx_i
    if display_origin == "lower":
        dy_abs_tiles = -dy_abs_tiles
        dy_inc_tiles = -dy_inc_tiles

    r_abs_tiles = np.sqrt(dx_abs_tiles**2 + dy_abs_tiles**2)
    r_inc_tiles = np.sqrt(dx_inc_tiles**2 + dy_inc_tiles**2)

    def _mean_std(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mu = np.nanmean(a.reshape(T, -1), axis=1)
        sd = np.nanstd(a.reshape(T, -1), axis=1)
        return mu.astype(np.float32), sd.astype(np.float32)

    dx_abs, std_dx_abs = _mean_std(dx_abs_tiles)
    dy_abs, std_dy_abs = _mean_std(dy_abs_tiles)
    r_abs, std_r_abs = _mean_std(r_abs_tiles)

    dx_inc, std_dx_inc = _mean_std(dx_inc_tiles)
    dy_inc, std_dy_inc = _mean_std(dy_inc_tiles)
    r_inc, std_r_inc = _mean_std(r_inc_tiles)

    temporal = {
        "abs": {
            "dx": dx_abs,
            "dy": dy_abs,
            "r": r_abs,
            "std_dx": std_dx_abs,
            "std_dy": std_dy_abs,
            "std_r": std_r_abs,
        },
        "inc": {
            "dx": dx_inc,
            "dy": dy_inc,
            "r": r_inc,
            "std_dx": std_dx_inc,
            "std_dy": std_dy_inc,
            "std_r": std_r_inc,
        },
        "qc": {
            "roi_grid_shape": (3, 3),
        },
    }

    meta: dict = {
        "kind": "speckle_stack_stats",
        "input_shape": (H, W),
        "stack_shape": (T, H, W),
        "n_frames": T,
        "display_origin": display_origin,
        "grain0": {k: grain0.get(k) for k in ("lx", "ly", "leq", "r")},
        "tracking": {
            "method": str(tracking_method),
            "backend": str(tracking_backend),
            "subpixel": bool(subpixel),
            "peak_mode": "abs",
            "search_area": "full_frame",
            "normalization": {"template": "zscore_local", "search": "zscore_global"},
            "roi_grain_factor": float(roi_grain_factor),
            "roi_size_yx": tuple(int(v) for v in roi_size_yx),
            "roi_step_factor": float(roi_step_factor),
            "roi_step_yx": tuple(int(v) for v in step_yx),
            "roi_labels": grid_labels,
            "roi_order": "row-major",
        },
    }

    out: dict = {"meta": meta, "full": out_full, "temporal": temporal}
    if out_tiles is not None:
        out["tiles"] = out_tiles

    if verbose:
        print("\rSpeckle stats loop: [##########] 100%", flush=True)

    if verbose:
        logger.info(
            "> speckle_stack_stats | frames=%d | roi=%dx%d | step=%d | elapsed=%s",
            T,
            roi_side,
            roi_side,
            step,
            elapsed_time(t0),
        )

    return out

def _stack_time_series(values: list[object]) -> object:
    """Stack per-frame outputs along a new leading time axis.

    Notes
    -----
    - Dicts are stacked recursively (same keys expected per frame).
    - NumPy arrays are stacked with np.stack(axis=0).
    - Scalars are stacked into a 1D NumPy array of length T.
    """
    if not values:
        raise ValueError("No values provided for stacking.")

    v0 = values[0]

    if isinstance(v0, dict):
        out: dict = {}
        keys = v0.keys()
        for k in keys:
            out[k] = _stack_time_series([v[k] for v in values])
        return out

    if isinstance(v0, np.ndarray):
        return np.stack([np.asarray(v) for v in values], axis=0)

    if isinstance(v0, (float, int, np.floating, np.integer, bool, np.bool_)):
        return np.asarray(values)

    return list(values)

# ---------------------------------------------------------------------------
# Grain
# ---------------------------------------------------------------------------

def grain(
    image: np.ndarray,
    *,
    fraction: float = 1.0 / np.e,
    radial_method: Literal["binned", "interpolated"] = "interpolated",
    verbose: bool = False,
) -> dict:
    """
    Compute speckle grain metrics from the autocorrelation peak.

    Parameters:
        image (np.ndarray):
            2D speckle intensity image.
        fraction (float):
            Threshold fraction for widths/radius (default: 1/e).
        radial_method (Literal["binned", "interpolated"]):
            Radial averaging method for leq (default: "interpolated").
        verbose (bool):
            If True, emit a concise, human-readable summary via the logging
            subsystem at INFO level. Default is False.
    Returns:
        Dictionary with keys:
            - lx: 1/e full width along x cut (pixels).
            - ly: 1/e full width along y cut (pixels).
            - leq: 1/e radius of the radially averaged autocorrelation (pixels).
            - r: anisotropy ratio lx / ly.
            - ac: 2D autocorrelation map (peak-normalized, shifted).
            - xlag: 1D lag axis for x (pixels).
            - ylag: 1D lag axis for y (pixels).

    Raises:
        ValueError
    """
    data = np.asarray(image, dtype=float)
    if data.ndim != 2:
        raise ValueError("image must be a 2D array.")
    if min(data.shape) < 128:
        raise ValueError("image too small for speckle grain metrics (min dimension < 128).")

    ac, xlag, ylag = autocorr2d(
        data,
        dx=1.0,
        dy=1.0,
        remove_mean=True,
        standardize=False,
        normalize="peak",
    )

    ac = np.asarray(ac)
    if np.iscomplexobj(ac):
        imag_max = float(np.max(np.abs(ac.imag)))
        real_max = float(np.max(np.abs(ac.real)))
        if imag_max > 1e-10 * max(real_max, 1.0):
            raise ValueError(
                f"autocorr2d returned significant imaginary part "
                f"(max|Im|={imag_max:.3e}, max|Re|={real_max:.3e})."
            )
        ac = ac.real

    iy, ix = np.unravel_index(int(np.argmax(ac)), ac.shape)

    y_cut = ac[:, ix]
    x_cut = ac[iy, :]

    ly_px, _ly_hit_edge = width_at_fraction(y_cut, fraction=fraction, center_index=iy)
    lx_px, _lx_hit_edge = width_at_fraction(x_cut, fraction=fraction, center_index=ix)

    if radial_method == "binned":
        rad, r = radial_mean_binned(ac)
    elif radial_method == "interpolated":
        rad, r = radial_mean_interpolated(ac)
    else:
        raise ValueError("radial_method must be 'binned' or 'interpolated'.")

    rad = np.asarray(rad, dtype=float)
    r = np.asarray(r, dtype=float)
    if rad.size < 2 or r.size < 2:
        raise ValueError("Radial profile is too short to estimate leq.")

    dr = float(r[1] - r[0])
    if dr <= 0:
        raise ValueError("Invalid radial sampling (non-positive dr).")

    dist_bins, _leq_hit_edge = distance_at_fraction_from_peak(rad, fraction=fraction, peak_index=0)
    leq_px = 2 * float(dist_bins) * dr

    lx_px = float(lx_px)
    ly_px = float(ly_px)
    r_aniso = float(lx_px / ly_px) if ly_px != 0 else float("inf")

    metrics = {
        "lx": lx_px,
        "ly": ly_px,
        "leq": float(leq_px),
        "r": float(r_aniso),
        "autocorr": np.asarray(ac, dtype=float),
        "xlag": np.asarray(xlag, dtype=float),
        "ylag": np.asarray(ylag, dtype=float),    
    }
    
    if verbose:
        logger.info(
            "> grain: lx=%.2f | ly=%.2f | lx/ly=%.2f | leq=%.2f ",
            metrics["lx"],
            metrics["ly"],
            metrics["r"],
            metrics["leq"],
        )

    return metrics

# ---------------------------------------------------------------------------
# Amplitude (visibility, contrast)
# ---------------------------------------------------------------------------

def amplitude(image: np.ndarray, verbose: bool = False) -> dict:
    """
    Compute amplitude-related speckle metrics from an intensity image.

    This function groups two commonly used amplitude fluctuation measures:

    1) Visibility: std(I) / mean(I). 

    2) robust Michelson contrast: (I_high - I_low) / (I_high + I_low), 
       where I_low and I_high are obtained from a percentile-based min/max range.

    Both metrics are dimensionless and complementary: visibility captures
    statistical fluctuations, while Michelson contrast captures dynamic
    range.

    Parameters:
        image (np.ndarray):
            2D speckle intensity image.
        verbose (bool):
            If True, emit a concise, human-readable summary via the logging
            subsystem at INFO level. Default is False.

    Returns:
        dict:
            Dictionary with keys:
                - "visibility": Speckle visibility (std / mean).
                - "contrast": Robust Michelson contrast.

    Raises:
        ValueError:
            If the input image is not 2D or if required intensity statistics
            are non-finite or non-positive.
    """

    img = np.asarray(image, dtype=float)
    if img.ndim != 2:
        raise ValueError("image must be a 2D array.")

    mu = float(np.nanmean(img))
    if not np.isfinite(mu) or mu <= 0.0:
        raise ValueError("Mean intensity must be positive and finite.")

    sigma = float(np.nanstd(img))
    visibility = sigma / mu

    vmin, vmax = percentile_minmax_range(img)
    denom = vmax + vmin
    if not np.isfinite(denom) or denom <= 0.0:
        raise ValueError("Invalid percentile range for Michelson contrast.")

    contrast = (vmax - vmin) / denom

    out = {"visibility": visibility, "contrast": contrast}

    if verbose:
        logger.info(
            "> visibility: %.2f | contrast: %.2f",
            visibility,
            contrast,
        )

    return out

# ---------------------------------------------------------------------------
# Spectral analysis
# ---------------------------------------------------------------------------

def bandwidth(image: np.ndarray, verbose: bool = False) -> dict[str, float]:
    """
    Compute spatial-frequency bandwidth metrics from the 2D PSD (pixel domain),
    including a spectral participation ratio (SPR) computed from the same PSD.

    This function characterizes how "broad" the speckle texture is in spatial-frequency space.
    All frequency quantities are returned in cycles/pixel (pixel domain).

    Metrics
    -------
    feq : float
        Equivalent (RMS) radial bandwidth, computed from the 2D PSD as:

            feq = sqrt( sum( f^2 * P ) / sum(P) )

        where f = sqrt(fx^2 + fy^2) and P is the PSD.

    f95 : float
        Encircled-energy radius in frequency space: the radial frequency such that 95% of the
        PSD energy is contained within f <= f95 (computed from the 2D PSD).

    sig_fx, sig_fy : float
        RMS bandwidth along x and y:

            sig_fx = sqrt( sum( fx^2 * P ) / sum(P) )
            sig_fy = sqrt( sum( fy^2 * P ) / sum(P) )

    rf : float
        Spectral anisotropy ratio:

            rf = sig_fx / sig_fy

    spr : float
        Spectral participation ratio (effective spectral DoF), computed from the normalized PSD:

            p_i = P_i / sum(P)
            spr = 1 / sum_i (p_i^2)

        This measures how many spectral bins effectively participate (higher = more spread/less concentrated).

    Mean removal and DC handling
    ----------------------------
    The image mean is removed before computing the PSD to reduce sensitivity to offsets and
    slow background pedestals. The DC bin (zero frequency) is set to zero as a safeguard
    against residual numerical DC.

    Notes on frequency support
    --------------------------
    Metrics are computed over an inscribed circular region in frequency space
    (FR <= min(max|fx|, max|fy|)) to avoid corner bins that may bias radial measures.

    Parameters:
        image (np.ndarray):
            2D speckle intensity image.
        verbose (bool):
            If True, emit a concise, human-readable summary via the logging
            subsystem at INFO level. Default is False.
    Returns:
        dict[str, float]:
            Dictionary with keys:
                - "feq": equivalent radial bandwidth (cycles/pixel)
                - "f95": 95% encircled-energy radius (cycles/pixel)
                - "sig_fx": RMS bandwidth along fx (cycles/pixel)
                - "sig_fy": RMS bandwidth along fy (cycles/pixel)
                - "rf": anisotropy ratio sig_fx / sig_fy
                - "spr": spectral participation ratio (dimensionless)

    Raises:
        ValueError:
            If image is not 2D, or if the PSD energy after mean/DC removal is not positive/finite.
    """
    img = np.asarray(image, dtype=float)
    if img.ndim != 2:
        raise ValueError("image must be a 2D array.")

    mu = float(np.nanmean(img))
    if not np.isfinite(mu):
        raise ValueError("image mean is not finite.")
    img = img - mu

    P, fx, fy = psd2d(img, dx=1.0, dy=1.0, scale=True)
    P = np.asarray(P, dtype=float)

    if P.ndim != 2:
        raise ValueError("psd2d returned a non-2D PSD (unexpected).")

    ny, nx = P.shape
    P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)

    P = P.copy()
    P[ny // 2, nx // 2] = 0.0

    FX, FY = np.meshgrid(np.asarray(fx, dtype=float), np.asarray(fy, dtype=float), indexing="xy")
    FR = np.sqrt(FX * FX + FY * FY)

    f_max = min(float(np.max(np.abs(fx))), float(np.max(np.abs(fy))))
    mask = FR <= f_max

    Pm = P[mask]
    FXm = FX[mask]
    FYm = FY[mask]
    FRm = FR[mask]

    total = float(np.sum(Pm))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("PSD energy is not positive/finite after mean/DC removal.")

    feq = float(np.sqrt(np.sum((FRm * FRm) * Pm) / total))
    sig_fx = float(np.sqrt(np.sum((FXm * FXm) * Pm) / total))
    sig_fy = float(np.sqrt(np.sum((FYm * FYm) * Pm) / total))
    rf = float(sig_fx / sig_fy) if sig_fy != 0.0 else float("inf")

    order = np.argsort(FRm)
    FRs = FRm[order]
    Ps = Pm[order]
    cdf = np.cumsum(Ps) / total
    idx = int(np.searchsorted(cdf, 0.95, side="left"))
    if idx >= FRs.size:
        idx = FRs.size - 1
    f95 = float(FRs[idx])

    p = Pm / total
    denom = float(np.sum(p * p))
    if not np.isfinite(denom) or denom <= 0.0:
        raise ValueError("Invalid SPR denominator (unexpected).")
    spr = float(1.0 / denom)

    spectral =  {
        "feq": feq,
        "f95": f95,
        "sig_fx": sig_fx,
        "sig_fy": sig_fy,
        "rf": rf,
        "spr": spr,
    }
    if verbose:
        logger.info(
            "> bandwidth: fx=%.4f | fy=%.4f | fx/fy=%.2f | feq=%.4f | f95=%.4f | spr=%.0f",
            spectral["sig_fx"],
            spectral["sig_fy"],
            spectral["rf"],
            spectral["feq"],
            spectral["f95"],
            spectral["spr"],
        )

    return spectral


