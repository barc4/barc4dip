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

from barc4dip.maths.radial import radial_mean_binned, radial_mean_interpolated
from barc4dip.maths.stats import distance_at_fraction_from_peak, width_at_fraction
from barc4dip.metrics.statistics import distribution_moments
from barc4dip.signal.corr import autocorr2d
from barc4dip.signal.fft import psd2d
from barc4dip.utils.range import percentile_minmax_range

logger = logging.getLogger(__name__)

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

    1) Visibility (a.k.a. speckle contrast):
       Defined as std(I) / mean(I). It quantifies relative intensity
       fluctuations around the mean and is directly linked to the number
       of coherent modes in fully developed speckle. This metric is
       sensitive to global intensity scaling but robust to spatial outliers
       when NaNs are present.

    2) Michelson contrast (robust form):
       Defined as (I_high - I_low) / (I_high + I_low), where I_low and I_high
       are obtained from a percentile-based min/max range. This emphasizes
       peak-to-valley modulation and is more sensitive to extreme values,
       making it useful for assessing modulation depth while remaining
       robust against hot/dead pixels.

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


def speckle_stats(
    image: np.ndarray,
    *,
    metrics: str | Sequence[str] = "all",
    tiles: bool = False,
    saturation_value: float | None = 65535.0,
    eps: float = 1e-6,
    verbose: bool = False,
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
        saturation_value (float | None):
            Passed to distribution_moments (default: 65535.0).
        eps (float):
            Passed to distribution_moments (default: 1e-6).
        verbose (bool):
            Verbose output for full-frame metrics only. Tile computations force
            verbose=False.

    Returns:
        dict:
            Dictionary with:
                - "full": dict of requested full-frame blocks.
                - "tiles": dict of requested tile blocks (only if feasible).

            Tile metrics are stored as:
                (mean_grid_3x3, std_grid_3x3)

            For direct 3x3 tiling, std_grid_3x3 is a 3x3 array of NaNs.

    Raises:
        TypeError:
            If image is not a NumPy array.
        ValueError:
            If image is not 2D.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("speckle_stats expects a numpy.ndarray")
    if image.ndim != 2:
        raise ValueError(f"Expected 2D array, got ndim={image.ndim}")

    h, w = image.shape
    groups = _normalize_metric_groups(metrics)

    out: dict = {
        "meta": {
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

    if not tiles:
        return out

    MIN_TILE_PX = 128
    mode = _choose_tiling_mode(h, w, min_tile_px=MIN_TILE_PX)
    if mode == "off":
        return out

    out["meta"].update(
        {
            "tile_grid_shape": (3, 3),
            "tile_labels": [["NW", "N", "NE"], ["W", "C", "E"], ["SW", "S", "SE"]],
            "tile_order": "row-major",
            "tile_mode": mode,
            "tile_shape_px": (
                int(h // 9),
                int(w // 9),
            )
            if mode == "subtiles_9x9"
            else (
                int(h // 3),
                int(w // 3),
            ),
        }
    )

    tiles_out: dict = {}

    def _nan_std_grid() -> np.ndarray:
        return np.full((3, 3), np.nan, dtype=float)

    def _pack(mean: np.ndarray, std: np.ndarray) -> dict:
        return {"mean": mean, "std": std}

    if mode == "subtiles_9x9":
        sub_edges_y = _split_edges(h, 9)
        sub_edges_x = _split_edges(w, 9)

        if "amplitude" in groups:
            vis_sub = np.empty((9, 9), dtype=float)
            con_sub = np.empty((9, 9), dtype=float)

        if "grain" in groups:
            g_sub = {k: np.empty((9, 9), dtype=float) for k in ("lx", "ly", "leq", "r")}

        stats_sub: dict[str, np.ndarray] | None = None

        if "bandwidth" in groups:
            bw_sub = {k: np.empty((9, 9), dtype=float) for k in ("spr", "feq", "f95", "sig_fx", "sig_fy", "rf")}

        for ry in range(9):
            y0, y1 = sub_edges_y[ry], sub_edges_y[ry + 1]
            for rx in range(9):
                x0, x1 = sub_edges_x[rx], sub_edges_x[rx + 1]
                tile = image[y0:y1, x0:x1]

                if "amplitude" in groups:
                    a = amplitude(tile, verbose=False)
                    vis_sub[ry, rx] = float(a["visibility"])
                    con_sub[ry, rx] = float(a["contrast"])

                if "grain" in groups:
                    g = grain(tile, verbose=False)
                    for k in g_sub:
                        g_sub[k][ry, rx] = float(g[k])

                if "stats" in groups:
                    try:
                        d = distribution_moments(tile, saturation_value=saturation_value, eps=eps, verbose=False)
                    except TypeError:
                        d = distribution_moments(tile, saturation_value=saturation_value, eps=eps)

                    if stats_sub is None:
                        stats_sub = {k: np.empty((9, 9), dtype=float) for k in d.keys()}
                    for k, v in d.items():
                        stats_sub[k][ry, rx] = float(v)

                if "bandwidth" in groups:
                    b = bandwidth(tile, verbose=False)
                    for k in bw_sub:
                        bw_sub[k][ry, rx] = float(b[k])

        if "amplitude" in groups:
            m, s = _aggregate_subtiles(vis_sub)
            tiles_out.setdefault("amplitude", {})["visibility"] = _pack(m, s)
            m, s = _aggregate_subtiles(con_sub)
            tiles_out["amplitude"]["contrast"] = _pack(m, s)

        if "grain" in groups:
            gt = tiles_out.setdefault("grain", {})
            for k, arr in g_sub.items():
                m, s = _aggregate_subtiles(arr)
                gt[k] = _pack(m, s)

        if "stats" in groups and stats_sub is not None:
            st = tiles_out.setdefault("stats", {})
            for k, arr in stats_sub.items():
                m, s = _aggregate_subtiles(arr)
                st[k] = _pack(m, s)

        if "bandwidth" in groups:
            bt = tiles_out.setdefault("bandwidth", {})
            for k, arr in bw_sub.items():
                m, s = _aggregate_subtiles(arr)
                bt[k] = _pack(m, s)

    elif mode == "tiles_3x3":
        edges_y = _split_edges(h, 3)
        edges_x = _split_edges(w, 3)

        if "amplitude" in groups:
            vis = np.empty((3, 3), dtype=float)
            con = np.empty((3, 3), dtype=float)

        if "grain" in groups:
            g3 = {k: np.empty((3, 3), dtype=float) for k in ("lx", "ly", "leq", "r")}

        stats3: dict[str, np.ndarray] | None = None

        if "bandwidth" in groups:
            bw3 = {k: np.empty((3, 3), dtype=float) for k in ("spr", "feq", "f95", "sig_fx", "sig_fy", "rf")}

        for ty in range(3):
            y0, y1 = edges_y[ty], edges_y[ty + 1]
            for tx in range(3):
                x0, x1 = edges_x[tx], edges_x[tx + 1]
                tile = image[y0:y1, x0:x1]

                if "amplitude" in groups:
                    a = amplitude(tile, verbose=False)
                    vis[ty, tx] = float(a["visibility"])
                    con[ty, tx] = float(a["contrast"])

                if "grain" in groups:
                    g = grain(tile, verbose=False)
                    for k in g3:
                        g3[k][ty, tx] = float(g[k])

                if "stats" in groups:
                    try:
                        d = distribution_moments(tile, saturation_value=saturation_value, eps=eps, verbose=False)
                    except TypeError:
                        d = distribution_moments(tile, saturation_value=saturation_value, eps=eps)

                    if stats3 is None:
                        stats3 = {k: np.empty((3, 3), dtype=float) for k in d.keys()}
                    for k, v in d.items():
                        stats3[k][ty, tx] = float(v)

                if "bandwidth" in groups:
                    b = bandwidth(tile, verbose=False)
                    for k in bw3:
                        bw3[k][ty, tx] = float(b[k])

        if "amplitude" in groups:
            tiles_out.setdefault("amplitude", {})["visibility"] = _pack(vis, _nan_std_grid())
            tiles_out["amplitude"]["contrast"] = _pack(con, _nan_std_grid())

        if "grain" in groups:
            gt = tiles_out.setdefault("grain", {})
            for k, arr in g3.items():
                gt[k] = _pack(arr, _nan_std_grid())

        if "stats" in groups and stats3 is not None:
            st = tiles_out.setdefault("stats", {})
            for k, arr in stats3.items():
                st[k] = _pack(arr, _nan_std_grid())

        if "bandwidth" in groups:
            bt = tiles_out.setdefault("bandwidth", {})
            for k, arr in bw3.items():
                bt[k] = _pack(arr, _nan_std_grid())

    else:
        raise RuntimeError(f"Unknown tiling mode: {mode!r}")

    if tiles_out:
        out["tiles"] = tiles_out
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


def _split_edges(length: int, n_parts: int) -> np.ndarray:
    return np.linspace(0, int(length), int(n_parts) + 1, dtype=int)


def _choose_tiling_mode(h: int, w: int, *, min_tile_px: int = 128) -> str:
    if h < 1 or w < 1:
        return "off"

    if (h // 9) >= min_tile_px and (w // 9) >= min_tile_px:
        return "subtiles_9x9"

    if (h // 3) >= min_tile_px and (w // 3) >= min_tile_px:
        return "tiles_3x3"

    return "off"


def _aggregate_subtiles(sub: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    sub = np.asarray(sub, dtype=float)
    mean = np.empty((3, 3), dtype=float)
    std = np.empty((3, 3), dtype=float)
    for by in range(3):
        for bx in range(3):
            block = sub[by * 3 : (by + 1) * 3, bx * 3 : (bx + 1) * 3].ravel()
            mean[by, bx] = float(np.mean(block))
            std[by, bx] = float(np.std(block, ddof=0))
    return mean, std
