# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2026 ESRF - the European Synchrotron

"""
speckle field metrics
"""
from __future__ import annotations

from typing import Literal, Sequence

import warnings

import numpy as np

from barc4dip.metrics.statistics import distribution_moments

from barc4dip.maths.radial import radial_mean_binned, radial_mean_interpolated
from barc4dip.maths.stats import distance_at_fraction_from_peak, width_at_fraction
from barc4dip.signal.fft import psd2d
from barc4dip.signal.corr import autocorr2d
from barc4dip.utils.range import percentile_minmax_range


def grain(
    image: np.ndarray,
    *,
    remove_mean: bool = True,
    standardize: bool = False,
    fraction: float = 1.0 / np.e,
    radial_method: Literal["binned", "interpolated"] = "binned",
    return_autocorr: bool = False,
) -> dict[str, float] | tuple[dict[str, float], dict[str, np.ndarray]]:
    """
    Compute speckle grain metrics from the autocorrelation peak (pixels only).

    Parameters:
        image (np.ndarray):
            2D speckle intensity image. Crop the ROI before calling.
        remove_mean (bool):
            If True, subtracts the patch mean before autocorrelation (default: True).
        standardize (bool):
            If True, computes correlation on (I - mean) / std (default: False).
        fraction (float):
            Threshold fraction for widths/radius (default: 1/e).
        radial_method (Literal["binned", "interpolated"]):
            Radial averaging method for leq (default: "binned").
        return_autocorr (bool):
            If True, also returns an autocorrelation dictionary with pixel lag axes (default: False).

    Returns:
        dict[str, float]:
            Dictionary with keys:
                - lx: 1/e full width along x cut (pixels).
                - ly: 1/e full width along y cut (pixels).
                - leq: 1/e radius of the radially averaged autocorrelation (pixels).
                - r: anisotropy ratio lx / ly.
        tuple[dict[str, float], dict[str, np.ndarray]]:
            If return_autocorr is True, returns (metrics, ac_dict) where ac_dict contains:
                - ac: 2D autocorrelation map (peak-normalized, shifted).
                - xlag: 1D lag axis for x (pixels).
                - ylag: 1D lag axis for y (pixels).

    Raises:
        ValueError
    """
    img = np.asarray(image, dtype=float)
    if img.ndim != 2:
        raise ValueError("image must be a 2D array.")
    if min(img.shape) < 8:
        raise ValueError("image too small for speckle grain metrics (min dimension < 8).")

    ac, xlag, ylag = autocorr2d(
        img,
        dx=1.0,
        dy=1.0,
        remove_mean=remove_mean,
        standardize=standardize,
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
    }

    if return_autocorr:
        ac_dict = {
            "ac": np.asarray(ac, dtype=float),
            "xlag": np.asarray(xlag, dtype=float),
            "ylag": np.asarray(ylag, dtype=float),
        }
        return metrics, ac_dict

    return metrics


def visibility(image: np.ndarray) -> float:
    """
    Compute speckle visibility (speckle contrast) as std/mean.

    Parameters:
        image (np.ndarray):
            2D speckle intensity image.

    Returns:
        float:
            Visibility (dimensionless), defined as nanstd(image) / nanmean(image).

    Raises:
        ValueError:
            If image is not 2D or if the mean intensity is not positive/finite.
    """
    img = np.asarray(image, dtype=float)
    if img.ndim != 2:
        raise ValueError("image must be a 2D array.")

    mu = float(np.nanmean(img))
    if not np.isfinite(mu) or mu <= 0.0:
        raise ValueError("Mean intensity must be positive and finite to compute visibility.")

    sigma = float(np.nanstd(img))
    return sigma / mu


def michelson_contrast(image: np.ndarray) -> float:
    """
    Compute a robust Michelson contrast using percentile-based min/max.

    Parameters:
        image (np.ndarray):
            2D speckle intensity image.

    Returns:
        float:
            Michelson contrast (dimensionless), defined as (I_high - I_low) / (I_high + I_low),
            where (I_low, I_high) come from percentile_minmax_range().

    Raises:
        ValueError:
            If image is not 2D or if the denominator is not positive/finite.
    """
    img = np.asarray(image, dtype=float)
    if img.ndim != 2:
        raise ValueError("image must be a 2D array.")

    vmin, vmax = percentile_minmax_range(img)
    denom = vmax + vmin
    if not np.isfinite(denom) or denom <= 0.0:
        raise ValueError("Invalid percentile range for Michelson contrast (non-positive/finite denominator).")

    return (vmax - vmin) / denom


def range_over_mean_contrast(image: np.ndarray) -> float:
    """
    Compute a robust range-over-mean contrast using percentile-based min/max.

    Parameters:
        image (np.ndarray):
            2D speckle intensity image.

    Returns:
        float:
            Range-over-mean contrast (dimensionless), defined as (I_high - I_low) / (2 * mean),
            where (I_low, I_high) come from percentile_minmax_range() and mean is nanmean(image).

    Raises:
        ValueError:
            If image is not 2D or if mean intensity is not positive/finite.
    """
    img = np.asarray(image, dtype=float)
    if img.ndim != 2:
        raise ValueError("image must be a 2D array.")

    mu = float(np.nanmean(img))
    if not np.isfinite(mu) or mu <= 0.0:
        raise ValueError("Mean intensity must be positive and finite to compute range-over-mean contrast.")

    vmin, vmax = percentile_minmax_range(img)
    return (vmax - vmin) / (2.0 * mu)


def spectral_participation_ratio(image: np.ndarray) -> float:
    """
    Compute an effective spectral degrees-of-freedom (DoF) score from the 2D PSD.

    This metric measures how broadly the speckle field occupies spatial-frequency space.
    It is computed as a participation ratio of the (normalized) power spectral density (PSD)
    and can be interpreted as an effective number of occupied spectral “modes” (frequency bins).

    Definition
    ----------
    Let P be the 2D PSD of the image and p_i = P_i / sum(P) its normalization over all bins.
    The spectral participation ratio is:

        SPR = 1 / sum_i (p_i^2)

    This quantity is commonly referred to as the (inverse) participation ratio (IPR) in signal
    processing and physics. It increases when spectral power is spread across many bins and
    decreases when power is concentrated in a small subset of frequencies.

    Practical interpretation (for speckle tracking)
    -----------------------------------------------
    - Higher SPR: richer/broader spatial-frequency content (more independent spectral components).
    - Lower SPR: narrowband/smoother texture (spectral power concentrated near a few frequencies).

    Mean removal and DC handling
    ----------------------------
    The image mean is removed before computing the PSD to reduce sensitivity to offsets and slow
    background pedestals. The DC bin (zero frequency) is also set to zero as a safeguard against
    residual numerical DC.

    Parameters:
        image (np.ndarray):
            2D speckle intensity image.

    Returns:
        float:
            Spectral participation ratio (dimensionless), an effective spectral DoF score.

    Raises:
        ValueError:
            If image is not 2D, or if the PSD energy after DC removal is not positive/finite.
    """
    img = np.asarray(image, dtype=float)
    if img.ndim != 2:
        raise ValueError("image must be a 2D array.")

    mu = float(np.nanmean(img))
    if not np.isfinite(mu):
        raise ValueError("image mean is not finite.")

    img = img - mu

    P, _fx, _fy = psd2d(img, dx=1.0, dy=1.0, scale=True)

    P = np.asarray(P, dtype=float)
    if P.ndim != 2:
        raise ValueError("psd2d returned a non-2D PSD (unexpected).")

    ny, nx = P.shape
    P = P.copy()
    P[ny // 2, nx // 2] = 0.0

    P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    total = float(np.sum(P))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("PSD energy is not positive/finite after mean removal and DC removal.")

    p = P.ravel() / total
    denom = float(np.sum(p * p))
    if not np.isfinite(denom) or denom <= 0.0:
        raise ValueError("Invalid participation denominator (unexpected).")

    return 1.0 / denom


def bandwidth(image: np.ndarray) -> dict[str, float]:
    """
    Compute spatial-frequency bandwidth metrics from the 2D PSD (pixel domain).

    This function characterizes how "broad" the speckle texture is in spatial-frequency space.
    All frequency quantities are returned in cycles/pixel (pixel domain), which is the relevant
    unit for speckle tracking performance (feature size in pixels, window sizing, etc.).

    Metrics
    -------
    feq : float
        Equivalent (RMS) radial bandwidth, computed from the 2D PSD as:

            feq = sqrt( sum( f^2 * P ) / sum(P) )

        where f = sqrt(fx^2 + fy^2) and P is the PSD.

    f95 : float
        Encircled-energy radius in frequency space: the radial frequency such that 95% of the
        PSD energy is contained within f <= f95 (computed from the 2D PSD, not from a radial mean).

    sig_fx, sig_fy : float
        RMS bandwidth along x and y:

            sig_fx = sqrt( sum( fx^2 * P ) / sum(P) )
            sig_fy = sqrt( sum( fy^2 * P ) / sum(P) )

    rf : float
        Spectral anisotropy ratio:

            rf = sig_fx / sig_fy

    Mean removal and DC handling
    ----------------------------
    The image mean is removed before computing the PSD to reduce sensitivity to offsets and
    slow background pedestals. The DC bin (zero frequency) is set to zero as a safeguard
    against residual numerical DC.

    Parameters:
        image (np.ndarray):
            2D speckle intensity image.

    Returns:
        dict[str, float]:
            Dictionary with keys:
                - "feq": equivalent radial bandwidth (cycles/pixel)
                - "f95": 95% encircled-energy radius (cycles/pixel)
                - "sig_fx": RMS bandwidth along fx (cycles/pixel)
                - "sig_fy": RMS bandwidth along fy (cycles/pixel)
                - "rf": anisotropy ratio sig_fx / sig_fy

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

    return {
        "feq": feq,
        "f95": f95,
        "sig_fx": sig_fx,
        "sig_fy": sig_fy,
        "rf": rf,
    }

def speckle_stats(
    data: np.ndarray,
    *,
    metrics: str | Sequence[str] = "all",
    tiles: bool = False,
    min_tile_px: int = 64,
    saturation_value: float | None = 65535.0,
    eps: float = 1e-6,
    return_full_cross_correlation: bool = True,
    warn_on_tile_fallback: bool = True,
) -> dict:
    """
    Compute speckle metrics on full-frame data and, optionally, on spatial tiles.

    Full-frame metrics are always computed. If tiles=True, the function attempts an
    effective 9x9 tiling (aggregated to 3x3 cardinal regions with mean+-std
    heterogeneity), falls back to plain 3x3 if needed, or disables tiling otherwise.
    All tiling decisions and fallbacks are recorded in the returned metadata.

    Parameters:
        data (np.ndarray):
            2D image (H, W) or 3D stack (N, H, W).
        metrics (str | Sequence[str]):
            Metric group(s) to compute. Allowed groups:
                - "amplitude": visibility and Michelson contrast.
                - "grain": grain metrics (lx, ly, leq, r). Full-frame may include autocorrelation.
                - "spectral": spectral participation ratio (spr) and bandwidth metrics.
                - "stats": distribution moments.
                - "all": computes all groups (default).
        tiles (bool):
            If True, compute metrics on tiles in addition to full-frame (default: False).
        min_tile_px (int):
            User-provided minimum tile size in pixels. This can only tighten the internal
            minimum size required for the requested metrics. The effective required minimum
            is max(internal_min_tile_px, min_tile_px) (default: 64).
        saturation_value (float | None):
            Passed to distribution_moments when "stats" is requested (default: 65535.0).
        eps (float):
            Passed to distribution_moments when "stats" is requested (default: 1e-6).
        return_full_cross_correlation (bool):
            If True and "grain" is requested, return full-frame autocorrelation (default: True).
            Autocorrelation is never returned for tiles.
        warn_on_tile_fallback (bool):
            If True, issue a warning if tiling falls back from 9×9 to 3×3 or is disabled (default: True).

    Returns:
        dict:
            Dictionary with top-level keys:
                - "meta": input and layout information (including tile labels if tiles=True).
                - "tiling": tiling decision metadata (present iff tiles=True).
                - "amplitude", "grain", "spectral", "stats": present iff requested.

            For 3D input, per-frame values are returned as NumPy arrays with leading axis (N, ...).
            For 2D input, the function behaves as N=1 and still returns arrays of shape (1, ...).

            Tile values are returned with shape (N, 3, 3). When used="9x9", tile outputs are
            dictionaries with "mean" and "std" (heterogeneity from sub-tiles). When used="3x3",
            "std" is None.

    Raises:
        TypeError:
            If data is not a NumPy array.
        ValueError:
            If data is not 2D or 3D.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("speckle_stats expects a numpy.ndarray")

    if data.ndim == 2:
        stack = data[None, ...]
    elif data.ndim == 3:
        stack = data
    else:
        raise ValueError(f"Expected 2D or 3D array, got ndim={data.ndim}")

    n_frames, h, w = stack.shape

    groups = _normalize_metric_groups(metrics)

    if "grain" in groups:
        internal_min = 128
        limiting_group = "grain"
    elif "spectral" in groups:
        internal_min = 64
        limiting_group = "spectral"
    else:
        internal_min = 32
        limiting_group = "amplitude/stats"

    if min_tile_px < 1:
        raise ValueError("min_tile_px must be >= 1")

    effective_min = max(int(internal_min), int(min_tile_px))

    out: dict = {
        "meta": {
            "input_shape": tuple(data.shape),
            "stack_shape": (int(n_frames), int(h), int(w)),
            "n_frames": int(n_frames),
            "requested_groups": sorted(groups),
        }
    }

    tiling_meta = None
    tile_mode = "off"
    used_subtiles = False

    if tiles:
        tile_labels = np.array(
            [["NW", "N", "NE"], ["W", "C", "E"], ["SW", "S", "SE"]],
            dtype=object,
        )
        out["meta"].update(
            {
                "tile_grid_shape": (3, 3),
                "tile_labels": tile_labels,
                "tile_order": "row-major",
            }
        )

        tile_mode, tile_shape_px, used_subtiles, warning_msg = _choose_tiling_mode(
            h,
            w,
            effective_min_tile_px=effective_min,
        )

        tiling_meta = {
            "requested": "9x9",
            "used": tile_mode,
            "internal_min_tile_px": int(internal_min),
            "user_min_tile_px": int(min_tile_px),
            "effective_min_tile_px": int(effective_min),
            "tile_shape_px": tile_shape_px,
            "limiting_metric_group": limiting_group,
            "warning": warning_msg,
        }
        out["tiling"] = tiling_meta

        if warn_on_tile_fallback and warning_msg is not None:
            warnings.warn(warning_msg, RuntimeWarning, stacklevel=2)

    if "amplitude" in groups:
        out["amplitude"] = {
            "full": {
                "visibility": np.empty((n_frames,), dtype=float),
                "contrast": np.empty((n_frames,), dtype=float),
            }
        }
        if tiles and tile_mode != "off":
            out["amplitude"]["tiles"] = {
                "visibility": {"mean": np.empty((n_frames, 3, 3), dtype=float), "std": None},
                "contrast": {"mean": np.empty((n_frames, 3, 3), dtype=float), "std": None},
            }

    if "spectral" in groups:
        out["spectral"] = {
            "full": {
                "spr": np.empty((n_frames,), dtype=float),
                "bandwidth": {
                    "feq": np.empty((n_frames,), dtype=float),
                    "f95": np.empty((n_frames,), dtype=float),
                    "sig_fx": np.empty((n_frames,), dtype=float),
                    "sig_fy": np.empty((n_frames,), dtype=float),
                    "rf": np.empty((n_frames,), dtype=float),
                },
            }
        }
        if tiles and tile_mode != "off":
            out["spectral"]["tiles"] = {
                "spr": {"mean": np.empty((n_frames, 3, 3), dtype=float), "std": None},
                "bandwidth": {
                    "feq": {"mean": np.empty((n_frames, 3, 3), dtype=float), "std": None},
                    "f95": {"mean": np.empty((n_frames, 3, 3), dtype=float), "std": None},
                    "sig_fx": {"mean": np.empty((n_frames, 3, 3), dtype=float), "std": None},
                    "sig_fy": {"mean": np.empty((n_frames, 3, 3), dtype=float), "std": None},
                    "rf": {"mean": np.empty((n_frames, 3, 3), dtype=float), "std": None},
                },
            }

    if "grain" in groups:
        out["grain"] = {
            "full": {
                "lx": np.empty((n_frames,), dtype=float),
                "ly": np.empty((n_frames,), dtype=float),
                "leq": np.empty((n_frames,), dtype=float),
                "r": np.empty((n_frames,), dtype=float),
            }
        }
        if return_full_cross_correlation:
            out["grain"]["full"]["autocorr"] = None

        if tiles and tile_mode != "off":
            out["grain"]["tiles"] = {
                "lx": {"mean": np.empty((n_frames, 3, 3), dtype=float), "std": None},
                "ly": {"mean": np.empty((n_frames, 3, 3), dtype=float), "std": None},
                "leq": {"mean": np.empty((n_frames, 3, 3), dtype=float), "std": None},
                "r": {"mean": np.empty((n_frames, 3, 3), dtype=float), "std": None},
            }

    if "stats" in groups:
        out["stats"] = {"full": None}
        if tiles and tile_mode != "off":
            out["stats"]["tiles"] = None

    if tiles and tile_mode != "off" and used_subtiles:
        _allocate_tile_std(out, groups, n_frames)

    ac_maps: list[np.ndarray] = []
    xlag_ref: np.ndarray | None = None
    ylag_ref: np.ndarray | None = None

    for i in range(n_frames):
        img = stack[i]

        if "amplitude" in groups:
            out["amplitude"]["full"]["visibility"][i] = visibility(img)
            out["amplitude"]["full"]["contrast"][i] = michelson_contrast(img)

        if "spectral" in groups:
            out["spectral"]["full"]["spr"][i] = spectral_participation_ratio(img)
            bw = bandwidth(img)
            for k in ("feq", "f95", "sig_fx", "sig_fy", "rf"):
                out["spectral"]["full"]["bandwidth"][k][i] = float(bw[k])

        if "grain" in groups:
            if return_full_cross_correlation:
                g, acd = grain(img, return_autocorr=True)
                ac_maps.append(np.asarray(acd["ac"]))
                if xlag_ref is None:
                    xlag_ref = np.asarray(acd["xlag"], dtype=float)
                    ylag_ref = np.asarray(acd["ylag"], dtype=float)
            else:
                g = grain(img, return_autocorr=False)

            out["grain"]["full"]["lx"][i] = float(g["lx"])
            out["grain"]["full"]["ly"][i] = float(g["ly"])
            out["grain"]["full"]["leq"][i] = float(g["leq"])
            out["grain"]["full"]["r"][i] = float(g["r"])

        if "stats" in groups:
            d = distribution_moments(img, saturation_value=saturation_value, eps=eps)
            if out["stats"]["full"] is None:
                out["stats"]["full"] = {k: np.empty((n_frames,), dtype=float) for k in d.keys()}
            for k, v in d.items():
                out["stats"]["full"][k][i] = float(v)

    if "grain" in groups and return_full_cross_correlation:
        if len(ac_maps) == n_frames and xlag_ref is not None and ylag_ref is not None:
            out["grain"]["full"]["autocorr"] = {
                "ac": np.stack(ac_maps, axis=0) if n_frames > 1 else ac_maps[0],
                "xlag": xlag_ref,
                "ylag": ylag_ref,
            }

    if tiles and tile_mode != "off":
        if used_subtiles:
            sub_edges_y = _split_edges(h, 9)
            sub_edges_x = _split_edges(w, 9)
            for i in range(n_frames):
                img = stack[i]
                if "amplitude" in groups:
                    vis_sub = np.empty((9, 9), dtype=float)
                    con_sub = np.empty((9, 9), dtype=float)
                if "spectral" in groups:
                    spr_sub = np.empty((9, 9), dtype=float)
                    bw_sub = {k: np.empty((9, 9), dtype=float) for k in ("feq", "f95", "sig_fx", "sig_fy", "rf")}
                if "grain" in groups:
                    g_sub = {k: np.empty((9, 9), dtype=float) for k in ("lx", "ly", "leq", "r")}
                if "stats" in groups:
                    stats_sub = None

                for ry in range(9):
                    y0, y1 = sub_edges_y[ry], sub_edges_y[ry + 1]
                    for rx in range(9):
                        x0, x1 = sub_edges_x[rx], sub_edges_x[rx + 1]
                        tile = img[y0:y1, x0:x1]

                        if "amplitude" in groups:
                            vis_sub[ry, rx] = visibility(tile)
                            con_sub[ry, rx] = michelson_contrast(tile)

                        if "spectral" in groups:
                            spr_sub[ry, rx] = spectral_participation_ratio(tile)
                            bw = bandwidth(tile)
                            for k in bw_sub:
                                bw_sub[k][ry, rx] = float(bw[k])

                        if "grain" in groups:
                            g = grain(tile, return_autocorr=False)
                            for k in g_sub:
                                g_sub[k][ry, rx] = float(g[k])

                        if "stats" in groups:
                            d = distribution_moments(tile, saturation_value=saturation_value, eps=eps)
                            if stats_sub is None:
                                stats_sub = {k: np.empty((9, 9), dtype=float) for k in d.keys()}
                            for k, v in d.items():
                                stats_sub[k][ry, rx] = float(v)

                if "amplitude" in groups:
                    m, s = _aggregate_subtiles(vis_sub)
                    out["amplitude"]["tiles"]["visibility"]["mean"][i] = m
                    out["amplitude"]["tiles"]["visibility"]["std"][i] = s
                    m, s = _aggregate_subtiles(con_sub)
                    out["amplitude"]["tiles"]["contrast"]["mean"][i] = m
                    out["amplitude"]["tiles"]["contrast"]["std"][i] = s

                if "spectral" in groups:
                    m, s = _aggregate_subtiles(spr_sub)
                    out["spectral"]["tiles"]["spr"]["mean"][i] = m
                    out["spectral"]["tiles"]["spr"]["std"][i] = s
                    for k in bw_sub:
                        m, s = _aggregate_subtiles(bw_sub[k])
                        out["spectral"]["tiles"]["bandwidth"][k]["mean"][i] = m
                        out["spectral"]["tiles"]["bandwidth"][k]["std"][i] = s

                if "grain" in groups:
                    for k in g_sub:
                        m, s = _aggregate_subtiles(g_sub[k])
                        out["grain"]["tiles"][k]["mean"][i] = m
                        out["grain"]["tiles"][k]["std"][i] = s

                if "stats" in groups and stats_sub is not None:
                    if out["stats"]["tiles"] is None:
                        out["stats"]["tiles"] = {k: {"mean": np.empty((n_frames, 3, 3), dtype=float), "std": np.empty((n_frames, 3, 3), dtype=float)} for k in stats_sub.keys()}
                    for k in stats_sub:
                        m, s = _aggregate_subtiles(stats_sub[k])
                        out["stats"]["tiles"][k]["mean"][i] = m
                        out["stats"]["tiles"][k]["std"][i] = s

        else:
            edges_y = _split_edges(h, 3)
            edges_x = _split_edges(w, 3)

            for i in range(n_frames):
                img = stack[i]
                for ty in range(3):
                    y0, y1 = edges_y[ty], edges_y[ty + 1]
                    for tx in range(3):
                        x0, x1 = edges_x[tx], edges_x[tx + 1]
                        tile = img[y0:y1, x0:x1]

                        if "amplitude" in groups:
                            out["amplitude"]["tiles"]["visibility"]["mean"][i, ty, tx] = visibility(tile)
                            out["amplitude"]["tiles"]["contrast"]["mean"][i, ty, tx] = michelson_contrast(tile)

                        if "spectral" in groups:
                            out["spectral"]["tiles"]["spr"]["mean"][i, ty, tx] = spectral_participation_ratio(tile)
                            bw = bandwidth(tile)
                            for k in ("feq", "f95", "sig_fx", "sig_fy", "rf"):
                                out["spectral"]["tiles"]["bandwidth"][k]["mean"][i, ty, tx] = float(bw[k])

                        if "grain" in groups:
                            g = grain(tile, return_autocorr=False)
                            out["grain"]["tiles"]["lx"]["mean"][i, ty, tx] = float(g["lx"])
                            out["grain"]["tiles"]["ly"]["mean"][i, ty, tx] = float(g["ly"])
                            out["grain"]["tiles"]["leq"]["mean"][i, ty, tx] = float(g["leq"])
                            out["grain"]["tiles"]["r"]["mean"][i, ty, tx] = float(g["r"])

                        if "stats" in groups:
                            d = distribution_moments(tile, saturation_value=saturation_value, eps=eps)
                            if out["stats"]["tiles"] is None:
                                out["stats"]["tiles"] = {k: {"mean": np.empty((n_frames, 3, 3), dtype=float), "std": None} for k in d.keys()}
                            for k, v in d.items():
                                out["stats"]["tiles"][k]["mean"][i, ty, tx] = float(v)

    return out


def _normalize_metric_groups(metrics: str | Sequence[str]) -> set[str]:
    if isinstance(metrics, str):
        m = metrics.strip().lower()
        if m == "all":
            return {"amplitude", "grain", "spectral", "stats"}
        return {m}
    groups = set()
    for item in metrics:
        if not isinstance(item, str):
            raise TypeError("metrics must be a str or a sequence of str")
        m = item.strip().lower()
        if m == "all":
            groups.update({"amplitude", "grain", "spectral", "stats"})
        else:
            groups.add(m)
    return groups


def _split_edges(length: int, n_parts: int) -> np.ndarray:
    return np.linspace(0, int(length), int(n_parts) + 1, dtype=int)


def _choose_tiling_mode(
    h: int,
    w: int,
    *,
    effective_min_tile_px: int,
) -> tuple[str, tuple[int, int] | None, bool, str | None]:

    edges_y_9 = _split_edges(h, 9)
    edges_x_9 = _split_edges(w, 9)
    tile_h_9 = int(np.min(np.diff(edges_y_9)))
    tile_w_9 = int(np.min(np.diff(edges_x_9)))
    if min(tile_h_9, tile_w_9) >= effective_min_tile_px:
        return "9x9", (tile_h_9, tile_w_9), True, None


    edges_y_3 = _split_edges(h, 3)
    edges_x_3 = _split_edges(w, 3)
    tile_h_3 = int(np.min(np.diff(edges_y_3)))
    tile_w_3 = int(np.min(np.diff(edges_x_3)))
    if min(tile_h_3, tile_w_3) >= effective_min_tile_px:
        msg = (
            f"tiles=True: falling back to 3x3 tiling because 9x9 subtiles are too small "
            f"(min subtile {min(tile_h_9, tile_w_9)} px < required {effective_min_tile_px} px)."
        )
        return "3x3", (tile_h_3, tile_w_3), False, msg

    msg = (
        f"tiles=True: tiling disabled because even 3x3 tiles are too small "
        f"(min tile {min(tile_h_3, tile_w_3)} px < required {effective_min_tile_px} px). "
        f"Returning full-frame metrics only."
    )
    return "off", None, False, msg


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


def _allocate_tile_std(out: dict, groups: set[str], n_frames: int) -> None:

    if "amplitude" in groups:
        out["amplitude"]["tiles"]["visibility"]["std"] = np.empty((n_frames, 3, 3), dtype=float)
        out["amplitude"]["tiles"]["contrast"]["std"] = np.empty((n_frames, 3, 3), dtype=float)

    if "spectral" in groups:
        out["spectral"]["tiles"]["spr"]["std"] = np.empty((n_frames, 3, 3), dtype=float)
        for k in out["spectral"]["tiles"]["bandwidth"].keys():
            out["spectral"]["tiles"]["bandwidth"][k]["std"] = np.empty((n_frames, 3, 3), dtype=float)

    if "grain" in groups:
        for k in ("lx", "ly", "leq", "r"):
            out["grain"]["tiles"][k]["std"] = np.empty((n_frames, 3, 3), dtype=float)

