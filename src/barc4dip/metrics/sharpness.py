# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2026 ESRF - the European Synchrotron
"""
Sharpness metrics.

Refer to:
S. Pertuz, D. Puig, and M. A. Garcia, 
"Analysis of focus measure operators for shape-from-focus," 
Pattern Recognition 46(5), 1415-1432 (2013). 
"""


from __future__ import annotations

import logging
from typing import Literal, Sequence, Union

import numpy as np
from scipy import ndimage

from ..maths.radial import radial_mean_binned, radial_mean_interpolated
from ..maths.stats import distance_at_fraction_from_peak, width_at_fraction
from ..signal.corr import autocorr2d
from ..signal.fft import psd2d
from ..utils import elapsed_time, now
from .common import (
    apply_display_origin,
    choose_tiling_mode,
    tiled_scalar_fields,
    tiles_meta,
)
from .statistics import distribution_moments

logger = logging.getLogger(__name__)

def sharpness_stats(
    image: np.ndarray,
    *,
    metrics: str | Sequence[str] = "all",
    tiles: bool = False,
    display_origin: Literal["upper", "lower"] = "lower",
    saturation_value: float | None = 65535.0,
    eps: float = 1e-6,
    verbose: bool = False,
) -> dict:
    """
    Compute sharpness metrics on a single 2D image.

    This aggregator mirrors the ergonomics and return schema of speckle_stats:
        - full-frame metrics under out["full"]
        - optional 3x3 tile metrics under out["tiles"] (same mean/std grid schema)
        - tiling policy, labels/order, and display_origin handling are delegated to
          the common tiling utilities.

    Parameters:
        image (np.ndarray):
            2D intensity image (H, W).
        metrics (str | Sequence[str]):
            Metric group(s) to compute:
                - "stats"
                - "gradient"
                - "laplacian"
                - "spectral"
                - "autocorrelation"
                - "eigenvalues"
                - "all" (default)
        tiles (bool):
            If True, also compute metrics on a 3x3 grid of tiles, where each
            tile is either:
                - aggregated from 9x9 sub-tiles (mean and std per 3x3 cell), or
                - computed directly on 3x3 tiles (std returned as NaNs).
            Tiles are only computed if the implied tile size meets MIN_TILE_PX
            under the common tiling policy.
        display_origin (Literal["upper", "lower"]):
            Defines the vertical origin convention used for analysis.
                - "lower" (default): detector-aligned convention.
                - "upper": NumPy convention. Index (0, 0) is the top-left pixel.
        saturation_value (float | None):
            Passed to distribution_moments for stats (default: 65535.0).
        eps (float):
            Passed to distribution_moments for stats (default: 1e-6).
        verbose (bool):
            Verbose output for full-frame metrics only. Tile computations force
            verbose=False.

    Returns:
        dict:
            Dictionary with:
                - "meta": metadata (including tiling metadata if tiles are computed)
                - "full": dict of requested full-frame blocks
                - "tiles": dict of requested tile blocks (only if feasible)

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
        raise TypeError("sharpness_stats expects a numpy.ndarray")
    if image.ndim != 2:
        raise ValueError(f"Expected 2D array, got ndim={image.ndim}")

    # Orientation ergonomics (shared across all metric aggregators).
    image = apply_display_origin(image, display_origin=display_origin)

    h, w = image.shape
    groups = _normalize_sharpness_groups(metrics)

    if verbose:
        logger.info("\nsharpness stats for a (h x w: %.0f x %.0f) image:", h, w)

    out: dict = {
        "meta": {
            "kind": "sharpness",
            "display_origin": display_origin,
            "input_shape": (int(h), int(w)),
            "requested_groups": sorted(groups),
        },
        "full": {},
    }

    # --- full-frame ---
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

    if "gradient" in groups:
        out["full"]["gradient"] = tenengrad(image, verbose=verbose)

    if "laplacian" in groups:
        out["full"]["laplacian"] = {"laplacian_variance": laplacian_variance(image, verbose=verbose)}

    if "spectral" in groups:
        out["full"]["spectral"] = {"spectral_entropy": spectral_entropy(image, verbose=verbose)}

    if "autocorrelation" in groups:
        out["full"]["autocorrelation"] = inverse_autocorr_width(image, verbose=verbose)

    if "eigenvalues" in groups:
        out["full"]["eigenvalues"] = eigenvalues(image, verbose=verbose)

    # --- tiles (same policy as speckle_stats via common.py) ---
    MIN_TILE_PX = 128
    mode, tile_shape_px = choose_tiling_mode(h, w, tiles=tiles, min_tile_px=MIN_TILE_PX)
    if mode == "off":
        if verbose:
            elapsed_time(t0)
        return out

    out["meta"].update(tiles_meta(h, w, tile_mode=mode, tile_shape_px=tile_shape_px))

    tiles_out: dict = {}

    if "stats" in groups:
        def _stats_tile(tile: np.ndarray) -> dict[str, float]:
            try:
                d = distribution_moments(tile, saturation_value=saturation_value, eps=eps, verbose=False)
            except TypeError:
                d = distribution_moments(tile, saturation_value=saturation_value, eps=eps)
            return {k: float(v) for k, v in d.items()}

        tiles_out["stats"] = tiled_scalar_fields(image, tile_mode=mode, compute_fn=_stats_tile)

    if "gradient" in groups:
        def _grad_tile(tile: np.ndarray) -> dict[str, float]:
            g = tenengrad(tile, verbose=False)
            return {
                "tenengrad": float(g["tenengrad"]),
                "ex": float(g["ex"]),
                "ey": float(g["ey"]),
                "re": float(g["re"]),
            }

        tiles_out["gradient"] = tiled_scalar_fields(image, tile_mode=mode, compute_fn=_grad_tile)

    if "laplacian" in groups:
        def _lap_tile(tile: np.ndarray) -> dict[str, float]:
            return {"laplacian_variance": float(laplacian_variance(tile, verbose=False))}

        tiles_out["laplacian"] = tiled_scalar_fields(image, tile_mode=mode, compute_fn=_lap_tile)

    if "spectral" in groups:
        def _spec_tile(tile: np.ndarray) -> dict[str, float]:
            return {"spectral_entropy": float(spectral_entropy(tile, verbose=False))}

        tiles_out["spectral"] = tiled_scalar_fields(image, tile_mode=mode, compute_fn=_spec_tile)

    if "autocorrelation" in groups:
        def _ac_tile(tile: np.ndarray) -> dict[str, float]:
            a = inverse_autocorr_width(tile, verbose=False)
            return {
                "sx": float(a["sx"]),
                "sy": float(a["sy"]),
                "seq": float(a["seq"]),
                "r": float(a["r"]),
            }

        tiles_out["autocorrelation"] = tiled_scalar_fields(image, tile_mode=mode, compute_fn=_ac_tile)

    if "eigenvalues" in groups:
        def _eig_tile(tile: np.ndarray) -> dict[str, float]:
            e = eigenvalues(tile, verbose=False)
            return {
                "eigenvalues": float(e["eigenvalues"]),
                "e1": float(e["e1"]),
                "e2": float(e["e2"]),
                "re": float(e["re"]),
            }

        tiles_out["eigenvalues"] = tiled_scalar_fields(image, tile_mode=mode, compute_fn=_eig_tile)

    if tiles_out:
        out["tiles"] = tiles_out

    if verbose:
        elapsed_time(t0)

    return out


def _normalize_sharpness_groups(metrics: str | Sequence[str]) -> set[str]:
    if isinstance(metrics, str):
        m = metrics.strip().lower()
        if m == "all":
            return {"stats", "gradient", "laplacian", "spectral", "autocorrelation", "eigenvalues"}
        return {m}

    groups: set[str] = set()
    for item in metrics:
        if not isinstance(item, str):
            raise TypeError("metrics must be a str or a sequence of str")
        m = item.strip().lower()
        if m == "all":
            groups.update({"stats", "gradient", "laplacian", "spectral", "autocorrelation", "eigenvalues"})
        else:
            groups.add(m)
    return groups

def tenengrad(
    image: np.ndarray,
    *,
    eps: float = 1e-12,
    verbose: bool = False,
) -> dict:
    """
    (GRA6)
    Compute Tenengrad sharpness and directional gradient energies.

    This metric is based on the Sobel gradient energy. It returns the mean
    squared gradient components along x and y, their sum (Tenengrad), and an
    anisotropy ratio suitable as a simple astigmatism proxy.

    Parameters:
        image (np.ndarray):
            2D intensity image.
        eps (float):
            Small value added to the denominator when computing the anisotropy
            ratio re = ex / (ey + eps) (default: 1e-12).
        verbose (bool):
            If True, emit a concise, human-readable summary via the logging
            subsystem at INFO level. Default is False.

    Returns:
        Dictionary with keys:
            - tenengrad: Mean of (Gx^2 + Gy^2).
            - ex: Mean of Gx^2 (horizontal gradient energy).
            - ey: Mean of Gy^2 (vertical gradient energy).
            - re: Anisotropy ratio ex / (ey + eps).

    Raises:
        ValueError:
            If image is empty, has ndim != 2, or contains no finite values.
    """
    data = np.asarray(image)

    if data.ndim != 2:
        raise ValueError(f"Expected 2D array, got ndim={data.ndim}")

    if data.size == 0:
        raise ValueError("tenengrad received an empty image.")

    finite = np.isfinite(data)
    if not np.any(finite):
        raise ValueError("tenengrad received image with no finite values.")

    x = np.asarray(data, dtype=float)

    gx = ndimage.sobel(x, axis=1, mode="reflect")
    gy = ndimage.sobel(x, axis=0, mode="reflect")

    gx2 = gx * gx
    gy2 = gy * gy

    ex = float(np.mean(gx2[finite]))
    ey = float(np.mean(gy2[finite]))
    ten = float(ex + ey)
    re = float(ex / (ey + float(eps)))

    out = {"tenengrad": ten, "ex": ex, "ey": ey, "re": re}

    if verbose:
        logger.info(
            "> tenengrad: %.6g | ex: %.6g | ey: %.6g | ex/ey: %.3f",
            out["tenengrad"],
            out["ex"],
            out["ey"],
            out["re"],
        )

    return out

def laplacian_variance(
    image: np.ndarray,
    *,
    verbose: bool = False,
) -> float:
    """
    (LAP4)
    Compute the population variance of the Laplacian of a 2D intensity image.

    This is a classic focus/sharpness operator (often referred to as "variance of
    Laplacian"). It highlights rapid intensity changes via the Laplacian and
    measures the dispersion of those values. Only finite values are considered.

    Parameters:
        image (np.ndarray):
            2D intensity image.
        verbose (bool):
            If True, emit a concise, human-readable summary via the logging
            subsystem at INFO level. Default is False.

    Returns:
        float:
            Population variance (ddof=0) of the Laplacian values.

    Raises:
        ValueError:
            If image is empty, has ndim != 2, or contains no finite values.
    """
    data = np.asarray(image)

    if data.ndim != 2:
        raise ValueError(f"Expected 2D array, got ndim={data.ndim}")

    if data.size == 0:
        raise ValueError("laplacian_variance received an empty image.")

    finite = np.isfinite(data)
    if not np.any(finite):
        raise ValueError("laplacian_variance received image with no finite values.")

    x = np.asarray(data, dtype=float)
    lap = ndimage.laplace(x, mode="reflect")

    var = float(np.var(lap[finite], ddof=0))

    if verbose:
        logger.info("> laplacian variance: %.6g", var)

    return var


def spectral_entropy(
    image: np.ndarray,
    *,
    remove_mean: bool = True,
    remove_dc: bool = True,
    eps: float = 1e-30,
    verbose: bool = False,
) -> float:
    """
    (not really in the paper, but this is the Shannon Entropy applied to the PSD)
    Compute the normalized spectral entropy of a 2D intensity image.

    The power spectral density (PSD) is normalized into a discrete probability
    distribution p_i over frequency bins, and Shannon entropy is computed:

        H = -sum(p_i * log(p_i))

    The result is normalized by log(M), where M is the number of bins used
    (after optional DC removal), yielding a value in [0, 1] in typical cases.

    Parameters:
        image (np.ndarray):
            2D intensity image. Must contain only finite values.
        remove_mean (bool):
            If True, subtract the image mean before computing the PSD. This
            reduces DC dominance (default: True).
        remove_dc (bool):
            If True, exclude the DC bin from the entropy computation
            (default: True).
        eps (float):
            Small positive value used to avoid log(0). Probabilities are
            clipped to >= eps (default: 1e-30).
        verbose (bool):
            If True, emit a concise summary via the logging subsystem
            at INFO level. Default is False.

    Returns:
        float:
            Normalized spectral entropy (dimensionless).

    Raises:
        ValueError:
            If image is empty, has ndim != 2, contains non-finite values,
            or has insufficient spectral support to compute entropy.
    """
    data = np.asarray(image)

    if data.ndim != 2:
        raise ValueError(f"Expected 2D array, got ndim={data.ndim}")
    if data.size == 0:
        raise ValueError("spectral_entropy received an empty image.")
    if not np.all(np.isfinite(data)):
        raise ValueError("spectral_entropy requires all values to be finite.")

    x = np.asarray(data, dtype=float)
    if remove_mean:
        x = x - float(np.mean(x))

    P, _fx, _fy = psd2d(x, scale=False)
    P = np.asarray(P, dtype=float)

    if np.any(P < 0):
        raise ValueError("psd2d returned negative PSD values (unexpected).")

    if remove_dc:
        cy = P.shape[0] // 2
        cx = P.shape[1] // 2
        P = P.copy()
        P[cy, cx] = 0.0

    s = float(np.sum(P))
    if not np.isfinite(s) or s <= 0.0:
        raise ValueError("PSD sum is non-positive; cannot compute spectral entropy.")

    p = P.ravel() / s

    if remove_dc:
        M = int(p.size - 1)
    else:
        M = int(p.size)

    if M < 2:
        raise ValueError("Insufficient number of spectral bins to compute normalized entropy.")

    p = np.clip(p, float(eps), None)
    H = float(-np.sum(p * np.log(p)))
    Hn = float(H / np.log(float(M)))

    if verbose:
        logger.info("> spectral_entropy: %.6g", Hn)

    return Hn


def inverse_autocorr_width(
    image: np.ndarray,
    *,
    fraction: float = 1.0 / np.e,
    radial_method: Literal["binned", "interpolated"] = "interpolated",
    min_size_px: int = 32,
    verbose: bool = False,
) -> dict:
    """
    Compute sharpness metrics from the inverse width of the normalized autocorrelation peak.

    This metric estimates correlation lengths from the 2D autocorrelation of the image
    (with mean removal and standardization), then returns inverse widths so that larger
    values indicate sharper images.

    Parameters:
        image (np.ndarray):
            2D intensity image.
        fraction (float):
            Threshold fraction for width estimation (default: 1/e).
        radial_method (Literal["binned", "interpolated"]):
            Radial averaging method for the equivalent width (default: "interpolated").
        min_size_px (int):
            Minimum allowed image size (min(H, W)) to compute the metric (default: 32).
        verbose (bool):
            If True, emit a concise, human-readable summary via the logging
            subsystem at INFO level. Default is False.

    Returns:
        Dictionary with keys:
            - sx: Inverse full width along x cut at `fraction` (1/pixels).
            - sy: Inverse full width along y cut at `fraction` (1/pixels).
            - seq: Inverse equivalent radial width at `fraction` (1/pixels).
            - r: Anisotropy ratio in width domain (lx / ly), matching grain() convention.

    Raises:
        ValueError:
            If image is not 2D, is empty, too small, or if widths cannot be estimated.
    """
    data = np.asarray(image, dtype=float)
    if data.ndim != 2:
        raise ValueError("image must be a 2D array.")
    if data.size == 0:
        raise ValueError("inverse_autocorr_width received an empty image.")
    if min(data.shape) < int(min_size_px):
        raise ValueError(
            f"image too small for inverse autocorrelation width "
            f"(min dimension < {int(min_size_px)})."
        )

    ac, _xlag, _ylag = autocorr2d(
        data,
        dx=1.0,
        dy=1.0,
        remove_mean=True,
        standardize=True,
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
        rad, r = radial_mean_interpolated(ac)
    elif radial_method == "interpolated":
        rad, r = radial_mean_interpolated(ac)
    else:
        raise ValueError("radial_method must be 'binned' or 'interpolated'.")

    rad = np.asarray(rad, dtype=float)
    r = np.asarray(r, dtype=float)
    if rad.size < 2 or r.size < 2:
        raise ValueError("Radial profile is too short to estimate equivalent width.")

    dr = float(r[1] - r[0])
    if dr <= 0.0:
        raise ValueError("Invalid radial sampling (non-positive dr).")

    dist_bins, _hit_edge = distance_at_fraction_from_peak(rad, fraction=fraction, peak_index=0)
    leq_px = 2.0 * float(dist_bins) * dr

    lx_px = float(lx_px)
    ly_px = float(ly_px)
    leq_px = float(leq_px)

    r_aniso = float(lx_px / ly_px) if ly_px != 0.0 else float("inf")

    out = {
        "sx": float(1.0 / lx_px) if lx_px != 0.0 else float("inf"),
        "sy": float(1.0 / ly_px) if ly_px != 0.0 else float("inf"),
        "seq": float(1.0 / leq_px) if leq_px != 0.0 else float("inf"),
        "r": r_aniso,
    }

    if verbose:
        logger.info(
            "> inv_ac_width: sx=%.4g | sy=%.4g | sx/sy=%.3g | seq=%.4g | r(lx/ly)=%.3g",
            out["sx"],
            out["sy"],
            float(out["sx"] / out["sy"]) if np.isfinite(out["sy"]) and out["sy"] != 0.0 else float("inf"),
            out["seq"],
            out["r"],
        )

    return out


def eigenvalues(
    image: np.ndarray,
    *,
    k: int = 5,
    eps: float = 1e-30,
    verbose: bool = False,
) -> dict:
    """
    (STA2)
    Compute a PCA/eigenvalues-based sharpness metric.

    This implements the STA2 focus measure described by Pertuz et al. The method is 
    based on the trace of the first k eigenvalues of a covariance matrix derived from an 
    energy-normalized, mean-removed image.

    Procedure:
        1) Energy normalization:
               I_tilde = I / sqrt(sum(I^2))
        2) Mean removal:
               J = I_tilde - mean(I_tilde)
        3) Covariance matrix:
               S = (J J^T) / (M*N - 1)
           where the image has shape (M, N).
        4) Sharpness measure:
               sta2 = sum of the first k eigenvalues of S.

    Eigenvalues are computed via singular value decomposition (SVD) of J,
    which is numerically equivalent to eigen-decomposition of S.

    This metric is invariant to global intensity scaling and captures
    structured variance in the image. Larger values indicate sharper images.

    Parameters:
        image (np.ndarray):
            2D intensity image. Must contain only finite values.
        k (int):
            Number of leading eigenvalues to sum (default: 5).
            Too small -> noise-dominated
            Too large -> oversmoothing, anisotropy washed out
            When applying it to the speckle fields, choose k~grain/2 
        eps (float):
            Small positive number used to guard divisions and degenerate cases
            (default: 1e-30).
        verbose (bool):
            If True, emit a concise, human-readable summary via the logging
            subsystem at INFO level. Default is False.

    Returns:
        Dictionary with keys:
            - eigenvalues: Sum of the first k eigenvalues of S.
            - e1: Largest eigenvalue.
            - e2: Second-largest eigenvalue (or 0 if unavailable).
            - re: Anisotropy proxy e1 / (e2 + eps).

    Raises:
        ValueError:
            If image is empty, has ndim != 2, contains non-finite values,
            if k < 1, or if image energy is zero.
    """
    data = np.asarray(image)

    if data.ndim != 2:
        raise ValueError(f"Expected 2D array, got ndim={data.ndim}")
    if data.size == 0:
        raise ValueError("eigenvalues received an empty image.")
    if not np.all(np.isfinite(data)):
        raise ValueError("eigenvalues requires all values to be finite.")

    k_eff = int(k)
    if k_eff < 1:
        raise ValueError("k must be >= 1.")

    x = np.asarray(data, dtype=float)

    energy = float(np.sqrt(np.sum(x * x)))
    if not np.isfinite(energy) or energy <= 0.0:
        raise ValueError("eigenvalues cannot normalize an all-zero image.")

    x_tilde = x / energy

    J = x_tilde - float(np.mean(x_tilde))

    M, N = J.shape
    denom = float(M * N - 1)
    if denom <= 0.0:
        raise ValueError("eigenvalues requires at least 2 pixels (M*N >= 2).")

    s = np.linalg.svd(J, full_matrices=False, compute_uv=False)
    eig = (s * s) / denom

    k_use = min(k_eff, int(eig.size))
    val = float(np.sum(eig[:k_use]))

    e1 = float(eig[0]) if eig.size >= 1 else 0.0
    e2 = float(eig[1]) if eig.size >= 2 else 0.0
    re = float(e1 / (e2 + float(eps)))

    out = {"eigenvalues": val, "e1": e1, "e2": e2, "re": re}

    if verbose:
        logger.info(
            "> eigenvalues: %.6g | e1: %.6g | e2: %.6g | e1/e2: %.3f | k=%d",
            out["eigenvalues"],
            out["e1"],
            out["e2"],
            out["re"],
            k_use,
        )

    return out
