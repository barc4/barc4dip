# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2026 ESRF - the European Synchrotron

"""
speckle field metrics
"""
from __future__ import annotations

from typing import Literal

import numpy as np

from barc4dip.math.radial import radial_mean_binned, radial_mean_interpolated
from barc4dip.math.stats import distance_at_fraction_from_peak, width_at_fraction
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
    leq_px = float(dist_bins) * dr

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
            "ac": np.asarray(ac),
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
