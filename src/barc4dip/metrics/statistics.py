# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2026 ESRF - the European Synchrotron

"""
Statistical metrics.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.stats import describe

logger = logging.getLogger(__name__)

def distribution_moments(
    image: np.ndarray,
    *,
    saturation_value: float | None = 65535.0,
    eps: float = 1e-6,
    verbose: bool = False,
) -> dict[str, float]:
    """
    Compute basic intensity distribution moments and simple diagnostics.

    This function operates on real-valued data (typically floats) and assumes
    scaling and dtype conversion are handled elsewhere in the pipeline.

    Parameters:
        image (np.ndarray):
            1D or 2D array of intensities. Values are flattened and treated as
            an i.i.d. sample (finite values only).
        saturation_value (float | None):
            Reference saturation level. If None, frac_sat is set to NaN
            (default: 65535.0).
        eps (float):
            Tolerance used to detect (near-)zero-valued pixels. Pixels with
            |value| <= eps are counted as zero (default: 1e-6).
        verbose (bool):
            If True, emit a concise, human-readable summary via the logging
            subsystem at INFO level. Default is False.

    Returns:
        dict[str, float]:
            Dictionary with keys:
                - mean: Mean intensity (finite values only).
                - variance: Intensity variance.
                - skewness: Skewness of the intensity distribution.
                - kurtosis: Kurtosis of the intensity distribution.
                - frac_zero: Fraction of finite pixels with |value| <= eps.
                - frac_sat: Fraction of finite pixels >= saturation_value,
                  or NaN if saturation_value is None.
                - SNRdB: Signal-to-noise proxy defined as 20*log10(mean / std).

    Raises:
        ValueError:
            If image is empty, has ndim not in {1, 2}, or contains no finite values.
    """
    data = np.asarray(image)
    if data.ndim not in (1, 2):
        raise ValueError(f"Expected 1D or 2D array, got ndim={image.ndim}")

    if data.size == 0:
        raise ValueError("distribution_moments received an empty image.")

    x = np.asarray(data, dtype=np.float64).ravel()
    finite = np.isfinite(x)

    if not np.any(finite):
        raise ValueError("distribution_moments received no finite values.")

    x = x[finite]

    desc = describe(x, axis=None)
    mean = float(desc.mean)
    variance = float(desc.variance)
    skewness = float(desc.skewness)
    kurtosis = float(desc.kurtosis)

    frac_zero = float(np.mean(np.abs(x) <= eps))

    if variance == 0.0:
        snr_db = float("inf") if mean > 0.0 else float("nan")
    else:
        snr_linear = mean / np.sqrt(variance)
        if snr_linear > 0.0:
            snr_db = float(20.0 * np.log10(snr_linear))
        elif snr_linear == 0.0:
            snr_db = float("-inf")
        else:
            snr_db = float("nan")

    if saturation_value is None:
        frac_sat = float("nan")
    else:
        frac_sat = float(np.mean(x >= float(saturation_value)))

    moments = {
        "mean": mean,
        "variance": variance,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "frac_zero": frac_zero,
        "frac_sat": frac_sat,
        "SNRdB": snr_db,
    }

    if verbose:
        std = float(np.sqrt(variance)) if variance >= 0.0 else float("nan")
        logger.info(
            "> moments: mean=%.0f | std=%.0f | skew=%.2f | kurt=%.2f | SNR=%.2f dB | zero=%.6f | sat=%.6f",
            moments["mean"],
            std,
            moments["skewness"],
            moments["kurtosis"],
            moments["SNRdB"],
            moments["frac_zero"],
            moments["frac_sat"],
        )
        
    return moments

