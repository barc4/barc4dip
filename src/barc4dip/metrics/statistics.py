# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2026 ESRF - the European Synchrotron

"""
Statistical metrics.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import describe

def distribution_moments(
    image: np.ndarray,
    *,
    saturation_value: float | None = 65535.0,
    eps: float = 1e-6,
) -> dict[str, float]:
    """
    Compute basic intensity distribution moments and simple diagnostics.

    This function operates on real-valued images (typically floats) and assumes
    scaling and dtype conversion are handled elsewhere in the pipeline.

    Parameters:
        image (np.ndarray):
            2D image (or any-shape array). Values are treated as a flat sample
            of intensities.
        saturation_value (float | None):
            Reference saturation level. If None, frac_sat is set to NaN
            (default: 65535.0).
        eps (float):
            Tolerance used to detect zero-valued pixels. Pixels with
            |value| <= eps are counted as zero (default: 1e-6).

    Returns:
        dict[str, float]:
            Dictionary with keys:
                - mean: Mean intensity.
                - variance: Intensity variance.
                - skewness: Skewness of the intensity distribution.
                - kurtosis: Kurtosis of the intensity distribution.
                - frac_zero: Fraction of finite pixels with |value| <= eps.
                - frac_sat: Fraction of finite pixels >= saturation_value,
                  or NaN if saturation_value is None.
                - SNRdB: Signal-to-noise proxy defined as 20*log10(mean / std).

    Raises:
        ValueError:
            If the image is empty or contains no finite values.
    """
    if image.size == 0:
        raise ValueError("distribution_moments received an empty image.")

    x = np.asarray(image, dtype=np.float64).ravel()
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

    return {
        "mean": mean,
        "variance": variance,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "frac_zero": frac_zero,
        "frac_sat": frac_sat,
        "SNRdB": snr_db,
    }

