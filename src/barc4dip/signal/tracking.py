# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2026 ESRF - the European Synchrotron

from __future__ import annotations

from typing import Callable, Literal

import numpy as np

from ..geometry.roi import embed_roi, roi_slices


_TrackingMethod = Literal["phase"]
_Tracker = Callable[..., tuple[float, float, float, float]]
_TRACKERS: dict[str, _Tracker] = {}

def _register(method: str) -> Callable[[_Tracker], _Tracker]:
    method_norm = method.strip().lower()

    def _decorator(fn: _Tracker) -> _Tracker:
        _TRACKERS[method_norm] = fn
        return fn

    return _decorator

def track_translation(
    template: np.ndarray,
    image: np.ndarray,
    *,
    slices_yx: tuple[slice, slice] | None = None,
    method: _TrackingMethod = "phase",
    backend: Literal["internal", "skimage"] = "internal",
    subpixel: bool = True,
    eps: float = 1e-9,
) -> tuple[float, float, float, float]:
    """
    Dispatcher for translation tracking methods.

    Parameters
    ----------
    template : np.ndarray
        2D template ROI (h, w).
    image : np.ndarray
        2D full frame image (H, W).
    slices_yx : tuple[slice, slice] | None
        Location of template within image coordinates. If None, assume centered.
    method : {"phase"}
        Tracking method identifier (default: "phase").
    backend : {"internal", "skimage"}
        Backend for the selected method (if applicable).
    subpixel : bool
        Enable subpixel refinement (default: True).
    eps : float
        Numerical stability constant.

    Returns
    -------
    tuple[float, float, float, float]
        (dy, dx, peak_value, snr)

    Raises
    ------
    ValueError
        If method is not registered.
    """
    method_norm = method.strip().lower()
    fn = _TRACKERS.get(method_norm)
    if fn is None:
        supported = ", ".join(sorted(_TRACKERS))
        raise ValueError(f"Unsupported tracking method: {method!r}. Supported: {supported}")

    return fn(
        template,
        image,
        slices_yx=slices_yx,
        backend=backend,
        subpixel=subpixel,
        eps=eps,
    )


@_register("phase")
def phase_correlation(
    template: np.ndarray,
    image: np.ndarray,
    *,
    slices_yx: tuple[slice, slice] | None = None,
    backend: Literal["internal", "skimage"] = "internal",
    subpixel: bool = True,
    eps: float = 1e-9,
) -> tuple[float, float, float, float]:
    """
    Estimate translation (dy, dx) by phase correlation of a template ROI vs full image.

    Conventions
    -----------
    NumPy convention (origin at upper-left):
    - +dy is downward, +dx is rightward.
    - Returned (dy, dx) is the shift to apply to the *template* to align it to the image.

    Parameters
    ----------
    template : np.ndarray
        2D template ROI (h, w).
    image : np.ndarray
        2D full frame image (H, W).
    slices_yx : tuple[slice, slice] | None
        Location of template within image coordinates. If None, the template is
        assumed centered in the image.
    backend : Literal["internal", "skimage"]
        "internal" uses FFT phase correlation + optional Taylor refinement.
        "skimage" uses skimage.registration.phase_cross_correlation.
    subpixel : bool
        If True (default), apply subpixel refinement (Taylor). If False, return
        integer-pixel shift.
    eps : float
        Numerical stability constant.

    Returns
    -------
    tuple[float, float, float, float]
        (dy, dx, peak_value, snr)

        For backend="skimage", peak_value and snr are returned as NaN.

    Raises
    ------
    ValueError
        If inputs are not 2D, or backend is unknown.
    ImportError
        If backend="skimage" is requested but scikit-image is not available.
    """
    tpl = _as_float2d(template, name="template")
    img = _as_float2d(image, name="image")

    H, W = img.shape
    h, w = tpl.shape

    if slices_yx is None:
        slices_yx = roi_slices((H, W), (h, w), center_yx=None, clip=False)

    img_z = _zscore2d(img, eps=eps)
    tpl_z = _zscore2d(tpl, eps=eps)

    tpl_pad = embed_roi(
        tpl_z,
        out_shape=(H, W),
        slices_yx=slices_yx,
        fill_value=0.0,
        dtype=np.float32,
    )

    if backend == "skimage":
        try:
            from skimage.registration import phase_cross_correlation
        except Exception as exc:
            raise ImportError("backend='skimage' requires scikit-image.") from exc

        up = 10 if subpixel else 1
        shift_yx, _, _ = phase_cross_correlation(img_z, tpl_pad, upsample_factor=up)
        dy = float(shift_yx[0])
        dx = float(shift_yx[1])
        return dy, dx, float("nan"), float("nan")

    if backend != "internal":
        raise ValueError("backend must be 'internal' or 'skimage'.")

    Fi = np.fft.fft2(img_z)
    Ft = np.fft.fft2(tpl_pad)

    prod = Fi * np.conj(Ft)
    cps = prod / (np.abs(prod) + eps)

    corr_cplx = np.fft.ifft2(cps)
    corr = np.fft.fftshift(corr_cplx)
    mag = np.abs(corr)
    max_i, max_j = np.unravel_index(np.argmax(mag), mag.shape)
    peak, snr = _corr_peak_quality(mag, peak_ij=(max_i, max_j), eps=eps)

    dy = float(max_i - (H // 2))
    dx = float(max_j - (W // 2))

    if subpixel:
        di, dj = _peak_subpixel_taylor(mag, peak_ij=(max_i, max_j))
        dy += di
        dx += dj

    return dy, dx, peak, snr

def _as_float2d(a: np.ndarray, *, name: str) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim != 2:
        raise ValueError(f"{name} must be a 2D array.")
    if not np.issubdtype(a.dtype, np.floating):
        a = a.astype(np.float32, copy=False)
    return a


def _zscore2d(a: np.ndarray, *, eps: float) -> np.ndarray:
    m = float(np.nanmean(a))
    s = float(np.nanstd(a))
    return (a - m) / (s + eps)


def _corr_peak_quality(
    corr: np.ndarray, *, peak_ij: tuple[int, int], eps: float
) -> tuple[float, float]:
    i, j = peak_ij
    peak = float(corr[i, j])
    bg = float(np.median(np.abs(corr)))
    snr = float(abs(peak) / (bg + eps))
    return peak, snr


def _peak_subpixel_taylor(
    corr: np.ndarray,
    *,
    peak_ij: tuple[int, int],
) -> tuple[float, float]:
    """
    Subpixel peak refinement using a 2D Taylor / quadratic approximation.

    Parameters
    ----------
    corr : np.ndarray
        2D correlation map with zero-lag at the center (fftshifted).
    peak_ij : tuple[int, int]
        Integer peak location (i, j).

    Returns
    -------
    tuple[float, float]
        Subpixel correction (di, dj) to add to the integer peak indices.

    Notes
    -----
    If the peak is on the border (no 3x3 neighborhood), returns (0.0, 0.0).
    """
    i, j = peak_ij
    ny, nx = corr.shape

    if i <= 0 or i >= ny - 1 or j <= 0 or j >= nx - 1:
        return 0.0, 0.0

    dy = (corr[i + 1, j] - corr[i - 1, j]) / 2.0
    dyy = (corr[i + 1, j] + corr[i - 1, j] - 2.0 * corr[i, j])

    dx = (corr[i, j + 1] - corr[i, j - 1]) / 2.0
    dxx = (corr[i, j + 1] + corr[i, j - 1] - 2.0 * corr[i, j])

    dxy = (
        corr[i + 1, j + 1]
        - corr[i + 1, j - 1]
        - corr[i - 1, j + 1]
        + corr[i - 1, j - 1]
    ) / 4.0

    det = (dxx * dyy - dxy * dxy)
    if det == 0.0:
        return 0.0, 0.0

    inv_det = 1.0 / det
    di = - (dyy * dx - dxy * dy) * inv_det
    dj = - (dxx * dy - dxy * dx) * inv_det

    return float(di), float(dj)


