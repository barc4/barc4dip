# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2026 ESRF

from __future__ import annotations

import logging
from typing import Literal, Sequence

import numpy as np

from ..utils import elapsed_time, now

logger = logging.getLogger(__name__)
_DeconvMethod = Literal["wiener", "rl", "uw"]


def deconvolve_psf(
    images: np.ndarray,
    *,
    sigma: float | Sequence[float],
    method: _DeconvMethod = "wiener",
    clip: bool = True,
    pad_mode: Literal["reflect"] = "reflect",
    balance: float | None = None,  # wiener
    num_iter: int = 50,  # richardson_lucy
    filter_epsilon: float | None = None,  # richardson_lucy (stability)
    reg: float | None = None,  # unsupervised_wiener
    user_params: dict | None = None,  # unsupervised_wiener
    is_real: bool = True,  # unsupervised_wiener
    # --- stack / perf ---
    parallel: bool = True,
    n_jobs: int | None = None,
    verbose: bool = False,
) -> np.ndarray:
    """
    Deconvolve detector PSF assuming a (possibly anisotropic) Gaussian blur.

    Supports 2D images (H, W) and 3D stacks (T, H, W). For stacks, frames are
    processed independently, optionally in parallel (threaded).

    Parameters
    ----------
    images
        Input image or image stack.
        - 2D array (H, W)
        - 3D array (T, H, W)
    sigma
        Gaussian PSF sigma in pixels.
        - float for isotropic PSF
        - (sy, sx) for anisotropic PSF
    method
        Deconvolution method:
        - "wiener": Wiener-Hunt deconvolution (fast, default)
        - "rl": Richardson-Lucy (iterative, positivity-friendly)
        - "uw": Unsupervised Wiener-Hunt (advanced; stochastic)
    clip
        Passed to scikit-image deconvolution functions. Note: in scikit-image,
        clip=True thresholds intermediate/output values to [-1, 1] for pipeline
        compatibility. This function internally normalizes the input to avoid
        destructive clipping for typical detector-valued images.
    pad_mode
        Padding mode before deconvolution to reduce boundary artifacts. Only
        "reflect" is supported by design.

    balance
        Wiener regularization parameter (scikit-image `wiener(..., balance=...)`).
        If None, a conservative default is used.
    num_iter
        Number of Richardson-Lucy iterations (scikit-image `richardson_lucy`).
    filter_epsilon
        Stabilization threshold for Richardson-Lucy divisions (optional).
    reg, user_params, is_real
        Parameters forwarded to scikit-image `unsupervised_wiener`.

    parallel, n_jobs
        If images is a stack, process frames in parallel using joblib threads.
        Serial mode is forced if parallel=False or n_jobs<=1.
    verbose
        If True, emit basic logging (and joblib verbosity in parallel mode).

    Returns
    -------
    np.ndarray
        Deconvolved image(s), float32, same shape as input.

    Raises
    ------
    TypeError
        If images is not a numpy.ndarray.
    ValueError
        If images is not 2D/3D, sigma invalid, or method unsupported.

    Notes
    -----
    - Gaussian PSF kernel size is `odd(max(5, ceil(6*sigma)))` per axis.
      This corresponds to truncation at ±3σ.
    - Input is normalized by max(abs(image)) per frame before calling scikit-image
      routines (to avoid clip=True destroying information), then rescaled back.
    """
    if not isinstance(images, np.ndarray):
        raise TypeError("deconvolve_psf expects a numpy.ndarray")
    if images.ndim not in {2, 3}:
        raise ValueError(f"images must be 2D (H, W) or 3D (T, H, W); got ndim={images.ndim}")

    sy, sx = _parse_sigma(sigma)
    psf = _gaussian_psf(sy, sx, min_size=5)

    if method not in {"wiener", "rl", "uw"}:
        raise ValueError(f"Unsupported method: {method!r}. Use 'wiener', 'rl', or 'uw'.")

    if pad_mode != "reflect":
        raise ValueError("Only pad_mode='reflect' is supported (by design).")

    if balance is None and method == "wiener":
        balance = 0.01

    img = images.astype(np.float32, copy=False)
    is_stack = img.ndim == 3

    def _one(frame: np.ndarray) -> np.ndarray:
        return _deconv_one_frame(
            frame,
            psf=psf,
            method=method,
            clip=clip,
            pad_mode=pad_mode,
            balance=balance,
            num_iter=num_iter,
            filter_epsilon=filter_epsilon,
            reg=reg,
            user_params=user_params,
            is_real=is_real,
        )

    if not is_stack:
        out2d = _one(img)
        if verbose:
            logger.info(
                "> deconvolve_psf | method=%s | sigma=(%.3f, %.3f) px | kernel=%dx%d",
                method,
                sy,
                sx,
                int(psf.shape[0]),
                int(psf.shape[1]),
            )
        return out2d.astype(np.float32, copy=False)

    T = int(img.shape[0])
    serial_mode = (not parallel) or (n_jobs is not None and int(n_jobs) <= 1)
    if parallel is True and n_jobs is None:
        n_jobs = -1
        
    if serial_mode:
        out = np.empty_like(img, dtype=np.float32)
        for t in range(T):
            out[t, :, :] = _one(img[t, :, :])
        if verbose:
            logger.info(
                "> deconvolve_psf | frames=%d | method=%s | sigma=(%.3f, %.3f) px | kernel=%dx%d | parallel=no",
                T,
                method,
                sy,
                sx,
                int(psf.shape[0]),
                int(psf.shape[1]),
            )
        return out

    from joblib import Parallel, delayed

    jb_verbose = 10 if verbose else 0
    prefer = "threads"

    frames = Parallel(n_jobs=n_jobs, prefer=prefer, verbose=jb_verbose)(
        delayed(_one)(img[t, :, :]) for t in range(T)
    )
    out = np.stack(frames, axis=0).astype(np.float32, copy=False)

    if verbose:
        logger.info(
            "> deconvolve_psf | frames=%d | method=%s | sigma=(%.3f, %.3f) px | kernel=%dx%d | parallel=yes | n_jobs=%s",
            T,
            method,
            sy,
            sx,
            int(psf.shape[0]),
            int(psf.shape[1]),
            str(n_jobs),
        )

    return out


def _parse_sigma(sigma: float | Sequence[float]) -> tuple[float, float]:
    if isinstance(sigma, (int, float, np.floating)):
        sy = float(sigma)
        sx = float(sigma)
    else:
        s = list(sigma)
        if len(s) != 2:
            raise ValueError("sigma must be a float or a length-2 sequence (sy, sx).")
        sy = float(s[0])
        sx = float(s[1])

    if not (np.isfinite(sy) and np.isfinite(sx)):
        raise ValueError("sigma values must be finite.")
    if sy <= 0 or sx <= 0:
        raise ValueError("sigma values must be > 0.")
    return sy, sx


def _odd(n: int) -> int:
    n = int(n)
    return n if (n % 2 == 1) else (n + 1)


def _gaussian_psf(sy: float, sx: float, *, min_size: int = 5) -> np.ndarray:
    ky = _odd(max(min_size, int(np.ceil(6.0 * sy))))
    kx = _odd(max(min_size, int(np.ceil(6.0 * sx))))

    y = np.arange(ky, dtype=np.float32) - (ky - 1) / 2.0
    x = np.arange(kx, dtype=np.float32) - (kx - 1) / 2.0
    yy, xx = np.meshgrid(y, x, indexing="ij")

    psf = np.exp(-0.5 * ((yy / sy) ** 2 + (xx / sx) ** 2)).astype(np.float32, copy=False)
    s = float(psf.sum())
    if not np.isfinite(s) or s <= 0:
        raise ValueError("Failed to build a valid Gaussian PSF (sum<=0).")
    psf /= s
    return psf


def _deconv_one_frame(
    frame: np.ndarray,
    *,
    psf: np.ndarray,
    method: _DeconvMethod,
    clip: bool,
    pad_mode: Literal["reflect"],
    balance: float | None,
    num_iter: int,
    filter_epsilon: float | None,
    reg: float | None,
    user_params: dict | None,
    is_real: bool,
) -> np.ndarray:
    from skimage import restoration

    if frame.ndim != 2:
        raise ValueError("Internal error: frame must be 2D.")

    py = int(psf.shape[0] // 2)
    px = int(psf.shape[1] // 2)
    padded = np.pad(frame, ((py, py), (px, px)), mode=pad_mode)

    scale = float(np.nanmax(np.abs(padded)))
    if not np.isfinite(scale) or scale == 0.0:
        out = np.zeros_like(padded, dtype=np.float32)
        return out[py:-py, px:-px]

    work = (padded / scale).astype(np.float32, copy=False)

    if method == "wiener":
        if balance is None:
            raise ValueError("balance must not be None for method='wiener'.")
        restored = restoration.wiener(work, psf, balance=float(balance), clip=bool(clip))
    elif method == "rl":
        if num_iter < 1:
            raise ValueError("num_iter must be >= 1 for method='rl'.")
        restored = restoration.richardson_lucy(
            work,
            psf,
            num_iter=int(num_iter),
            clip=bool(clip),
            filter_epsilon=filter_epsilon,
        )
    else:  # "uw"
        restored, _ = restoration.unsupervised_wiener(
            work,
            psf,
            reg=reg,
            user_params=user_params,
            is_real=bool(is_real),
            clip=bool(clip),
        )

    restored = restored.astype(np.float32, copy=False) * scale
    # crop back
    return restored[py:-py, px:-px]