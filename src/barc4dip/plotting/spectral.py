# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2026 ESRF

from __future__ import annotations

from typing import Literal, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .style import igor_cmap, srw_cmap, start_plotting


def plt_spectrum1d(
    curve: np.ndarray,
    axis: np.ndarray,
    title: str | None = None,
    *,
    k: float = 1.0,
    xlabel: str = "radius",
    ylabel: str = "value",
    xmin: float | None = None,
    xmax: float | None = None,
    ymin: float | None = None,
    ymax: float | None = None,
    logx: bool = False,
    logy: bool = False,
    cumulative: bool = False,
    percentiles: tuple[float, ...] | None = None,
    mask_center: bool = False,
) -> tuple[Figure, Axes, Axes | None]:
    """
    Plot a 1D spectral curve.

    This is intended for curves such as radial means of FFT, PSD, or
    autocorrelation maps.

    Parameters
    ----------
    axis : np.ndarray
        1D axis values.
    curve : np.ndarray
        1D curve corresponding to ``axis``.
    title : str | None
        Optional figure title.
    k : float
        Scaling factor for fonts and titles (passed to ``start_plotting``).
        Default is 1.0.
    xlabel : str
        Label for the x-axis.
    ylabel : str
        Label for the main y-axis.
    xmin, xmax : float | None
        X-axis limits.
    ymin, ymax : float | None
        Y-axis limits for the main curve.
    logx : bool
        If True, use logarithmic scaling on the x-axis.
    logy : bool
        If True, use logarithmic scaling on the y-axis.
    cumulative : bool
        If True, overlay the normalized cumulative integral on a
        secondary y-axis.
    percentiles : tuple[float, ...] | None
        Optional percentile values (in [0, 100]) to display as vertical
        markers along the x-axis. Percentiles are computed over the
        cumulative integral of the curve.
    mask_center : bool
        If True, set the first two elements of the plotted curve to ``np.nan``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure handle.
    ax : matplotlib.axes.Axes
        Main axes.
    ax2 : matplotlib.axes.Axes | None
        Secondary axes for cumulative curve if enabled, otherwise None.

    Raises
    ------
    TypeError
        If ``axis`` or ``curve`` are not numeric NumPy arrays.
    ValueError
        If inputs are not 1D, have different lengths, contain no finite
        values, or are inconsistent with logarithmic scaling.
    """
    if not isinstance(axis, np.ndarray):
        raise TypeError("plt_spectrum1d expects axis as a numpy.ndarray")
    if not isinstance(curve, np.ndarray):
        raise TypeError("plt_spectrum1d expects curve as a numpy.ndarray")
    if axis.ndim != 1:
        raise ValueError(f"plt_spectrum1d expects a 1D axis; got shape={axis.shape!r}")
    if curve.ndim != 1:
        raise ValueError(f"plt_spectrum1d expects 1D curve; got shape={curve.shape!r}")
    if axis.size != curve.size:
        raise ValueError(
            f"plt_spectrum1d expects axis and curve with the same length; "
            f"got {axis.size} and {curve.size}"
        )
    if not np.issubdtype(axis.dtype, np.number):
        raise TypeError(f"plt_spectrum1d expects a numeric axis; got dtype={axis.dtype}")
    if not np.issubdtype(curve.dtype, np.number):
        raise TypeError(f"plt_spectrum1d expects numeric curve; got dtype={curve.dtype}")

    m = np.isfinite(axis) & np.isfinite(curve)
    if not np.any(m):
        raise ValueError("plt_spectrum1d expects at least one finite sample")

    x = np.asarray(axis[m], dtype=float)
    y = np.asarray(curve[m], dtype=float)

    if mask_center:
        x = x[2:]
        y = y[2:]

    if x.size < 2:
        raise ValueError("plt_spectrum1d expects at least two finite samples")

    dx = np.diff(x)
    if np.any(dx == 0.0):
        raise ValueError("axis must be strictly monotonic")
    if not (np.all(dx > 0.0) or np.all(dx < 0.0)):
        raise ValueError("axis must be strictly monotonic")

    if logx and np.any(x <= 0.0):
        raise ValueError("logx=True requires strictly positive axis values")

    start_plotting(k)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))

    y_plot = y.copy()

    if logy:
        y_plot = np.where(y_plot > 0.0, y_plot, np.nan)
        if not np.any(np.isfinite(y_plot)):
            raise ValueError("logy=True requires at least one strictly positive finite value")

    ax.plot(
        x,
        y_plot,
        color="steelblue",
        linewidth=1.5,
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title:
        ax.set_title(title, fontsize=15 * k)

    if xmin is not None or xmax is not None:
        ax.set_xlim(
            left=float(x[0]) if xmin is None else float(xmin),
            right=float(x[-1]) if xmax is None else float(xmax),
        )

    if logx:
        ax.set_xscale("log")

    if logy:
        ax.set_yscale("log")
        if ymin is None:
            positive = y_plot[np.isfinite(y_plot) & (y_plot > 0.0)]
            if positive.size > 0:
                ax.set_ylim(bottom=max(0.5 * float(np.min(positive)), 1e-300))
        else:
            ax.set_ylim(bottom=ymin)
        if ymax is not None:
            ax.set_ylim(top=ymax)
    else:
        if ymin is not None:
            ax.set_ylim(bottom=ymin)
        if ymax is not None:
            ax.set_ylim(top=ymax)

    cdf = None
    x_cdf = None
    if x.size >= 2:
        y_nonneg = np.clip(y, a_min=0.0, a_max=None)
        dx_abs = np.abs(np.diff(x))
        increments = 0.5 * (y_nonneg[:-1] + y_nonneg[1:]) * dx_abs
        cdf = np.empty_like(y_nonneg)
        cdf[0] = 0.0
        cdf[1:] = np.cumsum(increments)
        if cdf[-1] > 0.0:
            cdf = cdf / cdf[-1]
        x_cdf = x

    if percentiles is not None and len(percentiles) > 0:
        p = np.asarray(percentiles, dtype=float)
        if np.any((p < 0) | (p > 100)):
            raise ValueError("percentiles must be in [0, 100]")
        if cdf is None or cdf[-1] <= 0.0:
            raise ValueError("percentiles require a curve with positive cumulative integral")

        xp = np.interp(p / 100.0, cdf, x_cdf)
        for xv in np.atleast_1d(xp):
            ax.axvline(float(xv), color="olive", linewidth=1.5)

    ax2: Axes | None = None
    if cumulative:
        if logy:
            ax.grid(True, which="both", linestyle=":", linewidth=0.5)
        else:
            ax.grid(True, which="both", axis="x", linestyle=":", linewidth=0.5)

        ax2 = ax.twinx()

        if cdf is None:
            raise ValueError("cumulative=True requires at least two finite samples")

        ax2.plot(
            x_cdf,
            cdf,
            color="darkred",
            linewidth=1.5,
        )
        ax2.set_ylabel("cumulative")
        ax2.set_ylim(-0.05, 1.05)
    else:
        ax.grid(True, which="both", linestyle=":", linewidth=0.5)

    return fig, ax, ax2


def plt_spectrum2d(
    data: np.ndarray,
    *,
    x: np.ndarray | None = None,
    y: np.ndarray | None = None,
    cuts: bool = True,
    show_phase: bool = True,
    log_intensity: bool = False,
    mask_center: bool = False,
    k: float = 1.0,
    cmap: str = "igor",
    vmin: float | None = None,
    vmax: float | None = None,
    xmin: float | None = None,
    xmax: float | None = None,
    ymin: float | None = None,
    ymax: float | None = None,
    xlabel: str = "x",
    ylabel: str = "y",
    title: str | None = None,
    display_origin: str = "lower",
) -> dict[str, dict[str, object] | None]:
    """
    Plot a 2D spectral map and optionally its central cuts.

    Real-valued inputs are plotted directly. Complex-valued inputs are plotted as
    magnitude, and optionally phase.

    Parameters
    ----------
    data : np.ndarray
        Input 2D array to display.
    x : np.ndarray | None, optional
        X-axis coordinates. If None, pixel/bin indices are used.
    y : np.ndarray | None, optional
        Y-axis coordinates. If None, pixel/bin indices are used.
    cuts : bool, optional
        If True, also plot the central horizontal and vertical cuts.
    show_phase : bool, optional
        If True and ``data`` is complex, also plot the phase map and phase cuts.
        Ignored for real-valued inputs.
    log_intensity : bool, optional
        If True, use logarithmic scaling for the intensity/magnitude map and
        semilog-y for the corresponding cuts.
    mask_center : bool, optional
        If True, mask only the central pixel block for display:
        ``5x5`` for odd sizes, ``4x4`` for even sizes, and mixed ``5x4`` or ``4x5``
        for mixed parity.
    k : float, optional
        Plot scaling factor passed to ``start_plotting``.
    cmap : str, optional
        Colormap name or the special keywords ``"srw"`` and ``"igor"``.
    vmin, vmax : float | None, optional
        Limits for the intensity/magnitude color scale.
    xmin, xmax, ymin, ymax : float | None, optional
        Axis limits.
    xlabel, ylabel : str, optional
        Axis labels for the 2D maps and cuts.
    intensity_label : str, optional
        Y-label for intensity/magnitude cuts and label for the corresponding colorbar.
    phase_label : str, optional
        Y-label for phase cuts and label for the phase colorbar.
    title : str | None, optional
        Figure title prefix.
    display_origin : str, optional
        Display origin passed to ``imshow``.

    Returns
    -------
    dict[str, dict[str, object] | None]
        Dictionary with figure and axes handles:
        ``{"intensity", "intensity_cuts", "phase", "phase_cuts"}``.

    Raises
    ------
    ValueError
        If the input shapes are inconsistent or if ``log_intensity=True`` but no
        strictly positive finite values are available for the intensity display.
    """
    arr = np.asarray(data)
    if arr.ndim != 2:
        raise ValueError(f"data must be a 2D array; got shape={arr.shape!r}")

    ny, nx = arr.shape
    x_axis = _resolve_axis(x, n=nx, name="x")
    y_axis = _resolve_axis(y, n=ny, name="y")

    start_plotting(k)
    cmap_obj = _resolve_cmap(cmap)
    extent = _imshow_extent(x_axis, y_axis)

    ix0 = nx // 2
    iy0 = ny // 2

    out: dict[str, dict[str, object] | None] = {
        "intensity": None,
        "intensity_cuts": None,
        "phase": None,
        "phase_cuts": None,
    }

    is_complex = np.iscomplexobj(arr)

    intensity = np.abs(arr) if is_complex else np.asarray(arr, dtype=float)
    intensity_plot = intensity.copy()
    if mask_center:
        _apply_center_mask_inplace(intensity_plot)

    norm_int, vmin_int, vmax_int, intensity_map = _resolve_intensity_norm(
        intensity_plot,
        log_intensity=log_intensity,
        vmin=vmin,
        vmax=vmax,
    )

    intensity_title = "Intensity - |A|$^2$" if not is_complex else "Magnitude - |A|"
    if title is not None:
        intensity_title = f"{title}"

    fig_int, ax_int, im_int, cbar_int = _plot_map(
        image=intensity_map,
        extent=extent,
        cmap=cmap_obj,
        norm=norm_int,
        vmin=vmin_int,
        vmax=vmax_int,
        xlabel=xlabel,
        ylabel=ylabel,
        zlabel=None,
        title=intensity_title,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        display_origin=display_origin,
    )
    out["intensity"] = {
        "fig": fig_int,
        "ax": ax_int,
        "image": im_int,
        "colorbar": cbar_int,
    }

    if cuts:
        intensity_h = intensity_plot[iy0, :]
        intensity_v = intensity_plot[:, ix0]

        fig_cuts, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        line_h = _plot_cut(
            ax1,
            x_axis,
            intensity_h,
            logy=log_intensity,
            xlabel=xlabel,
            ylabel=None,
            title="Hor. cut",
            xmin=xmin,
            xmax=xmax,
        )
        line_v = _plot_cut(
            ax2,
            y_axis,
            intensity_v,
            logy=log_intensity,
            xlabel=ylabel,
            ylabel=None,
            title="Ver. cut",
            xmin=ymin,
            xmax=ymax,
        )
        fig_cuts.tight_layout()
        out["intensity_cuts"] = {
            "fig": fig_cuts,
            "axes": (ax1, ax2),
            "lines": (line_h, line_v),
        }

    if not (is_complex and show_phase):
        return out

    phase_map = np.angle(arr)
    if mask_center:
        phase_map = phase_map.copy()
        _apply_center_mask_inplace(phase_map)

    phase_title = "Phase - $\\angle A$" if title is None else f"{title}"

    fig_phase, ax_phase, im_phase, cbar_phase = _plot_map(
        image=phase_map,
        extent=extent,
        cmap="coolwarm",
        norm=None,
        vmin=None,
        vmax=None,
        xlabel=xlabel,
        ylabel=ylabel,
        zlabel=None,
        title=phase_title,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        display_origin=display_origin,
    )
    out["phase"] = {
        "fig": fig_phase,
        "ax": ax_phase,
        "image": im_phase,
        "colorbar": cbar_phase,
    }

    if cuts:
        phase_h = phase_map[iy0, :]
        phase_v = phase_map[:, ix0]

        fig_pc, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        line_h = _plot_cut(
            ax1,
            x_axis,
            phase_h,
            logy=False,
            xlabel=xlabel,
            ylabel='rad',
            title=f"Hor. cut ({ylabel}=0)",
            xmin=xmin,
            xmax=xmax,
        )
        line_v = _plot_cut(
            ax2,
            y_axis,
            phase_v,
            logy=False,
            xlabel=ylabel,
            ylabel='rad',
            title=f"Ver. cut ({xlabel}=0)",
            xmin=ymin,
            xmax=ymax,
        )
        fig_pc.tight_layout()
        out["phase_cuts"] = {
            "fig": fig_pc,
            "axes": (ax1, ax2),
            "lines": (line_h, line_v),
        }

    return out


def _resolve_axis(axis: np.ndarray | None, *, n: int, name: str) -> np.ndarray:
    if axis is None:
        return np.arange(n, dtype=float)

    out = np.asarray(axis, dtype=float)
    if out.ndim != 1:
        raise ValueError(f"{name} must be a 1D array; got ndim={out.ndim}")
    if out.size != n:
        raise ValueError(f"{name} must have length {n}; got {out.size}")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} contains non-finite values")
    if n > 1:
        d = np.diff(out)
        if np.any(d == 0.0):
            raise ValueError(f"{name} must be strictly monotonic")
        if not (np.all(d > 0.0) or np.all(d < 0.0)):
            raise ValueError(f"{name} must be strictly monotonic")
    return out


def _resolve_cmap(cmap: str):
    if cmap == "srw":
        return srw_cmap
    if cmap == "igor":
        return igor_cmap
    return plt.get_cmap(cmap)


def _apply_center_mask_inplace(data: np.ndarray) -> None:
    ny, nx = data.shape
    wd = 2
    y0 = (ny - wd) // 2
    y1 = ny // 2 + wd
    x0 = (nx - wd) // 2
    x1 = nx // 2 + wd

    data[y0:y1, x0:x1] = np.nan


def _resolve_intensity_norm(
    data: np.ndarray,
    *,
    log_intensity: bool,
    vmin: float | None,
    vmax: float | None,
) -> tuple[LogNorm | None, float | None, float | None, np.ndarray]:
    out = np.asarray(data, dtype=float)

    if not log_intensity:
        return None, vmin, vmax, out

    positive = out[np.isfinite(out) & (out > 0.0)]
    if positive.size == 0:
        raise ValueError("log_intensity=True requires at least one strictly positive finite value")

    vmin_eff = vmin if (vmin is not None and vmin > 0.0) else float(np.min(positive))
    vmax_eff = vmax if (vmax is not None and vmax > vmin_eff) else float(np.max(positive))
    out = np.where(out > 0.0, out, np.nan)
    return LogNorm(vmin=vmin_eff, vmax=vmax_eff), None, None, out


def _imshow_extent(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    if x.size == 1:
        dx = 1.0
    else:
        dx = float(np.mean(np.diff(x)))

    if y.size == 1:
        dy = 1.0
    else:
        dy = float(np.mean(np.diff(y)))

    return (
        float(x[0] - 0.5 * dx),
        float(x[-1] + 0.5 * dx),
        float(y[0] - 0.5 * dy),
        float(y[-1] + 0.5 * dy),
    )


def _plot_map(
    *,
    image: np.ndarray,
    extent: tuple[float, float, float, float],
    cmap,
    norm: LogNorm | None,
    vmin: float | None,
    vmax: float | None,
    xlabel: str,
    ylabel: str,
    zlabel: str,
    title: str,
    xmin: float | None,
    xmax: float | None,
    ymin: float | None,
    ymax: float | None,
    display_origin: str,
) -> tuple[plt.Figure, plt.Axes, plt.AxesImage, plt.colorbar]:
    fig = plt.figure()
    ax = fig.add_subplot(111)

    im = ax.imshow(
        image,
        origin=display_origin,
        cmap=cmap,
        norm=norm,
        vmin=vmin,
        vmax=vmax,
        interpolation="none",
        aspect="equal",
        extent=extent,
    )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if xmin is not None:
        ax.set_xlim(left=xmin)
    if xmax is not None:
        ax.set_xlim(right=xmax)
    if ymin is not None:
        ax.set_ylim(bottom=ymin)
    if ymax is not None:
        ax.set_ylim(top=ymax)

    ax.grid(True, linestyle=":", linewidth=0.5)
    ax.tick_params(direction="in", top=True, right=True)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(zlabel)

    return fig, ax, im, cbar


def _plot_cut(
    ax: plt.Axes,
    axis: np.ndarray,
    values: np.ndarray,
    *,
    logy: bool,
    xlabel: str,
    ylabel: str | None,
    title: str,
    xmin: float | None,
    xmax: float | None,
):
    vals = np.asarray(values, dtype=float)

    if logy:
        vals = np.where(vals > 0.0, vals, np.nan)
        line = ax.semilogy(axis, vals, color="darkred", lw=1.5)[0]
    else:
        line = ax.plot(axis, vals, color="darkred", lw=1.5)[0]

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    ax.grid(True, linestyle=":", linewidth=0.5)
    ax.tick_params(direction="in", top=True, right=True)

    lo = float(axis[0]) if xmin is None else float(xmin)
    hi = float(axis[-1]) if xmax is None else float(xmax)
    ax.set_xlim(lo, hi)

    return line