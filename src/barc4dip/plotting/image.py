# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2026 ESRF

from __future__ import annotations

from typing import Literal, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .style import igor_cmap, srw_cmap, start_plotting


def plt_image(
    img: np.ndarray,
    title: str | None = None,
    *,
    k: float = 1.0,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "viridis",
    xmin: float | None = None,
    xmax: float | None = None,
    ymin: float | None = None,
    ymax: float | None = None,
    display_origin: Literal["upper", "lower"] = "lower",
    colorbar: bool = True,
    cbar_label: str | None = None,
    roi: slice | tuple[slice, slice] | None = None,
    roi_zoom: bool = False,
    roi_color: str = "orange",
    roi_lw: float = 1.75,
    roi_alpha: float = 0.95,
) -> Figure:
    """
    Plot an image in pixel coordinates with an optional size-matched colorbar,
    and optionally overlay a rectangular ROI defined by Python slices.

    Parameters
    ----------
    img : np.ndarray
        2D image to display.
    title : str | None
        Optional figure title.
    k : float
        Scaling factor for fonts and titles (passed to ``start_plotting``).
        Default is 1.0.
    vmin, vmax : float | None
        Minimum/maximum value for color scaling.
    cmap : str
        Colormap name or special keywords ('srw', 'igor').
    xmin, xmax, ymin, ymax : float | None
        Axis limits in pixels (applied after ROI zoom if enabled).
    display_origin : {"upper", "lower"}
        Display origin passed to ``imshow(origin=...)``.
    colorbar : bool
        If True, attach a colorbar whose height matches the image axes.
    cbar_label : str | None
        Optional colorbar label.
    roi : slice | (slice, slice) | None
        ROI selection in numpy indexing order (y, x). Examples:
            - roi=(slice(100, 200), slice(300, 500))
            - roi=np.s_[100:200, 300:500]
            - roi=slice(100, 200)  # interpreted as y slice, full x
        Steps must be 1 (or None).
    roi_zoom : bool
        If True, set axis limits to the ROI bounds (with the correct direction
        for the chosen ``display_origin``).
    roi_color, roi_lw, roi_alpha
        Style for ROI rectangle overlay.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure handle.
    """
    if img.ndim != 2:
        raise ValueError(f"image expects a 2D array; got shape={img.shape!r}")

    start_plotting(k)

    if cmap == "srw":
        cmap_obj = srw_cmap
    elif cmap == "igor":
        cmap_obj = igor_cmap
    else:
        cmap_obj = plt.get_cmap(cmap)

    ny, nx = img.shape
    fig_h = 5.0
    fig_w = fig_h * (nx / ny)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    extent = (0.0, float(nx), 0.0, float(ny))

    im = ax.imshow(
        img,
        origin=display_origin,
        cmap=cmap_obj,
        vmin=vmin,
        vmax=vmax,
        interpolation="none",
        aspect="equal",
        extent=extent,
    )

    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")

    if title:
        ax.set_title(title, fontsize=15 * k)


    if roi is not None:
        x0, y0, w, h, ysl, xsl = _roi_to_rect(roi, ny=ny, nx=nx)

        if roi_zoom:
            ax.set_xlim(left=float(xsl.start), right=float(xsl.stop))
            if display_origin == "lower":
                ax.set_ylim(bottom=float(ysl.start), top=float(ysl.stop))
            else:
                ax.set_ylim(bottom=float(ysl.stop), top=float(ysl.start))
        else:
            rect = Rectangle(
                (x0, y0),
                w,
                h,
                fill=False,
                edgecolor=roi_color,
                linewidth=roi_lw,
                alpha=roi_alpha,
            )
            ax.add_patch(rect)

        # if roi_zoom:
        #     ax.set_xlim(left=float(xsl.start), right=float(xsl.stop))
        #     if display_origin == "lower":
        #         ax.set_ylim(bottom=float(ysl.start), top=float(ysl.stop))
        #     else:
        #         ax.set_ylim(bottom=float(ysl.stop), top=float(ysl.start))

    if xmin is not None or xmax is not None:
        ax.set_xlim(left=xmin, right=xmax)
    if ymin is not None or ymax is not None:
        ax.set_ylim(bottom=ymin, top=ymax)

    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax)
        if cbar_label is not None:
            cbar.set_label(cbar_label)

    return fig

def _as_unit_step_slice(s: slice, *, n: int, name: str) -> slice:
    if not isinstance(s, slice):
        raise TypeError(f"{name} must be a slice; got {type(s)!r}")

    step = 1 if s.step is None else s.step
    if step != 1:
        raise ValueError(f"{name}.step must be 1 or None for a rectangular ROI; got {s.step!r}")

    start = 0 if s.start is None else int(s.start)
    stop = n if s.stop is None else int(s.stop)

    if start < 0:
        start = n + start
    if stop < 0:
        stop = n + stop

    start = max(0, min(n, start))
    stop = max(0, min(n, stop))

    if stop < start:
        start, stop = stop, start

    return slice(start, stop, 1)


def _roi_to_rect(
    roi: slice | tuple[slice, slice],
    *,
    ny: int,
    nx: int,
) -> tuple[float, float, float, float, slice, slice]:
    """
    Convert ROI slices to rectangle params in pixel coords.

    Returns
    -------
    x0, y0, w, h, yslice, xslice
    """
    if isinstance(roi, tuple):
        if len(roi) != 2:
            raise ValueError("roi tuple must be (yslice, xslice)")
        ysl, xsl = roi
    else:
        ysl, xsl = roi, slice(None)

    ysl = _as_unit_step_slice(ysl, n=ny, name="roi[0] (yslice)")
    xsl = _as_unit_step_slice(xsl, n=nx, name="roi[1] (xslice)")

    y0 = float(ysl.start)
    x0 = float(xsl.start)
    h = float(ysl.stop - ysl.start)
    w = float(xsl.stop - xsl.start)

    return x0, y0, w, h, ysl, xsl


def plt_tiles_metric(
    img: np.ndarray,
    stats: dict,
    metric_path: str | Sequence[str],
    *,
    title: str | None = None,
    k: float = 1.0,
    cmap: str = "gray",
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar: bool = True,
    cbar_label: str | None = None,
    show_std: bool = True,
    fmt: str = "{:.2f}",
    normalize: bool = False,
    display_origin: Literal["upper", "lower"] | None = None,
) -> Figure:
    """
    Plot an image and overlay a 3x3 tile grid with metric mean ± std.

    Parameters
    ----------
    img
        2D image (H, W).
    stats
        Full stats dictionary containing keys "meta" and "tiles".
    metric_path
        Metric path inside tiles, e.g. ("grain", "lx") or "grain.lx".
        This function reads:
            stats["tiles"][group][metric]["mean"] -> (3, 3)
            stats["tiles"][group][metric]["std"]  -> (3, 3)
    title
        Optional figure title.
    k
        Global scaling factor (fonts, etc.).
    cmap, vmin, vmax
        Colormap and scaling for the image.
    colorbar
        If True, add a colorbar (axes_grid1, size-matched to the image axes).
    cbar_label
        Optional colorbar label. This applies to the image only and is not
        modified by ``normalize``.
    show_std
        If True, display "mean ± std". If False, only "mean".
    fmt
        Format string for numbers, e.g. "{:.2f}" or "{:.3g}".
    normalize
        If True, normalize the displayed tile values by the central tile mean
        (tile "C"). Both mean and std are divided by the same central value.
        This only affects the text overlay, not the image or colorbar.
    display_origin
        If None, uses stats["meta"]["display_origin"] when available,
        otherwise defaults to "lower".

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure handle.

    Raises
    ------
    ValueError
        If inputs are invalid, or if normalization is requested but the
        central tile mean is not finite or is zero.
    KeyError
        If the requested metric path is missing from ``stats["tiles"]``.
    """
    if not isinstance(img, np.ndarray) or img.ndim != 2:
        raise ValueError(
            f"img must be a 2D numpy array; got {type(img)} shape={getattr(img, 'shape', None)!r}"
        )

    meta = stats.get("meta")
    tiles = stats.get("tiles")
    if not isinstance(meta, dict) or not isinstance(tiles, dict):
        raise ValueError("stats must contain dict keys 'meta' and 'tiles'")

    if isinstance(metric_path, str):
        parts = tuple(p for p in metric_path.replace("/", ".").split(".") if p)
    else:
        parts = tuple(metric_path)
    if len(parts) != 2:
        raise ValueError("metric_path must be like ('grain','lx') or 'grain.lx'")

    group, metric = parts

    units = meta.get("units", {})
    unit = None
    if isinstance(units, dict):
        group_units = units.get(group)
        if isinstance(group_units, dict):
            unit = group_units.get(metric)

    if normalize:
        metric_with_unit = f"{metric} [norm.]"
    else:
        if isinstance(unit, str) and unit.strip() != "":
            metric_with_unit = f"{metric} ({unit})"
        else:
            metric_with_unit = metric

    group_block = tiles.get(group)
    if not isinstance(group_block, dict):
        raise KeyError(f"tiles has no group {group!r}")

    metric_block = group_block.get(metric)
    if not isinstance(metric_block, dict):
        raise KeyError(f"tiles[{group!r}] has no metric {metric!r}")

    mean = metric_block.get("mean")
    std = metric_block.get("std")
    if not (isinstance(mean, np.ndarray) and mean.shape == (3, 3)):
        raise ValueError(
            f"Expected mean array with shape (3,3); got {type(mean)} shape={getattr(mean, 'shape', None)!r}"
        )
    if show_std:
        if not (isinstance(std, np.ndarray) and std.shape == (3, 3)):
            raise ValueError(
                f"Expected std array with shape (3,3); got {type(std)} shape={getattr(std, 'shape', None)!r}"
            )

    mean_disp = mean.astype(np.float64, copy=False)
    std_disp = std.astype(np.float64, copy=False) if isinstance(std, np.ndarray) else None

    if normalize:
        center_value = float(mean_disp[1, 1])
        if not np.isfinite(center_value):
            raise ValueError("Cannot normalize tile labels: central tile mean is not finite")
        if np.isclose(center_value, 0.0):
            raise ValueError("Cannot normalize tile labels: central tile mean is zero")

        mean_disp = mean_disp / center_value
        if std_disp is not None:
            std_disp = std_disp / center_value

    tile_labels = meta.get("tile_labels", None)
    if isinstance(tile_labels, np.ndarray) and tile_labels.shape == (3, 3):
        labels = tile_labels
    else:
        labels = np.array(
            [["NW", "N", "NE"], ["W", "C", "E"], ["SW", "S", "SE"]],
            dtype=object,
        )

    origin = display_origin
    if origin is None:
        origin = meta.get("display_origin", "lower")
    if origin not in ("upper", "lower"):
        origin = "lower"

    ny, nx = img.shape
    extent = (0.0, float(nx), 0.0, float(ny))

    fig_h = 5.0
    fig_w = fig_h * (nx / ny)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(
        img,
        origin=origin,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="none",
        aspect="equal",
        extent=extent,
    )

    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")

    if title is None:
        title = metric_with_unit
    ax.set_title(title, fontsize=14 * k)

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()

    edges_x = np.linspace(x0, x1, 4)
    edges_y = np.linspace(y1, y0, 4)

    grid_color = "orange"
    grid_lw = 1.5
    grid_alpha = 0.9
    text_color = "w"
    text_alpha = 0.95
    bbox_facecolor = "black"
    bbox_alpha = 0.60
    bbox_edgecolor = "black"
    bbox_lw = 0.0

    for x in edges_x:
        ax.plot([x, x], [y1, y0], "-", lw=grid_lw, alpha=grid_alpha, color=grid_color)
    for y in edges_y:
        ax.plot([x0, x1], [y, y], "-", lw=grid_lw, alpha=grid_alpha, color=grid_color)

    for iy in range(3):
        for ix in range(3):
            cx = 0.5 * (edges_x[ix] + edges_x[ix + 1])
            cy = 0.5 * (edges_y[iy] + edges_y[iy + 1])

            m = float(mean_disp[iy, ix])
            if show_std:
                s = float(std_disp[iy, ix])
                txt = f"{labels[iy, ix]}\n{fmt.format(m)} ± {fmt.format(s)}"
            else:
                txt = f"{labels[iy, ix]}\n{fmt.format(m)}"

            ax.text(
                cx,
                cy,
                txt,
                ha="center",
                va="center",
                fontsize=10 * k,
                color=text_color,
                alpha=text_alpha,
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    facecolor=bbox_facecolor,
                    alpha=bbox_alpha,
                    edgecolor=bbox_edgecolor,
                    linewidth=bbox_lw,
                ),
            )

    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax)
        if cbar_label is not None:
            cbar.set_label(cbar_label)

    return fig


def plt_histogram(
    img: np.ndarray,
    title: str | None = None,
    *,
    k: float = 1.0,
    bin_min: int = 0,
    bin_max: int = 65536,
    ymin: float | None = None,
    ymax: float | None = None,
    logy: bool = False,
    cumulative: bool = False,
    density: bool = False,
    percentiles: tuple[float, ...] | None = None,
) -> Figure:
    """
    Plot a histogram of pixel values from a 2D image.

    The histogram is computed over all finite pixel values. For floating-point
    images, values are clipped to ``[bin_min, bin_max]`` before histogramming
    to preserve the detector-style gray-level convention. For ``uint16`` images,
    the function enforces the "one bin per gray level" intent by requiring
    ``bins == (bin_max - bin_min)``.

    Parameters
    ----------
    img : np.ndarray
        2D numeric image array.
    title : str | None
        Optional figure title.
    k : float
        Scaling factor for fonts and titles (passed to ``start_plotting``).
        Default is 1.0.
    bin_min : int
        Minimum gray/value included in the histogram range.
    bin_max : int
        Maximum gray/value included in the histogram range.
    ymin, ymax : float | None
        Y-axis limits (counts or density).
    logy : bool
        If True, use logarithmic scaling on the y-axis.
    cumulative : bool
        If True, overlay the normalized cumulative distribution on a
        secondary y-axis.
    density : bool
        If True, normalize the histogram so that the integral equals 1.
        In this case, the y-axis represents probability density instead of counts.
    percentiles : tuple[float, ...] | None
        Optional percentile values (in [0, 100]) to display as vertical
        markers on the histogram.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure handle.

    Raises
    ------
    TypeError
        If ``img`` is not a numeric NumPy array.
    ValueError
        If ``img`` is not 2D, contains no finite values, or if bin
        configuration is inconsistent with uint16 detector semantics.
    """
    if not isinstance(img, np.ndarray):
        raise TypeError("histogram expects a numpy.ndarray")
    if img.ndim != 2:
        raise ValueError(f"histogram expects a 2D array; got shape={img.shape!r}")
    if not np.issubdtype(img.dtype, np.number):
        raise TypeError(f"histogram expects a numeric array; got dtype={img.dtype}")
    if bin_max <= bin_min:
        raise ValueError("require bin_max > bin_min")


    start_plotting(k)

    values = img.ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("histogram expects at least one finite value")

    if np.issubdtype(values.dtype, np.floating):
        values = np.clip(values, float(bin_min), float(bin_max))

    fig, ax = plt.subplots(figsize=(7.0, 4.5))

    counts, bin_edges, _ = ax.hist(
        values,
        bins=int(bin_max - bin_min),
        range=(bin_min, bin_max),
        histtype="step",
        linewidth=1.5,
        color="steelblue",
        alpha=1,
        density=density,
    )

    ax.set_xlabel("value")
    ax.set_ylabel("density" if density else "counts")
    ax.set_xlim(bin_min, bin_max)

    if title:
        ax.set_title(title, fontsize=15 * k)

    if ymin is not None or ymax is not None:
        ax.set_ylim(bottom=ymin, top=ymax)

    if logy:
        ax.set_yscale("log")
        if ymin is None:
            ax.set_ylim(bottom=0.5)
        else:
            ax.set_ylim(bottom=ymin)
        if ymax is not None:
            ax.set_ylim(top=ymax)
    else:
        if ymin is None:
            ax.set_ylim(bottom=0.0)
        else:
            ax.set_ylim(bottom=ymin)
        if ymax is not None:
            ax.set_ylim(top=ymax)

    if percentiles is not None and len(percentiles) > 0:
        p = np.asarray(percentiles, dtype=float)
        if np.any((p < 0) | (p > 100)):
            raise ValueError("percentiles must be in [0, 100]")
        pv = np.percentile(values, p)
        for x in np.atleast_1d(pv):
            ax.axvline(float(x), color="olive", linewidth=1.5)

    ax2: Axes | None = None
    if cumulative:
        if logy:
            ax.grid(True, which="both",linestyle=":", linewidth=0.5)
        else:
            ax.grid(True, which="both", axis='x', linestyle=":", linewidth=0.5)

        ax2 = ax.twinx()

        cdf = np.cumsum(counts)
        if cdf.size > 0 and cdf[-1] > 0:
            cdf = cdf / cdf[-1]

        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        ax2.plot(
            bin_centers,
            cdf,
            color="darkred",
            linewidth=1.5,
        )
        ax2.set_ylabel("cumulative")
        ax2.set_ylim(-0.05, 1.05)
    else:
        ax.grid(True, which="both", linestyle=":", linewidth=0.5)

    return fig