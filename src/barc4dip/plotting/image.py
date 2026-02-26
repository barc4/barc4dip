# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2026 ESRF

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
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
) -> tuple[Figure, Axes, object]:
    """
    Plot an image in pixel coordinates with an optional size-matched colorbar.

    Parameters
    ----------
    img : np.ndarray
        2D image to display.
    title : str | None
        Optional figure title.
    k : float
        Scaling factor for fonts and titles (passed to start_plotting). Default is 1.0.
    vmin, vmax : float | None
        Minimum/maximum value for color scaling.
    cmap : str
        Colormap name or special keywords ('srw', 'igor').
    xmin, xmax, ymin, ymax : float | None
        Axis limits in pixels.
    display_origin : {"upper", "lower"}
        Display origin passed to ``imshow(origin=...)``.
    colorbar : bool
        If True, attach a colorbar whose height matches the image axes.
    cbar_label : str | None
        Optional colorbar label.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure handle.
    ax : matplotlib.axes.Axes
        Axes handle.
    im : matplotlib.image.AxesImage
        The image artist returned by ``imshow``.
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

    im = ax.imshow(
        img,
        origin=display_origin,
        cmap=cmap_obj,
        vmin=vmin,
        vmax=vmax,
        interpolation="none",
        aspect="equal",
    )

    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")

    if title:
        ax.set_title(title, fontsize=15 * k)

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

    return fig, ax, im


def plt_speckle_tiles_metric(img: np.ndarray,
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
    display_origin: Literal["upper", "lower"] | None = None,
    grid_color: str = "orange",
    grid_lw: float = 1.5,
    grid_alpha: float = 0.9,
    text_color: str = "w",
    text_alpha: float = 0.95,
    bbox_facecolor: str = "black",
    bbox_alpha: float = 0.40,
    bbox_edgecolor: str = "none",
    bbox_lw: float = 0.0,
) -> tuple[Figure, Axes, object]:
    """
    Plot a speckle image and overlay a 3x3 tile grid with metric mean ± std.

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
        Optional colorbar label.
    show_std
        If True, display "mean ± std". If False, only "mean".
    fmt
        Format string for numbers, e.g. "{:.2f}" or "{:.3g}".
    display_origin
        If None, uses stats["meta"]["display_origin"] when available,
        otherwise defaults to "lower".
    grid_color, grid_lw, grid_alpha
        Grid styling.
    text_color, text_alpha, bbox_*
        Text and bbox styling.

    Returns
    -------
    fig, ax, im
        Matplotlib handles.
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
        title = f"{group}.{metric} (tiles)"
    ax.set_title(title, fontsize=14 * k)

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()

    edges_x = np.linspace(x0, x1, 4)
    edges_y = np.linspace(y1, y0, 4)

    for x in edges_x:
        ax.plot([x, x], [y1, y0], "-", lw=grid_lw, alpha=grid_alpha, color=grid_color)
    for y in edges_y:
        ax.plot([x0, x1], [y, y], "-", lw=grid_lw, alpha=grid_alpha, color=grid_color)

    for iy in range(3):
        for ix in range(3):
            cx = 0.5 * (edges_x[ix] + edges_x[ix + 1])
            cy = 0.5 * (edges_y[iy] + edges_y[iy + 1])

            m = float(mean[iy, ix])
            if show_std:
                s = float(std[iy, ix])
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

    return fig, ax, im


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
) -> tuple[Figure, Axes, Axes | None]:
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
    ax : matplotlib.axes.Axes
        Main histogram axes.
    ax2 : matplotlib.axes.Axes | None
        Secondary axes for cumulative curve if enabled, otherwise None.

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

    fig, ax = plt.subplots(figsize=(6.0, 3.5))

    counts, bin_edges, _ = ax.hist(
        values,
        bins=int(bin_max - bin_min),
        range=(bin_min, bin_max),
        histtype="step",
        linewidth=1.,
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

    return fig, ax, ax2