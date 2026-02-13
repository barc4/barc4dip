# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2026 ESRF - the European Synchrotron

""" 
Plotting routines
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .style import igor_cmap, srw_cmap, start_plotting


def uint16_image(
    img: np.ndarray,
    title: str = None,
    *,
    k: float = 1.0,
    vmin: float = None,
    vmax: float= None,
    cmap: str = "viridis",
    xmin: float = None,
    xmax: float = None,
    ymin: float = None,
    ymax: float = None,
) -> None:
    """
    Plot a 2D uint16 image in pixel coordinates with a colorbar.

    Parameters:
        img (np.ndarray): 2D uint16 image to display.
        title (str | None): Optional figure title.
        k (float): Scaling factor for fonts and titles. Default is 1.0.
        vmin (float | None): Minimum value for color scaling.
        vmax (float | None): Maximum value for color scaling.
        cmap (str): Colormap name or special keywords ('srw', 'igor').
        xmin (float | None): Minimum x-axis limit in pixels.
        xmax (float | None): Maximum x-axis limit in pixels.
        ymin (float | None): Minimum y-axis limit in pixels.
        ymax (float | None): Maximum y-axis limit in pixels.

    Returns:
        None
    """
    if img.ndim != 2:
        raise ValueError(
            f"plot_uint16 expects a 2D array; got shape={img.shape!r}"
        )

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
        origin="lower",
        extent=(0, nx, 0, ny),
        cmap=cmap_obj,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        aspect="equal",
    )

    ax.set_xlabel("x [px]")
    ax.set_ylabel("y [px]")

    if title:
        ax.set_title(title, fontsize=15 * k)

    if xmin is not None or xmax is not None:
        ax.set_xlim(left=xmin, right=xmax)
    if ymin is not None or ymax is not None:
        ax.set_ylim(bottom=ymin, top=ymax)

    im_ratio = ny / nx
    plt.colorbar(im, ax=ax, fraction=0.046 * im_ratio, pad=0.04)

    plt.show()


def uint16_histogram(
    img: np.ndarray,
    title: str = None,
    *,
    k: float = 1.0,
    bin_min: int = 0,
    bin_max: int = 65535,
    bins: int = 65535,
    ymin: float = None,
    ymax: float = None,
    logy: bool = False,
    cumulative: bool = False,
) -> None:
    """
    Plot a histogram of gray values from a uint16 image.

    Parameters:
        img (np.ndarray): 2D uint16 image.
        title (str | None): Optional figure title.
        k (float): Scaling factor for fonts and titles. Default is 1.0.
        bin_min (int): Minimum gray value included in bins.
        bin_max (int): Maximum gray value included in bins.
        bins (int): Number of histogram bins.
        ymin (float | None): Minimum y-axis value (counts).
        ymax (float | None): Maximum y-axis value (counts).
        logy (bool): If True, use logarithmic scale for counts.
        cumulative (bool): If True, plot normalized cumulative distribution.

    Returns:
        None
    """
    if img.dtype != np.uint16:
        raise TypeError(
            f"uint16_histogram expects uint16 image; got {img.dtype}"
        )

    start_plotting(k)

    values = img.ravel()

    fig, ax = plt.subplots(figsize=(6.0, 3.5))

    counts, bin_edges, _ = ax.hist(
        values,
        bins=bins,
        range=(bin_min, bin_max),
        histtype="step",
        linewidth=1.0,
        color="steelblue",
    )

    ax.set_xlabel("gray value [uint16]")
    ax.set_ylabel("counts")
    ax.set_xlim(bin_min, bin_max)

    if title:
        ax.set_title(title, fontsize=15 * k)

    if ymin is not None or ymax is not None:
        ax.set_ylim(bottom=ymin, top=ymax)

    if logy:
        ax.set_yscale("log")

    if cumulative:
        ax2 = ax.twinx()

        cdf = np.cumsum(counts)
        if cdf[-1] > 0:
            cdf /= cdf[-1]

        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        ax2.plot(
            bin_centers,
            cdf,
            color="darkred",
            linewidth=1.0,
        )

        ax2.set_ylabel("cumulative")
        ax2.set_ylim(-0.05, 1.05)

    plt.show()
