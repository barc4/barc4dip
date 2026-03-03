# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2026 ESRF

from __future__ import annotations

from typing import Literal

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .style import start_plotting


_TemporalKey = Literal["abs", "inc"]
_ViewKind = Literal["trajectory", "timeseries"]
_Uncertainty = Literal["none", "band", "errorbar"]


def _get_temporal_block(stack_stats: dict, temporal: _TemporalKey) -> dict:
    temporal_root = stack_stats.get("temporal")
    if not isinstance(temporal_root, dict):
        raise ValueError("stack_stats must contain dict key 'temporal'")
    block = temporal_root.get(temporal)
    if not isinstance(block, dict):
        raise ValueError(f"stack_stats['temporal'] must contain dict key {temporal!r}")
    return block


def _get_series(block: dict, key: str) -> np.ndarray:
    arr = block.get(key)
    if arr is None and key.startswith("std_"):
        alt = key.replace("std_", "") + "_std"
        arr = block.get(alt)
    if not isinstance(arr, np.ndarray):
        raise ValueError(f"temporal block missing numpy array key {key!r}")
    if arr.ndim != 1:
        raise ValueError(f"temporal[{key!r}] must be 1D; got shape={arr.shape!r}")
    return arr


def plt_displacement(
    stack_stats: dict,
    *,
    temporal: _TemporalKey = "abs",
    kind: _ViewKind = "trajectory",
    cmap: str = "viridis",
    show_path: bool = True,
    uncertainty: _Uncertainty = "none",
    k: float = 1.0,
    title: str | None = None,
) -> tuple[Figure, Axes | np.ndarray, object | None]:
    """
    Plot speckle displacement diagnostics from ``speckle_stack_stats`` output.

    Parameters
    ----------
    stack_stats : dict
        Output dictionary from ``speckle_stack_stats``. Must contain:
        ``stack_stats["temporal"][temporal]["dx"]``, ``["dy"]``, optionally ``["r"]``
        and corresponding uncertainties ``std_dx``, ``std_dy``, ``std_r``.
    temporal : {"abs","inc"}
        Choose displacement convention from ``stack_stats["temporal"]``.
    kind : {"trajectory","timeseries"}
        - "trajectory": XY path (dx vs dy) with time-colored points.
        - "timeseries": dx(t), dy(t), and optionally r(t) stacked subplots.
    cmap : str
        Matplotlib colormap name for time-colored points (trajectory mode).
    show_path : bool
        If True, draw a connected path behind the time-colored points (trajectory mode).
    uncertainty : {"none","band","errorbar"}
        Uncertainty rendering for time series when std arrays are available.
        Default is "band" (±1σ fill).
    k : float
        Plot styling scale (passed to ``start_plotting``).
    title : str | None
        Optional title. If None, a mode-dependent title is used.

    Returns
    -------
    fig : Figure
    ax_or_axes : Axes | np.ndarray
        Single Axes (trajectory) or array of Axes (time-series).
    artist : object | None
        Scatter artist for trajectory mode (for colorbar), else None.
    """

    start_plotting(k)

    block = _get_temporal_block(stack_stats, temporal=temporal)

    dx = _get_series(block, "dx").astype(float, copy=False)
    dy = _get_series(block, "dy").astype(float, copy=False)

    include_r = True
    r = None
    std_dx = std_dy = std_r = None

    if isinstance(block.get("r"), np.ndarray):
        r = _get_series(block, "r").astype(float, copy=False)

    if isinstance(block.get("std_dx"), np.ndarray) or isinstance(block.get("dx_std"), np.ndarray):
        std_dx = _get_series(block, "std_dx").astype(float, copy=False)
    if isinstance(block.get("std_dy"), np.ndarray) or isinstance(block.get("dy_std"), np.ndarray):
        std_dy = _get_series(block, "std_dy").astype(float, copy=False)
    if isinstance(block.get("std_r"), np.ndarray) or isinstance(block.get("r_std"), np.ndarray):
        std_r = _get_series(block, "std_r").astype(float, copy=False)

    n = dx.size
    if dy.size != n:
        raise ValueError(f"dx and dy must have same length; got {dx.size} and {dy.size}")
    if r is not None and r.size != n:
        raise ValueError(f"r must match dx length; got {r.size} vs {n}")
    if std_dx is not None and std_dx.size != n:
        raise ValueError(f"std_dx must match dx length; got {std_dx.size} vs {n}")
    if std_dy is not None and std_dy.size != n:
        raise ValueError(f"std_dy must match dy length; got {std_dy.size} vs {n}")
    if std_r is not None and std_r.size != n:
        raise ValueError(f"std_r must match dx length; got {std_r.size} vs {n}")

    m = np.ones(n, dtype=bool)
    drop_nan = True
    if drop_nan:
        m &= np.isfinite(dx) & np.isfinite(dy)
        if kind == "timeseries" and include_r and r is not None:
            m &= np.isfinite(r)
        if kind == "timeseries" and uncertainty != "none":
            if std_dx is not None:
                m &= np.isfinite(std_dx)
            if std_dy is not None:
                m &= np.isfinite(std_dy)
            if include_r and r is not None and std_r is not None:
                m &= np.isfinite(std_r)

    dxp = dx[m]
    dyp = dy[m]
    rp = r[m] if (r is not None and include_r) else None
    sdxp = std_dx[m] if (std_dx is not None and kind == "timeseries" and uncertainty != "none") else None
    sdyp = std_dy[m] if (std_dy is not None and kind == "timeseries" and uncertainty != "none") else None
    sdrp = std_r[m] if (std_r is not None and kind == "timeseries" and uncertainty != "none" and rp is not None) else None

    t = np.arange(dxp.size, dtype=float)

    if kind == "trajectory":
        fig_h = 6.0
        fig_w = 6.0
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        if show_path:
            ax.plot(dxp, dyp, linewidth=1., color='black')

        sc = ax.scatter(dxp, dyp, c=t, cmap=cmap, s=25, zorder=3, edgecolors='black', linewidths=0.5)

        ax.set_xlabel("dx (px)")
        ax.set_ylabel("dy (px)")
        if title is None:
            ax.set_title(f"Speckle displacement ({temporal})", fontsize=15 * k)
        else:
            ax.set_title(title, fontsize=15 * k)
        ax.set_aspect(1)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.08)
        cbar = fig.colorbar(sc, cax=cax)
        cbar.set_label("(frame)")

        ax.grid(True, alpha=0.3)
        return fig, ax, sc

    if kind != "timeseries":
        raise ValueError(f"unknown kind={kind!r}")

    nrows = 3
    fig_h = 7.0
    fig_w = 8.0

    fig, axes = plt.subplots(nrows=nrows, ncols=1, sharex=True, figsize=(fig_w, fig_h))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes], dtype=object)

    def _plot_series(ax: Axes, y: np.ndarray, clr: str, ystd: np.ndarray | None, label: str) -> None:
        ax.plot(t, y, linewidth=1., linestyle='-', color=clr,
                markerfacecolor='white', markeredgecolor=clr, markeredgewidth=1.1,
                marker="o", markersize=3,)
        if uncertainty != "none" and ystd is not None:
            if uncertainty == "band":
                ax.fill_between(t, y - ystd, y + ystd, alpha=0.2)
            elif uncertainty == "errorbar":
                ax.errorbar(t, y, yerr=ystd, fmt="none", elinewidth=0.8, capsize=0)
            else:
                raise ValueError(f"unknown uncertainty={uncertainty!r}")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)

    colors = ["darkred", "olive", "steelblue"]

    _plot_series(axes[0], dxp, colors[0], sdxp, "dx (px)")
    _plot_series(axes[1], dyp, colors[1], sdyp, "dy (px)")
    _plot_series(axes[2],  rp, colors[2], sdrp, "r (px)")

    axes[-1].set_xlabel("(frame)")

    if title is None:
        fig.suptitle(f"Speckle displacement ({temporal})", fontsize=15 * k)
    else:
        fig.suptitle(title, fontsize=15 * k)

    fig.tight_layout()
    return fig, axes, None