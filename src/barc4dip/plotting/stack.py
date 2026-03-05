# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2026 ESRF

from __future__ import annotations

from typing import Literal, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .style import start_plotting

_TemporalKey = Literal["abs", "inc"]
_ViewKind = Literal["trajectory", "timeseries"]
_Uncertainty = Literal["none", "band", "errorbar"]
_StatsScope = Literal["full", "tiles"]


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


def _parse_metric_path(metric_path: str | Sequence[str]) -> tuple[str, str]:
    if isinstance(metric_path, str):
        parts = tuple(p for p in metric_path.replace("/", ".").split(".") if p)
    else:
        parts = tuple(metric_path)
    if len(parts) != 2:
        raise ValueError("metric_path must be like ('grain','lx') or 'grain.lx'")
    group, metric = parts
    return str(group), str(metric)


def _default_tile_labels(meta: dict) -> np.ndarray:
    tile_labels = meta.get("tile_labels", None)
    if isinstance(tile_labels, np.ndarray) and tile_labels.shape == (3, 3):
        return tile_labels
    return np.array([["NW", "N", "NE"], ["W", "C", "E"], ["SW", "S", "SE"]], dtype=object)


def _plot_timeseries(
    ax: Axes,
    t: np.ndarray,
    y: np.ndarray,
    *,
    color: object,
    ylabel: str,
    label: str=None,
    uncertainty: _Uncertainty,
    ystd: np.ndarray | float | None,
    marker: str = "o",
    markersize: float = 3.0,
) -> None:
    ax.plot(
        t,
        y,
        linewidth=1.0,
        linestyle="-",
        color=color,
        markerfacecolor="white",
        markeredgecolor=color,
        markeredgewidth=1.1,
        marker=marker,
        markersize=markersize,
        label=label,
    )

    if uncertainty != "none" and ystd is not None:
        if uncertainty == "band":
            ax.fill_between(t, y - ystd, y + ystd, alpha=0.2, color=color)
        elif uncertainty == "errorbar":
            ax.errorbar(t, y, yerr=ystd, fmt="none", elinewidth=0.8, capsize=0, color=color)
        else:
            raise ValueError(f"unknown uncertainty={uncertainty!r}")

    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)


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

    meta = stack_stats.get("meta")
    if not isinstance(meta, dict):
        raise ValueError("stack_stats must contain dict key 'meta'")

    units = meta.get("units", {})
    unit_px = "px"
    if isinstance(units, dict):
        temporal_units = units.get("temporal")
        if isinstance(temporal_units, dict):
            u_dx = temporal_units.get("dx")
            if isinstance(u_dx, str) and u_dx.strip() != "":
                unit_px = u_dx

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
            ax.plot(dxp, dyp, linewidth=1.0, color="black")

        sc = ax.scatter(dxp, dyp, c=t, cmap=cmap, s=25, zorder=3, edgecolors="black", linewidths=0.5)

        ax.set_xlabel(f"dx ({unit_px})")
        ax.set_ylabel(f"dy ({unit_px})")
        if title is None:
            ax.set_title(f"speckle displacement ({temporal})", fontsize=15 * k)
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

    colors = ["darkred", "olive", "steelblue"]

    _plot_timeseries(axes[0], t, dxp, color=colors[0], ylabel=f"dx ({unit_px})", uncertainty=uncertainty, ystd=sdxp)
    _plot_timeseries(axes[1], t, dyp, color=colors[1], ylabel=f"dy ({unit_px})", uncertainty=uncertainty, ystd=sdyp)
    _plot_timeseries(axes[2], t, rp,  color=colors[2], ylabel=f"r ({unit_px})",  uncertainty=uncertainty, ystd=sdrp)

    axes[-1].set_xlabel("(frame)")

    if title is None:
        fig.suptitle(f"speckle displacement ({temporal})", fontsize=15 * k)
    else:
        fig.suptitle(title, fontsize=15 * k)

    fig.tight_layout()
    return fig, axes, None


def plt_speckle_stack_metric(
    stack_stats: dict,
    metric_path: str | Sequence[str],
    *,
    scope: _StatsScope = "full",
    uncertainty: _Uncertainty = "none",
    cmap: str = "tab10",
    color: str = "darkred",
    markers: Sequence[str] | None = None,
    k: float = 1.0,
    title: str | None = None,
) -> tuple[Figure, Axes, None]:
    """
    Plot a single metric as a time series from ``speckle_stack_stats`` output.

    This is a 1-panel analogue of ``plt_displacement(kind="timeseries")``:
    - scope="full": plot one curve from stack_stats["full"][group][metric]
      and (optionally) show uncertainty using a scalar std = np.nanstd(y).
    - scope="tiles": plot the 9 tile curves from
        stack_stats["tiles"][group][metric]["mean"] -> (T, 3, 3)
        stack_stats["tiles"][group][metric]["std"]  -> (T, 3, 3)
      using distinct colors and markers, and per-frame std.
    """
    start_plotting(k)

    if not isinstance(stack_stats, dict):
        raise TypeError("stack_stats must be a dict")

    meta = stack_stats.get("meta")
    if not isinstance(meta, dict):
        raise ValueError("stack_stats must contain dict key 'meta'")

    units = meta.get("units", {})

    group, metric = _parse_metric_path(metric_path)

    unit = None
    if isinstance(units, dict):
        group_units = units.get(group)
        if isinstance(group_units, dict):
            unit = group_units.get(metric)

    if isinstance(unit, str) and unit.strip() != "":
        metric_with_unit = f"{metric} ({unit})"
        ylabel = metric_with_unit
    else:
        metric_with_unit = metric
        ylabel = metric

    nrows = 1
    fig_h = 3.0
    fig_w = 9.0

    fig, ax = plt.subplots(nrows=nrows, ncols=1, sharex=True, figsize=(fig_w, fig_h))

    if title is None:
        if scope == "full":
            tlt = "from full image"
        else:
            tlt = "from tiled image"

        title = f"{metric} {tlt}"
    ax.set_title(title, fontsize=15 * k)
    ax.set_xlabel("(frame)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

    if scope == "full":
        full = stack_stats.get("full")
        if not isinstance(full, dict):
            raise ValueError("stack_stats must contain dict key 'full'")

        group_block = full.get(group)
        if not isinstance(group_block, dict):
            raise KeyError(f"full has no group {group!r}")

        y = group_block.get(metric)
        if not isinstance(y, np.ndarray):
            raise ValueError(f"Expected full[{group!r}][{metric!r}] as numpy array; got {type(y)}")
        if y.ndim != 1:
            raise ValueError(f"Expected 1D time series for full[{group!r}][{metric!r}]; got shape={y.shape!r}")

        t = np.arange(y.size, dtype=float)
        m = np.isfinite(y)
        yp = y[m]
        tp = t[m]

        ystd: float | None
        if uncertainty == "none":
            ystd = None
        else:
            ystd = float(np.nanstd(yp))

        _plot_timeseries(
            ax,
            tp,
            yp,
            color=color,
            ylabel=ylabel,
            uncertainty=uncertainty,
            ystd=ystd,
            marker="o",
            markersize=3.0,
        )

        return fig, ax, None

    if scope != "tiles":
        raise ValueError(f"unknown scope={scope!r}")

    tiles = stack_stats.get("tiles")
    if not isinstance(tiles, dict):
        raise ValueError("stack_stats must contain dict key 'tiles' for scope='tiles'")

    group_block = tiles.get(group)
    if not isinstance(group_block, dict):
        raise KeyError(f"tiles has no group {group!r}")

    metric_block = group_block.get(metric)
    if not isinstance(metric_block, dict):
        raise KeyError(f"tiles[{group!r}] has no metric {metric!r}")

    mean = metric_block.get("mean")
    std = metric_block.get("std")

    if not isinstance(mean, np.ndarray) or mean.ndim != 3 or mean.shape[1:] != (3, 3):
        raise ValueError(
            f"Expected tiles[{group!r}][{metric!r}]['mean'] with shape (T,3,3); "
            f"got {type(mean)} shape={getattr(mean, 'shape', None)!r}"
        )
    if uncertainty != "none":
        if not isinstance(std, np.ndarray) or std.ndim != 3 or std.shape != mean.shape:
            raise ValueError(
                f"Expected tiles[{group!r}][{metric!r}]['std'] with shape {mean.shape!r}; "
                f"got {type(std)} shape={getattr(std, 'shape', None)!r}"
            )

    labels = _default_tile_labels(meta)

    T = int(mean.shape[0])
    t_all = np.arange(T, dtype=float)

    if markers is None:
        markers = ("o", "s", "^", "v", "D", "P", "X", "<", ">")
    if len(markers) < 9:
        raise ValueError("markers must have length >= 9 (tiles mode)")

    cmap_obj = plt.get_cmap(cmap)
    colors = [cmap_obj(i / max(8, 1)) for i in range(9)]

    idx = 0
    for iy in range(3):
        for ix in range(3):
            y = mean[:, iy, ix].astype(float, copy=False)
            ystd_arr = None
            if uncertainty != "none":
                ystd_arr = std[:, iy, ix].astype(float, copy=False)

            m = np.isfinite(y)
            if ystd_arr is not None:
                m &= np.isfinite(ystd_arr)

            if not np.any(m):
                idx += 1
                continue
            _plot_timeseries(
                ax,
                t_all[m],
                y[m],
                color=colors[idx],
                ylabel=ylabel,
                label=str(labels[iy, ix]),
                uncertainty=uncertainty,
                ystd=ystd_arr[m] if ystd_arr is not None else None,
                marker=str(markers[idx]),
                markersize=3.0,
            )
            idx += 1
    if T > 1:
        xmin, xmax = ax.get_xlim()
        tmax = t_all[-1]
        ax.set_xlim(xmin, 1.18 * tmax)
    ax.legend(loc="center right", fontsize=9 * k, framealpha=0.85)
    return fig, ax, None