# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2026 ESRF

"""
Markdown reporting utilities (text-only logbook summaries).

The public entry point `logbook_report()` formats a compact Markdown report from
the dictionary returned by a metrics aggregator (e.g. speckles, stats, sharpness,
perceptual). 

"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

from ..utils import now

_LogbookFormatter = Callable[..., str]
_LOGBOOK_FORMATTERS: dict[str, _LogbookFormatter] = {}


def _register(kind: str) -> Callable[[_LogbookFormatter], _LogbookFormatter]:
    kind_norm = kind.strip().lower()

    def _decorator(fn: _LogbookFormatter) -> _LogbookFormatter:
        _LOGBOOK_FORMATTERS[kind_norm] = fn
        return fn

    return _decorator


def logbook_report(
    stats: dict,
    report_path: str | Path | None = None,
    *,
    complete: bool = False,
    notes: bool = False,
) -> str:
    """
    Build (and optionally write) a compact Markdown logbook summary.

    Parameters:
        stats (dict):
            Dictionary returned by a metrics aggregator.
        report_path (str | Path | None):
            If provided, write the Markdown text to this path.
        complete (bool):
            If True, include additional (more verbose) metric blocks.
            Default is False.
        notes (bool):
            If True, include explanatory notes. Default is False.

    Returns:
        str:
            Markdown text.

    Raises:
        TypeError:
            If stats is not a dict.
        ValueError:
            If required keys are missing, or the report kind cannot be resolved.
        FileNotFoundError:
            If report_path is provided but its parent directory does not exist.
    """
    if not isinstance(stats, dict):
        raise TypeError("logbook_report expects stats to be a dict")

    meta = stats.get("meta")
    if not isinstance(meta, dict):
        raise ValueError("stats must contain dict key 'meta'")

    resolved_kind = meta.get("kind")
    if not isinstance(resolved_kind, str) or not resolved_kind.strip():
        raise ValueError(
            "Cannot determine report kind. Set stats['meta']['kind']."
        )

    resolved_kind = resolved_kind.strip().lower()

    formatter = _LOGBOOK_FORMATTERS.get(resolved_kind)
    if formatter is None:
        supported = ", ".join(sorted(_LOGBOOK_FORMATTERS))
        raise ValueError(
            f"Unsupported report kind: {resolved_kind!r}. Supported: {supported}"
        )

    text = formatter(stats, complete=complete, notes=notes)

    if report_path is not None:
        report_path = Path(report_path)
        if not report_path.parent.exists():
            raise FileNotFoundError(
                f"Parent directory does not exist: {report_path.parent}"
            )
        report_path.write_text(text, encoding="utf-8")

    return text


@_register("speckles")
def _logbook_speckles(
    stats: dict,
    *,
    complete: bool = False,
    notes: bool = False,
) -> str:
    meta = stats.get("meta")
    full = stats.get("full")
    if not isinstance(meta, dict) or not isinstance(full, dict):
        raise ValueError("stats must contain dict keys 'meta' and 'full'")

    tiles = stats.get("tiles") if isinstance(stats.get("tiles"), dict) else None

    lines: list[str] = []
    lines.append("# Speckle summary")
    lines.append(f"{datetime.fromtimestamp(now()).strftime('%Y-%m-%d | %H:%M:%S')}")
    lines.append("")

    lines.append("## Metadata")

    input_shape = meta.get("input_shape", None)
    if (
        isinstance(input_shape, (tuple, list))
        and len(input_shape) == 2
        and all(isinstance(v, (int, np.integer)) for v in input_shape)
    ):
        lines.append(f"- Image shape: {int(input_shape[0])} x {int(input_shape[1])} px")
    else:
        lines.append("- Image shape: (unknown)")

    display_origin = meta.get("display_origin", "unknown")

    convention_map = {
        "lower": "detector-aligned, origin at bottom-left",
        "upper": "numpy-aligned, origin at top-left",
    }

    convention = convention_map.get(display_origin, "unknown")

    lines.append(f"- Image orientation: {display_origin} ({convention})")
    
    if "tile_grid_shape" in meta:
        tile_mode = meta.get("tile_mode", "unknown")
        tile_shape_px = meta.get("tile_shape_px", None)
        if (
            isinstance(tile_shape_px, (tuple, list))
            and len(tile_shape_px) == 2
            and all(isinstance(v, (int, np.integer)) for v in tile_shape_px)
        ):
            lines.append(
                f"- Tiles: {tile_mode}, tile shape: {int(tile_shape_px[0])} x {int(tile_shape_px[1])} px"
            )
        else:
            lines.append(f"- Tiles: {tile_mode}")
        if notes:
            tile_labels = meta.get("tile_labels", None)
            if tile_labels is not None:
                lines.append("- Tile order: row-major (NW, N, NE; W, C, E; SW, S, SE)")
                lines.append("")
                lines.append("Tile labels:")
                lines.append("```")
                lines.extend(_format_tile_labels(tile_labels))
                lines.append("```")

    lines.append("")

    if "amplitude" in full:
        amp = full["amplitude"]
        lines.append("## Amplitude (full image)")
        lines.append("```")
        lines.append(
            f"> visibility: {_f(amp.get('visibility'), 3)} | contrast: {_f(amp.get('contrast'), 3)}"
        )
        lines.append("```")
        lines.append("")

        _append_tiles_pair(
            lines,
            tiles,
            group="amplitude",
            key_left="visibility",
            title_left="Visibility (tiles)",
            fmt_left=("{:.3f}", "{:.3f}"),
            key_right="contrast",
            title_right="Contrast (tiles)",
            fmt_right=("{:.3f}", "{:.3f}"),
        )
        if notes:
            lines.append("Notes: ")
            lines.append(" - visibility: std(I)/mean(I).")
            lines.append(
                " - contrast: (I_high - I_low)/(I_high + I_low), where I_low and I_high"
            )
            lines.append(
                "   are obtained from a 99.5% percentile-based min/max range."
            )
            lines.append("")

    if "grain" in full:
        g = full["grain"]
        lines.append("## Grain (full image)")
        lines.append("```")
        lines.append(
            f"> grain: lx={_f(g.get('lx'), 2)} | ly={_f(g.get('ly'), 2)} | "
            f"lx/ly={_f(g.get('r'), 2)} | leq={_f(g.get('leq'), 2)}"
        )
        lines.append("```")
        lines.append("")

        _append_tiles_pair(
            lines,
            tiles,
            group="grain",
            key_left="lx",
            title_left="lx (tiles)",
            fmt_left=("{:.2f}", "{:.2f}"),
            key_right="ly",
            title_right="ly (tiles)",
            fmt_right=("{:.2f}", "{:.2f}"),
        )
        if complete:
            _append_tiles_pair(
                lines,
                tiles,
                group="grain",
                key_left="r",
                title_left="lx/ly (tiles)",
                fmt_left=("{:.2f}", "{:.2f}"),
                key_right="leq",
                title_right="leq (tiles)",
                fmt_right=("{:.2f}", "{:.2f}"),
            )
        if notes:
            lines.append("Notes: ")
            lines.append(" - units in pixel")
            lines.append(
                " - speckle grain metrics are computed from the autocorrelation peak"
            )
            lines.append(" - widths are given as 1/e values")
            lines.append(" - leq: 1/e radius of the radially averaged autocorrelation")
            lines.append("")

    if "stats" in full:
        s = full["stats"]
        lines.append("## Moments (full image)")
        lines.append("```")
        lines.append(
            f"> moments: mean={_f(s.get('mean'), 0)} | std={_f(s.get('std'), 0)} | "
            f"skew={_f(s.get('skewness'), 2)} | kurt={_f(s.get('kurtosis'), 2)} | "
            f"SNR={_f(s.get('SNRdB'), 2)} dB"
        )
        lines.append("```")
        lines.append("")

        _append_tiles_pair(
            lines,
            tiles,
            group="stats",
            key_left="mean",
            title_left="mean (tiles)",
            fmt_left=("{:.0f}", "{:.0f}"),
            key_right="std",
            title_right="std (tiles)",
            fmt_right=("{:.0f}", "{:.0f}"),
        )

        if complete:
            _append_tiles_pair(
                lines,
                tiles,
                group="stats",
                key_left="skewness",
                title_left="skewness (tiles)",
                fmt_left=("{:.2f}", "{:.2f}"),
                key_right="kurtosis",
                title_right="kurtosis (tiles)",
                fmt_right=("{:.2f}", "{:.2f}"),
            )
            _append_tiles_pair(
                lines,
                tiles,
                group="stats",
                key_left="SNRdB",
                title_left="SNR dB (tiles)",
                fmt_left=("{:.2f}", "{:.2f}"),
                key_right=None,
                title_right=None,
                fmt_right=None,
            )
        if notes:
            lines.append("Notes: ")
            lines.append(" - units in gray scale (uint16)")
            lines.append(" - **skewness** shows the *asymmetry* of the distribution.")
            lines.append(
                "    (if positive, the histogram has a longer “tail” on the right side; if negative, on the left)"
            )
            lines.append(" - **Kurtosis** shows the *peakedness* of the profile.")
            lines.append(
                "    (A Gaussian beam has kurtosis ≈ 0 in the “excess” convention,"
            )
            lines.append(
                "     if positive, the histogram has a sharper peak and heavier tails,"
            )
            lines.append(
                "     if neagtive, the histogram has a flatter, more top-hat-like profile)"
            )
            lines.append(" - SNR dB: 20*log10(mean/std)")
            lines.append("")

    if "bandwidth" in full:
        b = full["bandwidth"]
        lines.append("## Bandwidth (full image)")
        lines.append("```")
        lines.append(
            f"> bandwidth: fx={_f(b.get('sig_fx'), 4)} | fy={_f(b.get('sig_fy'), 4)} | "
            f"fx/fy={_f(b.get('rf'), 2)} | feq={_f(b.get('feq'), 4)} | "
            f"f95={_f(b.get('f95'), 4)}"
        )
        lines.append("```")
        lines.append("")

        _append_tiles_pair(
            lines,
            tiles,
            group="bandwidth",
            key_left="sig_fx",
            title_left="fx (tiles)",
            fmt_left=("{:.4f}", "{:.4f}"),
            key_right="sig_fy",
            title_right="fy (tiles)",
            fmt_right=("{:.4f}", "{:.4f}"),
        )

        if complete:
            _append_tiles_pair(
                lines,
                tiles,
                group="bandwidth",
                key_left="rf",
                title_left="fx/fy (tiles)",
                fmt_left=("{:.2f}", "{:.2f}"),
                key_right="feq",
                title_right="feq (tiles)",
                fmt_right=("{:.4f}", "{:.4f}"),
            )

            _append_tiles_pair(
                lines,
                tiles,
                group="bandwidth",
                key_left="f95",
                title_left="f95 (tiles)",
                fmt_left=("{:.4f}", "{:.4f}"),
                key_right=None,
                title_right=None,
                fmt_right=None,
            )
        if notes:
            lines.append("Notes: ")
            lines.append(" - units in cycles/pixel")
            lines.append(" - fx, fy: RMS bandwidth computed from the 2D PSD")
            lines.append(" - feq: radial RMS bandwidth computed from the 2D PSD")
            lines.append(
                " - f95: radial frequency such that 95% of the PSD energy is contained"
            )
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _f(x: object, ndigits: int) -> str:
    if x is None:
        return "nan"
    if isinstance(x, (int, float, np.floating)):
        if ndigits <= 0:
            return f"{float(x):.0f}"
        return f"{float(x):.{ndigits}f}"
    return str(x)


def _format_tile_labels(tile_labels: object) -> list[str]:
    arr = np.asarray(tile_labels, dtype=object)
    if arr.shape != (3, 3):
        return [str(tile_labels)]
    return [
        f"{arr[0,0]}  {arr[0,1]}  {arr[0,2]}",
        f"{arr[1,0]}   {arr[1,1]}  {arr[1,2]}",
        f"{arr[2,0]}  {arr[2,1]}  {arr[2,2]}",
    ]


def _append_tiles_pair(
    lines: list[str],
    tiles: dict | None,
    *,
    group: str,
    key_left: str,
    title_left: str,
    fmt_left: tuple[str, str],
    key_right: str | None,
    title_right: str | None,
    fmt_right: tuple[str, str] | None,
) -> None:
    """
    Append a tiles block to `lines`.

    Supports two modes:
    - Paired: left and right matrices printed side-by-side (mean±std).
    - Single: only left matrix printed (mean±std) when key_right is None.
    """
    if tiles is None:
        return

    g = tiles.get(group, None)
    if not isinstance(g, dict):
        return

    left = g.get(key_left, None)
    if not isinstance(left, dict) or "mean" not in left or "std" not in left:
        return

    Lm = np.asarray(left["mean"], dtype=float)
    Ls = np.asarray(left["std"], dtype=float)
    if Lm.shape != (3, 3) or Ls.shape != (3, 3):
        return

    # Single-matrix mode
    if key_right is None:
        matrix_lines = _format_single_matrix(Lm, Ls, fmt_left)
        lines.append(title_left)
        lines.append("```")
        lines.extend(matrix_lines)
        lines.append("```")
        lines.append("")
        return

    if title_right is None or fmt_right is None:
        return
    right = g.get(key_right, None)
    if not isinstance(right, dict) or "mean" not in right or "std" not in right:
        return

    Rm = np.asarray(right["mean"], dtype=float)
    Rs = np.asarray(right["std"], dtype=float)
    if Rm.shape != (3, 3) or Rs.shape != (3, 3):
        return

    matrix_lines, left_width, gap = _format_pair_matrices(
        Lm, Ls, fmt_left, Rm, Rs, fmt_right
    )

    header = title_left.ljust(left_width + gap) + title_right

    lines.append(header)
    lines.append("```")
    lines.extend(matrix_lines)
    lines.append("```")
    lines.append("")


def _format_single_matrix(
    mean: np.ndarray,
    std: np.ndarray,
    fmt: tuple[str, str],
) -> list[str]:
    fmt_m, fmt_s = fmt

    def cell(i: int, j: int) -> str:
        return (fmt_m.format(mean[i, j]) + "±" + fmt_s.format(std[i, j])).replace("nan", "nan")

    lines: list[str] = []
    for i in range(3):
        row = "  ".join(cell(i, j) for j in range(3))
        lines.append(row)
    return lines


def _format_pair_matrices(
    Lm: np.ndarray,
    Ls: np.ndarray,
    fmt_left: tuple[str, str],
    Rm: np.ndarray,
    Rs: np.ndarray,
    fmt_right: tuple[str, str],
    *,
    gap: int = 4,
) -> tuple[list[str], int, int]:
    lfm, lfs = fmt_left
    rfm, rfs = fmt_right

    def lcell(i: int, j: int) -> str:
        return (lfm.format(Lm[i, j]) + "±" + lfs.format(Ls[i, j])).replace("nan", "nan")

    def rcell(i: int, j: int) -> str:
        return (rfm.format(Rm[i, j]) + "±" + rfs.format(Rs[i, j])).replace("nan", "nan")

    Lrows = ["  ".join(lcell(i, j) for j in range(3)) for i in range(3)]
    Rrows = ["  ".join(rcell(i, j) for j in range(3)) for i in range(3)]

    left_width = max(len(s) for s in Lrows) if Lrows else 0

    lines = []
    for i in range(3):
        lines.append(Lrows[i].ljust(left_width) + (" " * gap) + Rrows[i])

    return lines, left_width, gap
