# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2026 ESRF

"""barc4dip.report.cli

Terminal-friendly CLI for speckle statistics + Markdown logbook report.

Example
-------
python -m barc4dip.report.cli --s speckles.tif --out report.md
python -m barc4dip.report.cli --s run.h5 --image_number 12 --out report.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..io import read_image
from ..metrics.speckles import speckle_stats
from ..preprocessing import flat_field_correction
from .markdown import logbook_report


def _is_h5(path: str) -> bool:
    sfx = Path(path).suffix.lower()
    return sfx in {".h5", ".hdf5"}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="barc4dip-speckles",
        description="Compute speckle metrics for a single image and write a Markdown logbook report.",
    )

    p.add_argument(
        "-s",
        "--speckle",
        dest="speckle_path",
        required=True,
        help="Path to speckle field image (.tif/.tiff/.edf/.h5/.hdf5).",
    )
    p.add_argument(
        "-n",
        "--image_number",
        dest="image_number",
        type=int,
        default=0,
        help="Frame index for HDF5 stacks (default: 0). Ignored for TIFF/EDF.",
    )

    p.add_argument(
        "-f",
        "--flat",
        dest="flat_path",
        default=None,
        help="Optional flat field image path.",
    )
    p.add_argument(
        "-d",
        "--dark",
        dest="dark_path",
        default=None,
        help="Optional dark field image path.",
    )

    p.add_argument(
        "-o",
        "--out",
        dest="out_path",
        default=None,
        help="Optional output Markdown filename (e.g. speckles_report.md).",
    )

    p.add_argument(
        "--no_tiles",
        dest="tiles",
        action="store_false",
        help="Disable 3x3 tiles computation.",
    )
    p.set_defaults(tiles=True)

    p.add_argument(
        "--complete",
        dest="complete",
        action="store_true",
        help="Include additional metric blocks in the Markdown report.",
    )
    p.add_argument(
        "--notes",
        dest="notes",
        action="store_true",
        help="Include brief explanatory notes in the Markdown report.",
    )

    p.add_argument(
        "--all",
        dest="all_groups",
        action="store_true",
        help="Compute all speckle metric groups (metrics='all').",
    )

    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    out_path = Path(args.out_path) if args.out_path is not None else None

    img_n = int(args.image_number)
    speckle_img_n = img_n if _is_h5(str(args.speckle_path)) else None

    speckles = read_image(str(args.speckle_path), image_number=speckle_img_n)

    flats = None
    if args.flat_path is not None:
        flats = read_image(str(args.flat_path))

    darks = None
    if args.dark_path is not None:
        darks = read_image(str(args.dark_path))

    if flats is not None or darks is not None:
        speckles = flat_field_correction(speckles, flats=flats, darks=darks)

    groups = "all" if bool(args.all_groups) else ("amplitude", "grain", "stats")

    metrics = speckle_stats(
        speckles,
        metrics=groups,
        tiles=bool(args.tiles),
        verbose=False,
    )

    text = logbook_report(
        metrics,
        report_path=out_path,
        complete=bool(args.complete),
        notes=bool(args.notes),
    )

    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
