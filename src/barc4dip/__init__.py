# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2026 ESRF - the European Synchrotron

from __future__ import annotations
from ._version import __version__

from . import io, maths, metrics, plotting, preprocessing, signal, utils
from .io import read_image, write_image
from .metrics import distribution_moments, speckle_stats
from .report import logbook_report

__all__ = [
    "__version__",
    # namespaces
    "io",
    "maths",
    "metrics",
    "plotting",
    "preprocessing",
    "signal",
    "utils",
    # curated functions
    "read_image",
    "write_image",
    "speckle_stats",
    "distribution_moments",
    "logbook_report",
]