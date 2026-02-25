# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2026 ESRF - the European Synchrotron

from __future__ import annotations

from . import geometry, io, maths, metrics, plotting, preprocessing, signal, utils
from ._version import __version__
from .io import read_image, write_image
from .metrics import (
    distribution_moments,
    sharpness_stats,
    speckle_stack_stats,
    speckle_stats,
)
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
    "geometry",
    # curated functions
    "read_image",
    "write_image",
    "speckle_stats",
    "sharpness_stats",
    "distribution_moments",
    "logbook_report",
]