# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2026 ESRF - the European Synchrotron

from __future__ import annotations
from ._version import __version__

from . import io, maths, metrics, plotting, preprocessing, report, utils
from .metrics import speckle_stats, distribution_moments
from .io import read_image, write_image

__all__ = [
    "__version__",
    # namespaces
    "io",
    "maths",
    "metrics",
    "plotting",
    "preprocessing",
    "report",
    "utils",
    # curated functions
    "read_image",
    "write_image",
    "speckle_stats",
    "distribution_moments",
]