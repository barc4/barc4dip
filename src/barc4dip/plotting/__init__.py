# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2026 ESRF - the European Synchrotron

"""
Unified public plotting API for barc4dip.
"""

from __future__ import annotations

from .image import (
    plt_histogram,
    plt_image,
    plt_tiles_metric,
)
from .spectral import plt_spectrum1d, plt_spectrum2d
from .stack import plt_displacement, plt_stack_metric
from .style import show

__all__ = [
    "plt_histogram",
    "plt_image",
    "plt_tiles_metric",
    "plt_spectrum1d",
    "plt_spectrum2d",
    "plt_displacement",
    "plt_stack_metric",
    "show",
]