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
from .stack import plt_displacement, plt_stack_metric
from .style import show

__all__ = [
    "plt_image",
    "plt_tiles_metric",
    "plt_histogram",
    "plt_displacement",
    "plt_stack_metric",
    "show",
]