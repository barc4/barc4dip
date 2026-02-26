# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2026 ESRF - the European Synchrotron

"""
Unified public plotting API for barc4dip.
"""

from __future__ import annotations

from .image import (
    plt_image,
    plt_speckle_tiles_metric,
    plt_histogram,
)

__all__ = [
    "plt_image",
    "plt_speckle_tiles_metric",
    "plt_histogram",
]