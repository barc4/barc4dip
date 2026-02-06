# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2026 ESRF - the European Synchrotron

"""
Unified public processing API for barc4dip.
"""

from __future__ import annotations

from .normalize import (
    flat_field_correction,
)

__all__ = [
    "flat_field_correction",
]