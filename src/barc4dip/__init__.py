# SPDX-License-Identifier: CECILL-2.1
from __future__ import annotations

from importlib.metadata import version as _version

__version__ = _version("barc4dip")

# Expose modules/packages for debugging (no eager deep imports)
from . import io
from . import utils
from . import plotting

# Keep these commented until they exist as packages/modules
# from . import geometry
# from . import preprocess
# from . import metrics
# from . import signal
# from . import math

__all__ = [
    "__version__",
    "io",
    "utils",
    "plotting",
    # "geometry",
    # "preprocess",
    # "metrics",
    # "signal",
    # "math",
]
