# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2026 ESRF - the European Synchrotron

""" 
image enhancement methods
"""

from __future__ import annotations

import cv2
import numpy as np


def clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to an image.

    Parameters:
        image (np.ndarray): Input image as a NumPy array.
        clip_limit (float, optional): Threshold for contrast limiting. Default is 2.0.
        tile_grid_size (tuple, optional): Size of grid for histogram equalization. Default is (8, 8).

    Returns:
        np.ndarray: The image with CLAHE applied.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)