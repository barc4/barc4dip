# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2026 ESRF - the European Synchrotron

""" 
filters - local operators
"""

from __future__ import annotations

import cv2
import numpy as np


def deconvolve(image: np.ndarray, kernel: np.ndarray, **kwargs) -> np.ndarray:
    """
    Apply deconvolution to an image using a kernel.

    Parameters:
        image (np.ndarray): Input image as a NumPy array.
        kernel (np.ndarray): The kernel to use for deconvolution.
        **kwargs: Additional keyword arguments.
            - iterations (int, optional): Number of iterations for deconvolution. Default is 10.

    Returns:
        np.ndarray: The deconvolved image.
    """
    iterations = kwargs.get('iterations', 10)
    return cv2.deconvolution(image, kernel, iterations=iterations)