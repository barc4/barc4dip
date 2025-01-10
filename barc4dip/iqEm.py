""" 
Image Quality Enhancement Methods
"""

__author__ = ['Rafael Celestre']
__contact__ = 'rafael.celestre@synchrotron-soleil.fr'
__license__ = 'CC BY-NC-SA 4.0'
__copyright__ = 'Synchrotron SOLEIL, Saint Aubin, France'
__created__ = '10/JAN/2025'
__changed__ = '10/JAN/2025'

from copy import copy
from typing import Union

import cv2
import numpy as np
from scipy.ndimage import median_filter

# ****************************************************************************
# ********************** image conversion
# ****************************************************************************

def convert_to_16bit_gray(image: np.ndarray, handle_outliers: bool = True) -> np.ndarray:
    """
    Convert an image to 16-bit grayscale.

    Parameters:
        image (np.ndarray): Input image as a NumPy array.
        handle_outliers (bool, optional): If True, apply a median filter to handle outliers before conversion. Default is True.

    Returns:
        np.ndarray: The converted 16-bit grayscale image.
    """
    if handle_outliers:
        bffr = copy(image)
        bffr = median_filter(image, size=3)
        imgmin = np.amin(bffr)
        imgmax = np.amax(bffr)
    else:
        imgmin = np.amin(image)
        imgmax = np.amax(image)

    image = (image - imgmin) / (imgmax - imgmin)
    clipped_image = np.clip(image, 0, 1)
    converted_image = (clipped_image * int(65535)).astype(np.uint16)
    return converted_image

# ****************************************************************************
# ********************** CLAHE
# ****************************************************************************

def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
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

# ****************************************************************************
# ********************** Deconvolution
# ****************************************************************************

def apply_deconvolution(image: np.ndarray, kernel: np.ndarray, **kwargs) -> np.ndarray:
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