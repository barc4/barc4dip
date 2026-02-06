# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2026 ESRF - the European Synchrotron

""" 
crop.py
"""

from __future__ import annotations

import numpy as np


def crop_to_square_center(array, constant=1.0):
    """
    Crops a 2D array to a square shape centered at the array's center,
    ensuring the number of pixels in the square is always odd.
    The size of the square is constant * min(dimensions of array).

    Parameters:
        array (np.ndarray): Input 2D array.
        constant (float): Factor to scale the square's size based on the smallest dimension.

    Returns:
        np.ndarray: Cropped square array with odd dimensions.
    """
    min_dim = min(array.shape)
    square_size = int(min_dim * constant)
    
    # Ensure square_size is odd
    if square_size % 2 == 0:
        square_size -= 1
    
    square_size = min(square_size, min(array.shape) | 1)
    
    center_y, center_x = np.array(array.shape) // 2
    
    half_size = square_size // 2
    start_y = max(center_y - half_size, 0)
    start_x = max(center_x - half_size, 0)
    
    end_y = min(start_y + square_size, array.shape[0])
    end_x = min(start_x + square_size, array.shape[1])
    
    start_y = end_y - square_size
    start_x = end_x - square_size
    
    cropped_array = array[start_y:end_y, start_x:end_x]
    
    return cropped_array
