# SPDX-License-Identifier: CECILL-2.1
# Copyright (c) 2026 ESRF - the European Synchrotron

from __future__ import annotations

""" 
Spectral analisys auxiliary functions
"""

__author__ = ['Rafael Celestre']
__contact__ = 'rafael.celestre@synchrotron-soleil.fr'
__license__ = 'CC BY-NC-SA 4.0'
__copyright__ = 'Synchrotron SOLEIL, Saint Aubin, France'
__created__ = '10/JAN/2025'
__changed__ = '10/JAN/2025'

import numpy as np
from numpy.fft import fft2, fftfreq, fftshift, ifft2
from scipy.interpolate import RegularGridInterpolator
from typing import Tuple

# ****************************************************************************
# ********************** FFT
# ****************************************************************************

def get_fft(image: np.ndarray, **kwargs) -> np.ndarray:
    """
    Calculate the 2D Fast Fourier Transform of a 16-bit image.

    Parameters:
        image (np.ndarray): A 16-bit single-channel image.

    Returns:
        np.ndarray: The FFT of the input image.
    """
    
    dx = kwargs.get('dx', 1)
    dy = kwargs.get('dy', 1)
   
    padding = kwargs.get('padding', False)

    nx, ny = image.shape

    if padding:
        pad_y = int(len(ny)/2)
        pad_x = int(len(nx)/2)
        image = np.pad(image, ((pad_x, pad_x), (pad_y, pad_y)), 'constant', constant_values=(0, 0))
        nx, ny = image.shape

    fft = fftshift(fft2(image))

    fx = fftshift(fftfreq(nx, dx))
    fy = fftshift(fftfreq(ny, dy))

    return fft, fx, fy

def get_power_spectrum(image: np.ndarray, **kwargs) -> np.ndarray:
    """
    Calculate the power spectrum of a 16-bit image.

    Parameters:
        image (np.ndarray): A 16-bit single-channel image.

    Returns:
        np.ndarray: The power spectrum of the input image.
    """
    dx = kwargs.get('dx', 1)
    dy = kwargs.get('dy', 1)

    fft, fx, fy = get_fft(image, **kwargs)
    psd = np.abs(fft)**2
    nx, ny = psd.shape

    psd *= (dx * dy / (nx * ny)) 
    return psd, fx, fy

# ****************************************************************************
# ********************** Cross-correlation
# ****************************************************************************

def get_cross_correlation(image1: np.ndarray, image2: np.ndarray, **kwargs) -> np.ndarray:
    """
    Calculate the cross-correlation of two 16-bit images.

    Parameters:
        image1 (np.ndarray): A 16-bit single-channel image.
        image2 (np.ndarray): A 16-bit single-channel image.

    Returns:
        np.ndarray: The cross-correlation of the input images.
    """

    fft1, fx1, fy1 = get_fft(image1, **kwargs)
    fft2, fx2, fy2 = get_fft(image2, **kwargs)

    image_product = fft1 * fft2.conj()
    conv = fftshift(ifft2(image_product))
    conv  = np.abs(conv)
    conv /= np.amax(conv)

    return conv, fx1, fy1

# ****************************************************************************
# ********************** Auto-correlation
# ****************************************************************************

def get_auto_correlation(image: np.ndarray, **kwargs) -> np.ndarray:
    """
    Calculate the auto-correlation of a 16-bit image.

    Parameters:
        image (np.ndarray): A 16-bit single-channel image.

    Returns:
        np.ndarray: The auto-correlation of the input image.
    """

    auto_correlation, fx, fy = get_cross_correlation(image, image, **kwargs)

    return auto_correlation, fx, fy

