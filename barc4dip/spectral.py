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

# ****************************************************************************
# ********************** Radial profile
# ****************************************************************************

def calc_radial_average(signal_2d: np.ndarray, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the radial average of a 2D signal.

    Args:
        signal_2d (np.ndarray): 2D array representing the signal to be radially averaged.
        x (np.ndarray): 1D array of x-axis coordinates.
        y (np.ndarray): 1D array of y-axis coordinates.

    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - Radially averaged values as a 1D array.
            - Corresponding radial distances as a 1D array.
    """
    def _polar_to_cartesian(theta: np.ndarray, rho: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x = rho * np.cos(theta)
        y = rho * np.sin(theta)
        return x, y

    x_start, x_end, nx = x[0], x[-1], x.size
    y_start, y_end, ny = y[0], y[-1], y.size
    x_center = 0.5 * (x_end + x_start)
    y_center = 0.5 * (y_end + y_start)

    range_r = [0, min(x_end - x_center, y_end - y_center)]
    range_theta = [0, 2 * np.pi]

    nr = int(nx * 0.5)
    ntheta = int((range_theta[1] - range_theta[0]) * 180)

    X = np.linspace(x_start, x_end, nx)
    Y = np.linspace(y_start, y_end, ny)

    R = np.linspace(range_r[0], range_r[1], nr)
    THETA = np.linspace(range_theta[0], range_theta[1], ntheta)

    R_grid, THETA_grid = np.meshgrid(R, THETA, indexing='ij')
    X_grid, Y_grid = _polar_to_cartesian(THETA_grid, R_grid)

    interp_func = RegularGridInterpolator((y, x), signal_2d, bounds_error=False, fill_value=0)

    points = np.column_stack([Y_grid.ravel(), X_grid.ravel()])
    values = interp_func(points).reshape(R_grid.shape)

    radial_avg = np.mean(values, axis=1)

    return radial_avg, R