""" 
Speckle Field Metrics
"""

__author__ = ['Rafael Celestre']
__contact__ = 'rafael.celestre@synchrotron-soleil.fr'
__license__ = 'CC BY-NC-SA 4.0'
__copyright__ = 'Synchrotron SOLEIL, Saint Aubin, France'
__created__ = '10/JAN/2025'
__changed__ = '10/JAN/2025'

import barc4dip.spectral as spc
import numpy as np
from scipy.signal import correlate2d
from scipy.optimize import curve_fit

# ****************************************************************************
# ********************** speckles
# ****************************************************************************

def get_speckle_metrics(image: np.ndarray, **kwargs) -> dict:
    """
    Calculate various metrics of the speckle field in an image.
    Parameters:
    -----------
    image : np.ndarray
        The input image for which the speckle metrics are to be calculated.
    **kwargs : dict, optional
        Additional keyword arguments.
        - roi (tuple): A tuple of four integers (y_start, y_end, x_start, x_end) defining the region of interest in the image. Default is None.
    Returns:
    --------
    dict
        A dictionary containing the following keys:
        - "visibility": The visibility of the image.
        - "michelson_contrast": The Michelson contrast of the image.
        - "speckle_grain_size": The speckle grain size of the image.
    """
    roi = kwargs.get('roi', None)
    if roi is not None:
        image = image[roi[0]:roi[1], roi[2]:roi[3]]
    
    visibility = get_visibility(image)
    contrast = get_michelson_contrast(image)
    grain_size = get_speckle_grain_size(image)
    
    return {"visibility": visibility, "contrast": contrast, "grain_size_h": grain_size[1], "grain_size_v": grain_size[0]}

def get_visibility(image: np.ndarray, **kwargs) -> np.ndarray:
    """
    Calculate the visibility of an image.

    Visibility is defined as the standard deviation divided by the mean of the pixel values in the image.

    Parameters:
    image (np.ndarray): The input image as a NumPy array.
    **kwargs: Additional keyword arguments.
        - roi (tuple): A tuple of four integers (y_start, y_end, x_start, x_end) defining the region of interest in the image. Default is None.
    Returns:
    np.ndarray: The visibility of the image.
    """
    roi = kwargs.get('roi', None)
    if roi is not None:
        image = image[roi[0]:roi[1], roi[2]:roi[3]]
    return np.std(image)/np.mean(image)

def get_michelson_contrast(image: np.ndarray, **kwargs) -> np.ndarray:
    """
    Calculate the Michelson contrast of an image.
    The Michelson contrast is defined as the difference between the maximum 
    and minimum pixel values in the image, divided by the sum of the maximum 
    and minimum pixel values.
    Parameters:
    image (np.ndarray): The input image as a NumPy array.
    **kwargs : dict, optional
        Additional keyword arguments:
        - roi (tuple): A tuple of four integers (y_start, y_end, x_start, x_end) defining the region of interest in the image. Default is None.
    Returns:
    np.ndarray: The Michelson contrast of the image.
    """
    roi = kwargs.get('roi', None)
    if roi is not None:
        image = image[roi[0]:roi[1], roi[2]:roi[3]]
    return (np.max(image) - np.min(image))/(np.max(image) + np.min(image))

def get_speckle_grain_size(image: np.ndarray, **kwargs) -> np.ndarray:
    """
    Calculate the speckle grain size of an image using the cross-correlation method.
    Parameters:
    -----------
    image : np.ndarray
        The input image for which the speckle grain size is to be calculated.
    **kwargs : dict, optional
        Additional keyword arguments:
        - return_autocorr (bool): If True, return the autocorrelation and frequency components along with the speckle grain size. Default is False.
        - dx (float): The pixel size in the x-direction. Default is 1.
        - dy (float): The pixel size in the y-direction. Default is 1.
        - roi (tuple): A tuple of four integers (y_start, y_end, x_start, x_end) defining the region of interest in the image. Default is None.

    Returns:
    --------
    np.ndarray
        A 1D array containing the vertical and horizontal speckle grain sizes.
    tuple (optional)
        If `return_autocorr` is True, returns a tuple containing the speckle grain sizes and a dictionary with the following keys:
        - "auto_correlation": The autocorrelation of the image.
        - "fx": The frequency components in the x-direction.
        - "fy": The frequency components in the y-direction.
    """

    roi = kwargs.get('roi', None)
    if roi is not None:
        image = image[roi[0]:roi[1], roi[2]:roi[3]]

    def width_at_half_max(profile):
        half_max = np.max(profile) / 2
        indices = np.where(profile >= half_max)[0]
        return indices[-1] - indices[0]
    
    return_autocorr = kwargs.get('return_autocorr', False)
    dx = kwargs.get('dx', 1)
    dy = kwargs.get('dy', 1)
    
    # Calculate the cross-correlation of the image
    autocorr, fx, fy = spc.get_auto_correlation(image, **kwargs)

    # Find the maximum of the cross-correlation
    mx_pos = np.asarray(np.unravel_index(np.argmax(autocorr), autocorr.shape))
    y_cut = autocorr[:, mx_pos[1]]
    x_cut = autocorr[mx_pos[0], :]

    # Calculate the FWHM of the cross-correlation cuts
    vertical_size = width_at_half_max(y_cut) * dy
    horizontal_size = width_at_half_max(x_cut) * dx

    if return_autocorr:
        return np.array([vertical_size, horizontal_size]), {"auto_correlation":autocorr, "fx":fx, "fy":fy}
    return np.array([vertical_size, horizontal_size])
    
def alternative_speckle_grain_size(image: np.ndarray, **kwargs):

    # Gaussian fitting function
    def gaussian(x, a, x0, sigma):
        return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

    # Function to compute FWHM from a Gaussian fit
    def fwhm_from_gaussian_fit(x, y):
        try:
            popt, _ = curve_fit(gaussian, x, y, p0=[y.max(), x[np.argmax(y)], 1])
            _, x0, sigma = popt
            fwhm = 2.355 * sigma  # FWHM = 2.355 * sigma for a Gaussian
            return fwhm
        except:
            return None
        
    dx = kwargs.get('dx', 1)
    dy = kwargs.get('dy', 1)

    x_axis = np.linspace(-image.shape[-1]/2, image.shape[-1]/2, image.shape[-1])*dx
    y_axis = np.linspace(-image.shape[-2]/2, image.shape[-2]/2, image.shape[-2])*dy
    
    # Compute auto-correlation
    autocorr = correlate2d(image, image, mode="full", boundary="wrap")

    # Get midpoints
    mid_y, mid_x = np.array(autocorr.shape) // 2

    # Extract cross-sections
    horizontal_profile = autocorr[mid_y, :]
    vertical_profile = autocorr[:, mid_x]

    # Fit horizontal FWHM
    horizontal_fwhm = fwhm_from_gaussian_fit(x_axis, horizontal_profile)

    # Fit vertical FWHM
    vertical_fwhm = fwhm_from_gaussian_fit(y_axis, vertical_profile)

    return np.array([vertical_fwhm, horizontal_fwhm])