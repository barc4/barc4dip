""" 
Image Quality Ranking Methods
"""
__author__ = ['Rafael Celestre']
__contact__ = 'rafael.celestre@synchrotron-soleil.fr'
__license__ = 'GPL-3.0'
__copyright__ = 'Synchrotron SOLEIL, Saint Aubin, France'
__created__ = '09/JAN/2025'
__changed__ = '09/JAN/2025'

from copy import copy

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
# ********************** sharpness metrics
# ****************************************************************************

def laplacian_variance_16bit(image: np.ndarray) -> float:
    """
    Calculate the variance of the Laplacian of a 16-bit image.

    Parameters:
        image (np.ndarray): A 16-bit single-channel image.

    Returns:
        float: The variance of the Laplacian of the image.
    """
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    variance = laplacian.var()
    return variance

def tenengrad_16bit(image: np.ndarray) -> float:
    """
    Calculate the Tenengrad sharpness measure of a 16-bit image.

    This is computed using the Sobel operator to find the gradient magnitude
    of the image, then taking the mean of the squared gradient magnitudes.

    Parameters:
        image (np.ndarray): A 16-bit single-channel image (grayscale).

    Returns:
        float: The mean of the squared gradient magnitudes, representing the sharpness of the image.
    """
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = sobel_x**2 + sobel_y**2
    return gradient_magnitude.mean()

def brenner_gradient_16bit(image: np.ndarray) -> int:
    """
    Calculate the Brenner gradient of a 16-bit image.

    This is a measure of image sharpness, calculated by shifting the image 
    by 5 pixels along the horizontal axis and computing the squared 
    difference between the original and shifted images.

    Parameters:
        image (np.ndarray): A 16-bit single-channel image.

    Returns:
        int: The sum of the squared differences, representing the Brenner gradient.
    """
    shifted = np.roll(image, -5, axis=1)
    gradient = (shifted - image) ** 2
    return np.sum(gradient)

def fourier_sharpness_16bit(image: np.ndarray) -> float:
    """
    Compute the sharpness of a 16-bit image using the Fourier Transform.

    This function calculates the Fourier Transform of the input image, shifts the zero-frequency
    component to the center, and computes the magnitude spectrum. It then defines a high-frequency
    threshold and calculates the proportion of high-frequency content as a measure of the image's sharpness.

    Parameters:
        image (np.ndarray): A 16-bit single-channel image.

    Returns:
        float: The proportion of high-frequency content in the image, representing its sharpness.
    """
    f_transform = np.fft.fft2(image)
    f_transform_shifted = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.abs(f_transform_shifted)
    threshold = np.percentile(magnitude_spectrum, 80)
    high_freq_energy = np.sum(magnitude_spectrum[magnitude_spectrum > threshold])
    return high_freq_energy

def energy_of_gradients_16bit(image: np.ndarray) -> float:
    """
    Calculate the energy of gradients for a 16-bit image using the Sobel operator.

    This function computes the gradients of the input image in both the x and y directions
    using the Sobel operator, then calculates the energy by summing the absolute values
    of these gradients.

    Parameters:
        image (np.ndarray): A 16-bit single-channel image.

    Returns:
        float: The energy of the gradients of the image.
    """
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    energy = np.sum(np.abs(sobel_x) + np.abs(sobel_y))
    return energy



