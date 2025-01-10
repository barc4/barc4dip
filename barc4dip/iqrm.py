""" 
Image Quality Ranking Methods
"""

__author__ = ['Rafael Celestre']
__contact__ = 'rafael.celestre@synchrotron-soleil.fr'
__license__ = 'CC BY-NC-SA 4.0'
__copyright__ = 'Synchrotron SOLEIL, Saint Aubin, France'
__created__ = '10/JAN/2025'
__changed__ = '10/JAN/2025'

from typing import Union

import cv2
import numpy as np
from scipy.stats import describe, entropy


# ****************************************************************************
# ********************** statistical moments
# ****************************************************************************

def get_statistical_metrics(image: np.ndarray) -> dict:
    """
    Calculate statistical moments and the Shannon entropy for a 16-bit image.

    Parameters:
        image (np.ndarray): A 16-bit single-channel image.

    Returns:
        dict: A dictionary containing the following statistical moments:
            - mean: The mean intensity of the image.
            - variance: The variance of the intensity values.
            - skewness: The skewness of the intensity distribution.
            - kurtosis: The kurtosis of the intensity distribution.
            - entropy: The Shannon entropy of the image.
    """
    mean, variance = describe(image, axis=None)[2:4]
    skewness = describe(image, axis=None).skewness
    kurtosis = describe(image, axis=None).kurtosis
    shannon_entropy = entropy(image, axis=None)

    return {
        'mean': mean,
        'variance': variance,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'entropy': shannon_entropy,
        'SNRdB': 10*np.log10(mean/variance)
        }

# ****************************************************************************
# ********************** sharpness metrics
# ****************************************************************************

def get_sharpness_metrics(image: np.ndarray, **kwargs) -> dict:
    """
    Calculate multiple sharpness metrics for a 16-bit image.

    This function calculates the Brenner gradient, energy of gradients, Fourier sharpness,
    Laplacian variance, and Tenengrad sharpness for the input image.

    Parameters:
        image (np.ndarray): A 16-bit single-channel image.

    Returns:
        dict: A dictionary containing the following sharpness metrics:
            - brenner_gradient: The Brenner gradient of the image.
            - energy_of_gradients: The energy of gradients of the image.
            - fourier_sharpness: The proportion of high-frequency content in the image.
            - laplacian_variance: The variance of the Laplacian of the image.
            - tenengrad: The Tenengrad sharpness measure of the image.
    """

    pixel_shift = kwargs.get('pixel_shift', 3)

    brenner_gradient_h = brenner_gradient_16bit(image, direction='h', pixel_shift=pixel_shift)
    brenner_gradient_v = brenner_gradient_16bit(image, direction='v', pixel_shift=pixel_shift)
    energy_of_gradients = energy_of_gradients_16bit(image)
    fourier_sharpness = fourier_sharpness_16bit(image)
    laplacian_variance = laplacian_variance_16bit(image)
    tenengrad = tenengrad_16bit(image)

    return {
        'brenner_gradient_h': brenner_gradient_h,
        'brenner_gradient_v': brenner_gradient_v,
        'energy_of_gradients': energy_of_gradients,
        'fourier_sharpness': fourier_sharpness,
        'laplacian_variance': laplacian_variance,
        'tenengrad': tenengrad
    }

def brenner_gradient_16bit(image: np.ndarray, direction: Union[int, str], pixel_shift: int = 1) -> int:
    """
    Calculate the Brenner gradient of a 16-bit image.

    The Brenner gradient is a measure of image sharpness, calculated by shifting the image 
    by a specified number of pixels along a given axis and computing the squared difference 
    between the original and shifted images.

    Parameters:
        image (np.ndarray): A 16-bit single-channel image.
        direction (Union[int, str], optional): The axis or direction for the gradient.
            - 0 or 'h', 'hor': Horizontal gradient.
            - 1 or 'v', 'ver': Vertical gradient.
        pixel_shift (int, optional): The number of pixels to shift. Default is 1.

    Returns:
        int: The sum of the squared differences, representing the Brenner gradient.
    """

    if isinstance(direction, str):
        direction = direction.lower()
        if direction.startswith('h'):
            direction = 1
        elif direction.startswith('v'):
            direction = 0
        else:
            raise ValueError("Direction must be 0, 1, 'h', 'hor', 'v', or 'ver'.")
    elif direction not in [0, 1]:
        raise ValueError("Direction must be 0, 1, 'h', 'hor', 'v', or 'ver'.")

    shifted = np.roll(image, -pixel_shift, axis=direction)

    if direction == 1:
        gradient = (shifted - image)[:, :-pixel_shift] ** 2
    else:
        gradient = (shifted - image)[:-pixel_shift, :] ** 2

    return np.sum(gradient)

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
