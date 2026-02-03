""" 
Read and write of tiff images
"""

__author__ = ['Rafael Celestre']
__contact__ = 'rafael.celestre@synchrotron-soleil.fr'
__license__ = 'CC BY-NC-SA 4.0'
__copyright__ = 'Synchrotron SOLEIL, Saint Aubin, France'
__created__ = '10/JAN/2025'
__changed__ = '10/JAN/2025'

import numpy as np
from barc4dip.src.barc4dip.ranking import convert_to_16bit_gray
from PIL import Image

# ****************************************************************************
# ********************** R/W of tiff images
# ****************************************************************************

def read_tif_image(image_path: str, **kwargs):
    """
    Reads a TIFF image from the given file path and processes it according to the provided options.
    Parameters:
    image_input (str): The file path to the TIFF image.
    **kwargs: Additional keyword arguments for processing the image.
        - convert16bit (bool): Whether to convert the image to 16-bit grayscale. Default is True.
        - limits (list): A list of four integers [xi, xf, yi, yf] specifying the cropping limits. Default is the full image.
        - handle_outliers (bool): Whether to handle outliers during the 16-bit conversion. Default is True.
    Returns:
    numpy.ndarray: The processed image array.
    Raises:
    ValueError: If the input is not a file path.
    Exception: If there is an error during image processing.
    Example:
    >>> img_array = read_tif_image("path/to/image.tif", convert16bit=True, limits=[0, 100, 0, 100])
    """

    convert16bit = kwargs.get("convert16bit", True)

    try:
        if isinstance(image_path, str):
            with Image.open(image_path) as img:
                img_array = np.array(img)
        else:
            raise ValueError("Input must be a file path")
        

        limits = kwargs.get('limits', [0, img_array.shape[1], 0, img_array.shape[0]])
        if limits is not None:
            xi, xf, yi, yf = limits
            img_array=img_array[yi:yf, xi:xf]

        if convert16bit:
            handle_outliers = kwargs.get("handle_outliers", True)
            img_array=convert_to_16bit_gray(img_array, handle_outliers)

        return img_array
        
    except Exception as e:
        print(f"Error plotting the image: {e}")


def save_tiff(data, image_path):
    """
    Save a numpy array as a TIFF image.

    Parameters:
    data (numpy.ndarray): The image data to be saved.
    image_path (str): The file path where the image will be saved.

    Returns:
    None
    """
    image = Image.fromarray(data)
    image.save(image_path)
    print("The image was saved to %s" % image_path)