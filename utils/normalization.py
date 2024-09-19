"""
Normalization is used to standardize the intensity values in an image by adjusting the range of
gray level values so that they extend in a desired range of values and improve the contrast of the
image. The main goal of normalization is to reduce the variance of the gray level value along the
ridges to facilitate subsequent processing steps
"""
import numpy as np

def normalize_pixel_array(image, m0, v0):
    """
    Normalize the entire image using vectorized operations.
    
    Args:
        image (numpy.ndarray): The input image to be normalized.
        m0 (float): Desired mean.
        v0 (float): Desired variance.

    Returns:
        numpy.ndarray: Normalized image.
    """
    # Compute the global mean (m) and variance (v)
    m = np.mean(image)
    v = np.var(image)

    # Vectorized calculation of deviation coefficient
    dev_coeff = np.sqrt(v0 * ((image - m) ** 2) / v)

    # Apply the normalization formula based on whether pixel is above or below the mean
    normalized_image = np.where(image > m, m0 + dev_coeff, m0 - dev_coeff)

    # Clip values to ensure they are in the 0-255 range
    normalized_image = np.clip(normalized_image, 0, 255)

    # Convert the result to uint8 for proper display
    return normalized_image.astype(np.uint8)

def normalize(image, m0, v0):
    """
    Wrapper function to normalize the image intensity using the vectorized approach.
    
    Args:
        image (numpy.ndarray): The input image.
        m0 (float): Desired mean.
        v0 (float): Desired variance.

    Returns:
        numpy.ndarray: The normalized image.
    """
    return normalize_pixel_array(image, m0, v0)


# def normalize_pixel(x, v0, v, m, m0):
#     """
#     From Handbook of Fingerprint Recognition pg 133
#     Normalize job used by Hong, Wan and Jain(1998)
#     similar to https://pdfs.semanticscholar.org/6e86/1d0b58bdf7e2e2bb0ecbf274cee6974fe13f.pdf equation 21
#     :param x: pixel value
#     :param v0: desired variance
#     :param v: global image variance
#     :param m: global image mean
#     :param m0: desired mean
#     :return: normilized pixel
#     """
#     dev_coeff = sqrt((v0 * ((x - m)**2)) / v)
#     return m0 + dev_coeff if x > m else m0 - dev_coeff

# def normalize(im, m0, v0):
#     m = np.mean(im)
#     v = np.std(im) ** 2
#     (y, x) = im.shape
#     normilize_image = im.copy()
#     for i in range(x):
#         for j in range(y):
#             normilize_image[j, i] = normalize_pixel(im[j, i], v0, v, m, m0)

#     return normilize_image
