import math
import numpy as np
import cv2 as cv

def calculate_angles(im, W, smooth=False):
    """
    Estimate anisotropy orientation.
    :param im: Input image (grayscale)
    :param W: Block size
    :param smooth: Boolean for applying smoothing on angles
    :return: Array of angles
    """
    (y, x) = im.shape

    # Sobel filtering to get gradients for the whole image
    Gx = cv.Sobel(im, cv.CV_64F, 1, 0, ksize=3)
    Gy = cv.Sobel(im, cv.CV_64F, 0, 1, ksize=3)

    result = np.zeros((y // W, x // W))

    for j in range(0, y - W, W):
        for i in range(0, x - W, W):
            block_Gx = Gx[j:j+W, i:i+W]
            block_Gy = Gy[j:j+W, i:i+W]

            # Compute sums of products
            nominator = 2 * np.sum(block_Gx * block_Gy)
            denominator = np.sum(block_Gx**2 - block_Gy**2)

            if nominator != 0 or denominator != 0:
                result[j // W, i // W] = (math.pi + math.atan2(nominator, denominator)) / 2
            else:
                result[j // W, i // W] = 0

    if smooth:
        result = smooth_angles(result)

    return result

def smooth_angles(angles):
    """
    Smooth the angle matrix using a Gaussian filter to remove noise.
    :param angles: Array of angles
    :return: Smoothed angle array
    """
    kernel = cv.getGaussianKernel(5, sigma=1)  # Use OpenCV's Gaussian Kernel
    kernel = kernel @ kernel.T  # Create 2D kernel

    cos_angles = np.cos(angles * 2)
    sin_angles = np.sin(angles * 2)

    cos_smoothed = cv.filter2D(cos_angles, -1, kernel)
    sin_smoothed = cv.filter2D(sin_angles, -1, kernel)

    smooth_angles = np.arctan2(sin_smoothed, cos_smoothed) / 2

    return smooth_angles

def get_line_ends(i, j, W, tang):
    """
    Compute the endpoints of a line given the orientation angle.
    :param i: X-coordinate of the block
    :param j: Y-coordinate of the block
    :param W: Block size
    :param tang: Tangent of the angle
    :return: Tuple with line start and end points
    """
    if abs(tang) <= 1:
        begin = (i, int(j - (W / 2) * tang + W / 2))
        end = (i + W, int(j + (W / 2) * tang + W / 2))
    else:
        begin = (int(i + W / 2 + W / (2 * tang)), j + W // 2)
        end = (int(i + W / 2 - W / (2 * tang)), j - W // 2)
    return begin, end

def visualize_angles(im, mask, angles, W):
    """
    Visualize orientation angles as lines on the image.
    :param im: Input image (grayscale)
    :param mask: Mask of the region of interest
    :param angles: Array of angles
    :param W: Block size
    :return: Visualization image with orientation lines
    """
    (y, x) = im.shape
    result = np.zeros_like(im, dtype=np.uint8)
    result = cv.cvtColor(result, cv.COLOR_GRAY2RGB)

    for i in range(0, x - W, W):
        for j in range(0, y - W, W):
            if np.sum(mask[j:j+W, i:i+W]) > (W * W * 0.8):  # Ensure mask threshold is met
                angle = angles[j // W, i // W]
                tang = math.tan(angle)
                begin, end = get_line_ends(i, j, W, tang)
                cv.line(result, begin, end, color=(150, 150, 150), thickness=1)

    return result
