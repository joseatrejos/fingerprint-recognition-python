"""
In order to eliminate the edges of the image and areas that are too noisy, segmentation is
necessary. It is based on the calculation of the variance of gray levels. For this purpose, the image
is divided into sub-blocks of (W × W) size’s and for each block the variance.
Then, the root of the variance of each block is compared with a threshold T, if the value obtained
is lower than the threshold, then the corresponding block is considered as the background of the
image and will be excluded by the subsequent processing.

The selected threshold value is T = 0.2 and the selected block size is W = 16

This step makes it possible to reduce the size of the useful part of the image and subsequently to
optimize the extraction phase of the biometric data.
"""
import numpy as np
import cv2 as cv

def normalise(img):
    """Normalize an image to have zero mean and unit variance."""
    return (img - np.mean(img)) / np.std(img)

def create_segmented_and_variance_images(im, w, threshold=0.2):
    """
    Segment an image by calculating the standard deviation in image blocks and applying a threshold to find the ROI.
    
    Args:
        im (numpy.ndarray): Input image.
        w (int): Size of the block (W × W) for variance calculation.
        threshold (float): Threshold for standard deviation to identify background areas.

    Returns:
        tuple: (segmented_image, normalized_image, mask)
            - segmented_image (numpy.ndarray): The segmented image where background regions are masked out.
            - normalized_image (numpy.ndarray): Normalized image (masked regions have zero mean, unit std).
            - mask (numpy.ndarray): Binary mask identifying the regions of interest (ROI).
    """
    (y, x) = im.shape

    # Threshold based on the overall image standard deviation
    threshold = np.std(im) * threshold

    # Initialize image variance map and mask
    image_variance = np.zeros(im.shape)
    segmented_image = im.copy()
    mask = np.ones_like(im)

    # Iterate over blocks and compute block standard deviation
    for i in range(0, x, w):
        for j in range(0, y, w):
            box = [i, j, min(i + w, x), min(j + w, y)]
            block = im[box[1]:box[3], box[0]:box[2]]
            block_stddev = np.std(block)
            image_variance[box[1]:box[3], box[0]:box[2]] = block_stddev

    # Apply threshold to create the mask (background areas will have std < threshold)
    mask[image_variance < threshold] = 0

    # Smooth the mask using morphological operations
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (w * 2, w * 2))
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    # Apply the mask to the segmented image
    segmented_image *= mask

    # Normalize the image only within the valid (non-background) regions
    im = normalise(im)
    valid_region = im[mask == 1]
    mean_val = np.mean(valid_region)
    std_val = np.std(valid_region)
    norm_img = (im - mean_val) / std_val

    return segmented_image, norm_img, mask
