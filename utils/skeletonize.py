import numpy as np
import cv2 as cv
from skimage.morphology import skeletonize as skelt

def skeletonize(image_input):
    """
    Skeletonization reduces binary objects to 1 pixel wide representations using Zhang-Suen algorithm.
    This helps extract minutiae from the thinned fingerprint ridges.
    :param image_input: 2d array uint8
    :return: Skeletonized image
    """
    # Invert the image so ridges are white and background is black
    image = np.zeros_like(image_input)
    image[image_input == 0] = 1.0
    
    # Perform Zhang-Suen skeletonization
    skeleton = skelt(image)

    # Initialize output image
    output = np.zeros_like(image_input)
    output[skeleton] = 255  # Mark skeleton with white pixels
    
    # Invert back the skeleton to match the original input format (ridges as black)
    cv.bitwise_not(output, output)
        
    return output


def thinning_morph(image, kernel):
    """
    Thinning image using morphological operations (erosion and dilation)
    :param image: 2d array uint8
    :param kernel: 3x3 2d array uint8
    :return: Thinned image
    """
    thinning_image = np.zeros_like(image)
    img = image.copy()

    while True:
        # Perform morphological erosion and dilation
        erosion = cv.erode(img, kernel, iterations=1)
        dilatation = cv.dilate(erosion, kernel, iterations=1)

        # Subtract the dilated image from the original to get the thinning result
        subs_img = np.subtract(img, dilatation)
        cv.bitwise_or(thinning_image, subs_img, thinning_image)

        # Update the image to the eroded version for the next iteration
        img = erosion.copy()

        # Stop when the image is fully eroded
        if np.sum(img) == 0:
            break

    # Shift down and compare with one pixel offset to clean up artifacts
    down = np.zeros_like(thinning_image)
    down[1:-1, :] = thinning_image[0:-2, :]
    down_mask = np.subtract(down, thinning_image)
    down_mask[0:-2, :] = down_mask[1:-1, :]
    cv.imshow('Down Shift Mask', down_mask)

    # Shift right and compare with one pixel offset
    left = np.zeros_like(thinning_image)
    left[:, 1:-1] = thinning_image[:, 0:-2]
    left_mask = np.subtract(left, thinning_image)
    left_mask[:, 0:-2] = left_mask[:, 1:-1]
    cv.imshow('Left Shift Mask', left_mask)

    # Combine left and down masks to clean artifacts
    cv.bitwise_or(down_mask, down_mask, thinning_image)

    # Invert the final thinned image to match the input format (black ridges)
    output = np.zeros_like(thinning_image)
    output[thinning_image < 250] = 255

    # Visualize the result of morphological thinning
    cv.imshow('Morphological Thinning Result', output)
    
    cv.waitKey(0)
    cv.destroyAllWindows()

    return output
