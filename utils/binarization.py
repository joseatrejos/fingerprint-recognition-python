import cv2 as cv
import numpy as np

def binarization(segmented_img: np.ndarray) -> np.ndarray:
    # Rescale the segmented image to the range [0, 255]
    segmented_img_rescaled = cv.normalize(segmented_img, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
 
    binary_img = cv.adaptiveThreshold(
        segmented_img_rescaled, 
        255, 
        cv.ADAPTIVE_THRESH_MEAN_C, 
        cv.THRESH_BINARY_INV, 
        15, 
        5
    )
    #     segmented_img_rescaled, 
    #     255, 
    #     cv.ADAPTIVE_THRESH_GAUSSIAN_C,
    #     cv.THRESH_BINARY, 
    #     11, 
    #     2
    # )

    return binary_img