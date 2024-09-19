import os, dotenv
import cv2 as cv
import numpy as np
from utils.poincare import calculate_singularities
from utils.segmentation import create_segmented_and_variance_images
from utils.normalization import normalize
from utils.gabor_filter import gabor_filter
from utils.frequency import ridge_freq
from utils import orientation
from utils.crossing_number import *
from tqdm import tqdm
from utils.skeletonize import skeletonize
from utils.show_image import show_image
from utils.binarization import binarization
from utils.output import output_images

dotenv.load_dotenv()

def preprocess_image(input_img: np.ndarray) -> np.ndarray:
    """
        Preprocess the image using the following pipeline: gaussian blur -> normalization -> segmentation -> binarization -> ridge orientation -> ridge frequency -> gabor filter -> thinning -> minutiaes -> singularities.

        Args:
            input_img (numpy.ndarray): The input image to be preprocessed.

        Returns:
            numpy.ndarray: The preprocessed image.
        
        Raises:
            ValueError: If the input image is not a numpy.ndarray.
    """
    block_size = 16

    # 1. Apply Gaussian Blur to reduce noise
    blurred_img = cv.GaussianBlur(input_img, (5, 5), 0)
    # show_image("Blurred Image", blurred_img)

    # 2. Normalization - removes the effects of sensor noise and finger pressure differences.
    normalized_img = normalize(blurred_img.copy(), np.mean(blurred_img), np.std(blurred_img))
    # show_image("Normalized Image", normalized_img)

    # 3. ROI Segmentation and Variance Calculation
    segmented_img, variance_img, mask = create_segmented_and_variance_images(normalized_img, block_size, 0.2)
    # show_image("Segmented Image", segmented_img)
    # show_image("Variance Image", variance_img)

    # 4. Binarization (thresholding) applied to the segmented image
    binary_img = binarization(segmented_img)
    # show_image("Binarized Image", binary_img)

    # 5. Ridge Orientation Calculation based on the binary image
    angles = orientation.calculate_angles(binary_img, W=block_size, smooth=True)
    orientation_img = orientation.visualize_angles(segmented_img, mask, angles, W=block_size)
    # show_image("Orientation Image", orientation_img)

    # 6. Ridge frequency estimation in Wavelet Domain using the variance image
    # TODO: This is not working as expected and returns a gray image
    freq = ridge_freq(variance_img, mask, angles, block_size, kernel_size=5, minWaveLength=5, maxWaveLength=15)
    # show_image("Ridge Frequency Image", freq)

    # 7. Gabor filtering - Enhance ridges using Gabor filter
    gabor_img = gabor_filter(variance_img, angles, freq)
    # show_image("Gabor Filtered Image", gabor_img)

    # 8. Skeletonization (thinning)
    thin_image = skeletonize(gabor_img)
    # show_image("Skeleton Image", thin_image)

    # 9. Minutiae detection
    minutiaes_img, minutiae_points = calculate_minutiaes(thin_image)
    # show_image("Minutiaes Image", minutiaes_img)

    # 10. Singularities detection
    singularities_img = calculate_singularities(thin_image, angles, 1, block_size, mask)
    # show_image("Singularities Image", singularities_img)

    # Visualize pipeline stage by stage
    output_imgs = [
        input_img, blurred_img, normalized_img, segmented_img, variance_img, 
        binary_img, orientation_img, freq, gabor_img, thin_image, 
        minutiaes_img, singularities_img
    ]

    results = output_images(output_imgs)
    
    return minutiaes_img, minutiae_points, results

def open_images(img_paths):
    return [cv.imread(img_path, cv.IMREAD_GRAYSCALE) for img_path in img_paths]

if __name__ == "__main__":
    # Open images
    img_paths = [
        f"""{os.getenv("IMAGE_PATH")}/huella1.png""", 
        f"""{os.getenv("IMAGE_PATH")}/huella2.png"""
    ]
    output_dir = "./output/"

    # Open images in grayscale
    images = open_images(img_paths)

    # Image pipeline
    processed_images = []
    minutiae_sets = []
    os.makedirs(output_dir, exist_ok=True)
    for i, img in enumerate(tqdm(images)):
        resulting_img, minutiae_points, results = preprocess_image(img)

        # Save image and minutiae points
        processed_images.append(resulting_img)
        minutiae_sets.append(minutiae_points)

        # Write results to disk in a single image
        cv.imwrite(os.path.join(output_dir, f"{i}.png"), results)

    # Match minutiae and compute dissimilarity
    dissimilarity_index = compute_dissimilarity(minutiae_sets[0], minutiae_sets[1])
    print(f"Dissimilarity Index: {dissimilarity_index}")
    