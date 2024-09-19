import os
import cv2 as cv
import numpy as np
from scipy.spatial import distance
from tqdm import tqdm

def minutiae_at(pixels, i, j, kernel_size):
    """
    Detect minutiae points using the Crossing Number method.
    """
    if pixels[i][j] == 1:
        if kernel_size == 3:
            cells = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
        else:
            cells = [(-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (0, 2), (1, 2), (2, 1), (2, 0)]

        values = [pixels[i + l][j + k] for k, l in cells]

        # count crossings (0 to 1 transitions)
        crossings = sum(abs(values[k] - values[k + 1]) for k in range(len(values) - 1)) // 2

        if crossings == 1:  # Ridge ending
            return "ending"
        if crossings == 3:  # Bifurcation
            return "bifurcation"

    return "none"


def calculate_minutiaes(im, kernel_size=3):
    """
    Calculate and return the minutiae points in the image using the Crossing Number method.
    """
    binary_image = np.zeros_like(im)
    binary_image[im < 10] = 1.0
    binary_image = binary_image.astype(np.int8)

    (y, x) = im.shape
    result = cv.cvtColor(im, cv.COLOR_GRAY2RGB)
    colors = {"ending": (150, 0, 0), "bifurcation": (0, 150, 0)}
    minutiae_points = []

    # Detect minutiae
    for i in range(1, x - kernel_size // 2):
        for j in range(1, y - kernel_size // 2):
            minutiae = minutiae_at(binary_image, j, i, kernel_size)
            if minutiae != "none":
                cv.circle(result, (i, j), radius=2, color=colors[minutiae], thickness=2)
                minutiae_points.append((i, j, 0))  # Placeholder for angle (orientation to be added later)

    return result, minutiae_points


def match_minutiaes(minutiae_A, minutiae_B, distance_threshold=10, angle_threshold=15):
    """
    Match minutiae points between two sets based on Euclidean distance and orientation.
    """
    matched_minutiae = 0
    used_B_indices = set()

    for minutia_A in minutiae_A:
        (x_A, y_A, angle_A) = minutia_A
        for i, minutia_B in enumerate(minutiae_B):
            if i in used_B_indices:
                continue  # Skip already matched minutiae
            (x_B, y_B, angle_B) = minutia_B

            if distance.euclidean((x_A, y_A), (x_B, y_B)) < distance_threshold and abs(angle_A - angle_B) < angle_threshold:
                matched_minutiae += 1
                used_B_indices.add(i)
                break

    return matched_minutiae


def compute_dissimilarity(minutiae_A, minutiae_B, distance_threshold=10, angle_threshold=15):
    """
    Compute the dissimilarity index between two fingerprints based on minutiae.
    """
    N_m = match_minutiaes(minutiae_A, minutiae_B, distance_threshold, angle_threshold)
    N_t = len(minutiae_A) + len(minutiae_B)

    if N_t == 0:
        return 1.0  # Avoid division by zero if no minutiae

    dissimilarity_index = 1 - (2 * N_m / N_t)
    return dissimilarity_index


def open_images(img_paths):
    """
    Open images from given paths and return them as grayscale images.
    """
    return [cv.imread(img_path, cv.IMREAD_GRAYSCALE) for img_path in img_paths]


if __name__ == "__main__":
    img_paths = [
        f"""{os.getenv("IMAGE_PATH")}/huella1.png""", 
        f"""{os.getenv("IMAGE_PATH")}/huella2.png"""
    ]
    output_dir = "./output/"

    # Open images in grayscale
    images = open_images(img_paths)

    # Process images
    processed_images = []
    minutiae_sets = []
    os.makedirs(output_dir, exist_ok=True)
    
    for i, img in enumerate(tqdm(images)):
        results, minutiae_points = preprocess_image(img)
        processed_images.append(results)
        minutiae_sets.append(minutiae_points)

    # Match minutiae and compute dissimilarity
    dissimilarity_index = compute_dissimilarity(minutiae_sets[0], minutiae_sets[1])
    print(f"Dissimilarity Index: {dissimilarity_index}")
