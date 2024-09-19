import cv2 as cv
import numpy as np

def output_images(output_imgs):
    # Resize all images to match dimensions before concatenation
    output_imgs = resize_to_match(output_imgs)

    for i in range(len(output_imgs)):
        if len(output_imgs[i].shape) == 2:  # Check if it's grayscale
            if output_imgs[i].dtype != np.uint8:  # Normalize and convert if needed
                output_imgs[i] = cv.normalize(output_imgs[i], None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
            output_imgs[i] = cv.cvtColor(output_imgs[i], cv.COLOR_GRAY2RGB)

    # Check for image shapes for debugging
    for img in output_imgs:
        print(f"Image shape: {img.shape}")

    try:
        # Ensure that both parts to be concatenated have the same width
        top_half = np.concatenate(output_imgs[:4], axis=1)
        bottom_half = np.concatenate(output_imgs[4:], axis=1)
        
        # Make sure the two halves have the same width by resizing the smaller one
        if top_half.shape[1] != bottom_half.shape[1]:
            target_width = max(top_half.shape[1], bottom_half.shape[1])
            if top_half.shape[1] < target_width:
                top_half = cv.resize(top_half, (target_width, top_half.shape[0]))
            else:
                bottom_half = cv.resize(bottom_half, (target_width, bottom_half.shape[0]))
        
        # Finally, concatenate top and bottom halves vertically
        results = np.concatenate([top_half, bottom_half], axis=0).astype(np.uint8)
        return results
    except ValueError as e:
        print(f"Concatenation error: {e}")
        raise

def resize_to_match(img_list):
    """Resize all images in img_list to match the dimensions of the largest image."""
    max_height = max(img.shape[0] for img in img_list)
    max_width = max(img.shape[1] for img in img_list)

    resized_images = []
    for img in img_list:
        resized_img = cv.resize(img, (max_width, max_height), interpolation=cv.INTER_LINEAR)
        resized_images.append(resized_img)

    return resized_images
