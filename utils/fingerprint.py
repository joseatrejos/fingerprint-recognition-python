import cv2 as cv
import numpy as np
import math
import scipy.ndimage
from skimage.morphology import skeletonize as skelt
from scipy.spatial import distance

class Fingerprint:

    def __init__(self):
        pass

    def normalize_pixel_array(self, image, m0, v0):
        """
        Normalize the entire image using vectorized operations. Normalization is used to standardize the intensity values in an image by adjusting the range of
        gray level values so that they extend in a desired range of values and improve the contrast of the
        image. The main goal of normalization is to reduce the variance of the gray level value along the
        ridges to facilitate subsequent processing steps
        
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

        if v == 0:
            return np.full(image.shape, m0, dtype=np.uint8)

        # Vectorized calculation of deviation coefficient
        dev_coeff = np.sqrt(v0 * ((image - m) ** 2) / v)

        # Apply the normalization formula based on whether pixel is above or below the mean
        normalized_image = np.where(image > m, m0 + dev_coeff, m0 - dev_coeff)

        # Clip values to ensure they are in the 0-255 range
        normalized_image = np.clip(normalized_image, 0, 255)

        # Convert the result to uint8 for proper display
        return normalized_image.astype(np.uint8)

    def normalize(self, image, m0, v0):
        """
        Wrapper function to normalize the image intensity using the vectorized approach.
        
        Args:
            image (numpy.ndarray): The input image.
            m0 (float): Desired mean.
            v0 (float): Desired variance.

        Returns:
            numpy.ndarray: The normalized image.
        """
        return self.normalize_pixel_array(image, m0, v0)


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
    def normalise(self, img):
        """Normalize an image to have zero mean and unit variance."""
        return (img - np.mean(img)) / np.std(img)

    def create_segmented_and_variance_images(self, im, w, threshold=0.2):
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
        global_threshold = np.std(im) * threshold

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
        mask[image_variance < global_threshold] = 0

        # Smooth the mask using morphological operations
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (w * 2, w * 2))
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

        # Apply the mask to the segmented image
        segmented_image *= mask

        # Normalize the image only within the valid (non-background) regions
        im = self.normalise(im)
        valid_region = im[mask == 1]
        mean_val = np.mean(valid_region)
        std_val = np.std(valid_region)
        norm_img = (im - mean_val) / std_val

        return segmented_image, norm_img, mask

    def binarization(self, segmented_img: np.ndarray, method: str = 'gaussian') -> np.ndarray:
        # Rescale the segmented image to the range [0, 255]
        segmented_img_rescaled = cv.normalize(segmented_img, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    
        if method == 'mean':
            binary_img = cv.adaptiveThreshold(
                segmented_img_rescaled, 
                255, 
                cv.ADAPTIVE_THRESH_MEAN_C, 
                cv.THRESH_BINARY_INV, 
                15, 
                5
            )
        elif method == 'gaussian':
            binary_img = cv.adaptiveThreshold(
                segmented_img_rescaled, 
                255, 
                cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv.THRESH_BINARY_INV, 
                11, 
                2
            )

        else:
            raise ValueError("Invalid method. Use 'mean' or 'gaussian'.")

        return binary_img

    def calculate_angles(self, im, W, smooth=False):
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
            result = self.smooth_angles(result)

        return result

    def smooth_angles(self, angles):
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

    def get_line_ends(self, i, j, W, tang):
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

    def visualize_angles(self, im, mask, angles, W):
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
                    begin, end = self.get_line_ends(i, j, W, tang)

                    # Ensure the line is within image boundaries
                    begin = (max(0, min(begin[0], x-1)), max(0, min(begin[1], y-1)))
                    end = (max(0, min(end[0], x-1)), max(0, min(end[1], y-1)))
                    
                    cv.line(result, begin, end, color=(150, 150, 150), thickness=1)

        return result
    
    def frequest(self, im, orientim, kernel_size, minWaveLength, maxWaveLength):
        """
        Estimate the fingerprint ridge frequency within a small block of a fingerprint image.
        """
        rows, cols = np.shape(im)
        
        # Find mean orientation within the block
        cosorient = np.mean(np.cos(2 * orientim))
        sinorient = np.mean(np.sin(2 * orientim))
        block_orient = math.atan2(sinorient, cosorient) / 2
        
        # Rotate the image block so that the ridges are vertical
        rotim = scipy.ndimage.rotate(im, block_orient / np.pi * 180 + 90, axes=(1, 0), reshape=True, order=3, mode='nearest')

        # Crop the image to avoid invalid regions
        cropsze = int(np.fix(rows / np.sqrt(2)))
        offset = int(np.fix((rows - cropsze) / 2))
        rotim = rotim[offset:offset + cropsze, offset:offset + cropsze]

        # Sum down the columns to get a projection of the grey values down the ridges
        ridge_sum = np.sum(rotim, axis=0)
        dilation = scipy.ndimage.grey_dilation(ridge_sum, kernel_size, structure=np.ones(kernel_size))
        ridge_noise = np.abs(dilation - ridge_sum)
        peak_thresh = 2
        maxpts = (ridge_noise < peak_thresh) & (ridge_sum > np.mean(ridge_sum))
        maxind = np.where(maxpts)
        _, no_of_peaks = np.shape(maxind)
        
        # Determine the spatial frequency of the ridges
        if no_of_peaks < 2:
            freq_block = np.zeros(im.shape)
        else:
            waveLength = (maxind[0][-1] - maxind[0][0]) / (no_of_peaks - 1)
            if minWaveLength <= waveLength <= maxWaveLength:
                freq_block = 1 / np.double(waveLength) * np.ones(im.shape)
            else:
                freq_block = np.zeros(im.shape)
        
        return freq_block

    def ridge_freq(self, im, mask, orient, block_size, kernel_size, minWaveLength, maxWaveLength):
        """
        Estimate the fingerprint ridge frequency across a fingerprint image.
        """
        rows, cols = im.shape
        freq = np.zeros((rows, cols))

        for row in range(0, rows - block_size + 1, block_size):
            for col in range(0, cols - block_size + 1, block_size):
                image_block = im[row:row + block_size, col:col + block_size]
                angle_block = orient[row // block_size, col // block_size]
                if angle_block:
                    freq[row:row + block_size, col:col + block_size] = self.frequest(image_block, angle_block, kernel_size,
                    minWaveLength, maxWaveLength)

        # Apply the mask to the frequency image
        freq = freq * mask

        # Calculate the median frequency of the non-zero elements
        freq_1d = np.reshape(freq, (1, rows * cols))
        ind = np.where(freq_1d > 0)
        ind = np.array(ind)
        ind = ind[1, :]
        non_zero_elems_in_freq = freq_1d[0, ind]
        medianfreq = np.median(non_zero_elems_in_freq) * mask

        return medianfreq
    

    """
    The principle of gabor filtering is to modify the value of the pixels of an image, generally in order to
    improve its appearance. In practice, it is a matter of creating a new image using the pixel values
    of the original image, in order to select in the Fourier domain the set of frequencies that make up
    the region to be detected. The filter used is the Gabor filter with even symmetry and oriented at 0 degrees.

    The resulting image will be the spatial convolution of the original (normalized) image and one of
    the base filters in the direction and local frequency from the two directional and frequency maps
    https://airccj.org/CSCP/vol7/csit76809.pdf pg.91
    """

    def gabor_filter(self, im, orient, freq, kx=0.65, ky=0.65):
        """
        Gabor filter is a linear filter used for edge detection. Gabor filter can be viewed as a sinusoidal plane of
        particular frequency and orientation, modulated by a Gaussian envelope.
        
        Args:
            im (np.ndarray): Input image.
            orient (np.ndarray): Orientation image.
            freq (np.ndarray): Frequency image.
            kx (float): Scaling factor for the x-axis.
            ky (float): Scaling factor for the y-axis.
        
        Returns:
            np.ndarray: Gabor filtered image.
        """
        angleInc = 3
        im = np.double(im)
        rows, cols = im.shape
        return_img = np.zeros((rows, cols))

        # Round the array of frequencies to the nearest 0.01 to reduce the
        # number of distinct frequencies we have to deal with.
        freq_1d = freq.flatten()
        frequency_ind = np.array(np.where(freq_1d > 0))
        non_zero_elems_in_freq = freq_1d[frequency_ind]
        non_zero_elems_in_freq = np.double(np.round((non_zero_elems_in_freq * 100))) / 100
        unfreq = np.unique(non_zero_elems_in_freq)

        # Generate filters corresponding to these distinct frequencies and
        # orientations in 'angleInc' increments.
        sigma_x = 1 / unfreq * kx
        sigma_y = 1 / unfreq * ky
        block_size = int(np.round(3 * np.max([sigma_x, sigma_y])))
        array = np.linspace(-block_size, block_size, 2 * block_size + 1)
        x, y = np.meshgrid(array, array)

        # Gabor filter equation
        reffilter = np.exp(-(((np.power(x, 2)) / (sigma_x * sigma_x) + (np.power(y, 2)) / (sigma_y * sigma_y)))) * np.cos(2 * np.pi * unfreq[0] * x)
        filt_rows, filt_cols = reffilter.shape
        gabor_filter = np.zeros((180 // angleInc, filt_rows, filt_cols))

        # Generate rotated versions of the filter.
        for degree in range(0, 180 // angleInc):
            rot_filt = scipy.ndimage.rotate(reffilter, -(degree * angleInc + 90), reshape=False)
            gabor_filter[degree] = rot_filt

        # Convert orientation matrix values from radians to an index value that corresponds to round(degrees/angleInc)
        maxorientindex = np.round(180 / angleInc)
        orientindex = np.round(orient / np.pi * 180 / angleInc)
        for i in range(rows // 16):
            for j in range(cols // 16):
                if orientindex[i][j] < 1:
                    orientindex[i][j] += maxorientindex
                if orientindex[i][j] > maxorientindex:
                    orientindex[i][j] -= maxorientindex

        # Find indices of matrix points greater than block_size from the image boundary
        block_size = int(block_size)
        valid_row, valid_col = np.where(freq > 0)
        finalind = np.where((valid_row > block_size) & (valid_row < rows - block_size) & (valid_col > block_size) & (valid_col < cols - block_size))

        for k in range(np.shape(finalind)[1]):
            r = valid_row[finalind[0][k]]
            c = valid_col[finalind[0][k]]
            img_block = im[r - block_size:r + block_size + 1, c - block_size:c + block_size + 1]
            return_img[r][c] = np.sum(img_block * gabor_filter[int(orientindex[r // 16][c // 16]) - 1])

        gabor_img = 255 - np.array((return_img < 0) * 255).astype(np.uint8)

        return gabor_img
        
    def skeletonize(self, image_input):
        """
        Skeletonization reduces binary objects to 1 pixel wide representations using Zhang-Suen algorithm.
        This helps extract minutiae from the thinned fingerprint ridges.
        :param image_input: 2d array uint8
        :return: Skeletonized image
        """
        # Ensure the input image is binary
        assert image_input.dtype == np.uint8, "Input image must be of type uint8"
        assert np.array_equal(np.unique(image_input), [0, 255]), "Input image must be binary (0 and 255)"

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
            # Ensure the input image is binary
            assert image.dtype == np.uint8, "Input image must be of type uint8"
            assert np.array_equal(np.unique(image), [0, 255]), "Input image must be binary (0 and 255)"

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

    def minutiae_at(self, pixels, i, j, kernel_size):
        """
        Detect minutiae points using the Crossing Number method and calculate the angle.
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
                angle = np.arctan2(values[1] - values[7], values[3] - values[5]) * 180 / np.pi
                return "ending", angle
            if crossings == 3:  # Bifurcation
                angle = np.arctan2(values[1] - values[7], values[3] - values[5]) * 180 / np.pi
                return "bifurcation", angle

        return "none", None

    def calculate_minutiaes(self, im, kernel_size=3):
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
                minutiae, angle = self.minutiae_at(binary_image, j, i, kernel_size)
                if minutiae != "none":
                    cv.circle(result, (i, j), radius=2, color=colors[minutiae], thickness=2)
                    minutiae_points.append((i, j, angle))

        return result, minutiae_points

    def open_images(self, img_paths):
        """
        Open images from given paths and return them as grayscale images.
        """
        return [cv.imread(img_path, cv.IMREAD_GRAYSCALE) for img_path in img_paths]