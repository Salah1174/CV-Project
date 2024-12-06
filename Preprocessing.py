import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import BarCodeExtraction 

def AdaptiveGaussian(img):
    #thresholding
    thresh1 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY, 31, 5)

    thresh2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 31, 5)

    # the window showing output images
    # with the corresponding thresholding
    # techniques applied to the input image
    cv2.imshow('Adaptive Mean', thresh1)
    cv2.imshow('Adaptive Gaussian', thresh2)


    # De-allocate any associated memory usage
    if cv2.waitKey(0) & 0xff == 27:
    	cv2.destroyAllWindows()
# from untitled4 import detect_salt_and_pepper_noise
# All images passes through this function 
# If salt and pepper true and freq domain true else false
def detect_salt_and_pepper_noise(image, black_threshold=20, white_threshold=80):
    total_pixels = image.size
    black_pixels = np.sum(image == 0)
    white_pixels = np.sum(image == 255)

    black_ratio = black_pixels / total_pixels * 100
    white_ratio = white_pixels / total_pixels * 100

    if black_ratio > black_threshold or white_ratio < white_threshold:
        return True

    return False

#if detect_salt_and_pepper_noise is true to determine whether it is salt and pepper(false) or freq domain(true)
def plot_time_domain(image_path, row=None, col=None):
    """
    Plots the time-domain representation of pixel intensities from an image.
    
    Args:
        image_path (str): Path to the input image.
        row (int, optional): Row index to extract pixel intensities. If None, the middle row is used.
        col (int, optional): Column index to extract pixel intensities. If None, the middle column is used.
    """
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image not found or unable to load!")
    
    # Get image dimensions
    rows, cols = image.shape
    
    # Determine which row or column to extract
    if row is None and col is None:
        row = rows // 2  # Default to the middle row
    elif row is not None and (row < 0 or row >= rows):
        raise ValueError(f"Row index out of bounds. Must be between 0 and {rows - 1}.")
    elif col is not None and (col < 0 or col >= cols):
        raise ValueError(f"Column index out of bounds. Must be between 0 and {cols - 1}.")
    
    # Extract the pixel intensities
    if row is not None:
        intensities = image[row, :]  # Pixel intensities from the row
        x = np.arange(cols)  # X-axis: column indices
        label = f"Row {row}"
    else:
        intensities = image[:, col]  # Pixel intensities from the column
        x = np.arange(rows)  # X-axis: row indices
        label = f"Column {col}"

    # Plot the time-domain representation
    plt.figure(figsize=(10, 5))
    plt.plot(x, intensities, label=label, color='b')
    plt.title(f"Time-Domain Representation of {label}")
    plt.xlabel("Pixel Index")
    plt.ylabel("Pixel Intensity")
    plt.grid()
    plt.legend()
    plt.show()
    return intensities


def analyze_peaks(signal, height=None, distance=50, tolerance=5, sampling_rate=1.0, plot=True):
    """
    Analyzes peaks in the time-domain signal and computes frequency domain characteristics.
    Returns the filter type based on the detected dominant frequencies.
    """
    # Dynamically adjust height if not provided
    if height is None:
        height = np.mean(signal) + 0.5 * np.std(signal)

    # Detect peaks
    peaks, _ = find_peaks(signal, height=height, distance=distance)

    if len(peaks) == 0:
        print(
            "No peaks detected. Consider adjusting the 'height' or 'distance' parameters.")
        return {
            'peaks': [],
            'peak_distances': [],
            'are_distances_equal': False,
            'dominant_frequencies': [],
            'magnitudes': [],
            'filter_type': None  # No filter type if no peaks are detected
        }

    # Calculate peak distances
    peak_distances = np.diff(peaks)
    if len(peak_distances) > 0:
        are_distances_equal = np.allclose(peak_distances, peak_distances[0], atol=tolerance)
    else:
        are_distances_equal = False

    # Compute FFT
    fft_result = np.fft.fft(signal)
    fft_magnitude = np.abs(fft_result)
    fft_frequencies = np.fft.fftfreq(len(signal), d=1/sampling_rate)

    # Consider positive frequencies
    positive_frequencies = fft_frequencies[:len(signal)//2]
    positive_magnitude = fft_magnitude[:len(signal)//2]

    # Detect peaks in FFT magnitude
    fft_peaks, _ = find_peaks(
        positive_magnitude, height=np.mean(positive_magnitude))
    dominant_frequencies = positive_frequencies[fft_peaks]
    dominant_magnitudes = positive_magnitude[fft_peaks]

    # Print the detected dominant frequencies and their magnitudes
    print("Detected Dominant Frequencies (Hz):", dominant_frequencies * 10000)
    print("Corresponding Magnitudes:", dominant_magnitudes)

    filter_type = None  # Default filter type is None

    # Determine the filter type based on the dominant frequencies
    if dominant_frequencies.size > 0:
        if dominant_frequencies[0] * 10000 < 60:
            filter_type = "Low-pass"
            print("Low Frequency: Low-pass filter likely used.")
        else:
            filter_type = "High-pass"
            print("High Frequency: High-pass filter likely used.")
    else:
        filter_type = "Low-pass"  # Default to low-pass if no dominant frequencies

    # Plot the signal and frequency domain if requested
    if plot:
        plt.figure(figsize=(12, 6))

        # Plot time-domain signal
        plt.subplot(2, 1, 1)
        plt.plot(signal, label='Signal')
        plt.plot(peaks, signal[peaks], 'ro', label='Detected Peaks')
        plt.title('Time-Domain Signal with Peaks')
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid()

        # Plot frequency-domain
        plt.subplot(2, 1, 2)
        plt.plot(positive_frequencies, positive_magnitude,
                 label='FFT Magnitude')
        plt.plot(dominant_frequencies, dominant_magnitudes,
                 'ro', label='Dominant Frequencies')
        plt.title('Frequency-Domain Analysis')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.legend()
        plt.grid()

        plt.tight_layout()
        plt.show()

    # Return analysis results including the filter type
    return {
        'peaks': peaks,
        'peak_distances': peak_distances,
        'are_distances_equal': are_distances_equal,
        'dominant_frequencies': dominant_frequencies,
        'magnitudes': dominant_magnitudes,
        'filter_type': filter_type
    }


def are_peaks_equally_spaced(signal, height=None, distance=50, tolerance=5):
    """
    Checks if the spaces (differences) between peaks in the given signal are approximately equal.

    Parameters:
        signal (numpy array): The input time-domain signal (1D array).
        height (float): Minimum height for a point to be considered a peak. Default is 150.
        distance (int): Minimum distance between consecutive peaks. Default is 50.
        tolerance (int): Allowed deviation for equal distances. Default is 5.

    Returns:
        bool: True if the differences between consecutive peaks are approximately equal, False otherwise.
    """
    # Detect peaks
    peaks, _ = find_peaks(signal, height=height, distance=distance)
    
    # Calculate peak distances
    peak_distances = np.diff(peaks)
    
    # Check if all distances are approximately equal
    if len(peak_distances) > 0:
        return np.allclose(peak_distances, peak_distances[0], atol=tolerance)
    else:
        return False

# if are_peaks_equally_spaced returns true
def try_highpass(dft_img, limit, gaussian: bool = False, keep_dc: bool = False):
    mask = ~give_me_circle_mask_nowww(dft_img.shape, limit)
    if (gaussian):
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
    if (keep_dc):
        mask[dft_img.shape[0]//2, dft_img.shape[1]//2] = 255
    dft_img_shifted = np.fft.fftshift(dft_img)
    dft_img_shifted_highpass = np.multiply(dft_img_shifted, mask)
    freqimg = plot_shifted_fft_and_ifft(dft_img_shifted_highpass)
    return freqimg


def try_lowpass(dft_img, limit, gaussian: bool = False):
    mask = give_me_circle_mask_nowww(dft_img.shape, limit)
    if (gaussian):
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
    dft_img_shifted = np.fft.fftshift(dft_img)
    dft_img_shifted_lowpass = np.multiply(dft_img_shifted, mask)
    freqimg = plot_shifted_fft_and_ifft(dft_img_shifted_lowpass)
    return freqimg


# if are_peaks_equally_spaced returns False
def noiseReductionSaltAndPeper(img):
    # img = cv2.imread(image_path, 0)

    if img is None:
        raise ValueError("Image not found.")

    # Taking a matrix of size 5 as the kernel
    kernel = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], np.uint8)

    vertical_kernel_3x3 = np.array([[1], [1], [1]], np.uint8)
    vertical_kernel_5x5 = np.array([[1], [1], [1], [1], [1], [1], [1]], np.uint8)

    img_dilation = cv2.dilate(img, kernel, iterations=1)
    img_erosion = cv2.erode(img_dilation, vertical_kernel_3x3, iterations=1)

    bilateral_filt_closing = cv2.bilateralFilter(
        img_erosion, d=9, sigmaColor=75, sigmaSpace=75
    )

    bilateral_filt_dil = cv2.bilateralFilter(
        img_dilation, d=9, sigmaColor=75, sigmaSpace=75
    )

    bilateral_filt_closing_ero = cv2.erode(
        bilateral_filt_closing, vertical_kernel_3x3, iterations=1
    )

    bilateral_filt_dil_ero = cv2.erode(
        bilateral_filt_dil, vertical_kernel_3x3, iterations=1
    )

    bilateral_filt_dil_ero_ero = cv2.erode(
        bilateral_filt_dil, vertical_kernel_5x5, iterations=1
    )

    # cv2.imshow("Input", img)
    # cv2.imshow('Dilation', img_dilation)
    # cv2.imshow('Erosion', img_erosion)
    # cv2.imshow('Bilateral Erosion', bilateral_filt_closing)  # Not Bad
    # cv2.imshow('Bilateral Dilation', bilateral_filt_dil)
    # cv2.imshow('Bilateral Closing Erosion', bilateral_filt_closing_ero)
    # cv2.imshow("Bilateral Dilation Erosion", bilateral_filt_dil_ero)
    # cv2.imshow("Bilateral Dilation Erosion Erosion", bilateral_filt_dil_ero_ero)

    # cv2.imshow('Laplacian Ero', img_dst_ero)

    # cv2.imshow('Median', img_dst_median)

    cv2.waitKey(0)

    return bilateral_filt_dil_ero_ero

    # laplacian = cv2.Laplacian(img_median, cv2.CV_64F)
    # laplacian_abs = cv2.convertScaleAbs(laplacian)
    # dst = cv2.Laplacian(img, CV_16S, ksize=kernel_size)


def sharpen_image(image):
    # Define the sharpening kernel
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

    # Apply the sharpening filter
    sharpened = cv2.filter2D(image, -1, kernel)

    return sharpened

def enhance_barcode(image):
    sharpened_image = sharpen_image(image)

    return sharpened_image

def give_me_circle_mask_nowww(mask_size, radius):
    mask = np.zeros(mask_size)
    cy = mask.shape[0] // 2
    cx = mask.shape[1] // 2
    return cv2.circle(mask, (cx, cy), radius, (255, 255, 255), -1).astype(np.uint8)

def plot_shifted_fft_and_ifft(dft_img_shifted):
    img = np.fft.ifft2(np.fft.ifftshift(dft_img_shifted))
    fig, (ax1, ax2) = plt.subplots(figsize=(10, 5), nrows=1, ncols=2)
    ax1.set(yticks=[0, img.shape[0]//2, img.shape[0] - 1],
            yticklabels=[-img.shape[0]//2, 0, img.shape[0]//2 - 1])
    ax1.set(xticks=[0, img.shape[1]//2, img.shape[1] - 1],
            xticklabels=[-img.shape[1]//2, 0, img.shape[1]//2 - 1])
    # ax1.imshow(np.abs(dft_img_shifted)**0.1, cmap='gray')
    # ax2.imshow(np.abs(img), cmap='gray')

    # img = np.abs(img.astype(np.uint16))
    img = np.abs(img)  # Get magnitude
    img = img.astype(np.uint16) 
    # plt.imsave("AX2.jpg",img)
    # img = cv2.imread("AX2.jpg")
    # cv2.imshow("AX2", img)
    # ax2 = ax2.astype(np.uint32)
    # ax2.imsave("Test Cases\\11 - bayza 5ales di bsara7a#3.jpg", np.abs(img), p)
    # plt.show()
    return img

# This is what im gonna use **********************************
# increase <20 w >220 
def increase_contrast(image):
    # Apply linear contrast stretching
    min_val, max_val = np.min(image), np.max(image)
    contrast_image = (image - min_val) * (255 / (max_val - min_val))
    contrast_image = np.uint8(contrast_image)  # Convert back to uint8 type

    return contrast_image

def calc_avg_intensity(image):
    return np.mean(image)

def apply_dynamic_threshold(image, avg_intensity):
    if (avg_intensity>120 and avg_intensity<130 ): #check if gray compressed (only one that causes issues with threshold is gray compressed)
      image = increase_contrast(image)
      avg_intensity = calc_avg_intensity(image)

    threshold_value = int(avg_intensity * 0.75)
    _, thresholded_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    return thresholded_image
def applydynamic_threshold(image, avg_intensity):
    if (avg_intensity>120 and avg_intensity<130 ): #check if gray compressed (only one that causes issues with threshold is gray compressed)
      image = increase_contrast(image)
      avg_intensity = calc_avg_intensity(image)

    threshold_value = int(avg_intensity * 0.75)
    _, thresholded_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    return thresholded_image

def preprocessing(img , image_path):

    avg_intensity =calc_avg_intensity(img)
    thresh = applydynamic_threshold(img,avg_intensity)
    is_salt_pepper =detect_salt_and_pepper_noise(thresh)

    print(img.shape)
    print(is_salt_pepper)
    if is_salt_pepper:
        intensities =plot_time_domain(image_path,300,300)
        print(is_salt_pepper)
        # Call the function
        results = analyze_peaks(intensities, height=150, distance=50, tolerance=5)
        peaks_equally_spaced = are_peaks_equally_spaced(intensities,height = 150)
        # Output results
        print("Peak indices:", results['peaks'])
        print("Peak distances:", results['peak_distances'])
        print("Are distances approximately equal?", results['are_distances_equal'])
        
        
        if peaks_equally_spaced:
            # avg_int2 = calc_avg_intensity(img)
            dft_img = np.fft.fft2(img)
            # dft_img_shift = np.fft.fftshift(dft_img)
            if results['filter_type'] == "High-pass":    
                freqimg = try_highpass(dft_img, 20, gaussian=False, keep_dc=True)
                # If try_highpass produces complex numbers, extract magnitude
                freqimg = np.abs(freqimg)
                # Ensure the image is in a uint8 format (0-255) for OpenCV compatibility
                freqimg = np.uint8(255 * (freqimg / np.max(freqimg)))  # Normalize to 0-255
                # If needed, ensure single-channel image
                if len(freqimg.shape) > 2:
                    freqimg = cv2.cvtColor(freqimg, cv2.COLOR_BGR2GRAY)
                contrast =increase_contrast(freqimg)
                BarCodeExtraction.detect_barcode(contrast)
                cv2.imshow("Frequency Domain", freqimg)

            elif results['filter_type'] == "Low-pass":
                freqimg = try_lowpass(dft_img, 150, gaussian=True)
                # If try_highpass produces complex numbers, extract magnitude
                freqimg = np.abs(freqimg)
                # Ensure the image is in a uint8 format (0-255) for OpenCV compatibility
                # Normalize to 0-255
                freqimg = np.uint8(255 * (freqimg / np.max(freqimg)))
                # If needed, ensure single-channel image
                if len(freqimg.shape) > 2:
                    freqimg = cv2.cvtColor(freqimg, cv2.COLOR_BGR2GRAY)
                contrast = increase_contrast(freqimg)
                BarCodeExtraction.detect_barcode(contrast)
                cv2.imshow("Frequency Domain", freqimg)
        else:
            avg_intensity =calc_avg_intensity(img)
            thresholded_image = apply_dynamic_threshold(img,avg_intensity)
            # cv2.imshow("Thresholded Image", thresholded_image)
            kernel = np.ones((3, 3), np.uint8)
            dil_img = cv2.dilate(thresholded_image, kernel, iterations=1)
            erode_img = cv2.erode(dil_img, kernel, iterations=1)
            
            # cv2.imshow("Dilated Image", erode_img)
            BarCodeExtraction.detect_barcode(erode_img)
    else:
        print("No Salt and Pepper Noise")
        
        avg_intensity =calc_avg_intensity(img)
        brightness_threshold = 250 
        
        if avg_intensity < brightness_threshold:
            thresholded_image = apply_dynamic_threshold(img,avg_intensity)
            # cv2.imshow("Thresholded Image", thresholded_image)
            kernel = np.ones((3, 3), np.uint8)
            dil_img = cv2.dilate(thresholded_image, kernel, iterations=1)
            erode_img = cv2.erode(dil_img, kernel, iterations=1)
            result =increase_contrast(erode_img)
            BarCodeExtraction.detect_barcode(result)
        else: 
            result =increase_contrast(img)
            BarCodeExtraction.detect_barcode(result)
    











