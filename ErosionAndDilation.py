# Python program to demonstrate erosion and
# dilation of images.
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import Decoder 

def AdaptiveGaussian(img):
	
    # path to input image is specified and
    # image is loaded with imread command
    # image1 = cv2.imread(image_path)

    # cv2.cvtColor is applied over the
    # image input with applied parameters
    # to convert the image in grayscale
    # img = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

    # applying different thresholding
    # techniques on the input image
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
def noiseReductionSaltAndPeper(image_path):
    img = cv2.imread(image_path, 0)

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

    # img_median = cv2.medianBlur(img_erosion, 3)

    # img_dilation_2 = cv2.dilate(img_median, vertical_kernel, iterations=1)
    # img_erosion_2 = cv2.erode(img_dilation_2, vertical_kernel, iterations=1)
    # img_median_2 = cv2.medianBlur(img_erosion_2, 3)

    # img_dst_dil = enhance_barcode(img_dilation_2)

    # img_dst_ero = enhance_barcode(img_erosion_2)

    # img_dst_median = enhance_barcode(img_median)

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

    # The first parameter is the original image,
    # kernel is the matrix with which image is
    # convolved and third parameter is the number
    # of iterations, which will determine how much
    # you want to erode/dilate a given image.

    # Step 2: Apply the Laplacian filter to detect edges
    # laplacian = cv2.Laplacian(img_median, cv2.CV_64F)

    # Step 3: Convert the result to uint8 (image format)
    # laplacian_abs = cv2.convertScaleAbs(laplacian)

    # dst = cv2.Laplacian(img, CV_16S, ksize=kernel_size)

def make_columns_uniform(image):
    height, width = image.shape
    num_parts = 10
    part_height = height // num_parts
    
    for x in range(width):
        part_most_common = []

        for part in range(num_parts):
            start_idx = part * part_height
            end_idx = (part + 1) * part_height if part != num_parts - 1 else height
            part_pixels = image[start_idx:end_idx, x]

            # Find  most common pixel value in the part
            unique, counts = np.unique(part_pixels, return_counts=True)
            
            for idx, item in enumerate(unique):
                if item >= 150:
                    unique[idx] = 255
                else:
                    unique[idx] = 0
            
            mode_pixel = unique[np.argmax(counts)]
            part_most_common.append(mode_pixel)
        
        # Determine the most common value among the parts
        final_unique, final_counts = np.unique(part_most_common, return_counts=True)
        most_common_pixel = final_unique[np.argmax(final_counts)]

        # Set all pixels in the column to the most common value
        image[:, x] = most_common_pixel
        
            
        # # Get the pixels in the current column
        # column_pixels = image[:, x]
        # cnt = [0,0]
        # # Find the most common pixel value in the column
        # unique, counts = np.unique(column_pixels, return_counts=True)
        # # mode_pixel = unique[np.argmax(counts)]

        # for idx, item in enumerate(unique):
        #     if(item >= 128):
        #         unique[idx] = 255
        #         cnt[0]+=1
        #     else:
        #         unique[idx] = 0
        #         cnt[1]+=1

        # print(unique)
        # print(unique[np.argmax(counts)])
        # # Set all pixels in the column to the most common value
        # image[:, x] = 0 if cnt[1] > cnt[0] else 255
    

    return image

# This function reorder the corners points appropriatly
# Helped significantly with warp function
def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = myPoints.sum(1)
    myPointsNew[1] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[0] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew

def detect_barcode(image):   
    
    # 1---->canny

    # blurred = cv2.GaussianBlur(image, (9, 9), 0)
    # edges = cv2.Canny(image, 50, 150)  
    # blurred = cv2.blur(edges, (9, 9))
    # _, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)

    #2---->blur before sobel

    blurred = cv2.GaussianBlur(image, (5, 5),0)
    # compute the Scharr gradient magnitude representation of the images in both the x and y direction 
    gradX = cv2.Sobel(blurred, ddepth = cv2.CV_32F, dx = 1, dy = 0, ksize = -1)
    gradY = cv2.Sobel(blurred, ddepth = cv2.CV_32F, dx = 0, dy = 1, ksize = -1)
    
    #3---->sobel bas

    # # compute the Scharr gradient magnitude representation of the images in both the x and y direction 
    # gradX = cv2.Sobel(image, ddepth = cv2.CV_32F, dx = 1, dy = 0, ksize = -1)
    # gradY = cv2.Sobel(image, ddepth = cv2.CV_32F, dx = 0, dy = 1, ksize = -1)

    # gradient 2, 3 only

    # subtract the y-gradient from the x-gradient
    gradient = cv2.subtract(gradX, gradY) 
    gradient = cv2.convertScaleAbs(gradient)
    # blur and threshold the image 
    blurred = cv2.blur(gradient, (9, 9))
    (_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)


    # construct a closing kernel and apply it to the thresholded image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # perform a series of erosions and dilations
    # cv2.imshow('closed1', closed)
    closed = cv2.erode(closed, None, iterations = 4)

    closed = cv2.dilate(closed, None, iterations = 4)

    # find the contours in the thresholded image,
    #  then sort the contours by their area,
    #  keeping only the largest one
    (cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnt = sorted(cnts, key = cv2.contourArea, reverse = True)[0] 
    
    if cv2.contourArea(cnt) > 0:  # Check if contour is valid

        # compute the rotated bounding box of the largest contour
        rect = cv2.minAreaRect(cnt)
        box = np.int32(cv2.boxPoints(rect))
        box = reorder(box)
        
       
        # Coordinates of each corner
        ax = box.item(0)
        ay = box.item(1)

        bx = box.item(2)
        by = box.item(3)

        cx = box.item(4)
        cy = box.item(5)

        dx = box.item(6)
        dy = box.item(7)

        # box coordinates
        x, y, w, h = cv2.boundingRect(box) 
        
        # print(box)
        border_threshold = 10
        # height, width = image.shape[:2]
        widthA = np.sqrt(((cx - dx) ** 2) + ((cy - dy) ** 2))
        widthB = np.sqrt(((ax - bx) ** 2) + ((ay - by) ** 2))
        width = max(int(widthA), int(widthB))
        
        heightA = np.sqrt(((ax - dx) ** 2) + ((ay - dy) ** 2))
        heightB = np.sqrt(((bx - cx) ** 2) + ((by - cy) ** 2))
        height = max(int(heightA), int(heightB))
        
        # print(width, height)
        # print (x, y, w, h)
        # print(ax, ay, bx, by, cx, cy, dx, dy)
        # if x < border_threshold:
        #     x = 0
        # if x + w > width - border_threshold:
        #     w = width - x
        # if y < border_threshold:
        #     y = 0
        # if y + h > height - border_threshold:
        #     h = height - y

        # Draw the extended bounding box
        # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Crop the barcode 
        # cropped_barcode = image[y:y+height, x:x+width] 

        
        pts1 = np.float32([[bx, by], [ax, ay], [cx, cy], [dx, dy]])
        pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
        
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        img_prespective = cv2.warpPerspective(image, matrix, (width, height))
        # cv2.circle(image, (ax,ay), 5, (0,0,255), 20)
        # cv2.circle(image, (bx,by), 1, (0,0,255), 10)
        # cv2.circle(image, (cx,cy), 1, (0,0,255), 10)
        # cv2.circle(image, (dx,dy), 1, (0,0,255), 10)
        
        # draw a bounding box arounded the detected barcode and display the image
        # cv2.drawContours(image, [box], -1, (0, 255, 0), 3) 


        # cv2.imshow('Gradient', gradient)
        # cv2.imshow('Blurred', blurred)
        # cv2.imshow('closed2', closed) 
        # cv2.imshow("Image", image) 
        # cv2.imshow('Cropped Barcode', cropped_barcode) 
        cv2.imshow("Warp", img_prespective) 
        cv2.imshow("Original", image)
        uniform_image = make_columns_uniform(img_prespective)
        cv2.imshow("Uniform Image", uniform_image)
        cv2.imwrite("UniformImage.jpg", uniform_image)
        decoded_digits = Decoder.decode_barcode()
        print(decoded_digits)
        cv2.waitKey(0)

def sharpen_image(image):
    # Define the sharpening kernel
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

    # Apply the sharpening filter
    sharpened = cv2.filter2D(image, -1, kernel)

    return sharpened

def enhance_barcode(image):
    # Step 1: Load the image (grayscale) after noise removal
    # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Step 2: Sharpen the image to improve clarity
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


# --------------------------------------------MAIN--------------------------------
# Uncomment the needed Test Case
# image_path ="Test Cases\\01 - lol easy.jpg" #Done
# image_path ="Test Cases\\02 - still easy.jpg" #Done
# image_path ="Test Cases\\03 - eda ya3am ew3a soba3ak mathazarsh.jpg" #Done
# image_path ="Test Cases\\04 - fen el nadara.jpg" #Decoder Can't detect the barcode
# image_path ="Test Cases\\05 - meen taffa el nour!!!.jpg" #Decoder Can't detect the barcode
# image_path ="Test Cases\\06 - meen fata7 el nour 333eenaaayy.jpg" #Decoder Can't detect the barcode
# image_path ="Test Cases\\07 - mal7 w felfel.jpg" #Decoder Can't detect the barcode
# image_path ="Test Cases\\08 - compresso espresso.jpg" #Decoder Can't detect the barcode
# image_path ="Test Cases\\09 - e3del el soora ya3ammm.jpg" #Decoder Can't detect the barcode
# image_path ="Test Cases\\10 - wen el kontraastttt.jpg" #Decoder Can't detect the barcode
image_path = "Test Cases\\11 - bayza 5ales di bsara7a.jpg" #Decoder Can't detect the barcode
img = cv2.imread(image_path, 0)


avg_intensity =calc_avg_intensity(img)
# thresholded_image = apply_dynamic_threshold(img,avg_intensity)
thresh = applydynamic_threshold(img,avg_intensity)
is_salt_pepper =detect_salt_and_pepper_noise(thresh)

# if is_salt_pepper:
#     is_salt_pepper= detect_salt_and_pepper_noise(img)
# print("salmaaa")
print(img.shape)
print(is_salt_pepper)
if is_salt_pepper:
    # is_salt_pepper =detect_salt_and_pepper_noise(is_salt_pepper)
    intensities =plot_time_domain(image_path,300,300)
    print(is_salt_pepper)
    # Call the function
    results = analyze_peaks(intensities, height=150, distance=50, tolerance=5)
    are_peaks_equally_spaced = are_peaks_equally_spaced(intensities,height = 150)
    # Output results
    print("Peak indices:", results['peaks'])
    print("Peak distances:", results['peak_distances'])
    print("Are distances approximately equal?", results['are_distances_equal'])
    
    if are_peaks_equally_spaced:
        avg_int2 = calc_avg_intensity(img)
        dft_img = np.fft.fft2(img)
        dft_img_shift = np.fft.fftshift(dft_img)
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
            detect_barcode(contrast)
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
            detect_barcode(contrast)
            cv2.imshow("Frequency Domain", freqimg)
    else:
        avg_intensity =calc_avg_intensity(img)
        thresholded_image = apply_dynamic_threshold(img,avg_intensity)
        kernel = np.ones((3, 3), np.uint8)
        dil_img = cv2.dilate(thresholded_image, kernel, iterations=1)
        detect_barcode(dil_img)
else:
    print("No Salt and Pepper Noise")
    result =increase_contrast(img)
    detect_barcode(result)
    


        

# else :





cv2.waitKey(0)
cv2.destroyAllWindows()





