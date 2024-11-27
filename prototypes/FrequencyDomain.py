import cv2
import numpy as np
import matplotlib.pyplot as plt

def noiseReductionFrequencyDomain(img):
    # Original image
    # f = cv2.imread("Test Cases\\11 - bayza 5ales di bsara7a.jpg", 0)
    cv2.imshow("Original Image", img)

    # Image in frequency domain
    F = np.fft.fft2(img)
    Fshift = np.fft.fftshift(F)

    # Filter: Low pass filter
    M, N = img.shape
    H = np.zeros((M, N), dtype=np.float32)
    D0 = 25
    for u in range(M):
        for v in range(N):
            D = np.sqrt((u - M/2)**2 + (v - N/2)**2)
            if D <= D0:
                H[u, v] = 1
            else:
                H[u, v] = 0

    # Ideal Low Pass Filtering
    Gshift = Fshift * H
    G = np.fft.ifftshift(Gshift)
    g_low_pass = np.abs(np.fft.ifft2(G))

    # Display Low-Pass Filtered Image
    cv2.imshow("Low-Pass Filtered Image", g_low_pass.astype(np.uint8))

    # Filter: High pass filter
    H = 1 - H

    # Ideal High Pass Filtering
    Gshift = Fshift * H
    G = np.fft.ifftshift(Gshift)
    g_high_pass = np.abs(np.fft.ifft2(G))

    # Display High-Pass Filtered Image
    cv2.imshow("High-Pass Filtered Image", g_high_pass.astype(np.uint8))
    cv2.imwrite("High_Pass.jpg", g_high_pass.astype(np.uint8))
    # Inverse the High-Pass Filtered Image
    g_inverse = 256 - g_high_pass

    # for pixel in range(g_inverse.shape[0]):
    #     for pixel2 in range(g_inverse.shape[1]):
    #         if g_inverse[pixel][pixel2] >= 255 :
    #             g_inverse[pixel][pixel2] = 0

    cv2.imshow("Inverted High-Pass Image", g_inverse.astype(np.uint8))
    cv2.imwrite("Inverted_High_Pass.jpg", g_inverse.astype(np.uint8))
    histogram = cv2.calcHist([g_inverse.astype(np.uint8)], [
        0], None, [256], [0, 256])

    # for intensity, frequency in enumerate(histogram):
        # print(f"Intensity: {intensity}, Frequency: {int(frequency)}")

    # Plot the histogram
    plt.figure()
    plt.title("Grayscale Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.plot(histogram)
    plt.xlim([0, 256])
    plt.show()

    # Wait for a key press to close windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    g_high_pass = g_high_pass.astype(np.uint8)
    return g_high_pass


def noiseReductionFrequencyDomain(img_path):
    # Load original image
    f = cv2.imread(img_path, 0)
    cv2.imshow("Original Image", f)

    # Image in frequency domain
    F = np.fft.fft2(f)
    Fshift = np.fft.fftshift(F)

    # Create Low-Pass Filter
    M, N = f.shape
    H_low = np.zeros((M, N), dtype=np.float32)
    D0 = 50  # Cutoff frequency
    for u in range(M):
        for v in range(N):
            D = np.sqrt((u - M / 2) ** 2 + (v - N / 2) ** 2)
            H_low[u, v] = 1 if D <= D0 else 0

    # Apply Low-Pass Filter
    Gshift_low = Fshift * H_low
    G_low = np.fft.ifftshift(Gshift_low)
    g_low_pass = np.abs(np.fft.ifft2(G_low))
    cv2.imshow("Low-Pass Filtered Image", g_low_pass.astype(np.uint8))

    # Create High-Pass Filter
    H_high = 1 - H_low

    # Apply High-Pass Filter
    Gshift_high = Fshift * H_high
    G_high = np.fft.ifftshift(Gshift_high)
    g_high_pass = np.abs(np.fft.ifft2(G_high))
    cv2.imshow("High-Pass Filtered Image", g_high_pass.astype(np.uint8))

    # Invert the High-Pass Filtered Image
    g_inverse = 256 - g_high_pass
    cv2.imshow("Inverted High-Pass Image", g_inverse.astype(np.uint8))

    # Apply Various Filters for Noise Reduction and Clarity
    # 1. Gaussian Blur
    gaussian_filtered = cv2.GaussianBlur(g_inverse, (15, 15), 0)
    cv2.imshow("Gaussian Blurred Image", gaussian_filtered)

    # 2. Bilateral Filter
    bilateral_filtered = cv2.bilateralFilter(
        g_inverse.astype(np.uint8), 9, 75, 75)
    cv2.imshow("Bilateral Filtered Image", bilateral_filtered)

    # 3. Edge Preserving Filter
    edge_preserving_filtered = cv2.edgePreservingFilter(
        g_inverse.astype(np.uint8), flags=1, sigma_s=60, sigma_r=0.4)
    cv2.imshow("Edge Preserving Filtered Image", edge_preserving_filtered)

    # Combine Filters (e.g., Gaussian + Edge Preserving)
    combined_filtered = cv2.addWeighted(
        gaussian_filtered.astype(np.uint8), 0.6,
        edge_preserving_filtered.astype(np.uint8), 0.4, 0)
    cv2.imshow("Combined Filtered Image", combined_filtered)

    # Plot histogram of the final processed image
    histogram = cv2.calcHist([combined_filtered], [0], None, [256], [0, 256])
    plt.figure()
    plt.title("Histogram of Processed Image")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.plot(histogram)
    plt.xlim([0, 256])
    plt.show()

    # Save the processed image if needed
    cv2.imwrite("Processed_Image.jpg", combined_filtered)

    # Wait for user to close windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# Call the function with an example image path
# noiseReductionFrequencyDomain("Test Cases\\11 - bayza 5ales di bsara7a.jpg")
def apply_notch_filter(Fshift, M, N, D0=50):
    """
    Applies a notch filter to remove periodic noise from the frequency domain.
    D0 is the radius of the notch filter in frequency space.
    """
    # Create the notch filter mask
    H = np.ones((M, N), dtype=np.float32)
    for u in range(M):
        for v in range(N):
            D = np.sqrt((u - M / 2) ** 2 + (v - N / 2) ** 2)
            if D <= D0:
                H[u, v] = 0  # Remove the low frequencies around the center
    return Fshift * H

def noise_reduction_barcode(img_path):
    # Load the image
    img = cv2.imread(img_path, 0)
    cv2.imshow("Original Image", img)

    # Fourier Transform
    F = np.fft.fft2(img)
    Fshift = np.fft.fftshift(F)
    M, N = img.shape

    # Step 1: Apply Notch Filter to remove periodic noise
    Fshift_no_noise = apply_notch_filter(Fshift, M, N, D0=50)

    # Step 2: Apply High Pass Filter to emphasize edges
    H_high = np.ones((M, N), dtype=np.float32)
    D0 = 30  # High-pass filter radius
    for u in range(M):
        for v in range(N):
            D = np.sqrt((u - M / 2) ** 2 + (v - N / 2) ** 2)
            if D <= D0:
                # Zero out low frequencies (remove smooth areas)
                H_high[u, v] = 0

    Fshift_high = Fshift_no_noise * H_high
    G_high = np.fft.ifftshift(Fshift_high)
    high_pass_image = np.abs(np.fft.ifft2(G_high))

    # Step 3: Apply Low Pass Filter to restore the smooth parts of the barcode (inner fillings)
    H_low = 1 - H_high  # Low-pass filter is the complement of the high-pass filter
    Fshift_low = Fshift_no_noise * H_low
    G_low = np.fft.ifftshift(Fshift_low)
    low_pass_image = np.abs(np.fft.ifft2(G_low))

    # Step 4: Combine high-pass and low-pass components to reconstruct the image
    alpha = 0.7
    beta = 0.3
    reconstructed_image = np.clip(
        alpha * high_pass_image + beta * low_pass_image, 0, 255).astype(np.uint8)

    # Step 5: Morphological operations to fill the barcode inner gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph_image = cv2.morphologyEx(
        reconstructed_image, cv2.MORPH_CLOSE, kernel)

    # Step 6: Thresholding to emphasize barcode lines
    _, threshold_image = cv2.threshold(
        morph_image, 128, 255, cv2.THRESH_BINARY)

    # Display results
    cv2.imshow("High-Pass Filtered Image", high_pass_image.astype(np.uint8))
    cv2.imshow("Low-Pass Filtered Image", low_pass_image.astype(np.uint8))
    cv2.imshow("Reconstructed Image", reconstructed_image)
    cv2.imshow("Morphologically Processed Image", morph_image)
    cv2.imshow("Thresholded Image", threshold_image)

    # Wait for a key press to close windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Path to the barcode image
img_path = "Test Cases\\11 - bayza 5ales di bsara7a.jpg"

# Run the noise reduction and barcode enhancement process
# noise_reduction_barcode(img_path)

def give_me_circle_mask_nowww(mask_size, radius):
    mask = np.zeros(mask_size)
    cy = mask.shape[0] // 2
    cx = mask.shape[1] // 2
    return cv2.circle(mask, (cx, cy), radius, (255, 255, 255), -1).astype(np.uint8)

def try_lowpass(dft_img, limit, gaussian: bool = False):
    mask = give_me_circle_mask_nowww(dft_img.shape, limit)
    if (gaussian):
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
    dft_img_shifted = np.fft.fftshift(dft_img)
    dft_img_shifted_lowpass = np.multiply(dft_img_shifted, mask)
    plot_shifted_fft_and_ifft(dft_img_shifted_lowpass)

def try_highpass(dft_img, limit, gaussian: bool = False, keep_dc: bool = False):
    mask = ~give_me_circle_mask_nowww(dft_img.shape, limit)
    if (gaussian):
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
    if (keep_dc):
        mask[dft_img.shape[0]//2, dft_img.shape[1]//2] = 255
    dft_img_shifted = np.fft.fftshift(dft_img)
    dft_img_shifted_highpass = np.multiply(dft_img_shifted, mask)
    plot_shifted_fft_and_ifft(dft_img_shifted_highpass)

def plot_shifted_fft_and_ifft(dft_img_shifted):
    img = np.fft.ifft2(np.fft.ifftshift(dft_img_shifted))
    fig, (ax1, ax2) = plt.subplots(figsize=(10, 5), nrows=1, ncols=2)
    ax1.set(yticks=[0, img.shape[0]//2, img.shape[0] - 1],yticklabels=[-img.shape[0]//2, 0, img.shape[0]//2 - 1])
    ax1.set(xticks=[0, img.shape[1]//2, img.shape[1] - 1],xticklabels=[-img.shape[1]//2, 0, img.shape[1]//2 - 1])
    ax1.imshow(np.abs(dft_img_shifted)**0.1, cmap='gray')
    ax2.imshow(np.abs(img), cmap='gray')


    img = np.abs(img.astype(np.uint16))
    # plt.imsave("AX2.jpg",img)
    # img = cv2.imread("AX2.jpg") 
    cv2.imshow("AX2", img)
    # ax2 = ax2.astype(np.uint32)
    # ax2.imsave("Test Cases\\11 - bayza 5ales di bsara7a#3.jpg", np.abs(img), p)
    plt.show()

def noiseReductionFrequencyDomain2(image_path, cutoff_scale=1.0):
    
    img = cv2.imread(image_path, 0)
    cv2.imshow("Original Image", img)

    # Image in frequency domain
    F = np.fft.fft2(img)
    Fshift = np.fft.fftshift(F)

    # Filter: Low pass filter
    M, N = img.shape
    H = np.zeros((M, N), dtype=np.float32)
    D0 = 25 * cutoff_scale  # Scale the cutoff frequency
    for u in range(M):
        for v in range(N):
            D = np.sqrt((u - M/2)**2 + (v - N/2)**2)
            H[u, v] = 1 if D <= D0 else 0

    # Apply Low-Pass Filter
    Gshift = Fshift * H
    G = np.fft.ifftshift(Gshift)
    g_low_pass = np.abs(np.fft.ifft2(G))
    cv2.imshow("Low-Pass Filtered Image", g_low_pass.astype(np.uint8))

    # Filter: High pass filter
    H = 1 - H

    # Apply High-Pass Filter
    Gshift = Fshift * H
    G = np.fft.ifftshift(Gshift)
    g_high_pass = np.abs(np.fft.ifft2(G))
    cv2.imshow("High-Pass Filtered Image", g_high_pass.astype(np.uint8))

    # Inverse the High-Pass Filtered Image
    g_inverse = 256 - g_high_pass
    cv2.imshow("Inverted High-Pass Image", g_inverse.astype(np.uint8))
    cv2.imwrite("Inverted_High_Pass.jpg", g_inverse.astype(np.uint8))

    # Wait for a key press to close windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    g_high_pass = g_high_pass.astype(np.uint8)
    return g_high_pass

def update_filter(scale):
    cutoff_scale = scale / 25  # Normalize scale
    filtered_image = noiseReductionFrequencyDomain2(
        "Test Cases\\11 - bayza 5ales di bsara7a.jpg", cutoff_scale)
    cv2.imshow("Filtered Image", filtered_image)

img = cv2.imread("Test Cases\\11 - bayza 5ales di bsara7a.jpg")
cv2.imshow("Original Image", img)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dft_img = np.fft.fft2(img_gray)
dft_img_shift = np.fft.fftshift(dft_img)
# plt.imshow(np.log(np.abs(dft_img_shift)), cmap='gray')
# plt.show()
try_highpass(dft_img, 20, gaussian=False, keep_dc=True)

# cv2.namedWindow("Filtered Image")
# cv2.createTrackbar("Cutoff Scale", "Filtered Image", 1, 100, update_filter)
# img = cv2.imread("Test Cases\\11 - bayza 5ales di bsara7a.jpg", 0)
# update_filter(10)  # Initial filter
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# noiseReductionFrequencyDomain(cv2.imread("C:\\ASU\\ASU\\Fall 2024\\Computer Vision\\TestCases\\12 - bayza 5ales di bsara7a#3.jpg", 0))

# noiseReductionFrequencyDomain2("C:\\ASU\\ASU\\Fall 2024\\Computer Vision\\TestCases\\11 - bayza 5ales di bsara7a.jpg", 1.0)

# # Load the image in grayscale
# image_path = "C:\\ASU\\ASU\\Fall 2024\\Computer Vision\\TestCases\\11 - bayza 5ales di bsara7a.jpg"
# image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# # Perform Fourier Transform to get the frequency domain
# dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
# dft_shift = np.fft.fftshift(dft)  # Shift the zero frequency to the center

# # Compute the magnitude spectrum for visualization
# magnitude_spectrum = 20 * \
#     np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

# # Create a mask to filter out the noise (manually specify noise positions)
# rows, cols = image.shape
# crow, ccol = rows // 2, cols // 2  # Center of the frequency domain

# # Create a circular mask to suppress noise frequencies
# mask = np.ones((rows, cols, 2), np.uint8)
# r = 30  # Radius of the mask for filtering
# cv2.circle(mask, (ccol, crow), r, (0, 0), -1)  # Keep low frequencies only

# # Apply the mask to the DFT
# fshift = dft_shift * mask

# # Perform inverse DFT to get the filtered image
# f_ishift = np.fft.ifftshift(fshift)
# filtered_image = cv2.idft(f_ishift)
# filtered_image = cv2.magnitude(
#     filtered_image[:, :, 0], filtered_image[:, :, 1])

# # Normalize the filtered image for display
# filtered_image = cv2.normalize(filtered_image, None, 0, 255, cv2.NORM_MINMAX)

# # Display the results
# plt.figure(figsize=(12, 6))

# plt.subplot(1, 3, 1)
# plt.imshow(image, cmap="gray")
# plt.title("Original Image")
# plt.axis("off")

# plt.subplot(1, 3, 2)
# plt.imshow(magnitude_spectrum, cmap="gray")
# plt.title("Magnitude Spectrum (FFT)")
# plt.axis("off")

# plt.subplot(1, 3, 3)
# plt.imshow(filtered_image, cmap="gray")
# plt.title("Filtered Image (Periodic Noise Removed)")
# plt.axis("off")

# plt.tight_layout()
# plt.show()

# original image
# f = cv2.imread(
#     "C:\\ASU\\ASU\\Fall 2024\\Computer Vision\\TestCases\\12 - bayza 5ales di bsara7a#3.jpg", 0)

# plt.imshow(f, cmap='gray')
# plt.axis('on')
# plt.show()

# # image in frequency domain
# F = np.fft.fft2(f)
# plt.imshow(np.log1p(np.abs(F)),
#            cmap='gray')
# plt.axis('on')
# plt.show()

# Fshift = np.fft.fftshift(F)
# plt.imshow(np.log1p(np.abs(Fshift)),
#            cmap='gray')
# plt.axis('on')
# plt.show()

# # Filter: Low pass filter
# M, N = f.shape
# H = np.zeros((M, N), dtype=np.float32)
# D0 = 50
# for u in range(M):
#     for v in range(N):
#         D = np.sqrt((u-M/2)**2 + (v-N/2)**2)
#         if D <= D0:
#             H[u, v] = 1
#         else:
#             H[u, v] = 0

# plt.imshow(H, cmap='gray')
# plt.axis('on')
# plt.show()

# # Ideal Low Pass Filtering
# Gshift = Fshift * H
# plt.imshow(np.log1p(np.abs(Gshift)),
#            cmap='gray')
# plt.axis('on')
# plt.show()

# # Inverse Fourier Transform
# G = np.fft.ifftshift(Gshift)
# plt.imshow(np.log1p(np.abs(G)),
#            cmap='gray')
# plt.axis('on')
# plt.show()

# g = np.abs(np.fft.ifft2(G))
# plt.imshow(g, cmap='gray')
# plt.axis('on')
# plt.show()


# # Filter: High pass filter
# H = 1 - H

# plt.imshow(H, cmap='gray')
# plt.axis('on')
# plt.show()

# # Ideal High Pass Filtering
# Gshift = Fshift * H
# plt.imshow(np.log1p(np.abs(Gshift)),
#            cmap='gray')
# plt.axis('on')
# plt.show()

# # Inverse Fourier Transform
# G = np.fft.ifftshift(Gshift)
# plt.imshow(np.log1p(np.abs(G)),
#            cmap='gray')
# plt.axis('on')
# plt.show()

# g = np.abs(np.fft.ifft2(G))
# plt.imshow(g, cmap='gray')
# plt.axis('on')
# plt.show()