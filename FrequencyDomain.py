import cv2
import numpy as np
import matplotlib.pyplot as plt



def noiseReductionFrequencyDomain(img):
    # Original image
    f = cv2.imread("Test Cases\\11 - bayza 5ales di bsara7a.jpg", 0)
    cv2.imshow("Original Image", f)

    # Image in frequency domain
    F = np.fft.fft2(f)
    Fshift = np.fft.fftshift(F)

    # Filter: Low pass filter
    M, N = f.shape
    H = np.zeros((M, N), dtype=np.float32)
    D0 = 50
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
    # cv2.imshow("Low-Pass Filtered Image", g_low_pass.astype(np.uint8))

    # Filter: High pass filter
    H = 1 - H

    # Ideal High Pass Filtering
    Gshift = Fshift * H
    G = np.fft.ifftshift(Gshift)
    g_high_pass = np.abs(np.fft.ifft2(G))

    # Display High-Pass Filtered Image
    cv2.imshow("High-Pass Filtered Image", g_high_pass.astype(np.uint8))

    # Inverse the High-Pass Filtered Image
    g_inverse = 256 - g_high_pass

    # for pixel in range(g_inverse.shape[0]):
    #     for pixel2 in range(g_inverse.shape[1]):
    #         if g_inverse[pixel][pixel2] >= 251 :
    #             g_inverse[pixel][pixel2] = 0

    cv2.imshow("Inverted High-Pass Image", g_inverse.astype(np.uint8))



    # low_freq_component = cv2.GaussianBlur(g_high_pass, (15, 15), 0)
    # reconstructed_image = g_high_pass + low_freq_component
    # reconstructed_image = np.clip(reconstructed_image, 0, 255).astype(np.uint8)
    # _, barcode_image = cv2.threshold(
    # reconstructed_image, 128, 255, cv2.THRESH_BINARY)

    # alpha = 2.5
    # beta = 0
    # reconstructed_image = cv2.convertScaleAbs(reconstructed_image, alpha=alpha, beta=beta)
    # reconstructed_image = cv2.bitwise_not(reconstructed_image)
    # # reconstructed_image = cv2.medianBlur(reconstructed_image, 3)
    # # reconstructed_image = cv2.bitwise_not(reconstructed_image)

    # cv2.imshow("Reconstructed Image", reconstructed_image)  

    histogram = cv2.calcHist([g_inverse.astype(np.uint8)], [
                             0], None, [256], [0, 256])

    for intensity, frequency in enumerate(histogram):
        print(f"Intensity: {intensity}, Frequency: {int(frequency)}")
    

    domainFilter = cv2.edgePreservingFilter(g_inverse, flags=1, sigma_s=60, sigma_r=0.6)
    cv2.imshow("Domain Filter", domainFilter)

    # Plot the histogram
    plt.figure()
    plt.title("Grayscale Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.plot(histogram)
    plt.xlim([0, 256])
    plt.show()

    # histogram1 = cv2.calcHist([g_high_pass.astype(np.uint8)], [
    #                          0], None, [256], [0, 256])

    # # for intensity, frequency in enumerate(histogram1):
    # #     print(f"Intensity: {intensity}, Frequency: {int(frequency)}")
    # # Plot the histogram
    # plt.figure()
    # plt.title("Grayscale Histogram")
    # plt.xlabel("Pixel Intensity")
    # plt.ylabel("Frequency")
    # plt.plot(histogram1)
    # plt.xlim([0, 256])
    # plt.show()


    # Wait for a key press to close windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()


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
    ax1.set(yticks=[0, img.shape[0]//2, img.shape[0] - 1],
            yticklabels=[-img.shape[0]//2, 0, img.shape[0]//2 - 1])
    ax1.set(xticks=[0, img.shape[1]//2, img.shape[1] - 1],
            xticklabels=[-img.shape[1]//2, 0, img.shape[1]//2 - 1])
    ax1.imshow(np.abs(dft_img_shifted)**0.1, cmap='gray')
    ax2.imshow(np.abs(img), cmap='gray')




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



noiseReductionFrequencyDomain(cv2.imread("C:\\ASU\\ASU\\Fall 2024\\Computer Vision\\TestCases\\12 - bayza 5ales di bsara7a#3.jpg", 0))


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