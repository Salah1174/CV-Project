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
    g_inverse = 255 - g_high_pass

    # for pixel in range(g_inverse.shape[0]):
    #     for pixel2 in range(g_inverse.shape[1]):
    #         if g_inverse[pixel][pixel2] < 228:
    #             g_inverse[pixel][pixel2] = 0



    cv2.imshow("Inverted High-Pass Image", g_inverse.astype(np.uint8))

    histogram = cv2.calcHist([g_inverse.astype(np.uint8)], [
                             0], None, [256], [0, 256])

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


noiseReductionFrequencyDomain(cv2.imread("C:\\ASU\\ASU\\Fall 2024\\Computer Vision\\TestCases\\12 - bayza 5ales di bsara7a#3.jpg", 0))


# # original image
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