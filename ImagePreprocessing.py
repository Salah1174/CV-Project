import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread(
    "C:\\ASU\\ASU\\Fall 2024\\Computer Vision\\CV-Project\\Barcode.jpg", cv.IMREAD_GRAYSCALE)

imgFilter_Median = cv.medianBlur(img, 5)


# imgFilter_Gaussian = cv.GaussianBlur(img, (5, 5), cv.BORDER_DEFAULT)


# img_arr = np.array(img)

# filter_size = 3


# def apply_median_filter(padded_image):
#     result = np.zeros((358, 220))
#     for i in range(0, padded_image.shape[0] - filter_size + 1):
#         for j in range(0, padded_image.shape[1] - filter_size + 1):
#             window = padded_image[i:i+filter_size, j:j+filter_size]
#             result[i, j] = np.median(window)
#     return result


# imgFilter = apply_median_filter(img_arr)

# plt.subplot(121), plt.imshow(img, cmap='gray')
plt.imshow(imgFilter_Gaussian, cmap='gray')
plt.show()
