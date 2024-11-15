import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread("C:\\ASU\\ASU\\Fall 2024\\Computer Vision\\CV-Project\\Barcode.jpg", cv.IMREAD_GRAYSCALE)

imgFilter = cv.medianBlur(img, 3)

plt.subplot(121), plt.imshow(img, cmap='gray')
plt.subplot(122), plt.imshow(imgFilter, cmap='gray')
plt.show()
