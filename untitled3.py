# -*- coding: utf-8 -*-
"""Untitled3.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1yJS3rNoFHpPeNQ1I6QWZGUYzvqvpWVSl
"""

import cv2
import numpy as np
from google.colab.patches import cv2_imshow

img = cv2.imread(
    "/content/Barcode.jpg", 0)

kernel = np.array([[0, 1, 0],
                   [0, 1, 0],
                   [0, 1, 0]],
                  np.uint8)
cv2_imshow(img)
median_filt = cv2.medianBlur(img, 3)
cv2_imshow(median_filt)

gaussian_filt = cv2.GaussianBlur(img, (5, 5), 0)
cv2_imshow(gaussian_filt)

bilateral_filt = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
cv2_imshow(bilateral_filt)



_, binary = cv2.threshold(median_filt, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2_imshow(binary)

kernelDil = np.array([[0, 1, 0],
                   [0, 1, 0],
                   [0, 1, 0]],
                  np.uint8)

kernelEro = np.array([[0, 1, 0],
                   [0, 1, 0],
                   [0, 1, 0],
                   [0, 1, 0]],
                  np.uint8)

dilate_img = cv2.dilate(binary, kernelDil, iterations=1)
cv2_imshow( dilate_img) #not bad

op = cv2.bitwise_and(dilate_img, gaussian_filt)
cv2_imshow(op)

_, binary = cv2.threshold(op, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2_imshow(binary) #good

erode_img = cv2.erode(dilate_img, kernelEro, iterations=1)
cv2_imshow(erode_img)

import cv2
import numpy as np

img = cv2.imread('/content/Barcode.jpg', cv2.IMREAD_GRAYSCALE)


bilateral_filt = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

from google.colab.patches import cv2_imshow


cv2_imshow(bilateral_filt)

dil = cv2.dilate(binary, kernelDil, iterations=2)
cv2_imshow(dil)
cv2_imshow(cv2.erode(dil, kernelEro, iterations=4))

_, binary = cv2.threshold(bilateral_filt, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2_imshow(binary)
median = cv2.medianBlur(binary, 3)
cv2_imshow(median) #good result
dil = cv2.dilate(median, kernelDil, iterations=2) # not bad
cv2_imshow(dil)
cv2_imshow(cv2.erode(dil, kernelEro, iterations=2)) #not bad

import cv2
import numpy as np

img = cv2.imread('/content/Barcode.jpg', cv2.IMREAD_GRAYSCALE)

bilateral_filt = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

_, thresh_img = cv2.threshold(bilateral_filt, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

kernel = np.ones((3,3), np.uint8)
dilated_img = cv2.dilate(thresh_img, kernel, iterations=1)
eroded_img = cv2.erode(dilated_img, kernel, iterations=1)

from google.colab.patches import cv2_imshow

cv2_imshow(img)
cv2_imshow(bilateral_filt)
cv2_imshow(thresh_img)
cv2_imshow(cv2.bitwise_or(thresh_img,dil))


