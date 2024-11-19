# Python program to demonstrate erosion and
# dilation of images.
import cv2
import numpy as np


def sharpen_image(image):
    # Define the sharpening kernel
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])

    # Apply the sharpening filter
    sharpened = cv2.filter2D(image, -1, kernel)

    return sharpened


def enhance_barcode(image):
    # Step 1: Load the image (grayscale) after noise removal
    # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Step 2: Sharpen the image to improve clarity
    sharpened_image = sharpen_image(image)

    return sharpened_image


# Reading the input image
img = cv2.imread(
    "C:\\ASU\\ASU\\Fall 2024\\Computer Vision\\CV-Project\\Barcode.jpg", 0)

# Taking a matrix of size 5 as the kernel
kernel = np.array([[0, 1, 0],
                   [0, 1, 0],
                   [0, 1, 0]],
                  np.uint8)

img_dilation = cv2.dilate(img, kernel, iterations=1)
img_erosion = cv2.erode(img_dilation, kernel, iterations=1)
img_median = cv2.medianBlur(img_erosion, 3)


img_dilation_2 = cv2.dilate(img_median, kernel, iterations=1)
img_erosion_2 = cv2.erode(img_dilation_2, kernel, iterations=1)
img_median_2 = cv2.medianBlur(img_erosion_2, 3)

img_dst_dil = enhance_barcode(img_dilation_2)

img_dst_ero = enhance_barcode(img_erosion_2)

img_dst_median = enhance_barcode(img_median)

cv2.imshow('Input', img)
cv2.imshow('Dilation', img_dilation_2)
cv2.imshow('Erosion', img_erosion_2)
cv2.imshow('Laplacian Dil', img_dst_dil)
cv2.imshow('Laplacian Ero', img_dst_ero)

cv2.imshow('Median', img_dst_median)


cv2.waitKey(0)


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
