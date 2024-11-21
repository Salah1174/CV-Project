# Python program to demonstrate erosion and
# dilation of images.
import cv2
import numpy as np

def detect_barcode(image):

# compute the Scharr gradient magnitude representation of the images 
# in both the x and y direction 
    gradX = cv2.Sobel(image, ddepth = cv2.CV_32F, dx = 1, dy = 0, ksize = -1)
    gradY = cv2.Sobel(image, ddepth = cv2.CV_32F, dx = 0, dy = 1, ksize = -1)

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
    cv2.imshow('closed1', closed)
    closed = cv2.erode(closed, None, iterations = 4)

    closed = cv2.dilate(closed, None, iterations = 4)

    # find the contours in the thresholded image, then sort the contours
    # by their area, keeping only the largest one

    (cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(cnts, key = cv2.contourArea, reverse = True)[0]

    # compute the rotated bounding box of the largest contour

    rect = cv2.minAreaRect(c)
    box = np.int32(cv2.boxPoints(rect))

    # draw a bounding box arounded the detected barcode and display the
    # image

    cv2.drawContours(image, [box], -1, (0, 255, 0), 3)



    cv2.imshow('Gradient', gradient)
    cv2.imshow('Blurred', blurred)
    cv2.imshow('closed2', closed)
    cv2.imshow("Image", img_dilation_2)


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
    "E:\\Computer_vision\\Project\\CV-Project\\Barcode.jpg", 0)

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
