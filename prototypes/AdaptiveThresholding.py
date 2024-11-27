# Python program to illustrate
# adaptive thresholding type on an image

# organizing imports
import cv2
import numpy as np

def AdaptiveGaussian(image_path):
	
    # path to input image is specified and
    # image is loaded with imread command
    image1 = cv2.imread(image_path)

    # cv2.cvtColor is applied over the
    # image input with applied parameters
    # to convert the image in grayscale
    img = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

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

def RegionFilling(img):
    # Threshold.
    # Set values equal to or above 220 to 0.
    # Set values below 220 to 255.

    th, im_th = cv2.threshold(img, 255, 255, cv2.THRESH_BINARY_INV)

    # Copy the thresholded image.
    im_floodfill = im_th.copy()

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.
    im_out = im_th | im_floodfill_inv

    # Display images.
    cv2.imshow("Thresholded Image", im_th)
    cv2.imshow("Floodfilled Image", im_floodfill)
    cv2.imshow("Inverted Floodfilled Image", im_floodfill_inv)
    cv2.imshow("Foreground", im_out)
    cv2.waitKey(0)

def AdaptiveGaussian(img1):

    # path to input image is specified and
    # image is loaded with imread command
    # image1 = cv2.imread(image_path)

    # cv2.cvtColor is applied over the
    # image input with applied parameters
    # to convert the image in grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # applying different thresholding
    # techniques on the input image
    # thresh1 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
    #                                 cv2.THRESH_BINARY, 199, 5)

    thresh2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 199, 5)

    # the window showing output images
    # with the corresponding thresholding
    # techniques applied to the input image
    # cv2.imshow('Adaptive Mean', thresh1)
    cv2.imshow('Adaptive Gaussian', thresh2)

    # De-allocate any associated memory usage
    if cv2.waitKey(0) & 0xff == 27:
    	cv2.destroyAllWindows()
AdaptiveGaussian("Test Cases\\11 - bayza 5ales di bsara7a.jpg")
# AdaptiveGaussian("High_Pass.jpg")
