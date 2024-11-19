import cv2
import numpy as np

# Load the image with salt and pepper noise
image_path = "C:\\ASU\\ASU\\Fall 2024\\Computer Vision\\CV-Project\\Barcode.jpg"
img = cv2.imread(image_path, cv2.IMREAD_COLOR)

# Convert to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Define a kernel
kernel = np.ones((3, 3), np.uint8)

# Apply morphological opening
opened_img = cv2.morphologyEx(gray_img, cv2.MORPH_CLOSE, kernel)

imgFilter = cv2.medianBlur(opened_img, 3)

# Display the original and processed images
cv2.imshow('Original Image', gray_img)
cv2.imshow('Opened Image', opened_img)
cv2.imshow('Median Image', imgFilter)

cv2.waitKey(0)
cv2.destroyAllWindows()
