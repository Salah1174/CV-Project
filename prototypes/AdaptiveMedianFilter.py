# import cv2
# import numpy as np
# from matplotlib import pyplot as plt


# def adaptive_median_filter(image, S_max):
#     padded_image = np.pad(
#         image, S_max // 2, mode='constant', constant_values=0)
#     output_image = np.copy(image)
#     rows, cols = image.shape

#     for i in range(rows):
#         for j in range(cols):
#             S = 3
#             while S <= S_max:
#                 # Extracting the subimage
#                 sub_img = padded_image[i:i + S, j:j + S]
#                 Z_min = np.min(sub_img)
#                 Z_max = np.max(sub_img)
#                 Z_m = np.median(sub_img)
#                 Z_xy = image[i, j]

#                 if Z_min < Z_m < Z_max:
#                     if Z_min < Z_xy < Z_max:
#                         output_image[i, j] = Z_xy
#                     else:
#                         output_image[i, j] = Z_m
#                     break
#                 else:
#                     S += 2
#             else:
#                 output_image[i, j] = Z_m

#     return output_image


# # Load the image
# image_path = "C:\\ASU\\ASU\\Fall 2024\\Computer Vision\\CV-Project\\Barcode.jpg"
# img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# # Apply adaptive median filtering
# S_max = 3  # Maximum window size
# filtered_img = adaptive_median_filter(img, S_max)

# # Display the original and filtered images
# plt.imshow(filtered_img, cmap='gray')
# plt.show()
# # cv2.imshow('Original Image', img)
# # cv2.imshow('Filtered Image', filtered_img)

# # cv2.waitKey(0)
# # cv2.destroyAllWindows()


# Median Spatial Domain Filtering


import cv2
import numpy as np


# Read the image
img_noisy1 = cv2.imread(
    "C:\\ASU\\ASU\\Fall 2024\\Computer Vision\\CV-Project\\Barcode.jpg", 0)

# Obtain the number of rows and columns
# of the image
m, n = img_noisy1.shape

# Traverse the image. For every 3X3 area,
# find the median of the pixels and
# replace the center pixel by the median
img_new1 = np.zeros([m, n])

for i in range(1, m-1):
    for j in range(1, n-1):
        temp = [img_noisy1[i-1, j-1],
                img_noisy1[i-1, j],
                img_noisy1[i-1, j + 1],
                img_noisy1[i, j-1],
                img_noisy1[i, j],
                img_noisy1[i, j + 1],
                img_noisy1[i + 1, j-1],
                img_noisy1[i + 1, j],
                img_noisy1[i + 1, j + 1]]
        
        temp = sorted(temp)
        img_new1[i, j] = temp[4]
img_new1 = img_new1.astype(np.uint8)
cv2.imwrite('new_median_filtered.jpg', img_new1)
