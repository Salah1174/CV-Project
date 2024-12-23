
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import find_peaks
# from ErosionAndDilation import *
# def reorder(myPoints):
#     myPoints = myPoints.reshape((4, 2))
#     myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
#     add = myPoints.sum(1)
#     myPointsNew[1] = myPoints[np.argmin(add)]
#     myPointsNew[3] = myPoints[np.argmax(add)]
#     diff = np.diff(myPoints, axis=1)
#     myPointsNew[0] = myPoints[np.argmin(diff)]
#     myPointsNew[2] = myPoints[np.argmax(diff)]
#     return myPointsNew
# def make_columns_uniform(image):
#     height, width = image.shape
#     num_parts = 6
#     part_height = height // num_parts
    
#     for x in range(width):
#         part_most_common = []

#         for part in range(num_parts):
#             start_idx = part * part_height
#             end_idx = (part + 1) * part_height if part != num_parts - 1 else height
#             part_pixels = image[start_idx:end_idx, x]

#             # Find  most common pixel value in the part
#             unique, counts = np.unique(part_pixels, return_counts=True)
            
#             for idx, item in enumerate(unique):
#                 if item >= 128:
#                     unique[idx] = 255
#                 else:
#                     unique[idx] = 0
            
#             mode_pixel = unique[np.argmax(counts)]
#             part_most_common.append(mode_pixel)
        
#         # Determine the most common value among the parts
#         final_unique, final_counts = np.unique(part_most_common, return_counts=True)
#         most_common_pixel = final_unique[np.argmax(final_counts)]

#         # Set all pixels in the column to the most common value
#         image[:, x] = most_common_pixel
        
            
#         # # Get the pixels in the current column
#         # column_pixels = image[:, x]
#         # cnt = [0,0]
#         # # Find the most common pixel value in the column
#         # unique, counts = np.unique(column_pixels, return_counts=True)
#         # # mode_pixel = unique[np.argmax(counts)]

#         # for idx, item in enumerate(unique):
#         #     if(item >= 128):
#         #         unique[idx] = 255
#         #         cnt[0]+=1
#         #     else:
#         #         unique[idx] = 0
#         #         cnt[1]+=1

#         # print(unique)
#         # print(unique[np.argmax(counts)])
#         # # Set all pixels in the column to the most common value
#         # image[:, x] = 0 if cnt[1] > cnt[0] else 255
    

#     return image




# def warp_barcode(image, contour):
#     """
#     Warp the detected barcode region to a rectangular image.
#     """
#     # Ensure the contour is reshaped to a 4x2 array
#     contour = contour.reshape(4, 2)

#     # Compute the bounding box of the barcode
#     rect = cv2.boundingRect(contour)
#     x, y, w, h = rect
#     cropped = image[y : y + h, x : x + w]
#     return cropped
# def preprocess_image(image):
#     """
#     Preprocess the image to highlight the barcode region.
#     """
#     # Load the image and convert to grayscale
#     # image = cv2.imread(image_path)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Apply a gradient to highlight the barcode region
#     gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1)))

#     # Threshold the gradient image
#     _, binary = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#     # Apply morphological closing to connect gaps in the barcode
#     closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3)))

#     return closed

# def find_barcode_contour(preprocessed_image):
#     # Ensure the image is grayscale
#     if len(preprocessed_image.shape) == 3:
#         preprocessed_image = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2GRAY)

#     # Find contours
#     contours, _ = cv2.findContours(preprocessed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # Find the largest contour
#     largest_contour = None
#     if contours:
#         largest_contour = max(contours, key=cv2.contourArea)

#     return largest_contour
# def display_result(image, contour):
#     """
#     Display the image with the detected barcode contour.
#     """
#     if contour is not None:
#         # Draw the contour on the image
#         cv2.drawContours(image, [contour], -1, (0, 255, 0), 3)
#         cv2.imshow("Detected Barcode", image)
#     else:
#         print("No barcode detected.")
#         cv2.imshow("Original Image", image)

# def center_image_on_blank(image):
#     """
#     Center an image on a fixed 600x600 white blank canvas using OpenCV.
    
#     Args:
#         image_path (str): Path to the input image.
        
#     Returns:
#         numpy.ndarray: The resulting image with the input image centered on a white background.
#     """
#     # Read the input image
#     # image = cv2.imread(image_path)
    
#     # Get the size of the input image
#     image_height, image_width = image.shape[:2]
    
#     image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
#     # Fixed canvas size
#     canvas_size = (800, 600)
    
#     # Create a white blank canvas
#     canvas = np.ones((canvas_size[1], canvas_size[0], 3), dtype=np.uint8) * 255  # 600x600 white canvas
    
#     # Calculate the position to place the input image to center it
#     x_offset = (canvas_size[0] - image_width) // 2
#     y_offset = (canvas_size[1] - image_height) // 2
    
#     # Check if the input image is larger than the canvas
#     if image_width > canvas_size[0] or image_height > canvas_size[1]:
#         # Resize the image to fit within the canvas while maintaining aspect ratio
#         scaling_factor = min(canvas_size[0] / image_width, canvas_size[1] / image_height)
#         new_width = int(image_width * scaling_factor)
#         new_height = int(image_height * scaling_factor)
#         image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
#         image_height, image_width = image.shape[:2]
#         x_offset = (canvas_size[0] - image_width) // 2
#         y_offset = (canvas_size[1] - image_height) // 2

#     # Place the input image on the canvas
#     canvas[y_offset:y_offset + image_height, x_offset:x_offset + image_width] = image
    
#     return canvas
# def barcode_detection_pipeline(image):
#     # Step 1: Preprocessing
#     preprocessed = preprocessing(image)

#     # Step 2: Find Largest Contour
#     largest_contour = find_barcode_contour(preprocessed)
#     if largest_contour is None:
#         print("No contour found.")
#         return None

#     # Step 3: Warp the Image
#     warped_image = warp(image, largest_contour)
#     if warped_image is None:
#         print("Warping failed.")
#         return None

#     # Step 4: Make Columns Uniform
#     uniform_image = make_columns_uniform(warped_image)

#     # Step 5: Center Image on Blank Canvas
#     centered_image = center_image_on_blank(uniform_image)

#     return centered_image

# def detect_barcode(image):   
#     cv2.imshow("Original", image)
    
#     largest_contour =preprocessing(image)

#     # barcode_contour =find_barcode_contour(closed1)
#     # cropped =warp_barcode(image,barcode_contour)
#     img_perspective=warp(image,largest_contour)
    
#     uniform_image = make_columns_uniform(img_perspective)
#     # center_image =center_image_on_blank(uniform_image)
#     # largest_contour2 =preprocessing(center_image)
#     # img_perspective2 =warp(center_image,largest_contour2)
#     # uniform_image = make_columns_uniform(img_perspective)
#     # barcode_contour =find_barcode_contour(closed2)
#     # cropped =warp_barcode(center_image,barcode_contour)
#     # img_perspective2 =warp_barcode(center_image,largest_contour2)
#     cv2.imshow("img_perspective", img_perspective)
#     cv2.imshow("uniform_image", uniform_image)
#     # cv2.imshow("center_image", center_image)
#     # cv2.imshow("cropped =", img_perspective2 )
#     # cv2.imshow("img_perspective2", img_perspective2)

    
#     return img_perspective

    
# # Uncomment the needed Test Case
# # image_path ="Test Cases\\01 - lol easy.jpg" #Done
# # image_path ="Test Cases\\02 - still easy.jpg" #Done
# # image_path ="Test Cases\\03 - eda ya3am ew3a soba3ak mathazarsh.jpg" #Done
# image_path ="Test Cases\\04 - fen el nadara.jpg" #Decoder Can't detect the barcode
# # image_path ="Test Cases\\05 - meen taffa el nour!!!.jpg" #Decoder Can't detect the barcode
# # image_path ="Test Cases\\06 - meen fata7 el nour 333eenaaayy.jpg" #Decoder Can't detect the barcode
# # image_path ="Test Cases\\07 - mal7 w felfel.jpg" #Decoder Can't detect the barcode
# # image_path ="Test Cases\\08 - compresso espresso.jpg" #Decoder Can't detect the barcode
# # image_path ="Test Cases\\09 - e3del el soora ya3ammm.jpg" #Decoder Can't detect the barcode
# # image_path ="Test Cases\\10 - wen el kontraastttt.jpg" #Decoder Can't detect the barcode
# # image_path = "Test Cases\\11 - bayza 5ales di bsara7a.jpg" #Decoder Can't detect the barcode
# img = cv2.imread(image_path, 0)

# denoised=noiseReductionSaltAndPeper(image_path)
# result =increase_contrast(denoised)
# # cv2.imshow("hello",result)
# uniform_image =detect_barcode(result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()