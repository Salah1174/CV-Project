import cv2
import numpy as np

import Decoder
def remove_initial_whites(pixels):
    first_black_index = pixels.find('1')
    if first_black_index == -1:
        return "", 0  
    return pixels[first_black_index:] ,first_black_index

    
def make_columns_uniform(image):
    height, width = image.shape
    num_parts = 10
    part_height = height // num_parts
    
    for x in range(width):
        part_most_common = []

        for part in range(num_parts):
            start_idx = part * part_height
            end_idx = (part + 1) * part_height if part != num_parts - 1 else height
            part_pixels = image[start_idx:end_idx, x]

            # Find  most common pixel value in the part
            unique, counts = np.unique(part_pixels, return_counts=True)
            
            for idx, item in enumerate(unique):
                if item >= 128:
                    unique[idx] = 255
                else:
                    unique[idx] = 0
            
            mode_pixel = unique[np.argmax(counts)]
            part_most_common.append(mode_pixel)
        
        # Determine the most common value among the parts
        final_unique, final_counts = np.unique(part_most_common, return_counts=True)
        most_common_pixel = final_unique[np.argmax(final_counts)]

        # Set all pixels in the column to the most common value
        image[:, x] = most_common_pixel
    

    return image

# This function reorder the corners points appropriatly
# Helped significantly with warp function
def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = myPoints.sum(1)
    myPointsNew[1] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[0] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew

def detect_barcode(image):   
    
    # 1---->canny

    # blurred = cv2.GaussianBlur(image, (9, 9), 0)
    # edges = cv2.Canny(image, 50, 150)  
    # blurred = cv2.blur(edges, (9, 9))
    # _, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)

    #2---->blur before sobel
    blurred = cv2.GaussianBlur(image, (5, 5),0)
    # compute the Scharr gradient magnitude representation of the images in both the x and y direction 
    gradX = cv2.Sobel(blurred, ddepth = cv2.CV_32F, dx = 1, dy = 0, ksize = -1)
    gradY = cv2.Sobel(blurred, ddepth = cv2.CV_32F, dx = 0, dy = 1, ksize = -1)
    
    #3---->sobel

    # # compute the Scharr gradient magnitude representation of the images in both the x and y direction 
    # gradX = cv2.Sobel(image, ddepth = cv2.CV_32F, dx = 1, dy = 0, ksize = -1)
    # gradY = cv2.Sobel(image, ddepth = cv2.CV_32F, dx = 0, dy = 1, ksize = -1)

    # for gradient 2, 3 only:

    # subtract the y-gradient from the x-gradient
    gradient = cv2.subtract(gradX, gradY) 
    gradient = cv2.convertScaleAbs(gradient)
    # blur and threshold the image 
    blurred = cv2.blur(gradient, (9, 9))
    (_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)
    # cv2.imshow("thresholded",thresh)


    # construct a closing kernel and apply it to the thresholded image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # perform a series of erosions and dilations
    # cv2.imshow('closed1', closed)
    closed = cv2.erode(closed, None, iterations = 4)

    closed = cv2.dilate(closed, None, iterations = 4)

    # find the contours in the thresholded image,
    #  then sort the contours by their area,
    #  keeping only the largest one
    (cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnt = sorted(cnts, key = cv2.contourArea, reverse = True)[0] 
    
    if cv2.contourArea(cnt) > 0:  # Check if contour is valid

        # compute the rotated bounding box of the largest contour
        rect = cv2.minAreaRect(cnt)
        box = np.int32(cv2.boxPoints(rect))
        box = reorder(box)
        
       
        # Coordinates of each corner
        ax = box.item(0)
        ay = box.item(1)

        bx = box.item(2)
        by = box.item(3)

        cx = box.item(4)
        cy = box.item(5)

        dx = box.item(6)
        dy = box.item(7)

        # box coordinates
        # x, y, w, h = cv2.boundingRect(box) 
        
        # print(box)
        # border_threshold = 10
        # height, width = image.shape[:2]
        
        widthA = np.sqrt(((cx - dx) ** 2) + ((cy - dy) ** 2))
        widthB = np.sqrt(((ax - bx) ** 2) + ((ay - by) ** 2))
        width = max(int(widthA), int(widthB))
        
        heightA = np.sqrt(((ax - dx) ** 2) + ((ay - dy) ** 2))
        heightB = np.sqrt(((bx - cx) ** 2) + ((by - cy) ** 2))
        height = max(int(heightA), int(heightB))
        
        # Draw the extended bounding box
        # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Crop the barcode 
        # cropped_barcode = image[y:y+height, x:x+width] 

        
        pts1 = np.float32([[bx, by], [ax, ay], [cx, cy], [dx, dy]])
        pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
        
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        img_prespective = cv2.warpPerspective(image, matrix, (width, height))
        


        # cv2.imshow('Gradient', gradient)
        # cv2.imshow('Blurred', blurred)
        # cv2.imshow('closed2', closed) 
        # cv2.imshow("Image", image) 
        # cv2.imshow('Cropped Barcode', cropped_barcode) 
        # cv2.imshow("Warp", img_prespective) 
        # cv2.imshow("Original", image)
        uniform_image = make_columns_uniform(img_prespective)
        # cv2.imshow("Uniform Image", uniform_image)
        cv2.imwrite("UniformImage.jpg", uniform_image)
        img = cv2.imread("UniformImage.jpg", cv2.IMREAD_GRAYSCALE) 
        # Get the average of each column in your image
        mean = img.mean(axis=0)
        # print(mean)
        # Set it to black or white based on its value
        mean[mean <= 127] = 1
        mean[mean > 128] = 0
        # Convert to string of pixels in order to loop over it
        pixels = ''.join(mean.astype(np.uint8).astype(str))
        pixels , ignore= remove_initial_whites(pixels)
        image_modified = img[ignore:, ignore:]
        cv2.imwrite("FinalImage.jpg", image_modified)
        cv2.imshow("Final Image", image_modified)
        decoded_digits = Decoder.decode_barcode()
        print(decoded_digits)
        cv2.waitKey(0)
    else:
        print("No barcode detected")
        cv2.waitKey(0)
