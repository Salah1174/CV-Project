# Python program to demonstrate erosion and
# dilation of images.
import cv2
import numpy as np

# This function reorder the corners points appropriatly 
# Helped significantly with warp function
def reorder(myPoints):    #salma yousef
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = myPoints.sum(1)
    myPointsNew[1] = myPoints[np.argmin(add)]
    myPointsNew[3] =myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[0] =myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew

def make_columns_uniform(image):
    # Get the dimensions of the image
    height, width = image.shape
    
    # Iterate through each column
    for x in range(width):
        # Get the pixels in the current column
        column_pixels = image[:, x]
        cnt = [0,0]
        # Find the most common pixel value in the column
        unique, counts = np.unique(column_pixels, return_counts=True)
        mode_pixel = unique[np.argmax(counts)]

        for idx, item in enumerate(unique):
            if(item >= 128):
                unique[idx] = 255
                cnt[0]+=1
            else:
                unique[idx] = 0
                cnt[1]+=1

        print(unique)
        print(unique[np.argmax(counts)])
        # Set all pixels in the column to the most common value
        image[:, x] = 0 if cnt[1] > cnt[0] else 255
    

    return image


def detect_barcode(image):   #salma yousef
    
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
    
    #3---->sobel bas

    # # compute the Scharr gradient magnitude representation of the images in both the x and y direction 
    # gradX = cv2.Sobel(image, ddepth = cv2.CV_32F, dx = 1, dy = 0, ksize = -1)
    # gradY = cv2.Sobel(image, ddepth = cv2.CV_32F, dx = 0, dy = 1, ksize = -1)

    # gradient 2, 3 only

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
        x, y, w, h = cv2.boundingRect(box) 
        
        # print(box)
        border_threshold = 10
        # height, width = image.shape[:2]
        widthA = np.sqrt(((cx - dx) ** 2) + ((cy - dy) ** 2))
        widthB = np.sqrt(((ax - bx) ** 2) + ((ay - by) ** 2))
        width = max(int(widthA), int(widthB))
        
        heightA = np.sqrt(((ax - dx) ** 2) + ((ay - dy) ** 2))
        heightB = np.sqrt(((bx - cx) ** 2) + ((by - cy) ** 2))
        height = max(int(heightA), int(heightB))
        
        print(width, height)
        print (x, y, w, h)
        print(ax, ay, bx, by, cx, cy, dx, dy)
        if x < border_threshold:
            x = 0
        if x + w > width - border_threshold:
            w = width - x
        if y < border_threshold:
            y = 0
        if y + h > height - border_threshold:
            h = height - y

        # Draw the extended bounding box
        # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Crop the barcode 
        cropped_barcode = image[y:y+height, x:x+width] #Salma Hisham

        
        pts1 = np.float32([[bx, by], [ax, ay], [cx, cy], [dx, dy]])
        pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        img_prespective = cv2.warpPerspective(image, matrix, (width, height))
        # cv2.circle(image, (ax,ay), 5, (0,0,255), 20)
        # cv2.circle(image, (bx,by), 1, (0,0,255), 10)
        # cv2.circle(image, (cx,cy), 1, (0,0,255), 10)
        # cv2.circle(image, (dx,dy), 1, (0,0,255), 10)
        
        # draw a bounding box arounded the detected barcode and display the image
        cv2.drawContours(image, [box], -1, (0, 255, 0), 3) 


        # cv2.imshow('Gradient', gradient)
        # cv2.imshow('Blurred', blurred)
        cv2.imshow('closed2', closed) #Salma Hisham
        cv2.imshow("Image", image) #salma hisham
        cv2.imshow('Cropped Barcode', cropped_barcode) #Salma Hisham
        cv2.imshow("Warp", img_prespective) #salma yousef

        # uniform_image = make_columns_uniform(img_prespective)
        # cv2.imshow("Uniform Image", uniform_image)
        
    cv2.waitKey(0)




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
    # "Test Cases\\01 - lol easy.jpg", 0)
    # "Barcode.jpg", 0)
    # "Test Cases\\02 - still easy.jpg", 0)
    # "Test Cases\\03 - eda ya3am ew3a soba3ak mathazarsh.jpg", 0)
    # "Test Cases\\04 - fen el nadara.jpg", 0)
    # "Test Cases\\05 - meen taffa el nour!!!.jpg", 0)
    # "Test Cases\\06 - meen fata7 el nour 333eenaaayy.jpg", 0)
    # "Test Cases\\07 - mal7 w felfel.jpg", 0)
    # "Test Cases\\08 - compresso espresso.jpg", 0)
    "Test Cases\\09 - e3del el soora ya3ammm.jpg", 0)
    # "Test Cases\\10 - wen el kontraastttt.jpg", 0)
    # "Test Cases\\11 - bayza 5ales di bsara7a.jpg", 0)
    # "Barcode_Noise_3.jpg", 0)



if img is None:
    raise ValueError("Image not found.")

# Taking a matrix of size 5 as the kernel
kernel = np.array([[0, 1, 0],
                   [0, 1, 0],
                   [0, 1, 0]],
                  np.uint8)

vertical_kernel_3x3 = np.array([[1], [1], [1]], np.uint8)
vertical_kernel_5x5 = np.array([[1], [1], [1], [1], [1], [1], [1]], np.uint8)


img_dilation = cv2.dilate(img, kernel, iterations=1)
img_erosion = cv2.erode(img_dilation, vertical_kernel_3x3, iterations=1)


bilateral_filt_closing = cv2.bilateralFilter(
    img_erosion, d=9, sigmaColor=75, sigmaSpace=75)


bilateral_filt_dil = cv2.bilateralFilter(
    img_dilation, d=9, sigmaColor=75, sigmaSpace=75)

bilateral_filt_closing_ero = cv2.erode(
    bilateral_filt_closing, vertical_kernel_3x3, iterations=1)

bilateral_filt_dil_ero = cv2.erode(
    bilateral_filt_dil, vertical_kernel_3x3, iterations=1)

bilateral_filt_dil_ero_ero = cv2.erode(
    bilateral_filt_dil, vertical_kernel_5x5, iterations=1)


# img_median = cv2.medianBlur(img_erosion, 3)



# img_dilation_2 = cv2.dilate(img_median, vertical_kernel, iterations=1)
# img_erosion_2 = cv2.erode(img_dilation_2, vertical_kernel, iterations=1)
# img_median_2 = cv2.medianBlur(img_erosion_2, 3)

# img_dst_dil = enhance_barcode(img_dilation_2)

# img_dst_ero = enhance_barcode(img_erosion_2)


# img_dst_median = enhance_barcode(img_median)

cv2.imshow('Input', img)
# cv2.imshow('Dilation', img_dilation)
# cv2.imshow('Erosion', img_erosion)
# cv2.imshow('Bilateral Erosion', bilateral_filt_closing)  # Not Bad
# cv2.imshow('Bilateral Dilation', bilateral_filt_dil)
# cv2.imshow('Bilateral Closing Erosion', bilateral_filt_closing_ero)
cv2.imshow('Bilateral Dilation Erosion', bilateral_filt_dil_ero)
cv2.imshow('Bilateral Dilation Erosion Erosion', bilateral_filt_dil_ero_ero)

# cv2.imshow('Laplacian Ero', img_dst_ero)

# cv2.imshow('Median', img_dst_median)
detect_barcode(bilateral_filt_dil_ero)


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
