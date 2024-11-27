import random
import cv2


def add_noise(img):

    # Getting the dimensions of the image
    row, col = img.shape
    print(row, col)
    # Randomly pick some pixels in the
    # image for coloring them white
    # Pick a random number between 300 and 10000
    number_of_pixels = int((row*col)/4)
    for i in range(number_of_pixels):

        # Pick a random y coordinate
        y_coord = random.randint(0, row - 1)

        # Pick a random x coordinate
        x_coord = random.randint(0, col - 1)

        # Color that pixel to white
        img[y_coord][x_coord] = 255

    # Randomly pick some pixels in
    # the image for coloring them black
    # Pick a random number between 300 and 10000
    number_of_pixels = random.randint(0, 255)
    for i in range(number_of_pixels):

        # Pick a random y coordinate
        y_coord = random.randint(0, row - 1)

        # Pick a random x coordinate
        x_coord = random.randint(0, col - 1)

        # Color that pixel to black
        img[y_coord][x_coord] = 0

    return img


# salt-and-pepper noise can
# be applied only to grayscale images
# Reading the color image in grayscale image
img = cv2.imread(
    "C:\\ASU\\ASU\\Fall 2024\\Computer Vision\\CV-Project\\Barcode_3.jpg", cv2.IMREAD_GRAYSCALE)

# Storing the image
cv2.imwrite('Barcode_Noise_3.jpg', add_noise(img))
