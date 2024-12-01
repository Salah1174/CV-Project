import numpy as np
import cv2 as cv


# 0 means narrow, 1 means wide
NARROW = "0"
WIDE = "1"
code11_widths = {
    "00110": "Stop/Start",
    "10001": "1",
    "01001": "2",
    "11000": "3",
    "00101": "4",
    "10100": "5",
    "01100": "6",
    "00011": "7",
    "10010": "8",
    "10000": "9",
    "00001": "0",
    "00100": "-",
}

img = cv.imread("UniformImage.jpg", cv.IMREAD_GRAYSCALE) 


# Get the average of each column in your image
mean = img.mean(axis=0)
# print(mean)

# Set it to black or white based on its value
mean[mean <= 127] = 1
mean[mean > 128] = 0

# Convert to string of pixels in order to loop over it
pixels = ''.join(mean.astype(np.uint8).astype(str))
# print(pixels)
for i in range(0, len(pixels), 8):
    print(pixels[i:i+8])


# Need to figure out how many pixels represent a narrow bar
narrow_bar_size = 0
for pixel in pixels:
    if pixel == "1":
        narrow_bar_size += 1
        print(narrow_bar_size)
    else:
        break


wide_bar_size = narrow_bar_size * 2

digits = []
pixel_index = 0
current_digit_widths = ""
skip_next = False



while pixel_index < len(pixels):


    

    if skip_next:
        pixel_index += narrow_bar_size
        skip_next = False
        continue

    count = 1
    try:                                                      # 1 1 1 1 0 0 0 0 1 1 1  1  1  1  1  1 
        while pixels[pixel_index] == pixels[pixel_index + 1]: # 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 
            count += 1
            pixel_index += 1
    except:
        pass
    pixel_index += 1

    current_digit_widths += NARROW if count == narrow_bar_size else WIDE

    if current_digit_widths in code11_widths:
        digits.append(code11_widths[current_digit_widths])
        current_digit_widths = ""
        skip_next = True  # Next iteration will be a separator, so skip it

print(digits)
