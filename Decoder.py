import numpy as np
import cv2 as cv


# def adjust_bar_sizes(pixels, narrow_bar_size, wide_bar_size):
#     max_wide_size = wide_bar_size * 1.5   #  Allow for slightly larger wide bars
#     bar_widths = []

#     current_width = 0
#     for pixel in pixels:
#         if pixel == "1":
#             if current_width > 0:
#                 current_width += 1
#                 bar_widths.append(current_width)
            
#             current_width = 0
#             current_width += 1
            
#         else:
#             if current_width > 0:
#                current_width += 1
#                bar_widths.append(current_width)
#             # bar_widths.append(current_width)
#             current_width = 0
#             current_width += 1
            

#     if current_width > 0:
#         bar_widths.append(current_width)

#     narrow_bars = [w for w in bar_widths if w <= narrow_bar_size]
#     wide_bars = [w for w in bar_widths if narrow_bar_size < w <= max_wide_size]

#     if wide_bars:
#         wide_bar_size = int(np.mean(wide_bars))

#     return narrow_bars, wide_bars

def detect_and_normalize_bar_sizes(pixels, narrow_bar_size, wide_bar_size,):
    bar_widths = []
    current_width = 1
    current_pixel = pixels[0]

    for pixel in pixels[1:]:
        if pixel == current_pixel:
            current_width += 1
            # print(f"current pixel is : {current_pixel}")
        else:
            bar_widths.append((current_pixel, current_width))
            print(f"current width is : {current_width}")
            current_width = 1
            current_pixel = pixel

    if current_width > 0:
        bar_widths.append((current_pixel, current_width))

    # narrow_bar_size = min(width for _, width in bar_widths)
    # wide_bar_size = 2* narrow_bar_size
    max_bar_size = max(width for _, width in bar_widths if width > narrow_bar_size)
    average = sum(width for _, width in bar_widths) / len(bar_widths)
    print(f"average is {average}")
    # print(f"Detected narrow bar size: {narrow_bar_size}")
    # print(f"Detected wide bar size: {wide_bar_size}")
    normalized_pixels = []
    
    for pixel, width in bar_widths:
        if width <= narrow_bar_size:
            normalized_pixels.append((pixel, narrow_bar_size))
        # elif narrow_bar_size < width <= wide_bar_size:
        #     normalized_pixels.append((pixel, wide_bar_size))
        elif width > narrow_bar_size:
            if width <= average:
                normalized_pixels.append((pixel, narrow_bar_size))
            elif average < width :
                # if width % wide_bar_size == 0:
                #     normalized_pixels.extend([(pixel, (2*narrow_bar_size))] * (width // wide_bar_size))
                # if width == (2*wide_bar_size):
                #     normalized_pixels.extend([(pixel, (2*narrow_bar_size))] * (width // wide_bar_size))
                # if width >= narrow_bar_size + wide_bar_size:
                #     normalized_pixels.extend([(pixel, wide_bar_size)] ) #* ((width - narrow_bar_size) // wide_bar_size)
                #     normalized_pixels.append((pixel, narrow_bar_size*((width - wide_bar_size) // narrow_bar_size)))
                # else:
                    normalized_pixels.extend([(pixel, wide_bar_size)])
            #     normalized_pixels.append((pixel, 2*narrow_bar_size))  
            # elif wide_bar_size < width < max_bar_size:
            #     normalized_pixels.append((pixel, 2*narrow_bar_size))
            # elif width % wide_bar_size == 0:
            #     normalized_pixels.extend([(pixel, (2*narrow_bar_size))] * (width // wide_bar_size))
            # elif width >= narrow_bar_size + wide_bar_size:
            #     normalized_pixels.extend([(pixel, (2*narrow_bar_size))] * (width // wide_bar_size))
            #     normalized_pixels.append((pixel, narrow_bar_size))
            # normalized_pixels.append((pixel, narrow_bar_size))
            # normalized_pixels.append((pixel, narrow_bar_size
            
    normalized_pixel_str = ''.join(
        ('1' * width) if pixel == '1' else ('0' * width) for pixel, width in normalized_pixels
    )

    return normalized_pixel_str
    
# def detect_bar_sizes(pixels):
#     bar_widths = []
#     current_width = 1
#     current_pixel = pixels[0]

#     for pixel in pixels[1:]:
#         if pixel == current_pixel:
#             current_width += 1
#         else:
#             bar_widths.append(current_width)
#             current_width = 1
#             current_pixel = pixel

#     if current_width > 0:
#         bar_widths.append(current_width)

#     narrow_bar_size = min(bar_widths)
#     wide_bar_size = max(w for w in bar_widths if w > narrow_bar_size)
#     return narrow_bar_size, wide_bar_size

def decode_barcode():
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
    print(pixels)
    # for i in range(0, len(pixels), 8):
    #     print(pixels[i:i+8])
    # Need to figure out how many pixels represent a narrow bar
    narrow_bar_size = 0
    for pixel in pixels:
        if pixel == "1":
            narrow_bar_size += 1
            print(narrow_bar_size)
        else:
            break
    
    wide_bar_size = narrow_bar_size * 2
    
    print(f"Detected narrow bar size: {narrow_bar_size}")
    print(f"Detected wide bar size: {wide_bar_size}")
    
    # narrow_bar_size, wide_bar_size = detect_bar_sizes(pixels)
    # pixels = detect_and_normalize_bar_sizes(pixels, narrow_bar_size, wide_bar_size)
    pixels = detect_and_normalize_bar_sizes(pixels, narrow_bar_size, wide_bar_size)
    # print(f"Detected narrow bar size: {narrow_bar_size}")
    # print(f"Detected wide bar size: {wide_bar_size}")
    
    print(f"Detected pixels: {pixels}")

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
        if count == narrow_bar_size: 
            current_digit_widths += NARROW
        elif count == wide_bar_size:
            current_digit_widths += WIDE
        # else:
        #     if count == narrow_bar_size + wide_bar_size:
        #         if current_digit_widths[-1] == NARROW:
        #             current_digit_widths += NARROW
        #             current_digit_widths += WIDE
        #         elif current_digit_widths[-1] == WIDE:
        #             current_digit_widths += WIDE
        #             current_digit_widths += NARROW
                
        #     elif count > narrow_bar_size + wide_bar_size:
        #             current_digit_widths += WIDE
        #             current_digit_widths += str(int(NARROW)*((count- wide_bar_size)//narrow_bar_size))
        #         # if count == (wide_bar_size *2):    
        #         #     current_digit_widths += (2*WIDE)
        #         # else :
        #             # current_digit_widths += str((count / narrow_bar_size)*int(NARROW))
            # else:
            #     print("Invalid barcode")
            #     break
         
        print(current_digit_widths)
        if current_digit_widths in code11_widths:
            digits.append(code11_widths[current_digit_widths])
            current_digit_widths = ""
            skip_next = True  # Next iteration will be a separator, so skip it
    print(digits)