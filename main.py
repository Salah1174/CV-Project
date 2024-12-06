import cv2
import Preprocessing
# --------------------------------------------MAIN--------------------------------
# Uncomment the needed Test Case

# image_path ="Test Cases\\01 - lol easy.jpg" #Done
# image_path ="Test Cases\\02 - still easy.jpg" #Done
# image_path ="Test Cases\\03 - eda ya3am ew3a soba3ak mathazarsh.jpg" #Done
# image_path ="Test Cases\\04 - fen el nadara.jpg" #Decoder Can't detect the barcode
# image_path ="Test Cases\\05 - meen taffa el nour!!!.jpg" #Decoder Can't detect the barcode
# image_path ="Test Cases\\06 - meen fata7 el nour 333eenaaayy.jpg" #Decoder Can't detect the barcode
# image_path ="Test Cases\\07 - mal7 w felfel.jpg" #Decoder Can't detect the barcode
# image_path ="Test Cases\\08 - compresso espresso.jpg" #Decoder Can't detect the barcode
# image_path ="Test Cases\\09 - e3del el soora ya3ammm.jpg" #Decoder Can't detect the barcode
# image_path ="Test Cases\\10 - wen el kontraastttt.jpg" #Decoder Can't detect the barcode
image_path = "Test Cases\\11 - bayza 5ales di bsara7a.jpg" #Decoder Can't detect the barcode

img = cv2.imread(image_path, 0)
cv2.imshow("Original Image", img)
Preprocessing.preprocessing(img, image_path)

cv2.waitKey(0)
cv2.destroyAllWindows()