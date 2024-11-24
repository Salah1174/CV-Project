img = cv2.imread("Warp.jpg")
cv2.imshow("Original Image", img)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dft_img = np.fft.fft2(img_gray)
dft_img_shift = np.fft.fftshift(dft_img)
# plt.imshow(np.log(np.abs(dft_img_shift)), cmap='gray')
# plt.show()

try_highpass(dft_img, 20, gaussian=False, keep_dc=True)