import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    img = cv2.imread("anh_toi_1.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    result = equalize_hist(gray)

    plt.figure(figsize=(30, 20))
    plt.subplot(121), plt.imshow(img)
    plt.subplot(122), plt.imshow(result, cmap='gray')
    plt.show()

def equalize_hist(img):
    height, width = img.shape
    total_pixels = height * width

    # Tính histogram
    hist = [0] * 256
    for i in range(height):
        for j in range(width):
            value = img[i, j]
            hist[value] += 1

    # Tính PDF và CDF
    pdf = [val / total_pixels for val in hist]
    cdf = [0] * 256
    cdf[0] = pdf[0]
    for i in range(1, 256):
        cdf[i] = cdf[i - 1] + pdf[i]

    # Tạo bảng tra cứu (Look-Up Table - LUT)
    l = 256
    trans = [0] * l
    for i in range(l):
        trans[i] = int(round((l - 1) * cdf[i]))

    # Ánh xạ ảnh mới
    equalized_img = np.copy(img)
    for i in range(height):
        for j in range(width):
            equalized_img[i, j] = trans[img[i, j]]

    return equalized_img

if __name__ == "__main__":
    main()