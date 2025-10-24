# Cân bằng lược đồ xám cho ảnh màu RGB
# Giải pháp: Chuyển đổi không gian màu RGB sang không gian màu khác có khả năng tách biệt giữa kênh độ sáng (Luminance)
# và kênh màu sắc (Chrominance). Các không gian màu phổ biến cho việc này là HSV (hoặc HSL) và YCbCr.
# Cách làm này đảm bảo chỉ tăng cường độ tương phản (độ sáng) của ảnh mà không làm ảnh hưởng đến màu săc gốc của nó.

import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    img = cv2.imread("../images/img7.2.jpg")
    result = use_hsv(img)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(30, 20))
    plt.subplot(121), plt.imshow(img)
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(result)
    plt.title('Result'), plt.xticks([]), plt.yticks([])
    plt.show()

# HSV giữ màu sắc gốc tốt, trực quan.
def use_hsv(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)

    v_eq = equalize_hist(v)

    hsv_img = cv2.merge((h, s, v_eq))
    return cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

# YCbCr hiệu quả cho nén ảnh nhưng dễ gây trôi màu, bạc màu.
def use_ycbcr(img):
    ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    y, cr, cb = cv2.split(ycbcr)

    y_eq = equalize_hist(y)

    ycbcr_img = cv2.merge((y_eq, cr, cb))
    return cv2.cvtColor(ycbcr_img, cv2.COLOR_YCrCb2BGR)

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

    # Tạo bảng tra (LUT)
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