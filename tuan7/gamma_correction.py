import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    gray = cv2.imread('anh_toi_1.png', cv2.IMREAD_GRAYSCALE)

    gamma = 0.5

    result = adjust_gamma(gray, gamma=gamma)

    plt.figure(figsize=(12, 8))  # Tạo một cửa sổ hình vẽ mới

    plt.figure(figsize=(30, 20))
    plt.subplot(122), plt.imshow(result, cmap='gray')
    plt.show()

def adjust_gamma(image, gamma=1.0):
    table = np.array([
        ((i / 255.0) ** gamma) * 255
        for i in range(256)
    ]).astype("uint8")

    return cv2.LUT(image, table)

if __name__ == '__main__':
    main()