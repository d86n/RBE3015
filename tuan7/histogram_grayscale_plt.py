import cv2
import numpy as np
import matplotlib.pyplot as plt


def main():
    img = cv2.imread("../images/img7.1.jpg")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Chạy hàm cân bằng của bạn
    result = equalize_hist(gray)

    # --- Phần 1: Hiển thị Ảnh ---
    plt.figure(figsize=(20, 10))

    plt.subplot(121), plt.imshow(gray, cmap='gray')
    plt.title('Ảnh Gốc (Grayscale)', fontsize=16)
    plt.xticks([]), plt.yticks([])

    plt.subplot(122), plt.imshow(result, cmap='gray')
    plt.title('Ảnh Kết Quả (Sau Cân Bằng)', fontsize=16)
    plt.xticks([]), plt.yticks([])

    # --- Phần 2: Hiển thị Histogram ---
    plt.figure(figsize=(12, 8))  # Tạo một cửa sổ hình vẽ mới

    # Vẽ histogram cho ảnh xám gốc (màu xanh)
    plt.hist(gray.ravel(), bins=256, range=[0, 256], color='blue', alpha=0.6, label='Histogram ảnh đầu vào')

    # Vẽ histogram cho ảnh kết quả (màu đỏ)
    plt.hist(result.ravel(), bins=256, range=[0, 256], color='red', alpha=0.6, label='Histogram ảnh đầu ra')

    plt.title('So sánh histogram trước và sau cân bằng', fontsize=16)
    plt.xlabel('Mức xám', fontsize=12)
    plt.ylabel('Số lượng điểm ảnh', fontsize=12)
    plt.legend()
    plt.xlim([0, 256])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig("histogram_compare.jpg")
    plt.show()

    psnr = cv2.PSNR(gray, result)

    print(f"Giá trị PSNR giữa ảnh gốc và ảnh kết quả là: {psnr:.2f} dB")


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