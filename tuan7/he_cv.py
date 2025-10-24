import cv2
import matplotlib.pyplot as plt
import numpy as np

# --- Tải ảnh ---
# Tải ảnh trực tiếp ở dạng ảnh xám
img = cv2.imread('../images/img7.1.jpg', cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Lỗi: Không thể tải ảnh.")
else:
    # --- 1. Áp dụng Cân bằng Histogram (HE) ---
    # Đây là toàn bộ thuật toán!
    eq_img = cv2.equalizeHist(img)

    # --- 2. Hiển thị ảnh ---
    plt.figure(figsize=(12, 8))

    # Ảnh gốc
    plt.subplot(2, 1, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Ảnh gốc (Original)')
    plt.axis('off')

    # Ảnh đã xử lý HE
    plt.subplot(2, 1, 2)
    plt.imshow(eq_img, cmap='gray')
    plt.title('Đã cân bằng Histogram (HE)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # # Tùy chọn: Lưu ảnh
    # cv2.imwrite('he_equalized_image.jpg', eq_img)