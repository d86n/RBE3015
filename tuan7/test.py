import cv2
import numpy as np
import matplotlib.pyplot as plt


def adjust_gamma(image, gamma=1.0):
    """
    Hàm chuẩn:
    gamma > 1 (ví dụ 2.2) sẽ làm SÁNG ảnh.
    gamma < 1 (ví dụ 0.5) sẽ làm TỐI ảnh.
    """
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

# --- Tải ảnh ---
# !!! LƯU Ý: Hãy sửa lại đường dẫn cho đúng với cấu trúc thư mục của bạn
# (ví dụ: 'tuan7/anh_goc.png')
try:
    img_goc = cv2.imread('anh_goc.png', cv2.IMREAD_GRAYSCALE)
    img_toi1 = cv2.imread('anh_toi_1.png', cv2.IMREAD_GRAYSCALE)
    img_toi2 = cv2.imread('anh_toi_2.png', cv2.IMREAD_GRAYSCALE)

    # Đảm bảo 3 ảnh cùng kích thước để so sánh
    h, w = img_goc.shape
    img_toi1 = cv2.resize(img_toi1, (w, h))
    img_toi2 = cv2.resize(img_toi2, (w, h))

    if img_goc is None or img_toi1 is None or img_toi2 is None:
        raise Exception("Không thể tải file ảnh, hãy kiểm tra lại đường dẫn!")

except Exception as e:
    print(f"Lỗi: {e}")
    exit()

# --- Xử lý Ảnh Tối 1 (Chỉ dùng Gamma) ---
gamma_value_1 = 5  # Giá trị gamma mạnh để làm sáng
final_toi1 = adjust_gamma(img_toi1, gamma=gamma_value_1)

# --- Xử lý Ảnh Tối 2 (Kết hợp CLAHE + Gamma) ---
# Bước 1: Dùng CLAHE để cứu chi tiết
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
clahe_toi2 = clahe.apply(img_toi2)

# Bước 2: Dùng Gamma nhẹ để tăng độ sáng tổng thể
gamma_value_2 = 1.5  # Giá trị gamma nhẹ
final_toi2 = adjust_gamma(clahe_toi2, gamma=gamma_value_2)

cv2.imshow("", final_toi1)
cv2.waitKey(0)
cv2.destroyAllWindows()