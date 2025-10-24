import cv2
import matplotlib.pyplot as plt

# --- 1. Tải ảnh ---
# Đảm bảo bạn thay 'path_to_your_dark_image.jpg' bằng đường dẫn thực tế đến ảnh của bạn.
# cv2.IMREAD_GRAYSCALE: Tải ảnh trực tiếp ở dạng ảnh xám (grayscale)
img = cv2.imread('C:/Users/Dien/Downloads/aaa.jpg', cv2.IMREAD_GRAYSCALE)

# Kiểm tra xem ảnh đã được tải thành công chưa
if img is None:
    print("Lỗi: Không thể tải ảnh. Vui lòng kiểm tra lại đường dẫn.")
else:
    # --- 2. Tạo đối tượng CLAHE ---
    # clipLimit: Ngưỡng giới hạn tương phản.
    #   Giá trị cao hơn sẽ cho độ tương phản mạnh hơn, nhưng có thể tăng nhiễu.
    #   Giá trị 2.0 là một khởi đầu tốt.
    # tileGridSize: Kích thước của lưới ô (ví dụ: 8x8).
    #   Ảnh sẽ được chia thành các ô 8x8 để xử lý cục bộ.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # --- 3. Áp dụng CLAHE cho ảnh xám ---
    cl_img = clahe.apply(img)

    # --- 4. Hiển thị kết quả (sử dụng Matplotlib) ---
    plt.figure(figsize=(12, 6))

    # Hiển thị ảnh gốc
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Ảnh gốc (Original)')
    plt.axis('off') # Ẩn trục tọa độ

    # Hiển thị ảnh đã xử lý CLAHE
    plt.subplot(1, 2, 2)
    plt.imshow(cl_img, cmap='gray')
    plt.title('Ảnh đã xử lý CLAHE (CLAHE Applied)')
    plt.axis('off')

    # Hiển thị cửa sổ
    plt.tight_layout()
    plt.show()

    # # Tùy chọn: Lưu ảnh kết quả ra file
    # cv2.imwrite('clahe_processed_image.jpg', cl_img)