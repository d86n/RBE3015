import cv2
import matplotlib.pyplot as plt

# --- 1. Tải ảnh ---
# Đảm bảo bạn thay 'path_to_your_dark_image.jpg' bằng đường dẫn thực tế đến ảnh của bạn.
# cv2.IMREAD_GRAYSCALE: Tải ảnh trực tiếp ở dạng ảnh xám (grayscale)
img = cv2.imread('../images/img7.1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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
    clahe_img = clahe.apply(gray)

    # --- 4. Hiển thị kết quả (sử dụng Matplotlib) ---

    plt.figure(figsize=(12, 8))  # Tạo một cửa sổ hình vẽ mới

    # Vẽ histogram cho ảnh xám gốc (màu xanh)
    plt.hist(gray.ravel(), bins=256, range=[0, 256], color='blue', alpha=0.6, label='Histogram ảnh đầu vào')

    # Vẽ histogram cho ảnh kết quả (màu đỏ)
    plt.hist(clahe_img.ravel(), bins=256, range=[0, 256], color='red', alpha=0.6, label='Histogram ảnh đầu ra')

    plt.title('So sánh histogram trước và sau cân bằng', fontsize=16)
    plt.xlabel('Mức xám', fontsize=12)
    plt.ylabel('Số lượng điểm ảnh', fontsize=12)
    plt.legend()
    plt.xlim([0, 256])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig("histogram_compare_2.jpg")
    plt.show()

    psnr = cv2.PSNR(gray, clahe_img)

    print(f"Giá trị PSNR giữa ảnh gốc và ảnh kết quả là: {psnr:.2f} dB")

    # # Tùy chọn: Lưu ảnh kết quả ra file
    # cv2.imwrite('clahe_processed_image.jpg', cl_img)