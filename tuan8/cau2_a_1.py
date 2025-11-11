import cv2
import numpy as np

# --- 1. Tải ảnh ---
image = cv2.imread("img1_result.jpg")

if image is None:
    print("Lỗi: Không thể tải ảnh. Vui lòng kiểm tra đường dẫn.")
else:
    # --- 2. (MỚI) Làm mờ ảnh để giảm nhiễu trước khi xử lý màu ---
    # Kernel (5,5) là kích thước tốt để bắt đầu, có thể thử (7,7) nếu vẫn nhiễu
    image_blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # --- 3. Chuyển đổi sang không gian màu HSV (sử dụng ảnh đã làm mờ) ---
    hsv_image = cv2.cvtColor(image_blurred, cv2.COLOR_BGR2HSV)

    # --- 4. Xác định dải màu xanh lam (Blue) trong HSV ---
    # Có thể nới rộng một chút dải màu nếu cần (ví dụ: lower_blue[0] giảm 5, upper_blue[0] tăng 5)
    lower_blue = np.array([90, 100, 100])
    upper_blue = np.array([130, 255, 255])

    # --- 5. Tạo mặt nạ (mask) cho màu xanh ---
    mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

    # --- 6. (CẢI THIỆN) Lọc nhiễu và làm mịn mask bằng các phép biến đổi hình thái học ---
    # Kernel lớn hơn sẽ có tác dụng làm mịn mạnh hơn, nhưng cũng có thể làm mất chi tiết nhỏ
    kernel = np.ones((7, 7), np.uint8)

    # Phép Mở (Open): Loại bỏ các đốm nhiễu nhỏ bên ngoài đối tượng
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Phép Đóng (Close): Lấp đầy các lỗ nhỏ bên trong đối tượng và làm mịn các cạnh
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # --- 7. Tìm các đường viền (contours) ---
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # --- 8. Xử lý contour tìm được ---
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)

        if cv2.contourArea(largest_contour) > 100:
            # --- (Tùy chọn) approxPolyDP để đơn giản hóa contour nếu vẫn quá chi tiết ---
            # epsilon = 0.01 * cv2.arcLength(largest_contour, True) # 1% chu vi
            # approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
            # contour_to_draw = approx_contour
            contour_to_draw = largest_contour  # Mặc định vẫn dùng contour gốc

            # --- 9. Tìm tâm 2D bằng Moments ---
            M = cv2.moments(contour_to_draw)

            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                center_2d = (cX, cY)

                # --- 10. Vẽ kết quả lên ảnh gốc ---
                cv2.drawContours(image, [contour_to_draw], -1, (0, 255, 0), 2)
                cv2.circle(image, center_2d, 7, (0, 0, 255), -1)
                cv2.putText(image, f"Tam: ({cX}, {cY})", (cX - 50, cY - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                print(f"Đã tìm thấy vật thể màu xanh.")
                print(f"Tọa độ tâm 2D: ({cX}, {cY})")

                # Lưu ảnh kết quả (vẫn giữ lại)
                cv2.imwrite("ket_qua_tam_2d_cai_thien.jpg", image)

                # Hiển thị ảnh mask (rất hữu ích để kiểm tra)
                cv2.imshow("Mask", mask)
                cv2.imshow("Ket Qua", image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            else:
                print("Lỗi: m00 = 0, không thể tính tâm.")
        else:
            print("Không tìm thấy vật thể nào đủ lớn.")
    else:
        print("Không tìm thấy vật thể màu xanh nào trong dải màu đã chọn.")