import cv2
import numpy as np

# --- 1. Tải ảnh ---
image = cv2.imread("img1_result.jpg")

if image is None:
    print("Lỗi: Không thể tải ảnh. Vui lòng kiểm tra đường dẫn.")
else:
    # --- 2. Chuyển đổi sang không gian màu HSV ---
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # --- 3. Xác định dải màu xanh lam (Blue) trong HSV ---
    lower_blue = np.array([90, 100, 100])
    upper_blue = np.array([130, 255, 255])

    # --- 4. Tạo mặt nạ (mask) cho màu xanh ---
    mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

    # --- 5. (Tùy chọn) Lọc nhiễu cho mask ---
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # --- 6. Tìm các đường viền (contours) ---
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # --- 7. Xử lý contour tìm được ---
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)

        if cv2.contourArea(largest_contour) > 100:
            M = cv2.moments(largest_contour)

            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                center_2d = (cX, cY)

                cv2.drawContours(image, [largest_contour], -1, (0, 255, 0), 2)
                cv2.circle(image, center_2d, 7, (0, 0, 255), -1)
                cv2.putText(image, f"Tam: ({cX}, {cY})", (cX - 50, cY - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                print(f"Đã tìm thấy vật thể màu xanh.")
                print(f"Tọa độ tâm 2D: ({cX}, {cY})")

                # Lưu ảnh kết quả (vẫn giữ lại)
                cv2.imwrite("ket_qua_tam_2d.jpg", image)

                # ========== PHẦN SỬA ĐỔI ĐỂ HIỂN THỊ ẢNH ==========

                # Hiển thị ảnh mask (hữu ích để gỡ lỗi dải màu)
                # cv2.imshow("Mask", mask)

                # Hiển thị ảnh kết quả cuối cùng
                cv2.imshow("Ket Qua", image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                # ===================================================

            else:
                print("Lỗi: m00 = 0, không thể tính tâm.")
        else:
            print("Không tìm thấy vật thể nào đủ lớn.")
    else:
        print("Không tìm thấy vật thể màu xanh nào trong dải màu đã chọn.")