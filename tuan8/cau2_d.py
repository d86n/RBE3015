import cv2
import numpy as np


# ====================================================================
# PHẦN 1: CÁC HÀM XỬ LÝ
# ====================================================================

def find_object_center(image):
    """Tìm tâm 2D (pixel) của vật thể màu xanh."""
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 100, 100])
    upper_blue = np.array([130, 255, 255])

    mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 100:
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                return cX, cY
    return None


def get_pixel_to_cm_mapping(calibration_image_path, square_size_cm, chessboard_size):
    """
    Tính H và H_inv dùng gốc (0,0) MẶC ĐỊNH (TRÊN-TRÁI).
    Hàm này đã được SỬA LỖI để xử lý trường hợp thứ tự góc bị đảo ngược (xoay 180 độ).
    """

    # Tọa độ thế giới (objp) luôn giả định gốc TRÊN-TRÁI
    objp_default = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp_default[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp_default = objp_default * square_size_cm
    world_points_2d_default = objp_default[:, :2]

    image_calib = cv2.imread(calibration_image_path)
    if image_calib is None:
        print(f"Lỗi: Không thể tải ảnh hiệu chuẩn '{calibration_image_path}'.")
        return None, None

    gray_calib = cv2.cvtColor(image_calib, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray_calib, chessboard_size, None)

    if ret:
        print(f"Đã tìm thấy bàn cờ {chessboard_size}.")

        # ========== SỬA LỖI XOAY 180 ĐỘ ==========
        # Kiểm tra xem góc đầu tiên (corners[0]) hay góc cuối (corners[-1])
        # gần với gốc (0,0) của ảnh hơn.

        # Tọa độ pixel của góc đầu tiên
        corner_0_px = corners[0][0]
        # Tọa độ pixel của góc cuối cùng
        corner_last_px = corners[-1][0]

        # Tính khoảng cách (Manhattan) đến gốc (0,0) của ảnh
        dist_0 = corner_0_px[0] + corner_0_px[1]
        dist_last = corner_last_px[0] + corner_last_px[1]

        # Nếu góc cuối cùng (dist_last) gần gốc (0,0) hơn góc đầu tiên (dist_0),
        # điều đó có nghĩa là OpenCV đã quét từ Dưới-Phải -> Trên-Trái.
        if dist_last < dist_0:
            print("Phát hiện thứ tự góc bị đảo ngược (BR->TL). Đang tự động sửa...")
            # Đảo ngược toàn bộ mảng corners để nó trở về thứ tự (TL->BR)
            corners = corners[::-1]
        # ==========================================

        # Tính ma trận H
        H, _ = cv2.findHomography(world_points_2d_default, corners)
        H_inv = np.linalg.inv(H)

        return H, H_inv
    else:
        print(f"LỖI: Không thể tìm thấy bàn cờ trong ảnh '{calibration_image_path}'.")
        print(f"Hãy kiểm tra lại CHESSBOARD_SIZE={chessboard_size} (10x7 ô vuông).")
        return None, None

    # ====================================================================


# PHẦN 2: THIẾT LẬP VÀ CHẠY
# (Logic phần này GIỮ NGUYÊN như lần trước, vì nó đã đúng)
# ====================================================================

# --- A. DỮ LIỆU CỦA BẠN ---
SQUARE_SIZE_CM = 2.9
AXIS_LENGTH_CM = 5.0
CHESSBOARD_SIZE = (9, 6)  # (cols=9, rows=6) cho 10x7 ô vuông

name = 'img10'

# --- B. ĐƯỜNG DẪN CỦA BẠN ---
IMAGE_CALIBRATION = "ban_co_result.jpg"
IMAGE_OBJECT = f"{name}_result.jpg"

# --- C. BƯỚC 1: Tính toán ánh xạ (Gốc trên-trái, đã sửa lỗi xoay) ---
H_default, H_inv_default = get_pixel_to_cm_mapping(IMAGE_CALIBRATION, SQUARE_SIZE_CM, CHESSBOARD_SIZE)

if H_default is not None and H_inv_default is not None:
    print("Đã tính toán thành công ma trận ánh xạ (gốc trên-trái).")

    # --- D. BƯỚC 2: Xử lý ảnh có vật thể ---
    image_obj = cv2.imread(IMAGE_OBJECT)
    if image_obj is None:
        print(f"Lỗi: Không thể tải ảnh vật thể '{IMAGE_OBJECT}'.")
    else:
        object_pixel_center = find_object_center(image_obj)

        if object_pixel_center is not None:
            cX, cY = object_pixel_center
            print(f"Tìm thấy tâm vật thể tại pixel: ({cX}, {cY})")

            # --- E. BƯỚC 3: Chuyển đổi tọa độ vật thể (Pixel -> CM) ---

            # 1. Chuyển đổi pixel sang (cm) với gốc TRÊN-TRÁI
            pixel_vec = np.array([cX, cY, 1], dtype=np.float32)
            world_vec_scaled = np.dot(H_inv_default, pixel_vec)
            X_default = world_vec_scaled[0] / world_vec_scaled[2]
            Y_default = world_vec_scaled[1] / world_vec_scaled[2]

            # 2. Tính chiều cao tối đa của bàn cờ (cm)
            max_Y_cm = (CHESSBOARD_SIZE[1] - 1) * SQUARE_SIZE_CM  # (rows - 1) * size

            # 3. Chuyển đổi sang gốc DƯỚI-TRÁI
            X_final = X_default
            Y_final = max_Y_cm - Y_default  # Lật trục Y

            print("--- KẾT QUẢ ---")
            print(f"Tọa độ thế giới thực (X, Y): ({X_final:.2f} cm, {Y_final:.2f} cm)")
            print(f"(Gốc (0, 0) là góc trong cùng bên DƯỚI-TRÁI của bàn cờ)")

            # --- F. VẼ TRỤC TỌA ĐỘ (CM -> Pixel) ---

            # 1. Định nghĩa các điểm (cm) của hệ trục DƯỚI-TRÁI
            #    nhưng biểu diễn chúng trong hệ tọa độ (cm) TRÊN-TRÁI
            origin_BL_cm = np.array([0, max_Y_cm, 1], dtype=np.float32)
            x_axis_BL_cm = np.array([AXIS_LENGTH_CM, max_Y_cm, 1], dtype=np.float32)
            y_axis_BL_cm = np.array([0, max_Y_cm - AXIS_LENGTH_CM, 1], dtype=np.float32)

            # 2. Chiếu các điểm này ra pixel bằng H_default
            origin_px_scaled = np.dot(H_default, origin_BL_cm)
            x_axis_px_scaled = np.dot(H_default, x_axis_BL_cm)
            y_axis_px_scaled = np.dot(H_default, y_axis_BL_cm)

            # 3. Chuẩn hóa
            origin_px = (int(origin_px_scaled[0] / origin_px_scaled[2]),
                         int(origin_px_scaled[1] / origin_px_scaled[2]))

            x_axis_px = (int(x_axis_px_scaled[0] / x_axis_px_scaled[2]),
                         int(x_axis_px_scaled[1] / x_axis_px_scaled[2]))

            y_axis_px = (int(y_axis_px_scaled[0] / y_axis_px_scaled[2]),
                         int(y_axis_px_scaled[1] / y_axis_px_scaled[2]))

            # 4. Vẽ trục
            cv2.arrowedLine(image_obj, origin_px, x_axis_px, (0, 0, 255), 2)
            cv2.putText(image_obj, "X (cm)", (x_axis_px[0] + 5, x_axis_px[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.arrowedLine(image_obj, origin_px, y_axis_px, (0, 255, 0), 2)
            cv2.putText(image_obj, "Y (cm)", (y_axis_px[0], y_axis_px[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # --- G. VẼ KẾT QUẢ VẬT THỂ ---
            cv2.circle(image_obj, object_pixel_center, 7, (0, 255, 0), -1)
            text = f"({X_final:.1f}cm, {Y_final:.1f}cm)"
            cv2.putText(image_obj, text, (object_pixel_center[0] - 60, object_pixel_center[1] - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # --- H. HIỂN THỊ ---
            cv2.imshow("Ket Qua The Gioi Thuc (Goc Duoi-Trai)", image_obj)
            cv2.imwrite(f'ket_qua_{name}.jpg', image_obj)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        else:
            print(f"Lỗi: Không tìm thấy vật thể màu xanh trong ảnh '{IMAGE_OBJECT}'.")