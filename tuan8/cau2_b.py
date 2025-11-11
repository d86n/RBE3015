import cv2
import numpy as np


# ====================================================================
# PHẦN 1: CÁC HÀM XỬ LÝ
# ====================================================================

def find_object_center(image):
    """Tìm tâm 2D (pixel) của vật thể màu xanh."""

    # HÀM NÀY DÙNG BGR2HSV (ĐÚNG)
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
                return (cX, cY)
    return None


def get_pixel_to_cm_mapping(calibration_image_path, square_size_cm, chessboard_size):
    """
    Tính ma trận Homography H (cm -> pixel) và H_inv (pixel -> cm).
    Gốc tọa độ (0,0) được đặt ở góc DƯỚI-TRÁI.
    """

    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    grid_coords = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

    max_y_index = chessboard_size[1] - 1
    grid_coords_remapped = np.copy(grid_coords)
    grid_coords_remapped[:, 1] = max_y_index - grid_coords[:, 1]

    objp[:, :2] = grid_coords_remapped
    objp = objp * square_size_cm
    world_points_2d = objp[:, :2]

    image_calib = cv2.imread(calibration_image_path)
    if image_calib is None:
        print(f"Lỗi: Không thể tải ảnh hiệu chuẩn '{calibration_image_path}'.")
        return None, None

    # ========== SỬA LỖI ==========
    # HÀM NÀY PHẢI DÙNG BGR2GRAY (Ảnh xám)
    gray_calib = cv2.cvtColor(image_calib, cv2.COLOR_BGR2GRAY)
    # ============================

    ret, corners = cv2.findChessboardCorners(gray_calib, chessboard_size, None)

    if ret:
        print("Đã tìm thấy bàn cờ trong ảnh hiệu chuẩn.")
        H, _ = cv2.findHomography(world_points_2d, corners)
        H_inv = np.linalg.inv(H)
        return H, H_inv
    else:
        print(f"Lỗi: Không thể tìm thấy bàn cờ trong ảnh hiệu chuẩn '{calibration_image_path}'.")
        print(f"Hãy kiểm tra lại CHESSBOARD_SIZE={chessboard_size} có khớp với ảnh không.")
        return None, None


# ====================================================================
# PHẦN 2: THIẾT LẬP VÀ CHẠY
# ====================================================================

# --- A. DỮ LIỆU CỦA BẠN ---
SQUARE_SIZE_CM = 2.9
CHESSBOARD_SIZE = (9, 6)  # 9 góc ngang, 6 góc dọc
AXIS_LENGTH_CM = 5.0

# --- B. ĐƯỜNG DẪN CỦA BẠN ---
IMAGE_CALIBRATION = "ban_co_result.jpg"
IMAGE_OBJECT = "img1_result.jpg"

# --- C. BƯỚC 1: Tính toán ánh xạ ---
H, H_inv = get_pixel_to_cm_mapping(IMAGE_CALIBRATION, SQUARE_SIZE_CM, CHESSBOARD_SIZE)

if H is not None and H_inv is not None:
    print("Đã tính toán thành công ma trận ánh xạ (gốc dưới-trái).")

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
            pixel_vec = np.array([cX, cY, 1], dtype=np.float32)
            world_vec_scaled = np.dot(H_inv, pixel_vec)
            X_cm = world_vec_scaled[0] / world_vec_scaled[2]
            Y_cm = world_vec_scaled[1] / world_vec_scaled[2]

            print("--- KẾT QUẢ ---")
            print(f"Tọa độ thế giới thực (X, Y): ({X_cm:.2f} cm, {Y_cm:.2f} cm)")
            print(f"(Gốc (0, 0) là góc trong cùng bên DƯỚI-TRÁI của bàn cờ)")

            # --- F. VẼ TRỤC TỌA ĐỘ (CM -> Pixel) ---
            origin_cm = np.array([0, 0, 1], dtype=np.float32)
            x_axis_cm = np.array([AXIS_LENGTH_CM, 0, 1], dtype=np.float32)
            y_axis_cm = np.array([0, AXIS_LENGTH_CM, 1], dtype=np.float32)

            origin_px_scaled = np.dot(H, origin_cm)
            x_axis_px_scaled = np.dot(H, x_axis_cm)
            y_axis_px_scaled = np.dot(H, y_axis_cm)

            origin_px = (int(origin_px_scaled[0] / origin_px_scaled[2]),
                         int(origin_px_scaled[1] / origin_px_scaled[2]))

            x_axis_px = (int(x_axis_px_scaled[0] / x_axis_px_scaled[2]),
                         int(x_axis_px_scaled[1] / x_axis_px_scaled[2]))

            y_axis_px = (int(y_axis_px_scaled[0] / y_axis_px_scaled[2]),
                         int(y_axis_px_scaled[1] / y_axis_px_scaled[2]))

            cv2.arrowedLine(image_obj, origin_px, x_axis_px, (0, 0, 255), 2)
            cv2.putText(image_obj, "X (cm)", (x_axis_px[0] + 5, x_axis_px[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.arrowedLine(image_obj, origin_px, y_axis_px, (0, 255, 0), 2)
            cv2.putText(image_obj, "Y (cm)", (y_axis_px[0], y_axis_px[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # --- G. VẼ KẾT QUẢ VẬT THỂ ---
            cv2.circle(image_obj, object_pixel_center, 7, (255, 0, 0), -1)
            text = f"({X_cm:.1f}cm, {Y_cm:.1f}cm)"
            cv2.putText(image_obj, text, (cX - 60, cY - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # --- H. HIỂN THỊ ---
            cv2.imshow("Ket Qua The Gioi Thuc (Goc Duoi-Trai)", image_obj)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        else:
            print(f"Lỗi: Không tìm thấy vật thể màu xanh trong ảnh '{IMAGE_OBJECT}'.")