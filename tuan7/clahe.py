import cv2
import numpy as np

img_path = 'anh_toi_1.png'

img = cv2.imread(img_path)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. Kiểm tra xem ảnh có được tải thành công không

    # 3. Tạo một đối tượng CLAHE
    # clipLimit: Ngưỡng giới hạn độ tương phản. 
    #            Đây là giá trị quan trọng nhất. 
    #            Giá trị 2.0 là một khởi đầu tốt.
    # tileGridSize: Kích thước của lưới (ví dụ: 8x8 pixels) mà ảnh
    #               sẽ được chia ra để cân bằng cục bộ.
clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))

    # 4. Áp dụng CLAHE cho ảnh xám
clahe_img = clahe.apply(gray)

cv2.imwrite("anh_toi_1_kq.jpg", clahe_img)
