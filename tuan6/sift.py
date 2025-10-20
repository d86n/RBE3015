import cv2

def main():
    image = cv2.imread('../images/img6.2.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()

    keypoints, descriptors = sift.detectAndCompute(gray, None)

    image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    print(f"Tổng số điểm đặc trưng SIFT tìm được: {len(keypoints)}")
    print(f"Kích thước của ma trận descriptors SIFT: {descriptors.shape}") # (số keypoint, 128)

    cv2.imwrite('sift_keypoints.jpg', image_with_keypoints)
    cv2.namedWindow('image', flags=cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 900, 900)
    cv2.imshow('image', image_with_keypoints)
    cv2.waitKey(0)
    cv2.destroyAllWindows()