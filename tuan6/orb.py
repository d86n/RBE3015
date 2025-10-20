import cv2
from matplotlib import pyplot as plt

def main():
    image = cv2.imread('../images/img6.2.jpg')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=1000)

    keypoints, descriptors = orb.detectAndCompute(gray_image, None)

    image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0), flags=0)

    print(f"Tổng số điểm đặc trưng ORB tìm được: {len(keypoints)}")
    print(f"Kích thước của ma trận descriptors ORB: {descriptors.shape}")  # (số keypoint, 32)

    plt.figure(figsize=(30, 20))
    plt.imshow(cv2.cvtColor(image_with_keypoints, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
