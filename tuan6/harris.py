import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    img = cv2.imread("../images/img6.1.jpg")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    dest = cv2.cornerHarris(gray, 2, 3, 0.06)

    img[dest > 0.02 * dest.max()] = [0, 0, 255]

    result = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(38, 21))
    plt.subplot(121), plt.imshow(img_rgb)
    plt.axis('off')
    plt.subplot(122), plt.imshow(result)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()