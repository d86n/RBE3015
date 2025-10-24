import cv2
import numpy as np

def main():
    gray = cv2.imread('../images/img7.7.jpg', cv2.IMREAD_GRAYSCALE)

    gamma = 0.5

    result = adjust_gamma(gray, gamma=gamma)

    cv2.imshow('original', gray)
    cv2.imshow('adjusted', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def adjust_gamma(image, gamma=1.0):
    gamma = gamma
    table = np.array([
        ((i / 255.0) ** gamma) * 255
        for i in range(256)
    ]).astype("uint8")

    return cv2.LUT(image, table)

if __name__ == '__main__':
    main()