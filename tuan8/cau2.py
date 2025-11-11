import cv2
import numpy as np

def main():
    calib_data = np.load('calib_data_2.npz')
    mtx = calib_data['mtx']
    dist = calib_data['dist']

    print(mtx)
    print('----------')
    print(dist)

    name = 'img10'

    img = cv2.imread(f'{name}.jpg')
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))

    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
    dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

    # crop the image
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    cv2.imwrite(f'{name}_result.jpg', dst)

if __name__ == '__main__':
    main()