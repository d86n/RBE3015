import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    img = cv2.imread("../images/img5.2.jpg")
    img = cv2.resize(img, (128, 128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    result = calc_dct_2d(img)

    dct_log_spectrum = np.log(np.abs(result) + 1)

    img_lp, img_hp = apply_filter(result, 16)

    plt.subplot(221), plt.imshow(img, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(222), plt.imshow(dct_log_spectrum, cmap='gray')
    plt.title('DCT'), plt.xticks([]), plt.yticks([])
    plt.subplot(223), plt.imshow(img_lp, cmap='gray')
    plt.title('DCT Low Pass'), plt.xticks([]), plt.yticks([])
    plt.subplot(224), plt.imshow(img_hp, cmap='gray')
    plt.title('DCT High Pass'), plt.xticks([]), plt.yticks([])
    plt.show()

def calc_dct_1d(x):
    n = len(x)
    output_x = np.zeros(n, dtype=np.float64)
    for i in range(n):
        sum_val = 0.0
        for j in range(n):
            cos_term = np.cos(np.pi / n * (j + 0.5) * i)
            sum_val += x[j] * cos_term
        output_x[i] = sum_val
    return output_x

def calc_dct_2d(image):
    m, n = image.shape
    temp = np.zeros((m, n), dtype=np.float64)

    for i in range(m):
        temp[i, :] = calc_dct_1d(image[i, :])

    dct_image = np.zeros((m, n), dtype=np.float64)

    for j in range(n):
        dct_image[:, j] = calc_dct_1d(temp[:, j])

    return dct_image

def calc_idct_1d(x):
    n = len(x)
    output_x = np.zeros(n, dtype=np.float64)
    for i in range(n):
        sum_val = x[0] / 2.0
        for j in range(1, n):
            cos_term = np.cos(np.pi / n * (i + 0.5) * j)
            sum_val += x[j] * cos_term
        output_x[i] = sum_val
    return output_x * (2.0 / n)

def calc_idct_2d(dct_image):
    m, n = dct_image.shape
    temp = np.zeros((m, n), dtype=np.float64)

    for i in range(m):
        temp[i, :] = calc_idct_1d(dct_image[i, :])

    idct_image = np.zeros((m, n), dtype=np.float64)

    for j in range(n):
        idct_image[:, j] = calc_idct_1d(temp[:, j])

    return idct_image

def apply_filter(dct_image, block_size):
    m, n = dct_image.shape
    mask_lp = np.zeros((m, n), np.float32)
    mask_lp[0:block_size, 0:block_size] = 1

    mask_hp = 1 - mask_lp

    dct_lp = dct_image * mask_lp

    img_back_lp = calc_idct_2d(dct_lp)

    dct_hp = dct_image * mask_hp
    img_back_hp = calc_idct_2d(dct_hp)

    return img_back_lp, normalize(img_back_hp)

def normalize(image):
    min_val = np.min(image)
    max_val = np.max(image)

    if max_val == min_val:
        return np.zeros(image.shape, dtype=np.uint8)

    normalized_image = (image.astype(np.float32) - min_val) * (255.0 / (max_val - min_val))

    return normalized_image.astype(np.uint8)

if __name__ == "__main__":
    main()