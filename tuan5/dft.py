import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    img = cv2.imread("../images/img5.1.jpg")
    img = cv2.resize(img, (128, 128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result = calc_dft_2d(img)

    dft_shifted_image = shift_dft_2d(result)
    magnitude_spectrum = np.log(np.abs(dft_shifted_image) + 1)

    img_lp, img_hp = apply_filter(dft_shifted_image)

    plt.subplot(221), plt.imshow(img, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(222), plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.subplot(223), plt.imshow(img_lp, cmap='gray')
    plt.title('Low Pass Filter'), plt.xticks([]), plt.yticks([])
    plt.subplot(224), plt.imshow(img_hp, cmap='gray')
    plt.title('High Pass Filter'), plt.xticks([]), plt.yticks([])
    plt.show()

def calc_dft_1d(x):
    n = len(x)
    output_x = np.zeros(n, dtype=np.complex128)
    for i in range(n):
        sum_val = 0.0
        for j in range(n):
            angle = -2 * np.pi * i * j / n
            sum_val += x[j] * np.exp(1j * angle)
        output_x[i] = sum_val
    return output_x

def calc_dft_2d(image):
    m, n = image.shape

    temp = np.zeros((m, n), dtype=np.complex128)

    for i in range(m):
        temp[i, :] = calc_dft_1d(image[i, :])

    dft_image = np.zeros((m, n), dtype=np.complex128)

    for j in range(n):
        dft_image[:, j] = calc_dft_1d(temp[:, j])

    return dft_image

def calc_idft_1d(x):
    n = len(x)
    output_x = np.zeros(n, dtype=np.complex128)
    for i in range(n):
        sum_val = 0.0
        for j in range(n):
            angle = 2 * np.pi * i * j / n
            sum_val += x[j] * np.exp(1j * angle)
        output_x[i] = sum_val

    return output_x / n

def calc_idft_2d(dft_image):
    m, n = dft_image.shape
    temp = np.zeros((m, n), dtype=np.complex128)

    for i in range(m):
        temp[i, :] = calc_idft_1d(dft_image[i, :])

    idft_image = np.zeros((m, n), dtype=np.complex128)

    for j in range(n):
        idft_image[:, j] = calc_idft_1d(temp[:, j])

    return idft_image

def shift_dft_2d(dft_image):
    m, n = dft_image.shape
    mid_m, mid_n = m // 2, n // 2

    dft_shifted_image = np.zeros_like(dft_image)

    top_left = dft_image[0:mid_m, 0:mid_n]
    top_right = dft_image[0:mid_m, mid_n:n]
    bottom_left = dft_image[mid_m:m, 0:mid_n]
    bottom_right = dft_image[mid_m:m, mid_n:n]

    dft_shifted_image[0:mid_m, 0:mid_n] = bottom_right
    dft_shifted_image[bottom_right.shape[0]:m, bottom_right.shape[1]:n] = top_left
    dft_shifted_image[0:mid_m, mid_n:n] = bottom_left
    dft_shifted_image[mid_m:m, 0:mid_n] = top_right

    return dft_shifted_image

def apply_filter(dft_shifted_image):
    m, n = dft_shifted_image.shape
    mid_m, mid_n = m // 2, n // 2
    cutoff_radius = 30

    mask_lp = np.zeros((m, n), np.float32)
    cv2.circle(mask_lp, (mid_m, mid_n), cutoff_radius, (1, 1), -1)

    mask_hp = 1 - mask_lp

    filtered_image_lp = dft_shifted_image * mask_lp
    f_ishift_lp = shift_dft_2d(filtered_image_lp)
    img_back_lp = calc_idft_2d(f_ishift_lp)
    img_back_lp = np.abs(img_back_lp)

    filtered_image_hp = dft_shifted_image * mask_hp
    f_ishift_hp = shift_dft_2d(filtered_image_hp)
    img_back_hp = calc_idft_2d(f_ishift_hp)
    img_back_hp = np.abs(img_back_hp)

    return img_back_lp, img_back_hp

if __name__ == "__main__":
    main()