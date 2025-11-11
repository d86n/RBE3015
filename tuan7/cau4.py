import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    anh_goc = cv2.imread('anh_goc.png', cv2.IMREAD_GRAYSCALE)
    anh_toi_1 = cv2.imread('anh_toi_1.png')
    anh_toi_1 = cv2.cvtColor(anh_toi_1, cv2.COLOR_BGR2GRAY)
    anh_toi_2 = cv2.imread('anh_toi_2.png', cv2.IMREAD_GRAYSCALE)

    hist_1 = equalize_hist(anh_toi_1)
    hist_2 = equalize_hist(anh_toi_2)

    clahe_1 = clahe(anh_toi_1, 2.0)
    clahe_2 = clahe(anh_toi_2, 2.0)

    gamma_1 = adjust_gamma(anh_toi_1, 0.5)
    gamma_2 = adjust_gamma(anh_toi_2, 0.5)

    clahe_gamma_1 = clahe(gamma_1, 10.0)
    clahe_gamma_2 = clahe(gamma_2, 0.5)

    show(anh_goc, clahe_gamma_1, clahe_gamma_2, 'Gamma Correction + CLAHE')

def show(anh_goc, anh_1, anh_2, ten):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(anh_goc, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title('Ảnh gốc')
    axes[0].axis('off')

    psnr_1 = cv2.PSNR(anh_goc, anh_1)
    psnr_2 = cv2.PSNR(anh_goc, anh_2)

    axes[1].imshow(anh_1, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title(f'Ảnh tối 1 áp dụng {ten} ({round(psnr_1, 2)})')
    axes[1].axis('off')

    axes[2].imshow(anh_2, cmap='gray', vmin=0, vmax=255)
    axes[2].set_title(f'Ảnh tối 2 áp dụng {ten} ({round(psnr_2, 2)})')
    axes[2].axis('off')

    plt.tight_layout()

    plt.savefig(f'kq_{ten}')
    plt.show()

def equalize_hist(img):
    height, width = img.shape
    total_pixels = height * width

    # Tính histogram
    hist = [0] * 256
    for i in range(height):
        for j in range(width):
            value = img[i, j]
            hist[value] += 1

    # Tính PDF và CDF
    pdf = [val / total_pixels for val in hist]
    cdf = [0] * 256
    cdf[0] = pdf[0]
    for i in range(1, 256):
        cdf[i] = cdf[i - 1] + pdf[i]

    # Tạo bảng tra cứu (Look-Up Table - LUT)
    l = 256
    trans = [0] * l
    for i in range(l):
        trans[i] = int(round((l - 1) * cdf[i]))

    # Ánh xạ ảnh mới
    equalized_img = np.copy(img)
    for i in range(height):
        for j in range(width):
            equalized_img[i, j] = trans[img[i, j]]

    return equalized_img

def clahe(img, clip_limit):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))

    clahe_img = clahe.apply(img)

    return clahe_img

def adjust_gamma(image, gamma=1.0):
    table = np.array([
        ((i / 255.0) ** gamma) * 255
        for i in range(256)
    ]).astype("uint8")

    return cv2.LUT(image, table)

if __name__ == '__main__':
    main()
