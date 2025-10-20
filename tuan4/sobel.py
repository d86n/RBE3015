import numpy as np

def main():
    pass

def convolve(image, kernel):
    img_h, img_w = image.shape
    ker_h, ker_w = kernel.shape

    pad_h = ker_h // 2
    pad_w = ker_w // 2

    padded_img = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')

    output = np.zeros_like(image, dtype=np.float64)

    for i in range(img_h):
        for j in range(img_w):
            region = padded_img[i:i+ker_h, j:j+ker_w]
            output[i, j] = np.sum(region * kernel)

    return output

def sobel(image):
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]], dtype=np.float64
    )

    sobel_y = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]], dtype=np.float64
    )

    gx = convolve(image, sobel_x)
    gy = convolve(image, sobel_y)

    gradient = np.sqrt(gx ** 2 + gy ** 2)

    gx_norm = np.clip(np.abs(gx), 0, 255).astype(np.uint8)
    gy_norm = np.clip(np.abs(gy), 0, 255).astype(np.uint8)

    gradient_norm = np.clip(np.abs(gradient), 0, 255).astype(np.uint8)

    return gx_norm, gy_norm, gradient_norm

if __name__ == "__main__":
    main()