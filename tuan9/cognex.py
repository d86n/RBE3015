import cv2

def main():
    checkerboard = cv2.imread('img/10mm.jpg')
    gray = cv2.cvtColor(checkerboard, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    row_index = h // 2
    profile = gray[row_index, :]

    current_color = profile[0]
    pixel_count = 0
    edge_size_pixels = 0
    found = False

    for pixel_value in profile:
        if pixel_value == current_color:
            pixel_count += 1
        else:
            edge_size_pixels = pixel_count
            found = True
            break

    if found:
        print(edge_size_pixels)

if __name__ == '__main__':
    main()