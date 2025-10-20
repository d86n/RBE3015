import cv2
import matplotlib.pyplot as plt

def main():
    img1 = cv2.imread('../images/img6.3.jpg')
    img2 = cv2.imread('../images/img6.4.jpg')

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=1000)

    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    ratio_thresh = 0.75

    for m, n in matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    img_matches_rgb = cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(30, 20))
    plt.imshow(img_matches_rgb)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
