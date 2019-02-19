import cv2
import numpy as np


def main():
    imgpath1 = 'stealie2.png'
    imgpath2 = 'test3.jpg'

    img1 = cv2.imread(imgpath1, cv2.IMREAD_COLOR)
    img1 = cv2.resize(img1, (784, 784), cv2.INTER_CUBIC)
    img2 = cv2.imread(imgpath2, cv2.IMREAD_COLOR)
    img2 = cv2.resize(img2, (784, 784), cv2.INTER_CUBIC)

    img1gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img1gray, 127, 155, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    print(len(contours))
    mask = np.zeros_like(img1)
    circle_ind = 24
    bolt_ind = 25
    cv2.drawContours(mask, contours, circle_ind, (255, 255, 255), -1)
    cv2.drawContours(mask, contours, bolt_ind, (0, 0, 0), -1)
    mask = np.squeeze(mask[:, :, 0]).astype(np.uint8)

    img2 = np.clip(img2 - 10, 0, 255)
    img1[mask == 255] = img2[mask == 255]
    cv2.imwrite('dreamstealie.png', img1)


if __name__ == '__main__':
    main()