import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def on_change(pos):
    pass


def main():
    src_H = cv.imread("data/sample_H.png", cv.IMREAD_GRAYSCALE)
    src_L = cv.imread("data/sample_L.png", cv.IMREAD_GRAYSCALE)
    src_H = cv.resize(src_H, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv.INTER_LINEAR)
    src_L = cv.resize(src_L, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv.INTER_LINEAR)
    cv.namedWindow("dst")
    cv.createTrackbar("brightness", "dst", 0, 255, on_change)
    cv.createTrackbar("minus_br", "dst", 0, 255, on_change)
    cv.createTrackbar("normalize", "dst", 0, 1, on_change)
    cv.createTrackbar("H-L", "dst", 0, 1, on_change)

    cv.setTrackbarPos("minus_br", "dst", 0)
    cv.setTrackbarPos("brightness", "dst", 0)
    cv.setTrackbarPos("normalize", "dst", 0)
    cv.setTrackbarPos("H-L", "dst", 0)

    while cv.waitKey(1) != ord('q'):
        br = cv.getTrackbarPos("brightness", "dst")
        minus_br = cv.getTrackbarPos("minus_br", "dst")
        normalize = cv.getTrackbarPos("normalize", "dst")
        H_L = cv.getTrackbarPos("H-L", "dst")

        dst = cv.add(src_H, br)
        dst = cv.subtract(dst, minus_br)
        if normalize == 1:
            dst = cv.normalize(dst, None, 0, 255, cv.NORM_MINMAX)
        if H_L == 1:
            dst = np.clip((dst - src_L), 0, 255)

        cv.imshow("dst", dst)

    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
