import time

import cv2
import numpy as np
from skimage.feature import hog
from skimage.filters import threshold_multiotsu
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt


# Load video capture from file
cap = cv2.VideoCapture('thermal.mp4')




#fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
fgbg = cv2.bgsegm.createBackgroundSubtractorLSBP()
fgbg = cv2.bgsegm.createBackgroundSubtractorGSOC()
fgbg = cv2.bgsegm.createBackgroundSubtractorGMG(20, 0.7)
fgbg = cv2.bgsegm.createBackgroundSubtractorCNT(2, True)
fgbg = cv2.createBackgroundSubtractorMOG2()


def display(window_name, image, grid_pos = (0, 0)):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, image)
    cv2.resizeWindow(window_name, 384, 512)
    cv2.moveWindow(window_name, 384 * grid_pos[0], (512 + 30) * grid_pos[1])


def otsu_variance(image, t1, t2):
    """Calculate variance for multi Otsu threshold."""
    c0 = image[image <= t1]
    c1 = image[(image > t1) & (image <= t2)]
    c2 = image[image > t2]

    w0 = len(c0) / len(image)
    w1 = len(c1) / len(image)
    w2 = len(c2) / len(image)

    var_within_class = w0 * np.var(c0) + w1 * np.var(c1) + w2 * np.var(c2)
    return var_within_class

def multi_otsu_optimized(image, step=10):
    """Calculate multi Otsu threshold with skipping steps."""
    fn_min = np.inf
    thresh = [-1, -1]

    for t1 in range(0, 128, step):
        for t2 in range(t1+1, 256, step):
            variance = otsu_variance(image, t1, t2)
            if variance < fn_min:
                fn_min = variance
                thresh = [t1, t2]

    return thresh

def divide_image(image, t1, t2):
    """Divide image into three regions."""
    region1 = image <= t1
    region2 = (image > t1) & (image <= t2)
    region3 = image > t2
    return region1, region2, region3

def show_divided_image(image, region1, region2, region3):
    multiotsu_img_optimized = np.zeros_like(image)
    multiotsu_img_optimized[region1] = 0    # Assign gray level 85 to the pixels in region 1
    multiotsu_img_optimized[region2] = 128   # Assign gray level 170 to the pixels in region 2
    multiotsu_img_optimized[region3] = 255   # Assign gray level 255 to the pixels in region 3
    return multiotsu_img_optimized

while(cap.isOpened()):
    ret, frame = cap.read()
    display('RGB Frame', frame, (0, 0))

    # resize to 192x256
    frame = cv2.resize(frame, (192, 256))
    # convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    clean = np.zeros_like(frame)


    thresholds = threshold_multiotsu(gray, classes=3)
    region_cold, region_mid, region_hot = divide_image(gray, thresholds[0], thresholds[1])
    mask_cold = np.zeros_like(gray)
    mask_mid = np.zeros_like(gray)
    mask_hot = np.zeros_like(gray)
    mask_cold[region_cold] = 255
    mask_cold = cv2.morphologyEx(mask_cold, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    mask_mid[region_mid] = 255
    mask_hot[region_hot] = 255
    mask_hot = cv2.morphologyEx(mask_hot, cv2.MORPH_OPEN, np.ones((3, 11), np.uint8))

    contours, _ = cv2.findContours(mask_hot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        # fit and rotate rectangle
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)
        center = rect[0]
        cv2.circle(frame, (int(center[0]), int(center[1])), 3, (0, 0, 255), -1)

        # fit and draw line
        [vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
        lefty = int((-x * vy / vx) + y)
        righty = int(((gray.shape[1] - x) * vy / vx) + y)
        cv2.line(frame, (gray.shape[1] - 1, righty), (0, lefty), (0, 0, 255), 1)
        cv2.fillPoly(clean, [box], (0, 0, 255))





    display('mask_cold', mask_cold, (0, 1))
    display('mask_mid', mask_mid, (1, 1))
    display('mask_hot', mask_hot, (2, 1))


    # detect contours
    contours_cold, _ = cv2.findContours(mask_cold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # detect squares
    squares = []
    for cnt in contours_cold:
        epsilon = 0.1 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 4:
            squares.append(approx)
    # draw squares
    cv2.drawContours(frame, squares, -1, (255, 0, 0), 2)
    cv2.fillPoly(clean, squares, (255, 0, 0))
    display('Squares', frame, (2, 0))


    display('Clean', clean, (3, 0))







    k = cv2.waitKey(0) & 0xff
    # if q
    if k == ord('q'):
        break

    # if space
    if k == ord('c'):
        cv2.waitKey(5000)

cap.release()
cv2.destroyAllWindows()

