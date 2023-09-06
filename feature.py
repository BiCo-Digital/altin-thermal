from collections import deque

import itertools
import cv2
import numpy as np
from imutils.object_detection import non_max_suppression

# create named window and move it - 400px in x dir
cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
# show window
cv2.imshow('Frame', np.zeros((256, 192, 3), np.uint8))
cv2.moveWindow("Frame", -512, 0)

# resize window
cv2.resizeWindow("Frame", 384, 512)

# Capture the frame from the video source while 1
cap = cv2.VideoCapture('output8_crop.mp4')

# bgsub = cv2.bgsegm.createBackgroundSubtractorLSBP()
# bgsub = cv2.bgsegm.createBackgroundSubtractorGSOC()
# bgsub = cv2.bgsegm.createBackgroundSubtractorGMG(20, 0.7)
bgsub = cv2.bgsegm.createBackgroundSubtractorCNT(2, True)
bgsub = cv2.createBackgroundSubtractorMOG2(100, 40)

# show first frame and allow user to select ROI
#_, f = cap.read()
#f = cv2.flip(f, 0)
#r = cv2.selectROI("Frame", f, False)

queue = deque(maxlen=200)


def plot_data_on_canvas(data):
    # Parameters for the canvas
    canvas_width = 1280
    canvas_height = 400
    margin = 50

    # Create a blank canvas
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

    x_coords = np.linspace(margin, canvas_width - margin, len(data)).astype(int)
    # y_coords = (canvas_height - margin) - (np.array(data) * (canvas_height - 2 * margin)).astype(int)

    min = np.min(data)
    max = np.max(data)
    delta = max - min
    y_coords = (canvas_height - margin) - (((np.array(data) - min) / delta) * (canvas_height - 2 * margin)).astype(int)

    for i in range(1, len(data)):
        cv2.line(canvas, (x_coords[i - 1], y_coords[i - 1]), (x_coords[i], y_coords[i]), (0, 255, 0), 2)
    cv2.line(canvas, (margin, margin), (margin, canvas_height - margin), (0, 0, 0), 1)
    cv2.line(canvas, (margin, canvas_height - margin), (canvas_width - margin, canvas_height - margin), (0, 0, 0), 1)
    return canvas

hot = cv2.imread('hot.png')
hot = cv2.cvtColor(hot, cv2.COLOR_BGR2GRAY)
cold = cv2.imread('cold.png')
cold = cv2.cvtColor(cold, cv2.COLOR_BGR2GRAY)


check_line_y = 170

while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:

        # flip frame
        frame = cv2.flip(frame, 0)

        # crop frame
        # frame = frame[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]

        # encode frame as grayscale
        frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        result_hot = cv2.matchTemplate(frameGray, hot, cv2.TM_CCOEFF_NORMED)
        result_cold = cv2.matchTemplate(frameGray, cold, cv2.TM_CCOEFF_NORMED)

        debug = cv2.cvtColor(frameGray, cv2.COLOR_GRAY2BGR)

        hot_w, hot_h = hot.shape[::-1]
        (hot_ys, hot_xs) = np.where(result_hot >= 0.7)
        hot_rects = np.array([[x, y, x + hot_w, y + hot_h] for (x, y) in zip(hot_xs, hot_ys)])
        hot_rects = non_max_suppression(hot_rects, probs=None, overlapThresh=0.8)
        hot_mask = np.zeros(frameGray.shape, np.uint8)
        for (x, y, x2, h2) in hot_rects:
            cv2.rectangle(hot_mask, (x, y), (x2, h2), (255, 255, 255), -1)

        cold_w, cold_h = cold.shape[::-1]
        (cold_ys, cold_xs) = np.where(result_cold >= 0.7)
        cold_rects = []
        for (x, y) in zip(cold_xs, cold_ys):
            rect = ((int(x + cold_w // 2), int(y + cold_h // 2)), (cold_w, cold_h), 0)
            if len(cold_rects) == 0:
                cold_rects.append(tuple(rect))
                continue

            merged = False
            for other_rect in cold_rects:
                # test intersection
                overlap_status, _ = cv2.rotatedRectangleIntersection(rect, other_rect)
                if overlap_status == cv2.INTERSECT_FULL or overlap_status == cv2.INTERSECT_PARTIAL:
                    # merge rectangles
                    cold_rects.remove(other_rect)
                    rect = cv2.minAreaRect(np.concatenate((cv2.boxPoints(rect), cv2.boxPoints(other_rect))))
                    cold_rects.append(tuple(rect))
                    merged = True
                    break

            if not merged:
                cold_rects.append(tuple(rect))
        cold_rects = np.array([[x, y, x + cold_w, y + cold_h] for (x, y) in zip(cold_xs, cold_ys)])
        cold_rects = non_max_suppression(cold_rects, probs=None, overlapThresh=0.8)
        cold_mask = np.zeros(frameGray.shape, np.uint8)
        for (x, y, x2, h2) in cold_rects:
            cv2.rectangle(cold_mask, (x, y), (x2, h2), (255, 255, 255), -1)



        for (x, y, x2, h2) in hot_rects:
            cv2.rectangle(debug, (x, y), (x2, h2), (0, 0, 255), 2)

        for (x, y, x2, h2) in cold_rects:
            cv2.rectangle(debug, (x, y), (x2, h2), (255, 0, 0), 2)

        # draw line
        cv2.line(debug, (0, check_line_y), (debug.shape[1], check_line_y), (0, 255, 0), 1)



        cv2.imshow('Frame', debug)
        cv2.imshow('Hot Mask', hot_mask)
        cv2.imshow('Cold Mask', cold_mask)

        # Press q on keyboard to  exit
        if cv2.waitKey(40) & 0xFF == ord('q'):
            break
    # Break the loop
    else:
        break

cap.release()
