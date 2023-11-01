import os
from collections import deque

import itertools
import cv2
import numpy as np
from imutils import skeletonize
from imutils.object_detection import non_max_suppression

# load frame.png, hot.png and using template matching find hot object, show it
frame = cv2.imread('frame.png', cv2.IMREAD_GRAYSCALE)
hot = cv2.imread('hot.png', cv2.IMREAD_GRAYSCALE)

w, h = hot.shape[::-1]
result = cv2.matchTemplate(frame, hot, cv2.TM_CCOEFF_NORMED)
threshold = 0.7
(ys, xs) = np.where(result > threshold)
rects = np.array([[x, y, x + w, y + h] for (x, y) in zip(xs, ys)])
rects = non_max_suppression(rects, probs=None, overlapThresh=0.3)
mask = np.zeros(frame.shape, np.uint8)
for (x, y, x2, h2) in rects:
    cv2.rectangle(mask, (x, y), (x2, h2), (255, 255, 255), -1)

#mask = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=1)

# find contours
contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# draw on frame
for c in contours:
    cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)

# show frame
cv2.imshow('frame', frame)


cv2.waitKey(0)
