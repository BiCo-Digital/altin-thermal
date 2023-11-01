import os
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
cap = cv2.VideoCapture('output10.mkv')
# skip to 1000th frame
#cap.set(cv2.CAP_PROP_POS_FRAMES, 100*25)



check_line_y = 128


# Object Counter, counts hot, cold and their ends (hot_end, cold_end)
class ObjectCounter:
    def __init__(self):

        self.cleared = False
        self.hot_last = False
        self.cold_last = False

        self.hot_mask = None
        self.cold_mask = None

        self.not_changed_frame_count = 0

        self.queue = deque(maxlen=3)

    def process(self, hot_mask, cold_mask):

        # check if at y=170 there is a hot or cold object determined by the mask
        hot = hot_mask[check_line_y, :]
        cold = cold_mask[check_line_y, :]
        hot = np.any(hot)
        cold = np.any(cold)



        # check if last hot_mask or cold_mask was the same as the current one
        if np.array_equal(self.hot_mask, hot_mask) and np.array_equal(self.cold_mask, cold_mask):
            self.not_changed_frame_count += 1
        else:
            self.not_changed_frame_count = 0

        # if the last 5 frames were the same, clear the queue
        if self.not_changed_frame_count > 5:
            self.queue.clear()
            print('✋✋✋✋✋✋✋✋')


        # save current hot_mask and cold_mask
        self.hot_mask = hot_mask
        self.cold_mask = cold_mask


        if (hot and self.hot_last) or (cold and self.cold_last):
            self.print()
            return

        if hot and not self.hot_last:
            self.hot_last = True
            self.queue.append(1)
            self.cold_last = False

        if cold and not self.cold_last:
            self.cold_last = True
            self.queue.append(0)
            self.hot_last = False

        if not hot and not cold:
            self.hot_last = False
            self.cold_last = False



        # remove from queue end sequences of 1, 0, 1
        if len(self.queue) >= 303:
            if self.queue[-1] == 1 and self.queue[-2] == 0 and self.queue[-3] == 1:
                self.queue.pop()
                self.queue.pop()
                self.queue.pop()

        if len(self.queue) >= 3:
            if self.queue[-1] == 1 and self.queue[-2] == 1 and self.queue[-3] == 1:
                self.fire_alarm()



        self.print()

    def fire_alarm(self):
        print('‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️')

    def print(self):
        # print queue as emojis
        print(''.join(['🔴' if x == 1 else '🔵' for x in self.queue]))



def find_query_in_frame(query, frame, sensitivity, overlapTresh):
    result = cv2.matchTemplate(frame, query, cv2.TM_CCOEFF_NORMED)
    w, h = query.shape[::-1]
    (ys, xs) = np.where(result > sensitivity)
    rects = np.array([[x, y, x + w, y + h] for (x, y) in zip(xs, ys)])
    #rects = non_max_suppression(rects, probs=None, overlapThresh=overlapTresh)
    mask = np.zeros(frame.shape, np.uint8)
    for (x, y, x2, h2) in rects:
        cv2.rectangle(mask, (x, y), (x2, h2), (255, 255, 255), -1)

    mask = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=1)


    return mask



object_counter = ObjectCounter()
hot = cv2.imread('hot.png', cv2.IMREAD_GRAYSCALE)
hot_15_deg = cv2.imread('hot_15_deg.png', cv2.IMREAD_GRAYSCALE)
hot_350_deg = cv2.imread('hot_350_deg.png', cv2.IMREAD_GRAYSCALE)

cold = cv2.imread('cold.png', cv2.IMREAD_GRAYSCALE)
cold_small = cv2.imread('cold_small.png', cv2.IMREAD_GRAYSCALE)



while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:

        # flip frame
        frame = cv2.flip(frame, 0)

        #rotate frame 90 degrees
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # crop frame
        # frame = frame[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]

        # resize to 192x256
        # frame = cv2.resize(frame, (192, 256), interpolation=cv2.INTER_AREA)

        # encode frame as grayscale
        frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


        hot_mask = find_query_in_frame(hot, frameGray, 0.7, 0.8)
        hot_15_deg_mask = find_query_in_frame(hot_15_deg, frameGray, 0.7, 0.8)
        hot_350_deg_mask = find_query_in_frame(hot_350_deg, frameGray, 0.7, 0.8)
        hot_mask = cv2.bitwise_or(hot_mask, hot_15_deg_mask)
        hot_mask = cv2.bitwise_or(hot_mask, hot_350_deg_mask)


        cold_mask = find_query_in_frame(cold, frameGray, 0.7, 0.3)
        cold_small_mask = find_query_in_frame(cold_small, frameGray, 0.7, 0.3)
        cold_mask = cv2.bitwise_or(cold_mask, cold_small_mask)

        # TODO: DEBUG
        debug = frame.copy()
        if True:
            #border only hot mask contours
            hot_canny = cv2.Canny(hot_mask, 100, 200)
            hot_contours, _ = cv2.findContours(hot_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(debug, hot_contours, -1, (0, 0, 255), 2)

            cold_canny = cv2.Canny(cold_mask, 100, 200)
            cold_contours, _ = cv2.findContours(cold_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(debug, cold_contours, -1, (255, 0, 0), 2)

            # TODO: DEBUG
            cv2.line(debug, (0, check_line_y), (debug.shape[1], check_line_y), (0, 255, 0), 1)

        object_counter.process(hot_mask, cold_mask)

        cv2.imshow('Frame', debug)
        cv2.imshow('Hot Mask', hot_mask)
        cv2.imshow('Cold Mask', cold_mask)

        # print number of frame
        print(cap.get(cv2.CAP_PROP_POS_FRAMES) // 25)

        cv2.imwrite('frame.png', frame)

        # press space to pause
        if cv2.waitKey(15) & 0xFF == ord(' '):
            while True:
                if cv2.waitKey(10) & 0xFF == ord(' '):
                    break

        # press q to quit
        if cv2.waitKey(15) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break

cap.release()
