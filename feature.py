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
cv2.moveWindow("Frame", 512, 0)

# resize window
cv2.resizeWindow("Frame", 384, 512)

# Capture the frame from the video source while 1
cap = cv2.VideoCapture('thermal.mp4')

# skip to frame 100*25
#cap.set(cv2.CAP_PROP_POS_FRAMES, 100 * 25)
#cap.set(cv2.CAP_PROP_POS_FRAMES, 450 * 25)

# bgsub = cv2.bgsegm.createBackgroundSubtractorLSBP()
# bgsub = cv2.bgsegm.createBackgroundSubtractorGSOC()
# bgsub = cv2.bgsegm.createBackgroundSubtractorGMG(20, 0.7)
bgsub = cv2.bgsegm.createBackgroundSubtractorCNT(2, True)
bgsub = cv2.createBackgroundSubtractorMOG2(100, 40)

# show first frame and allow user to select ROI
# _, f = cap.read()
# f = cv2.flip(f, 0)
# r = cv2.selectROI("Frame", f, False)

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


hot = cv2.imread('hot_clean.png')
hot = cv2.cvtColor(hot, cv2.COLOR_BGR2GRAY)
cold = cv2.imread('cold.png')
cold = cv2.cvtColor(cold, cv2.COLOR_BGR2GRAY)

check_line_y = 140


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
                print('‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️')



        self.print()

    def print(self):
        # print queue as emojis
        print(''.join(['🔴' if x == 1 else '🔵' for x in self.queue]))

object_counter = ObjectCounter()

while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:

        # flip frame
        # frame = cv2.flip(frame, 0)

        # crop frame
        # frame = frame[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]

        # resize to 192x256
        frame = cv2.resize(frame, (192, 256), interpolation=cv2.INTER_AREA)

        # encode frame as grayscale
        frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        result_hot = cv2.matchTemplate(frameGray, hot, cv2.TM_CCOEFF_NORMED)
        # get the best match position and treshold value
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result_hot)

        best_treshold = min_val

        result_cold = cv2.matchTemplate(frameGray, cold, cv2.TM_CCOEFF_NORMED)

        debug = cv2.cvtColor(frameGray, cv2.COLOR_GRAY2BGR)

        hot_w, hot_h = hot.shape[::-1]
        (hot_ys, hot_xs) = np.where(result_hot > 0.6)
        hot_rects = np.array([[x, y, x + hot_w, y + hot_h] for (x, y) in zip(hot_xs, hot_ys)])
        hot_rects = non_max_suppression(hot_rects, probs=None, overlapThresh=0.8)
        hot_mask = np.zeros(frameGray.shape, np.uint8)
        for (x, y, x2, h2) in hot_rects:
            cv2.rectangle(hot_mask, (x, y), (x2, h2), (255, 255, 255), -1)

        cold_w, cold_h = cold.shape[::-1]
        (cold_ys, cold_xs) = np.where(result_cold >= 0.7)
        cold_rects = np.array([[x, y, x + cold_w, y + cold_h] for (x, y) in zip(cold_xs, cold_ys)])
        cold_rects = non_max_suppression(cold_rects, probs=None, overlapThresh=0.8)
        cold_mask = np.zeros(frameGray.shape, np.uint8)
        for (x, y, x2, h2) in cold_rects:
            cv2.rectangle(cold_mask, (x, y), (x2, h2), (255, 255, 255), -1)

        # TODO: DEBUG
        if True:
            for (x, y, x2, h2) in hot_rects:
                cv2.rectangle(debug, (x, y), (x2, h2), (0, 0, 255), 2)

            for (x, y, x2, h2) in cold_rects:
                cv2.rectangle(debug, (x, y), (x2, h2), (255, 0, 0), 2)

            # TODO: DEBUG
            cv2.line(debug, (0, check_line_y), (debug.shape[1], check_line_y), (0, 255, 0), 1)

        object_counter.process(hot_mask, cold_mask)

        cv2.imshow('Frame', debug)
        cv2.imshow('Hot Mask', hot_mask)
        cv2.imshow('Cold Mask', cold_mask)

        # print number of frame
        print(cap.get(cv2.CAP_PROP_POS_FRAMES) // 25)


        # press space to pause
        if cv2.waitKey(40) & 0xFF == ord(' '):
            while True:
                if cv2.waitKey(10) & 0xFF == ord(' '):
                    break
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    # exit program if q is pressed
                    os._exit(0)

    # Break the loop
    else:
        break

cap.release()
