import os
import pickle
from collections import deque

import itertools
import cv2
import numpy as np
import skimage
from imutils.object_detection import non_max_suppression
from skimage.exposure import exposure
from skimage.feature import local_binary_pattern, hog
from skimage.filters import threshold_multiotsu
from sklearn import svm
from sklearn.preprocessing import StandardScaler

# create named window and move it - 400px in x dir
cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
# show window
cv2.imshow('Frame', np.zeros((256, 192, 3), np.uint8))
cv2.moveWindow("Frame", 20, 0)

# resize window
cv2.resizeWindow("Frame", 800, 600)

# Capture the frame from the video source while 1
cap = cv2.VideoCapture('output8.mp4')
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
        cv2.waitKey(0)

    def print(self):
        # print queue as emojis
        print(''.join(['🔴' if x == 1 else '🔵' for x in self.queue]))


def _find_query_in_frame(self, query, frame, sensitivity, footprint):

    """
            Finds the query image in the frame image using template matching.

            Parameters:
            - query: the query image (grayscale or color)
            - frame: the frame image (grayscale or color)
            - sensitivity: threshold for the match (higher values mean more sensitive)
            - footprint: tuple defining the neighborhood size for finding local minima

            Returns:
            - List of tuples representing the potential matches. Each tuple contains:
              - x: x-coordinate of the top-left corner of the bounding box
              - y: y-coordinate of the top-left corner of the bounding box
              - w: width of the bounding box
              - h: height of the bounding box
            """
    # Ensure the images are in grayscale
    if len(query.shape) == 3:
        query = cv2.cvtColor(query, cv2.COLOR_BGR2GRAY)
    if len(frame.shape) == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Get the size of the template
    w, h = query.shape[::-1]

    # Apply template matching using TM_SQDIFF_NORMED method
    result = cv2.matchTemplate(frame, query, cv2.TM_SQDIFF_NORMED)

    # Find local minima in the match result
    local_minima = (result == cv2.erode(result, np.ones(footprint)))

    # List to store possible positions
    positions = []

    # Add the possible positions to the list
    for y in range(result.shape[0]):
        for x in range(result.shape[1]):
            if local_minima[y, x] and result[y, x] < 1 - sensitivity:
                positions.append((x, y, w, h))

    mask = np.zeros(frame.shape, np.uint8)
    for (x, y, w, h) in positions:
        cv2.rectangle(mask, (x, y), (x + w, y + h), (255, 255, 255), -1)

    mask = cv2.dilate(mask, np.ones((3, 8), np.uint8), iterations=1)
    # convex hull
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        #hull = cv2.convexHull(c)
        #cv2.drawContours(mask, [hull], 0, (255, 255, 255), -1)
        cv2.drawContours(mask, [c], 0, (255, 255, 255), -1)


    return mask


LEFT_GUIDE_X = 50
RIGHT_GUIDE_X = 140
BOTTOM_GUIDE_Y = 220


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        # if contains debug, skip
        if 'debug' in filename:
            continue
        img = cv2.imread(os.path.join(folder,filename), cv2.IMREAD_COLOR)
        if img is not None:
            images.append(img)
    return images

images_in_folder = load_images_from_folder('fire/fireeee')
index = 0

cold_areas = []


HOT_TRESHOLD = 200
HOT_AREA_TRESHOLD = 120
COLD_TRESHOLD = 190
COLD_AREA_TRESHOLD = 150
COLD_AREA_MAX_TRESHOLD = 1500

def classify_using_filters(frame):
    # test gray
    if len(frame.shape) == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # crop to guides
    original_shape = frame.shape
    frame = frame[:, LEFT_GUIDE_X:RIGHT_GUIDE_X]
    frame = frame[:BOTTOM_GUIDE_Y, :]


    hot_mask = cv2.bilateralFilter(frame, 15, 30, 30)
    hot_mask[hot_mask < HOT_TRESHOLD] = 0
    hot_mask[hot_mask >= HOT_TRESHOLD] = 255

    hot_mask = cv2.erode(hot_mask, np.ones((1, 7), np.uint8), iterations=1)
    hot_mask = cv2.dilate(hot_mask, np.ones((1, 7), np.uint8), iterations=1)
    # join these two functions erosion and dilation into one single function called opening
    #hot_mask = cv2.morphologyEx(hot_mask, cv2.MORPH_OPEN, np.ones((1, 7), np.uint8))

    contours, _ = cv2.findContours(hot_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hot_mask = np.zeros(original_shape, np.uint8)
    for c in contours:
        hull = cv2.convexHull(c)
        area = cv2.contourArea(hull)
        if area > HOT_AREA_TRESHOLD:
            hull[:, :, 0] += LEFT_GUIDE_X
            cv2.drawContours(hot_mask, [hull], 0, (255, 255, 255), -1)



    # Apply linear contrast stretching
    #frame = cv2.convertScaleAbs(frame, alpha=-2.1, beta=50)
    #frame = np.array(255 * (frame / 255) ** 20, dtype='uint8')
    #frameRGB = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

    #do local histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    frame = clahe.apply(frame)

    #frame = skimage.exposure.equalize_adapthist(frame, clip_limit=0.03)


    cv2.imshow('cold_contrast', frame.copy())

    cold_mask = cv2.bilateralFilter(frame, 15, 30, 30)
    cold_mask = cv2.medianBlur(cold_mask, 15)
    #mean = cold_mask.copy()
    cold_mask = cv2.bitwise_not(cold_mask)
    cv2.imshow('cold_mask_bilateral', cold_mask.copy())


    cold_mask[cold_mask < COLD_TRESHOLD] = 0
    cold_mask[cold_mask >= COLD_TRESHOLD] = 255

    cold_mask = cv2.erode(cold_mask, np.ones((3, 3), np.uint8), iterations=1)
    cold_mask = cv2.dilate(cold_mask, np.ones((3, 3), np.uint8), iterations=1)

    cv2.imshow('cold_mask_threshold', cold_mask.copy())

    canny = cv2.Canny(cold_mask, 100, 200)
    cv2.imshow('cold_mask_canny', canny.copy())


    contours, _ = cv2.findContours(cold_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cold_mask = np.zeros(original_shape, np.uint8)

    for c in contours:
        approx = cv2.approxPolyDP(c, 0.038 * cv2.arcLength(c, True), True)
        hull = cv2.convexHull(approx)
        x,y,w,h = cv2.boundingRect(c)
        area = w * h
        if COLD_AREA_TRESHOLD < area < COLD_AREA_MAX_TRESHOLD:
            # shift bbox
            #cv2.drawContours(cold_mask, [bbox], 0, (255, 255, 255), -1)
            #draw rectangle with left offset
            cv2.rectangle(cold_mask, (x + LEFT_GUIDE_X, y), (x + w + LEFT_GUIDE_X, y + h), (255, 255, 255), -1)

            cold_areas.append(area)



    return hot_mask, cold_mask

def classify_using_filters_with_svm(frame):
    hot_mask, cold_mask = classify_using_filters(frame)

    # load svm for hot
    hot_svm = None
    with open('svm_model_hot.pkl', 'rb') as f:
        hot_svm = pickle.load(f)

    # load svm for cold
    cold_svm = None
    with open('svm_model_cold.pkl', 'rb') as f:
        cold_svm = pickle.load(f)

    # get contours for hot_mask
    contours_hot, _ = cv2.findContours(hot_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hot_mask = np.zeros(frame.shape, np.uint8)
    for c in contours_hot:
        rect = cv2.minAreaRect(c)
        box = np.int0(cv2.boxPoints(rect))
        x = box[0][0]

        w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
        if h > w:
            w, h = h, w
        # predict with svm
        if hot_svm is not None:
            prediction = hot_svm.predict([[w, h]])
            if prediction == 1:
                cv2.drawContours(hot_mask, [c], 0, (255, 255, 255), -1)

    return hot_mask, cold_mask

# create deque of 3 elements
hot_samples = deque(maxlen=10000)

object_counter = ObjectCounter()
while (cap.isOpened()):
    ret, frame = cap.read()

    #cv2.waitKey(0)
    #frame = images_in_folder[index]
    index += 1
    if ret == True:

        # flip frame
        # frame = cv2.flip(frame, 0)

        #rotate frame 90 degrees

        #frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # crop frame
        # frame = frame[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]

        # resize to 192x256
        # frame = cv2.resize(frame, (192, 256), interpolation=cv2.INTER_AREA)



        # crop to guides
        original_frame = frame.copy()
        #frame = frame[:, LEFT_GUIDE_X:RIGHT_GUIDE_X]
        #frame = frame[:BOTTOM_GUIDE_Y, :]

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)




        frameRGB = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        hot_mask, cold_mask = classify_using_filters_with_svm(frame)

        object_counter.process(hot_mask, cold_mask)


        frameRGB[hot_mask == 255] = (0, 0, 255)
        frameRGB[cold_mask == 255] = (255, 0, 0)
        frameRGB[check_line_y, :] = (0, 255, 0)


        cv2.imshow('Frame', frameRGB)
        cv2.imshow('hot_mask', hot_mask)
        cv2.imshow('cold_mask', cold_mask)


        # print number of frame
        print(cap.get(cv2.CAP_PROP_POS_FRAMES) // 25 , cap.get(cv2.CAP_PROP_FRAME_COUNT) // 25)


        # press space to pause

        if cv2.waitKey(15) & 0xFF == ord(' '):
            while True:
                if cv2.waitKey(5) & 0xFF == ord(' '):
                    break
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break

    # Break the loop
    else:
        break




# calculate cold_areas statistics
cold_areas = np.array(cold_areas)

print('cold_areas', cold_areas)
print('cold_areas mean', np.mean(cold_areas))
print('cold_areas std', np.std(cold_areas))
print('cold_areas min', np.min(cold_areas))
print('cold_areas max', np.max(cold_areas))
print('cold_areas median', np.median(cold_areas))
print('cold_areas 25th percentile', np.percentile(cold_areas, 25))
print('cold_areas 75th percentile', np.percentile(cold_areas, 75))
print('cold_areas 90th percentile', np.percentile(cold_areas, 90))
print('cold_areas 95th percentile', np.percentile(cold_areas, 95))
print('cold_areas 99th percentile', np.percentile(cold_areas, 99))
print('cold_areas 99.9th percentile', np.percentile(cold_areas, 99.9))

import matplotlib.pyplot as plt
plt.hist(cold_areas, bins=100)
plt.show()

