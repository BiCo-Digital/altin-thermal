import time
import uuid
from datetime import datetime

import cv2
import numpy as np
from gpiozero import OutputDevice, CPUTemperature
import threading
import platform
from skimage.filters import threshold_multiotsu
from collections import deque
from imutils.object_detection import non_max_suppression
import os
import pickle
from sklearn import svm

def is_mac():
    return platform.system() == "Darwin"


def use_video_file():
    return is_mac()


DIR_PATH = os.path.dirname(os.path.realpath(__file__))

WIDTH = 320
HEIGHT = 480
RELAY_GPIO = 26

P2Pro_resolution = (256, 384)
CAM_WIDTH = 256
CAM_HEIGHT = 384

VIDEO_HEIGHT = 256
VIDEO_WIDTH = 192

CHECK_LINE_Y = VIDEO_HEIGHT // 2

GUIDE_DISTANCE = 36
LEFT_GUIDE_X = 55
RIGHT_GUIDE_X = 140
BOTTOM_GUIDE_Y = 220


# create enum for LIVE, TRESHOLD, OVERLAY, DEBUG, SETTINGS
class Mode:
    LIVE = 0
    TRESHOLD = 1
    OVERLAY = 2
    DEBUG = 3
    SETTINGS = 4


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

    # apply bilateral filter
    frame = cv2.bilateralFilter(frame, 15, 30, 30)

    hot_frame = frame.copy()
    #hot_frame = cv2.bilateralFilter(hot_frame, 15, 30, 30)
    hot_frame[hot_frame < HOT_TRESHOLD] = 0
    hot_frame[hot_frame >= HOT_TRESHOLD] = 255

    hot_frame = cv2.morphologyEx(hot_frame, cv2.MORPH_OPEN, np.ones((1, 7), np.uint8))

    hot_contours, _ = cv2.findContours(hot_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hot_mask = np.zeros(original_shape, np.uint8)
    for c in hot_contours:
        hull = cv2.convexHull(c)
        area = cv2.contourArea(hull)
        if area > HOT_AREA_TRESHOLD:
            hull[:, :, 0] += LEFT_GUIDE_X
            cv2.drawContours(hot_mask, [hull], 0, (255, 255, 255), -1)


    # Apply linear contrast stretching
    # frame = cv2.convertScaleAbs(frame, alpha=-2.1, beta=50)
    # frame = np.array(255 * (frame / 255) ** 20, dtype='uint8')
    # frameRGB = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

    # do local histogram equalization
    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    #frame = clahe.apply(frame)

    # frame = skimage.exposure.equalize_adapthist(frame, clip_limit=0.03)


    cold_frame = frame.copy()
    cold_frame = cv2.convertScaleAbs(cold_frame, alpha=-2.1, beta=50)
    #cold_frame = cv2.bilateralFilter(cold_frame, 15, 30, 30)
    cold_frame = cv2.medianBlur(cold_frame, 15)
    cold_frame = cv2.bitwise_not(cold_frame)

    cold_frame[cold_frame < COLD_TRESHOLD] = 0
    cold_frame[cold_frame >= COLD_TRESHOLD] = 255

    cold_frame = cv2.morphologyEx(cold_frame, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


    cold_contours, _ = cv2.findContours(cold_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cold_mask = np.zeros(original_shape, np.uint8)
    for c in cold_contours:
        #approx = cv2.approxPolyDP(c, 0.038 * cv2.arcLength(c, True), True)
        #hull = cv2.convexHull(approx)
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if COLD_AREA_TRESHOLD < area < COLD_AREA_MAX_TRESHOLD:
            cv2.rectangle(cold_mask, (x + LEFT_GUIDE_X, y), (x + w + LEFT_GUIDE_X, y + h), (255, 255, 255), -1)

    return hot_mask, cold_mask


# load svm for hot
hot_svm = None
with open(DIR_PATH + '/svm_model_hot.pkl', 'rb') as f:
    hot_svm = pickle.load(f)

def classify_using_filters_with_svm(frame):
    hot_mask, cold_mask = classify_using_filters(frame)

    # get contours for hot_mask
    contours_hot, _ = cv2.findContours(hot_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hot_mask = np.zeros(frame.shape, np.uint8)
    for c in contours_hot:
        rect = cv2.minAreaRect(c)
        box = np.intp(cv2.boxPoints(rect))
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


class ObjectCounter:
    def __init__(self):

        self.cleared = False
        self.hot_last = False
        self.cold_last = False

        self.hot_mask = None
        self.cold_mask = None

        self.not_changed_frame_count = 0

        self.is_frame_ok = True

        self.queue = deque(maxlen=3)
        self.hot_image = cv2.imread(DIR_PATH + '/hot.png', cv2.IMREAD_GRAYSCALE)
        self.hot_15_deg_image = cv2.imread(DIR_PATH + '/hot_15_deg.png', cv2.IMREAD_GRAYSCALE)
        self.hot_350_deg_image = cv2.imread(DIR_PATH + '/hot_350_deg.png', cv2.IMREAD_GRAYSCALE)

        self.cold_image = cv2.imread(DIR_PATH + '/cold.png', cv2.IMREAD_GRAYSCALE)
        self.cold_small_image = cv2.imread(DIR_PATH + '/cold_small.png', cv2.IMREAD_GRAYSCALE)

    def _find_query_in_frame_back(self, query, frame, sensitivity, overlapTresh):
        result = cv2.matchTemplate(frame, query, cv2.TM_CCOEFF_NORMED)
        w, h = query.shape[::-1]
        (ys, xs) = np.where(result > sensitivity)
        rects = np.array([[x, y, x + w, y + h] for (x, y) in zip(xs, ys)])
        # rects = non_max_suppression(rects, probs=None, overlapThresh=overlapTresh)
        mask = np.zeros(frame.shape, np.uint8)
        for (x, y, x2, h2) in rects:
            cv2.rectangle(mask, (x, y), (x2, h2), (255, 255, 255), -1)

        # mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)

        return mask

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
            hull = cv2.convexHull(c)
            cv2.drawContours(mask, [hull], 0, (255, 255, 255), -1)

        return mask

    def _create_mask(self, region, morph_op=None, kernel=None):
        mask = np.zeros_like(region, dtype=np.uint8)
        mask[region] = 255
        if morph_op:
            mask = cv2.morphologyEx(mask, morph_op, kernel)
        return mask

    def _create_threshold_masks(self, gray, thresholds):
        region_cold = gray <= thresholds[0]
        region_mid = (gray > thresholds[0]) & (gray <= thresholds[1])
        region_hot = gray > thresholds[1]
        mask_cold = self._create_mask(region_cold, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        mask_mid = self._create_mask(region_mid)
        mask_hot = self._create_mask(region_hot, cv2.MORPH_OPEN, np.ones((3, 11), np.uint8))
        return mask_cold, mask_mid, mask_hot

    def classify(self, frame):

        # assure that frame is 192x256 and grayscale
        # frame = cv2.resize(frame, (192, 256))
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        hot_mask, cold_mask = classify_using_filters_with_svm(frame)

        if np.all(hot_mask == 0):
            self.hot_mask = None
            return

        mid_mask = cv2.bitwise_not(cv2.bitwise_or(hot_mask, cold_mask))

        # if hot or cold mask is empty, return
        if np.all(hot_mask == 0) and np.all(cold_mask == 0):
            return

        # check if at y=170 there is a hot or cold object determined by the mask
        hot = hot_mask[CHECK_LINE_Y, :]
        cold = cold_mask[CHECK_LINE_Y, :]
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

        self.is_frame_ok = True

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
                self.is_frame_ok = False

        self.print()

    def is_scene_ok(self):
        return self.is_frame_ok

    def get_debug_image(self, frame, width=192, height=256):
        if self.hot_mask is None or self.cold_mask is None:
            return frame if frame is not None else np.zeros((height, width, 3), np.uint8)

        debug = frame if frame is not None else np.zeros((height, width, 3), np.uint8)

        d_h = self.hot_mask
        d_c = self.cold_mask
        if width != 192 or height != 256:
            d_h = cv2.resize(self.hot_mask, (width, height))
            d_c = cv2.resize(self.cold_mask, (width, height))

        # get contours from masks
        contours_hot, _ = cv2.findContours(d_h, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_cold, _ = cv2.findContours(d_c, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)




        # draw contours
        cv2.drawContours(debug, contours_hot, -1, (0, 0, 255), -1)
        cv2.drawContours(debug, contours_cold, -1, (255, 0, 0), 2)
        return debug

    def print(self):
        # print queue as emojis
        if self.is_scene_ok():
            print(''.join(['🔴' if x == 1 else '🔵' for x in self.queue]))
        else:
            print(''.join(['🔴' if x == 1 else '🔵' for x in self.queue]), '💨💨💨💨💨💨💨')


class App:
    def __init__(self, window_title, video_source='./output8_crop2.mp4'):
        self.cpu = CPUTemperature() if not is_mac() else None
        self.window_title = window_title
        self.video_source = video_source
        self.init_video_source()
        self.init_window()
        self.show_settings = False
        self.display_mode = Mode.DEBUG
        self.trigger_delay = 0.1
        self.trigger_duration = 0.5
        self.soup_classifier = ObjectCounter()
        if not is_mac():
            self.relay = OutputDevice(RELAY_GPIO, active_high=False)
            self.relay.off()
        self.isAlreadyFiring = False
        self.firingMessage = 'CEKAM...'
        self.runLoop()

    def init_video_source(self):
        if use_video_file():
            self.vid = cv2.VideoCapture(self.video_source)
        else:
            self.vid = cv2.VideoCapture(0)
            self.vid.set(cv2.CAP_PROP_CONVERT_RGB, 0)

    def init_window(self):
        cv2.namedWindow(self.window_title, cv2.WINDOW_NORMAL)
        if is_mac():
            cv2.resizeWindow(self.window_title, WIDTH, HEIGHT)
            cv2.imshow(self.window_title, np.zeros((HEIGHT, WIDTH, 3), np.uint8))
            cv2.moveWindow(self.window_title, -500, 0)
        else:
            cv2.setWindowProperty(self.window_title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.moveWindow(self.window_title, -1, -1)
        cv2.setMouseCallback(self.window_title, self.onMouseClick)

    def grab_frame(self):
        ret, frame = self.vid.read()

        if use_video_file():
            if self.vid.get(cv2.CAP_PROP_POS_FRAMES) == self.vid.get(cv2.CAP_PROP_FRAME_COUNT):
                self.vid.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # rotate 90
            # frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            frame = cv2.resize(frame, (192, 256))
            # if self.video_source contains output
            if self.video_source.startswith('./output'):
                frame = cv2.flip(frame, 0)
            return ret, frame
        else:
            frame_mid_pos = int(len(frame) / 2)
            picture_data = frame[0:frame_mid_pos]
            thermal_data = frame[frame_mid_pos:]
            yuv_picture = np.frombuffer(picture_data, dtype=np.uint8).reshape((CAM_HEIGHT // 2, CAM_WIDTH, 2))
            rgb_picture = cv2.cvtColor(yuv_picture, cv2.COLOR_YUV2RGB_YUY2)
            rgb_picture = cv2.rotate(rgb_picture, cv2.ROTATE_90_COUNTERCLOCKWISE)
            return ret, rgb_picture

            thermal_picture_16 = np.frombuffer(thermal_data, dtype=np.uint16).reshape((CAM_HEIGHT // 2, CAM_WIDTH))
            thermal_picture_8 = cv2.normalize(thermal_picture_16, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            thermal_picture_8 = cv2.rotate(thermal_picture_8, cv2.ROTATE_90_COUNTERCLOCKWISE)

    def runLoop(self):
        frame_rate = 25
        prev_t = 0

        while True:
            time_elapsed = time.time() - prev_t
            if time_elapsed < 1. / frame_rate:
                continue
            prev_t = time.time()

            if self.display_mode == Mode.SETTINGS:
                self.showSettings()
                continue

            ret, frame = self.grab_frame()
            if ret:
                self.soup_classifier.classify(frame)
                is_ok = self.soup_classifier.is_scene_ok()
                if not is_ok:
                    self.fireRelay(frame)

                debug_image = self.soup_classifier.get_debug_image(frame)
                debug_image = self.drawGuidelines(debug_image)


                cv2.putText(debug_image, str(round((time.time() - prev_t) * 1000)), (7, 23), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                cv2.putText(debug_image, "ms", (4, 43), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                if self.cpu is not None:
                    cv2.putText(debug_image, str(int(self.cpu.temperature)), (147, 23), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                    cv2.putText(debug_image, "C", (153, 45), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

                if self.isAlreadyFiring:
                    cv2.putText(debug_image, self.firingMessage, (10, debug_image.shape[0] - 10), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)

                cv2.imshow(self.window_title, debug_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def fireRelay(self, frame):
        if self.isAlreadyFiring:
            return

        # save frame to file when firing with timestamp
        str_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        cv2.imwrite(DIR_PATH + '/fire/fired_' + str_timestamp + '.jpg', frame)
        overlay = cv2.addWeighted(frame, 1, self.soup_classifier.get_debug_image(192, 256), 1, 0)
        cv2.imwrite(DIR_PATH + '/fire/fired_' + str_timestamp + '_debug.jpg', overlay)

        def fire_on_thread():
            self.firingMessage = 'CEKAM...'
            time.sleep(self.trigger_delay)
            self.firingMessage = 'FOUKAM'
            self.relay.on()
            time.sleep(self.trigger_duration)
            self.relay.off()
            self.isAlreadyFiring = False

        def fire_debug():
            self.firingMessage = 'CEKAM...'
            time.sleep(self.trigger_delay)
            self.firingMessage = 'FOUKAM'
            time.sleep(self.trigger_duration)
            self.isAlreadyFiring = False

        t = threading.Thread(target=fire_on_thread if not is_mac() else fire_debug)
        t.start()
        self.isAlreadyFiring = True

    def drawGuidelines2(self, frame):
        cv2.line(frame, (115, 10), (100, 396), (0, 255, 0), 2)
        cv2.line(frame, (235, 10), (250, 396), (0, 255, 0), 2)
        cv2.line(frame, (0, 395), (WIDTH, 395), (0, 255, 0), 2)
        return frame

    def drawGuidelines(self, frame, w=192, h = 256):
        # Constants
        x_center = w // 2
        d_top = 120 // 2
        d_bottom = 150 // 2

        # Draw lines sloped using distance from top and bottom
        # cv2.line(frame, (x_center - d_top, 0), (x_center - d_bottom, 396), (0, 255, 0), 2)
        # cv2.line(frame, (x_center + d_top, 0), (x_center + d_bottom, 396), (0, 255, 0), 2)

        # Draw line LEFT_GUIDE_X and RIGHT_GUIDE_X, keep in mind that GUIDES are calclulated for width 192, not w
        cv2.line(frame, (int(LEFT_GUIDE_X / 192 * w), 0), (int(LEFT_GUIDE_X / 192 * w), h), (0, 255, 0), 2)
        cv2.line(frame, (int(RIGHT_GUIDE_X / 192 * w), 0), (int(RIGHT_GUIDE_X / 192 * w), h), (0, 255, 0), 2)

        cv2.line(frame, (0, int(h * CHECK_LINE_Y / 256)), (w, int(h * CHECK_LINE_Y / 256)), (0, 255, 0), 2)

        return frame

    def onMouseClick(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print('Mouse clicked at: ', x, y)
            if self.display_mode == Mode.SETTINGS:
                third_width = WIDTH // 3
                third_height = HEIGHT // 3
                if self.didHitButton(0, 40, third_width, third_width, x, y):
                    self.trigger_delay -= 0.1
                    if self.trigger_delay < 0:
                        self.trigger_delay = 0
                if self.didHitButton(third_width * 2, 40, third_width, third_width, x, y):
                    self.trigger_delay += 0.1
                if self.didHitButton(0, 40 + third_width + 40, third_width, third_width, x, y):
                    self.trigger_duration -= 0.1
                    if self.trigger_duration < 0:
                        self.trigger_duration = 0
                if self.didHitButton(third_width * 2, 40 + third_width + 40, third_width, third_width, x, y):
                    self.trigger_duration += 0.1
                if self.didHitButton(0, 80 + third_width * 2 + 40, WIDTH, third_width, x, y):
                    self.display_mode = Mode.DEBUG
            else:
                self.display_mode = Mode.SETTINGS

        if event == cv2.EVENT_LBUTTONUP:
            print('Mouse released at: ', x, y)

    def drawButton(self, frame, x, y, width, height, text):
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 5)
        cv2.putText(frame, text, (x + width // 2 - 10, y + height // 2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    def didHitButton(self, x, y, width, height, mouse_x, mouse_y):
        return x <= mouse_x <= x + width and y <= mouse_y <= y + height

    def showSettings(self):
        frame = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
        third_width = WIDTH // 3
        cv2.putText(frame, 'ZPOZDENI', (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        self.drawButton(frame, 0, 40, third_width, third_width, '-')
        self.drawButton(frame, third_width, 40, third_width, third_width, str(round(self.trigger_delay, 1)))
        self.drawButton(frame, third_width * 2, 40, third_width, third_width, '+')
        cv2.putText(frame, 'TRVANI', (0, 40 + third_width + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        self.drawButton(frame, 0, 40 + third_width + 40, third_width, third_width, '-')
        self.drawButton(frame, third_width, 40 + third_width + 40, third_width, third_width, str(round(self.trigger_duration, 1)))
        self.drawButton(frame, third_width * 2, 40 + third_width + 40, third_width, third_width, '+')
        self.drawButton(frame, 0, 80 + third_width * 2 + 40, WIDTH, third_width, 'START')
        cv2.imshow(self.window_title, frame)
        cv2.waitKey(20)


app = App("Thermal Camera")
