import time
import uuid
from datetime import datetime

import cv2
import numpy as np
from gpiozero import OutputDevice
import threading
import platform
from skimage.filters import threshold_multiotsu
from collections import deque
from imutils.object_detection import non_max_suppression
import os

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

CHECK_LINE_Y = CAM_WIDTH // 2 + 30

GUIDE_DISTANCE = 36
LEFT_GUIDE_X = 55
RIGHT_GUIDE_X = 145

# create enum for LIVE, TRESHOLD, OVERLAY, DEBUG, SETTINGS
class Mode:
    LIVE = 0
    TRESHOLD = 1
    OVERLAY = 2
    DEBUG = 3
    SETTINGS = 4


class MovingAverage:
    """
    Computes the moving average and median for the last 'size' data points.
    """

    def __init__(self, size: int):
        self.size = size
        self.queue = deque(maxlen=size)
        self.sum = 0

    def next(self, val: int):
        if len(self.queue) == self.queue.maxlen:
            self.sum -= self.queue[0]

        self.queue.append(val)
        self.sum += val

    def average(self) -> int:
        if not self.queue:
            return None
        return int(self.sum / len(self.queue))

    def median(self) -> int:
        if not self.queue:
            return None
        return int(np.median(list(self.queue)))


class Box:
    def __init__(self, contour, cold=False):
        self.uuid = uuid.uuid4()
        self.contour = contour
        self.rect = cv2.minAreaRect(contour)
        self.center = self.rect[0]
        self.width = max(self.rect[1])
        self.height = min(self.rect[1])
        self.angle = self.rect[2]
        self.area = self.width * self.height
        self.aspect_ratio = self.width / self.height
        self.corners = np.int0(cv2.boxPoints(self.rect))
        self.cold = cold

    @staticmethod
    def create_centered_box(relative_height=0.1):
        frame_width = CAM_HEIGHT // 2
        frame_height = CAM_WIDTH
        tall = int(frame_height * relative_height)
        points = np.array([
            [0, frame_height // 2 + tall // 2],
            [0, frame_height // 2 - tall // 2],
            [frame_width, frame_height // 2 - tall // 2],
            [frame_width, frame_height // 2 + tall // 2]
        ])
        return Box(points)

    def clone_from(self, other_box):
        attrs = ['contour', 'rect', 'center', 'width', 'height', 'angle', 'area', 'aspect_ratio', 'corners']
        for attr in attrs:
            setattr(self, attr, getattr(other_box, attr))

    def is_overlapping(self, other_box):
        overlap_status, _ = cv2.rotatedRectangleIntersection(self.rect, other_box.rect)
        return overlap_status != 0

    def compute_distance_to(self, other_box):
        return np.linalg.norm(np.array(self.center) - np.array(other_box.center))


class SceneManager:
    def __init__(self):
        self.frame_count = 0
        self.cold_boxes = []
        self.avg_cold_box_area = MovingAverage(100)
        self.hot_boxes = []
        self.avg_hot_box_area = MovingAverage(100)
        self.checkpoint = Box.create_centered_box(0.1)
        self.safe_area = Box.create_centered_box(0.9)
        self.ordered_boxes = []

    def is_scene_ok(self):
        count_of_hot_boxes_in_a_row = 0
        for b in self.ordered_boxes:
            if b.cold:
                count_of_hot_boxes_in_a_row = 0
            else:
                count_of_hot_boxes_in_a_row += 1
                if count_of_hot_boxes_in_a_row >= 4:
                    return False
        return True

    def update_avg_box_area(self, boxes, avg_box_area):
        for box in boxes:
            avg_box_area.next(box.area)

    def filter_boxes(self, boxes, avg_box_area):
        # check if median is not None
        if avg_box_area.median() is None:
            return boxes

        lower_bound = 0.3 * avg_box_area.median()
        upper_bound = 3 * avg_box_area.median()
        return [box for box in boxes if lower_bound < box.area < upper_bound]

    def merge_and_sort_boxes(self):
        self.ordered_boxes = self.cold_boxes + self.hot_boxes
        self.ordered_boxes.sort(key=lambda box: -box.center[1])

    def update(self, new_cold_boxes, new_hot_boxes):
        self.frame_count += 1

        self.update_avg_box_area(new_cold_boxes, self.avg_cold_box_area)
        self.update_avg_box_area(new_hot_boxes, self.avg_hot_box_area)

        self.cold_boxes = self.filter_boxes(new_cold_boxes, self.avg_cold_box_area)
        self.hot_boxes = self.filter_boxes(new_hot_boxes, self.avg_hot_box_area)

        self.merge_and_sort_boxes()

        return self.is_scene_ok()

    def draw_scene(self):
        clean = np.zeros((256, 192, 3), np.uint8)

        cv2.fillPoly(clean, [self.safe_area.corners], (80, 0, 0))
        cv2.fillPoly(clean, [self.checkpoint.corners], (0, 255, 0) if self.is_scene_ok() else (0, 0, 255))

        for box in self.ordered_boxes:
            color = (255, 0, 0) if box.cold else (0, 0, 255)
            cv2.drawContours(clean, [box.corners], 0, color, 2)

        return clean


class SoupClassifier2:


    def __init__(self):
        self.scene = SceneManager()
        self.hot = cv2.imread('hot.png', cv2.IMREAD_GRAYSCALE)
        self.hot_15_deg = cv2.imread('hot_15_deg.png', cv2.IMREAD_GRAYSCALE)
        self.hot_350_deg = cv2.imread('hot_350_deg.png', cv2.IMREAD_GRAYSCALE)

        self.cold = cv2.imread('cold.png', cv2.IMREAD_GRAYSCALE)

    def _convert_to_grayscale(self, frame):
        if len(frame.shape) == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

    def _create_threshold_masks(self, gray, thresholds):
        region_cold = gray <= thresholds[0]
        region_mid = (gray > thresholds[0]) & (gray <= thresholds[1])
        region_hot = gray > thresholds[1]
        mask_cold = self._create_mask(region_cold, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        mask_mid = self._create_mask(region_mid)
        mask_hot = self._create_mask(region_hot, cv2.MORPH_OPEN, np.ones((3, 11), np.uint8))
        return mask_cold, mask_mid, mask_hot

    def _create_mask(self, region, morph_op=None, kernel=None):
        mask = np.zeros_like(region, dtype=np.uint8)
        mask[region] = 255
        if morph_op:
            mask = cv2.morphologyEx(mask, morph_op, kernel)
        return mask

    def _get_boxes_from_contours(self, contours, cold=False):
        boxes = []
        for cnt in contours:
            if cold:
                epsilon = 0.1 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                if cv2.isContourConvex(approx) and len(approx) == 4:
                    boxes.append(Box(cnt, cold=True))
            else:
                boxes.append(Box(cnt))
        return boxes

    def _find_query_in_frame(self, query, frame, sensitivity, overlapTresh):
        result = cv2.matchTemplate(frame, query, cv2.TM_CCOEFF_NORMED)
        w, h = query.shape[::-1]
        (ys, xs) = np.where(result > sensitivity)
        rects = np.array([[x, y, x + w, y + h] for (x, y) in zip(xs, ys)])
        rects = non_max_suppression(rects, probs=None, overlapThresh=overlapTresh)
        mask = np.zeros(frame.shape, np.uint8)
        for (x, y, x2, h2) in rects:
            cv2.rectangle(mask, (x, y), (x2, h2), (255, 255, 255), -1)

        #mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)

        return mask

    def classify(self, frame):
        #frame = cv2.resize(frame, (192, 256))
        gray = self._convert_to_grayscale(frame)



        mask_hot_0 = self._find_query_in_frame(self.hot, gray, 0.7, 0.3)
        mask_hot_15 = self._find_query_in_frame(self.hot_15_deg, gray, 0.7, 0.3)
        mask_hot_350 = self._find_query_in_frame(self.hot_350_deg, gray, 0.7, 0.3)
        mask_hot = cv2.bitwise_or(mask_hot_0, mask_hot_15)
        mask_hot = cv2.bitwise_or(mask_hot, mask_hot_350)

        mask_cold = self._find_query_in_frame(self.cold, gray, 0.7, 0.3)

        mask_mid = cv2.bitwise_not(cv2.bitwise_or(mask_hot, mask_cold))


        debug_frame = np.zeros_like(gray)
        debug_frame[mask_hot > 0] = 255
        debug_frame[mask_mid > 0] = 128
        debug_frame[mask_cold > 0] = 0
        debug_frame = cv2.cvtColor(debug_frame, cv2.COLOR_GRAY2BGR)

        #thresholds = threshold_multiotsu(gray, classes=3)
        #mask_cold, mask_mid, mask_hot = self._create_threshold_masks(gray, thresholds)
        hot_boxes = self._get_boxes_from_contours(cv2.findContours(mask_hot, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0])
        cold_boxes = self._get_boxes_from_contours(cv2.findContours(mask_cold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0], cold=True)

        return self.scene.update(cold_boxes, hot_boxes), debug_frame

    def is_scene_ok(self):
        return self.scene.is_scene_ok()

    def get_debug_image(self, width=None, height=None):
        if width is not None:
            return cv2.resize(self.scene.draw_scene(), (width, height))
        else:
            return self.scene.draw_scene()


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
        #rects = non_max_suppression(rects, probs=None, overlapThresh=overlapTresh)
        mask = np.zeros(frame.shape, np.uint8)
        for (x, y, x2, h2) in rects:
            cv2.rectangle(mask, (x, y), (x2, h2), (255, 255, 255), -1)

        #mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)

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
        frame = cv2.resize(frame, (192, 256))
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # crop frame to LEFT_GUIDE_X:RIGHT_GUIDE_X
        frame = frame[:, LEFT_GUIDE_X:RIGHT_GUIDE_X]

        # gauss
        frame = cv2.GaussianBlur(frame, (5, 5), 0)


        hot_mask = self._find_query_in_frame(self.hot_image, frame, sensitivity=0.85, footprint=(6,6))
        hot_mask_15 = self._find_query_in_frame(self.hot_15_deg_image, frame, sensitivity=0.85, footprint=(6,6))
        #hot_mask_350 = self._find_query_in_frame(self.hot_350_deg_image, frame, 0.7, 0.3)
        hot_mask = cv2.bitwise_or(hot_mask, hot_mask_15)
        #hot_mask = cv2.bitwise_or(hot_mask, hot_mask_350)

        if np.all(hot_mask == 0):
            self.hot_mask = None
            return


        thresholds = threshold_multiotsu(cv2.GaussianBlur(frame, (5, 5), 0), classes=3)
        region_cold = frame <= thresholds[0]
        cold_mask = self._create_mask(region_cold, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

        cold_mask = self._find_query_in_frame(self.cold_image, frame, sensitivity=0.10, footprint=(3,3))
        #cold_mask_small = self._find_query_in_frame(self.cold_small_image, frame, sensitivity=0.75, footprint=(6,6))
        #cold_mask = cv2.bitwise_or(cold_mask, cold_mask_small)

        cv2.imshow('cold_mask', cold_mask)



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

    def get_debug_image(self, width=192, height=256):
        if self.hot_mask is None or self.cold_mask is None:
            return np.zeros((height, width, 3), np.uint8)

        debug = np.zeros((height, width, 3), np.uint8)

        # add hot_mask to debug image at position LEFT_GUIDE_X
        big_hot_mask = np.zeros((256, 192), np.uint8)
        big_hot_mask[:, LEFT_GUIDE_X:RIGHT_GUIDE_X] = self.hot_mask

        # add cold_mask to debug image at position LEFT_GUIDE_X
        big_cold_mask = np.zeros((256, 192), np.uint8)
        big_cold_mask[:, LEFT_GUIDE_X:RIGHT_GUIDE_X] = self.cold_mask

        d_h = cv2.resize(big_hot_mask, (width, height))
        d_c = cv2.resize(big_cold_mask, (width, height))

        # get contours from masks
        contours_hot, _ = cv2.findContours(d_h, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_cold, _ = cv2.findContours(d_c, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # draw contours
        cv2.drawContours(debug, contours_hot, -1, (0, 0, 255), 3)
        cv2.drawContours(debug, contours_cold, -1, (255, 0, 0), 3)
        return debug

    def print(self):
        # print queue as emojis
        print(''.join(['🔴' if x == 1 else '🔵' for x in self.queue]))


class App:
    def __init__(self, window_title, video_source='./output8.mp4'):
        self.window_title = window_title
        self.video_source = video_source
        self.init_video_source()
        self.init_window()
        self.show_settings = False
        self.display_mode = Mode.OVERLAY
        self.trigger_delay = 0.5
        self.trigger_duration = 1
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
            thermal_picture_16 = np.frombuffer(thermal_data, dtype=np.uint16).reshape((CAM_HEIGHT // 2, CAM_WIDTH))
            thermal_picture_8 = cv2.normalize(thermal_picture_16, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            rgb_picture = cv2.rotate(rgb_picture, cv2.ROTATE_90_COUNTERCLOCKWISE)
            thermal_picture_8 = cv2.rotate(thermal_picture_8, cv2.ROTATE_90_COUNTERCLOCKWISE)

            return ret, rgb_picture

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
                treshold_frame = self.soup_classifier.get_debug_image(WIDTH, HEIGHT)
                if not is_ok:
                    self.fireRelay(frame)

                scaled_frame = cv2.resize(frame, (WIDTH, HEIGHT))
                if len(scaled_frame.shape) == 2:
                    scaled_frame = cv2.cvtColor(scaled_frame, cv2.COLOR_GRAY2RGB)
                dbg = self.soup_classifier.get_debug_image(WIDTH, HEIGHT)
                if len(dbg.shape) == 2:
                    dbg = cv2.cvtColor(dbg, cv2.COLOR_GRAY2RGB)
                overlay = cv2.addWeighted(scaled_frame, 1, dbg, 1, 0)
                cv2.putText(overlay, str(round((time.time() - prev_t) * 1000)) + ' ms', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                overlay = self.drawGuidelines(overlay)
                if self.isAlreadyFiring:
                    cv2.putText(overlay, self.firingMessage, (10, HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                if self.display_mode == Mode.LIVE:
                    cv2.imshow(self.window_title, frame)
                elif self.display_mode == Mode.TRESHOLD:
                    cv2.imshow(self.window_title, treshold_frame)
                elif self.display_mode == Mode.OVERLAY:
                    cv2.imshow(self.window_title, overlay)
                elif self.display_mode == Mode.DEBUG:
                    cv2.imshow(self.window_title, dbg)

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

    def drawGuidelines(self, frame):
        # Constants
        x_center = WIDTH // 2
        d_top = 120 // 2
        d_bottom = 150 // 2

        # Draw lines sloped using distance from top and bottom
        #cv2.line(frame, (x_center - d_top, 0), (x_center - d_bottom, 396), (0, 255, 0), 2)
        #cv2.line(frame, (x_center + d_top, 0), (x_center + d_bottom, 396), (0, 255, 0), 2)

        # Draw line LEFT_GUIDE_X and RIGHT_GUIDE_X, keep in mind that GUIDES are calclulated for width 192, not WIDTH
        cv2.line(frame, (int(LEFT_GUIDE_X / 192 * WIDTH), 0), (int(LEFT_GUIDE_X / 192 * WIDTH), HEIGHT), (0, 255, 0), 2)
        cv2.line(frame, (int(RIGHT_GUIDE_X / 192 * WIDTH), 0), (int(RIGHT_GUIDE_X / 192 * WIDTH), HEIGHT), (0, 255, 0), 2)


        cv2.line(frame, (0, int(HEIGHT * CHECK_LINE_Y / 256)), (WIDTH, int(HEIGHT * CHECK_LINE_Y / 256)), (0, 255, 0), 2)

        return frame

    def onMouseClick(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print('Mouse clicked at: ', x, y)
            if self.display_mode == Mode.LIVE:
                self.display_mode = Mode.TRESHOLD
            elif self.display_mode == Mode.TRESHOLD:
                self.display_mode = Mode.OVERLAY
            elif self.display_mode == Mode.OVERLAY:
                self.display_mode = Mode.DEBUG
            elif self.display_mode == Mode.DEBUG:
                self.display_mode = Mode.SETTINGS
            elif self.display_mode == Mode.SETTINGS:
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
                    self.display_mode = Mode.LIVE
            else:
                self.display_mode = Mode.LIVE

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
