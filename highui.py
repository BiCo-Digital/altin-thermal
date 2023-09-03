import time
import uuid

import cv2
import numpy as np
from gpiozero import OutputDevice
import threading
import platform
from skimage.filters import threshold_multiotsu
from collections import deque


def is_mac():
    return platform.system() == "Darwin"


def use_video_file():
    return is_mac()


WIDTH = 320
HEIGHT = 480
RELAY_GPIO = 26

P2Pro_resolution = (256, 384)
CAM_WIDTH = 256
CAM_HEIGHT = 384


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


class SoupClassifier:
    def __init__(self):
        self.scene = SceneManager()

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

    def classify(self, frame):
        frame = cv2.resize(frame, (192, 256))
        gray = self._convert_to_grayscale(frame)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        thresholds = threshold_multiotsu(gray, classes=3)
        mask_cold, mask_mid, mask_hot = self._create_threshold_masks(gray, thresholds)

        hot_boxes = self._get_boxes_from_contours(cv2.findContours(mask_hot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])
        cold_boxes = self._get_boxes_from_contours(cv2.findContours(mask_cold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], cold=True)

        debug_frame = np.zeros_like(gray)
        debug_frame[mask_hot > 0] = 255
        debug_frame[mask_mid > 0] = 128
        debug_frame[mask_cold > 0] = 0
        debug_frame = cv2.cvtColor(debug_frame, cv2.COLOR_GRAY2BGR)

        return self.scene.update(cold_boxes, hot_boxes), debug_frame

    def is_scene_ok(self):
        return self.scene.is_scene_ok()

    def get_debug_image(self, width=None, height=None):
        if width is not None:
            return cv2.resize(self.scene.draw_scene(), (width, height))
        else:
            return self.scene.draw_scene()


class App:
    def __init__(self, window_title, video_source='./thermal.mp4'):
        self.window_title = window_title
        self.video_source = video_source
        self.init_video_source()
        self.init_window()
        self.show_settings = is_mac()
        self.display_mode = Mode.SETTINGS if is_mac() else Mode.LIVE
        self.trigger_delay = 0.5
        self.trigger_duration = 1
        self.soup_classifier = SoupClassifier()
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
            cv2.moveWindow(self.window_title, 0, 0)
        else:
            cv2.setWindowProperty(self.window_title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.moveWindow(self.window_title, -1, -1)
        cv2.setMouseCallback(self.window_title, self.onMouseClick)

    def grab_frame(self):
        ret, frame = self.vid.read()

        if use_video_file():
            if self.vid.get(cv2.CAP_PROP_POS_FRAMES) == self.vid.get(cv2.CAP_PROP_FRAME_COUNT):
                self.vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame = cv2.resize(frame, (CAM_WIDTH, CAM_HEIGHT))
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

            return ret, thermal_picture_8

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
                is_ok, treshold_frame = self.soup_classifier.classify(frame)
                if not is_ok:
                    self.fireRelay()

                scaled_frame = cv2.resize(frame, (WIDTH, HEIGHT))
                if len(scaled_frame.shape) == 2:
                    scaled_frame = cv2.cvtColor(scaled_frame, cv2.COLOR_GRAY2RGB)
                dbg = self.soup_classifier.get_debug_image(WIDTH, HEIGHT)
                overlay = cv2.addWeighted(scaled_frame, 0.5, dbg, 0.3, 0)
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

    def fireRelay(self):
        if self.isAlreadyFiring:
            return
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
        cv2.line(frame, (x_center - d_top, 0), (x_center - d_bottom, 396), (0, 128, 0), 2)
        cv2.line(frame, (x_center + d_top, 0), (x_center + d_bottom, 396), (0, 128, 0), 2)

        cv2.line(frame, (0, 395), (WIDTH, 395), (0, 128, 0), 2)

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
