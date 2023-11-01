import time
import uuid
from collections import deque

import cv2
import numpy as np
from skimage.feature import hog
from skimage.filters import threshold_multiotsu
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt


def display(window_name, image, grid_pos=(0, 0)):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, image)
    cv2.resizeWindow(window_name, 384, 512)
    cv2.moveWindow(window_name, 384 * grid_pos[0] - 0, (512 + 30) * grid_pos[1])


class MovingAverage:
    def __init__(self, size):
        self.size = size
        self.queue = deque(maxlen=size)
        self.sum = 0
        self.avg = 0
        self.med = 0

    def next(self, val):
        if len(self.queue) == self.queue.maxlen:
            self.sum -= self.queue[0]
        self.queue.append(val)
        self.sum += val
        self.avg = self.sum / len(self.queue)
        self.med = np.median(self.queue)

    def average(self):
        if len(self.queue) < self.size:
            return None
        return int(self.avg)

    def median(self):
        return int(self.med)


class Box:
    uuid = None

    contour = None
    rect = None
    center = None
    width = None
    height = None
    angle = None
    area = None
    aspect_ratio = None
    corners = None

    cold = False

    def __init__(self, contour, cold=False):
        self.uuid = uuid.uuid4()

        self.contour = contour
        self.rect = cv2.minAreaRect(contour)
        self.center = self.rect[0]
        self.width = max(self.rect[1][0], self.rect[1][1])
        self.height = min(self.rect[1][0], self.rect[1][1])
        self.angle = self.rect[2]
        self.area = self.width * self.height
        self.aspect_ratio = self.width / self.height
        self.corners = cv2.boxPoints(self.rect)
        self.corners = np.int0(self.corners)

        self.cold = cold

    # static method that creates a box in middle of frame with height of 10 pixels and width of full frame
    @staticmethod
    def create_middle_box(t = 0.1):
        width = 192
        height = 256
        tall = int(256 * t)
        points = np.array([[0, int(height / 2) + int(tall / 2)], [0, int(height / 2) - int(tall / 2)], [width, int(height / 2) - int(tall / 2)], [width, int(height / 2) + int(tall / 2)]])
        box = Box(points)
        return box

    def copy_from(self, other_box):
        self.contour = other_box.contour
        self.rect = other_box.rect
        self.center = other_box.center
        self.width = other_box.width
        self.height = other_box.height
        self.angle = other_box.angle
        self.area = other_box.area
        self.aspect_ratio = other_box.aspect_ratio
        self.corners = other_box.corners

    def is_overlaping(self, other_box):
        overlap = cv2.rotatedRectangleIntersection(self.rect, other_box.rect)
        if overlap[0] == 0:
            return False
        else:
            return True

    def distance_to(self, other_box):
        return np.linalg.norm(np.array(self.center) - np.array(other_box.center))


class SceneManager:
    frame_count = 0
    cold_boxes = []
    avg_cold_box_area = MovingAverage(100)
    hot_boxes = []
    avg_hot_box_area = MovingAverage(100)

    checkpoint = Box.create_middle_box(0.1)
    safe_area = Box.create_middle_box(0.9)

    ordered_boxes = []
    masked_boxes = []

    def __init__(self):
        pass


    def is_scene_ok(self):
        is_ok = False

        # there must not be 4 hot boxes in a row in reduced boxes
        count_of_hot_boxes_in_a_row = 0
        for b in self.ordered_boxes:
            if b.cold:
                count_of_hot_boxes_in_a_row = 0
            else:
                count_of_hot_boxes_in_a_row += 1
            if count_of_hot_boxes_in_a_row >= 4:
                return False

        if count_of_hot_boxes_in_a_row < 4:
            return True

        return is_ok

    def update(self, new_cold_boxes, new_hot_boxes):
        self.frame_count += 1

        s_time = time.time()

        # update average cold box width
        for b in new_cold_boxes:
            self.avg_cold_box_area.next(b.area)
        # update average hot box width
        for b in new_hot_boxes:
            self.avg_hot_box_area.next(b.area)

        # filter out boxes that are too small ( < 0.3 * avg_cold_box_area )
        self.cold_boxes = [b for b in new_cold_boxes if 0.3 * self.avg_cold_box_area.median() < b.area < 3 * self.avg_cold_box_area.median()]
        self.hot_boxes = [b for b in new_hot_boxes if 0.3 * self.avg_hot_box_area.median() < b.area < 3 * self.avg_hot_box_area.median()]


        # merge cold and hot boxes into one list and order them by y coordinate
        self.ordered_boxes = self.cold_boxes + self.hot_boxes
        self.ordered_boxes.sort(key=lambda box: -box.center[1])



        # determine, if scene is ok
        is_ok = self.is_scene_ok()


        #print(self.avg_cold_box_area.median() , self.avg_hot_box_area.median(), '   ', '✅' if is_ok else '❌', '    ', ['🔵' if box.cold else '⚫️' for box in self.ordered_boxes])

        # print processing time in ms
        #print('scene classify time: ', (time.time() - s_time) * 1000, 'ms')
        return is_ok

    def draw_scene(self):
        clean = np.zeros((256, 192, 3), np.uint8)

        cv2.fillPoly(clean, [self.safe_area.corners], (10, 0, 0))
        cv2.fillPoly(clean, [self.checkpoint.corners], (0, 255, 0) if self.is_scene_ok() else (0, 0, 255))

        for box in self.ordered_boxes:
            if box.cold:
                cv2.drawContours(clean, [box.corners], 0, (255, 0, 0), 2)
            else:
                cv2.drawContours(clean, [box.corners], 0, (0, 0, 255), 2)


        return clean


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
        for t2 in range(t1 + 1, 256, step):
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
    multiotsu_img_optimized[region1] = 0  # Assign gray level 85 to the pixels in region 1
    multiotsu_img_optimized[region2] = 128  # Assign gray level 170 to the pixels in region 2
    multiotsu_img_optimized[region3] = 255  # Assign gray level 255 to the pixels in region 3
    return multiotsu_img_optimized


scene = SceneManager()

class SoupClassifier:
    def __init__(self):
        self.scene = SceneManager()

    def classify(self, frame):
        time_start = time.time()
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


        # HOT HOT HOT HOT HOT HOT HOT HOT HOT HOT HOT HOT HOT HOT HOT HOT HOT HOT HOT HOT HOT HOT HOT HOT HOT HOT HOT HOT HOT HOT
        hot_boxes = []
        contours, _ = cv2.findContours(mask_hot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            box = Box(cnt, cold=False)
            hot_boxes.append(box)

            # TODO: DEBUG
            cv2.drawContours(frame, [box.corners], 0, (0, 0, 255), 2)
            center = box.center
            cv2.circle(frame, (int(center[0]), int(center[1])), 3, (0, 0, 255), -1)
            [vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
            lefty = int((-x * vy / vx) + y)
            righty = int(((gray.shape[1] - x) * vy / vx) + y)
            cv2.line(frame, (gray.shape[1] - 1, righty), (0, lefty), (0, 0, 255), 1)
            cv2.fillPoly(clean, [box.corners], (0, 0, 255))
            cv2.putText(frame, str(int(box.area)), (int(center[0]), int(center[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # COLD COLD COLD COLD COLD COLD COLD COLD COLD COLD COLD COLD COLD COLD COLD COLD COLD COLD COLD COLD COLD COLD COLD COLD
        cold_boxes = []
        contours_cold, _ = cv2.findContours(mask_cold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours_cold:
            epsilon = 0.1 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            # is convex
            if cv2.isContourConvex(approx):
                if len(approx) == 4:
                    # get min box
                    box = Box(cnt, cold=True)
                    cold_boxes.append(box)

        # print processing time in ms
        # print('feature extraction time: ', (time.time() - time_start) * 1000, 'ms')
        return self.scene.update(cold_boxes, hot_boxes)


    def is_scene_ok(self):
        return self.scene.is_scene_ok()

    def get_debug_image(self, width, height):
        if width is not None:
            return cv2.resize(self.scene.draw_scene(), (width, height))
        else:
            return self.scene.draw_scene()

