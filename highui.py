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

WIDTH = 320
HEIGHT = 480
RELAY_GPIO = 26



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
        region_cold = gray <= thresholds[0]
        region_mid = (gray > thresholds[0]) & (gray <= thresholds[1])
        region_hot = gray > thresholds[1]
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



# create a named window using opencv and show the image in the window so it fills the window. next add 2 buttons over the image and show the window
class App:

    # initialize the app
    def __init__(self, window_title, video_source='./thermal.mp4'):
        self.window_title = window_title
        self.video_source = video_source

        # open video source
        self.initVideoFromFile(self.video_source)

        # create a canvas that can fit the above video source size
        self.canvas = cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_title, WIDTH, HEIGHT)
        cv2.moveWindow(window_title, 0, 0)

        # create mouse click callback
        cv2.setMouseCallback(window_title, self.onMouseClick)
        self.show_settings = True
        self.trigger_delay = 0.5 # seconds
        self.trigger_duration = 1 # seconds

        # create a classifier
        self.soup_classifier = SoupClassifier()

        # create a relay
        if not is_mac():
            self.relay = OutputDevice(RELAY_GPIO)
            self.relay.off()
        self.isAlreadyFiring = False

        self.runLoop()

    def initVideoFromFile(self, video_source):
        self.vid = cv2.VideoCapture(video_source)

    def grabFrameFromFile(self):
        # if last frame from file, rewind
        if self.vid.get(cv2.CAP_PROP_POS_FRAMES) == self.vid.get(cv2.CAP_PROP_FRAME_COUNT):
            self.vid.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # read frame
        ret, frame = self.vid.read()

        # resize
        frame = cv2.resize(frame, (WIDTH, HEIGHT))

        return ret, frame


    def initVideoFromCamera(self, camera_source=0):
        self.vid = cv2.VideoCapture(camera_source)
        self.vid.set(cv2.CAP_PROP_CONVERT_RGB, 0)

    def grabFrame(self):
        # read frame
        ret, frame = self.vid.read()
        return ret, frame


    def runLoop(self):
        # cap framerate to 25
        frame_rate = 25
        prev_t = 0

        while True:
            # Make sure we don't go over the framerate cap
            time_elapsed = time.time() - prev_t
            if time_elapsed < 1./frame_rate:
                continue
            prev_t = time.time()

            # if settings are shown, show them and continue
            if self.show_settings:
                self.showSettings()
                continue


            # get a frame from the video source
            ret, frame = self.grabFrameFromFile()
            if ret:
                is_ok = self.soup_classifier.classify(frame)

                if not is_ok:
                    self.fireRelay()

                dbg = self.soup_classifier.get_debug_image(WIDTH, HEIGHT)
                overlay = cv2.addWeighted(frame, 0.5, dbg, 0.5, 0)


                # show procceesing time on frame, in ms (rounded)
                cv2.putText(overlay, str(round((time.time() - prev_t) * 1000)) + ' ms', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # draw guidelines
                overlay = self.drawGuidelines(overlay)
                # write FIRE on bottom of frame if relay is on
                if self.isAlreadyFiring:
                    cv2.putText(overlay, 'FOUKAM', (10, HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                # show the frame
                cv2.imshow(self.window_title, overlay)


            if cv2.waitKey(1) & 0xFF == ord('q'):
                break



    def fireRelay(self):
        if self.isAlreadyFiring:
            return

        # fire relay for trigger_duration seconds but wait trigger_delay seconds before firing using threading library
        def fire_on_thread():
            time.sleep(self.trigger_delay)
            self.relay.on()
            time.sleep(self.trigger_duration)
            self.relay.off()
            self.isAlreadyFiring = False

        def fire_debug():
            print("GOING TO FIRE IN " + str(self.trigger_delay) + " SECONDS")
            time.sleep(self.trigger_delay)
            print("FIRING FOR " + str(self.trigger_duration) + " SECONDS")
            time.sleep(self.trigger_duration)
            print("STOPPING FIRE")
            self.isAlreadyFiring = False



        t = threading.Thread(target=fire_on_thread if not is_mac() else fire_debug)
        t.start()
        self.isAlreadyFiring = True





    def drawGuidelines(self, frame):
        cv2.line(frame, (115, 10), (100, 396), (0, 255, 0), 2)
        cv2.line(frame, (235, 10), (250, 396), (0, 255, 0), 2)
        cv2.line(frame, (0, 395), (WIDTH, 395), (0, 255, 0), 2)
        return frame



    def onMouseClick(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print('Mouse clicked at: ', x, y)
            if self.show_settings:
                # check if click was inside a button
                third_width = WIDTH // 3
                third_height = HEIGHT // 3

                # check if click was inside minus button
                if self.didHitButton(0, 40, third_width, third_width, x, y):
                    self.trigger_delay -= 0.1
                    if self.trigger_delay < 0:
                        self.trigger_delay = 0

                # check if click was inside plus button
                if self.didHitButton(third_width * 2, 40, third_width, third_width, x, y):
                    self.trigger_delay += 0.1

                # check if click was inside minus button
                if self.didHitButton(0, 40 + third_width + 40, third_width, third_width, x, y):
                    self.trigger_duration -= 0.1
                    if self.trigger_duration < 0:
                        self.trigger_duration = 0

                # check if click was inside plus button
                if self.didHitButton(third_width * 2, 40 + third_width + 40, third_width, third_width, x, y):
                    self.trigger_duration += 0.1

                # check if click was inside start button
                if self.didHitButton(0, 80 + third_width * 2 + 40, WIDTH, third_width, x, y):
                    print('START')
                    self.show_settings = False

            else:
                self.show_settings = True




        if event == cv2.EVENT_LBUTTONUP:
            print('Mouse released at: ', x, y)

    def drawButton(self, frame, x, y, width, height, text):
        #draw button with border of 5 pixels and text in the middle
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 5)
        cv2.putText(frame, text, (x + width // 2 - 10, y + height // 2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    def didHitButton(self, x, y, width, height, mouse_x, mouse_y):
        if mouse_x >= x and mouse_x <= x + width and mouse_y >= y and mouse_y <= y + height:
            return True
        return False

    def showSettings(self):
        frame = np.zeros((HEIGHT, WIDTH, 3), np.uint8)

        # draw two rows consisting of 2 buttons each and a text label between them, the buttons are used to set the delay and duration of the trigger using + and - buttons
        # the text label shows the current delay and duration

        # draw first row (minus button, delay label, plus button)
        third_width = WIDTH // 3

        # draw minus button (left) as a rectangle with a minus sign in it third_width wide and 100 high
        cv2.putText(frame, 'ZPOZDENI', (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        self.drawButton(frame, 0, 40, third_width, third_width, '-')
        self.drawButton(frame, third_width, 40, third_width, third_width, str(round(self.trigger_delay, 1)))
        self.drawButton(frame, third_width * 2, 40, third_width, third_width, '+')

        # draw second row (minus button, duration label, plus button)
        cv2.putText(frame, 'TRVANI', (0, 40 + third_width + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        self.drawButton(frame, 0, 40 + third_width + 40, third_width, third_width, '-')
        self.drawButton(frame, third_width, 40 + third_width + 40, third_width, third_width, str(round(self.trigger_duration, 1)))
        self.drawButton(frame, third_width * 2, 40 + third_width + 40, third_width, third_width, '+')

        # draw start button
        self.drawButton(frame, 0, 80 + third_width * 2 + 40, WIDTH, third_width, 'START')





        # show the frame
        cv2.imshow(self.window_title, frame)
        cv2.waitKey(20)


app = App("Thermal Camera")
