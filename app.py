from collections import deque
import os
import shutil
import time
import cv2
import numpy as np
import datetime
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder, Quality
from picamera2.outputs import CircularOutput

os.chdir('/home/matejnevlud/')

# Configure camera for 2304, 1296 mode
picam2 = Picamera2()
video_config = picam2.create_video_configuration({'size': (256, 192), 'format': 'XBGR8888'},
                                                 raw={'size': (2304, 1296)},
                                                 controls={'NoiseReductionMode': 0, 'FrameRate': 50})
picam2.configure(video_config)
picam2.start_preview()
encoder = H264Encoder()
encoder.output = CircularOutput()
picam2.encoder = encoder
picam2.start()
picam2.start_encoder(encoder=encoder, quality=Quality.LOW)

thermalcamera = Picamera2(1)
thermalcamera.configure(thermalcamera.create_video_configuration(raw=True))
thermalcamera.start()

time.sleep(2)

bgsub = cv2.bgsegm.createBackgroundSubtractorCNT(minPixelStability=1, useHistory=False, maxPixelStability=2, isParallel=True)

# named fullscreen window on ubutn
# cv2.namedWindow('frame', cv2.WND_PROP_FULLSCREEN)
# cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
DIR_NAME = '/home/matejnevlud/thermal_images_' + timestamp
# os.makedirs(DIR_NAME, exist_ok=True)
os.makedirs('XXX', exist_ok=True)

# Capture 100 frames and calculate FPS
last_time = time.time()
frame_count = 0
color_frame = np.zeros((192, 256, 3), dtype=np.uint8)
thermal_frame = np.zeros((192, 256), dtype=np.float32)

# create deques for storing max value
# and min value for the last 100 frames
max_t_deque = deque(maxlen=100)
min_t_deque = deque(maxlen=100)
mean_t_deque = deque(maxlen=100)

DEQUE_LEN = 200
soup_area_deque = deque(maxlen=200)
soup_t_deque = deque(maxlen=200)
soup_min_t_deque = deque(maxlen=200)
soup_avg_t_deque = deque(maxlen=200)
soup_delta_t_deque = deque(maxlen=200)
last_soup_timestamp = datetime.datetime.now()

last_soup_min_t_z_score = 0


def preprocess_pi_frame(pi_frame):
    pi_frame = cv2.cvtColor(pi_frame, cv2.COLOR_BGR2RGB)
    pi_frame = cv2.rotate(pi_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # cv2.imshow('pi_frame0', pi_frame)

    # pts1 = np.float32([[75, 0], [160, 0], [67, 207], [180, 220]])
    # pts2 = np.float32([[0, 0], [192, 0], [0, 340], [192, 340]])
    pts1 = np.float32([[60, 58], [172, 55], [55, 215], [190, 225]])
    pts2 = np.float32([[0, 0], [192, 0], [0, 256], [192, 256]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    pi_frame = cv2.warpPerspective(pi_frame, M, (192, 256), flags=cv2.INTER_NEAREST)

    # pyramid mean shift filtering
    # pi_frame = cv2.pyrMeanShiftFiltering(pi_frame, 3, 3)
    # pi_frame = cv2.bilateralFilter(pi_frame, 15, 30, 90)
    cv2.imshow('pi_frame2', pi_frame)

    pi_frame = cv2.GaussianBlur(pi_frame, (5, 5), 0)

    return pi_frame


def detect_soups(pi_frame_foreground):
    contours, hierarchy = cv2.findContours(pi_frame_foreground.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pi_frame_foreground = np.zeros((256, 192), dtype=np.uint8)
    soups_rects = []
    for cnt in contours:
        hull = cv2.convexHull(cnt)
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)

        cv2.drawContours(pi_frame_foreground, [cnt], 0, (64, 64, 64), -1)

        if area > 3000 and w > 80 and h > w and y > 0 and y + h < 256:
            soup_area_deque.append(area)
            cv2.drawContours(pi_frame_foreground, [hull], 0, (255, 255, 255), -1)
            cv2.rectangle(pi_frame_foreground, (x, y), (x + w, y + h), (255, 255, 255), 2)
            soups_rects.append((x, y, w, h))

    return pi_frame_foreground, soups_rects


def preprocess_thermal_frame(frame_usb):
    frame_mid_pos = int(len(frame_usb) / 2)

    thermal_buffer = frame_usb[frame_mid_pos:]
    thermal_picture_u16 = np.frombuffer(thermal_buffer, dtype=np.uint16).reshape((192, 256))
    thermal_picture_u16 = cv2.rotate(thermal_picture_u16, cv2.ROTATE_90_COUNTERCLOCKWISE)
    thermal_picture_u16 >>= 2
    thermal_picture_f32 = thermal_picture_u16.astype(np.float32)
    thermal_picture_f32 /= 16
    thermal_picture_f32 -= 273.15

    thermal_picture_f32 = np.clip(thermal_picture_f32, 0, 50)

    return thermal_picture_f32


pi_frames_deque = deque(maxlen=10 * 25)
usb_frames_deque = deque(maxlen=10 * 25)
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
DIR_NAME = 'thermal_images_' + timestamp


def add_frames_to_buffer(pi_frame, usb_frame):
    pi_frames_deque.append(pi_frame)
    usb_frames_deque.append(usb_frame)


def save_frame_buffers_to_disk():
    print('saving frames to disk')
    os.makedirs(DIR_NAME, exist_ok=True)
    for i, (pi_frame, usb_frame) in enumerate(zip(pi_frames_deque, usb_frames_deque)):
        cv2.imwrite(f'{DIR_NAME}/frame_{i:04d}.png', pi_frame)
        with open(f'{DIR_NAME}/frame_{i:04d}.raw', 'wb') as f:
            f.write(usb_frame)
    # archive the folder
    shutil.make_archive(DIR_NAME, 'zip', DIR_NAME)
    # delete the folder
    shutil.rmtree(DIR_NAME)


last_thermal_frame = None
fire = False
while 1:
    frame_count += 1
    fire = False

    pi_frame = picam2.capture_array("main")
    frame_usb = thermalcamera.capture_array()
    add_frames_to_buffer(pi_frame, frame_usb)

    pi_frame = preprocess_pi_frame(pi_frame)

    pi_frame = bgsub.apply(pi_frame)
    pi_frame = cv2.dilate(pi_frame, np.ones((7, 7), np.uint8), iterations=2)
    pi_frame = cv2.erode(pi_frame, np.ones((7, 7), np.uint8), iterations=2)

    pi_frame, soups_rects = detect_soups(pi_frame)
    cv2.imshow('pi_frame3', pi_frame)

    thermal_frame = preprocess_thermal_frame(frame_usb)

    # check if thermal_frame changed since last frame
    if np.array_equal(last_thermal_frame, thermal_frame):
        print('✋✋✋✋✋✋✋✋✋✋✋✋✋✋', end='\r')
        continue
    last_thermal_frame = thermal_frame.copy()

    #  check if last_soup_timestamp is older than 10 seconds
    if (datetime.datetime.now() - last_soup_timestamp).total_seconds() > 5 and len(soup_min_t_deque) > 0:
        print('🗑️ 🗑️ 🗑️ 🗑️ 🗑️ 🗑️ 🗑️ 🗑️  Resetting soup deques                                           ', end='\n')

        soup_area_deque.clear()
        soup_t_deque.clear()
        soup_min_t_deque.clear()
        soup_avg_t_deque.clear()
        soup_delta_t_deque.clear()
        last_soup_timestamp = datetime.datetime.now()

    # calculate min and max values for the last 100 frames
    max_t_deque.append(thermal_frame.max())
    min_t_deque.append(thermal_frame.min())

    # remove soups_rects from thermal_frame
    soups_frames = []
    thermal_frame_without_soups = thermal_frame.copy()
    soups_frame = np.zeros(thermal_frame.shape, dtype=np.float32)
    for x, y, w, h in soups_rects:
        soup_frame = thermal_frame[y:y + h, x:x + w].copy()

        soup_mean_t = np.nanmean(soup_frame)
        soup_t_deque.append(soup_mean_t)

        soup_max_t = np.nanmax(soup_frame)
        soup_min_t = np.nanmin(soup_frame)
        soup_avg_t = np.nanmean(soup_frame)
        soup_avg_t_deque.append(soup_avg_t)
        soup_min_t_deque.append(soup_min_t)
        soup_delta_t = soup_max_t - soup_min_t
        soup_delta_t_deque.append(soup_delta_t)

        soup_delta_t_z_score = (soup_delta_t - np.nanmean(soup_delta_t_deque)) / np.nanstd(soup_delta_t_deque)
        soup_min_t_z_score = (soup_min_t - np.nanmean(soup_min_t_deque)) / np.nanstd(soup_min_t_deque)

        if ((soup_min_t_z_score > 4.5 or soup_min_t - np.nanmean(soup_min_t_deque) > 4) and len(soup_min_t_deque) > DEQUE_LEN / 2):
            fire = True
            short_timestamp = datetime.datetime.now().strftime("%H-%M-%S")
            if soup_min_t - np.nanmean(soup_min_t_deque) > 4:
                short_timestamp += '_HOTTER'
            print(f'🔥🔥🔥🔥🔥🔥🔥🔥 min_t: {soup_min_t:.2f}, soup_avg_t {soup_avg_t:.2f},   soup_min_t_z_score: {soup_min_t_z_score:.2f} [{short_timestamp}]', )
            cv2.imwrite(f'XXX/frame_{short_timestamp}_{str(int(soup_min_t_z_score))}.png', cv2.applyColorMap(cv2.normalize(soup_frame, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U), cv2.COLORMAP_JET))

            # prepare and write small log file with this detection next to the image
            with open(f'XXX/frame_{short_timestamp}_{str(int(soup_min_t_z_score))}.txt', 'w') as f:
                f.write(f'queue_min_t: {np.nanmean(soup_min_t_deque):.2f}, queue_delta_t {np.nanmean(soup_delta_t_deque):.2f}, last_min_t_z_score: {last_soup_min_t_z_score:.2f}\n')
                f.write(f'soup_min_t: {soup_min_t:.2f}, soup_delta_t {soup_delta_t:.2f}, soup_min_t_z_score: {soup_min_t_z_score:.2f}, soup_avg_t: {soup_avg_t:.2f}\n')

        last_soup_min_t_z_score = soup_min_t_z_score

        thermal_frame_without_soups[y:y + h, x:x + w] = np.nan

        # paste soup_frame into soups_frame
        soups_frame[y:y + h, x:x + w] = soup_frame

        last_soup_timestamp = datetime.datetime.now()

    # calculate mean temperature of thermal_frame but filter out 0 values
    mean_t_deque.append(np.nanmean(thermal_frame_without_soups))

    # overlay frame on thermal image
    thermal_picture_colored = cv2.applyColorMap(cv2.normalize(thermal_frame, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U), cv2.COLORMAP_JET)
    frame = cv2.cvtColor(pi_frame, cv2.COLOR_GRAY2RGB)
    frame = cv2.addWeighted(frame, 0.6, thermal_picture_colored, 0.6, 0)
    if fire:
        # draw red circle in top right corner to indicate fire detection
        cv2.circle(frame, (frame.shape[1] - 20, 20), 12, (0, 0, 255), -1)

    cv2.imshow('frame', frame)

    # calculate FPS
    now = time.time()
    fps = 1 / (now - last_time)
    last_time = now
    print(f'fps: {fps:.2f}, soup_min_t: {np.nanmean(soup_min_t_deque):.2f}, soup_avg_t {np.nanmean(soup_avg_t_deque):.2f}', f'🍲 soup_min_t_z_score: {last_soup_min_t_z_score:.2f}' if len(soups_rects) > 0 else '  ', end='\r')

    #  print O if no soups detected else 4, using one print statement

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

picam2.stop()
cv2.destroyAllWindows()

save_frame_buffers_to_disk()