from collections import deque
import os
import shutil
import time
import cv2
import numpy as np
import datetime
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder, Quality, JpegEncoder
from picamera2.outputs import CircularOutput
import threading
from gpiozero import OutputDevice
import requests
import pytz

from ht301_hacklib import device_info, info

os.chdir('/home/matejnevlud/')
timezone = pytz.timezone("Europe/Prague")

# load settings file if exists
# if not, create default settings file

# load settings file if exists

THERMAL_WIDTH = 384
THERMAL_HEIGHT = 288





def init_camera():
    # Configure camera for 2304, 1296 mode
    picam2 = Picamera2()
    video_config = picam2.create_video_configuration({'size': (THERMAL_WIDTH // 2, THERMAL_HEIGHT // 2), 'format': 'XBGR8888'},
                                                     raw={'size': (2304, 1296)},
                                                     controls={'NoiseReductionMode': 0, 'FrameRate': 50})
    picam2.configure(video_config)
    picam2.start_preview()
    encoder = JpegEncoder()
    encoder.output = CircularOutput()
    picam2.encoder = encoder
    picam2.start()
    picam2.start_encoder(encoder=encoder, quality=Quality.LOW)


    thermalcamera = Picamera2(1)
    thermalcamera.configure(thermalcamera.create_video_configuration(raw=True))
    thermalcamera.start()

    time.sleep(2)

    return picam2, thermalcamera
picam2, thermalcamera = init_camera()

bgsub = cv2.bgsegm.createBackgroundSubtractorCNT(minPixelStability=1, useHistory=False, maxPixelStability=2, isParallel=True)
#bgsub = cv2.createBackgroundSubtractorMOG2(history=1, varThreshold=32, detectShadows=False)


# named fullscreen window on ubutn
cv2.namedWindow('thermal_picture_colored', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('thermal_picture_colored', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)



TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
DIR_NAME = '/home/matejnevlud/thermal_images_' + TIMESTAMP
os.makedirs('XXX', exist_ok=True)

# Capture 100 frames and calculate FPS
last_time = time.time()
frame_count = 0
color_frame = np.zeros((THERMAL_HEIGHT, THERMAL_WIDTH, 3), dtype=np.uint8)
thermal_frame = np.zeros((THERMAL_HEIGHT, THERMAL_WIDTH), dtype=np.float32)


# create deques for storing max value
# and min value for the last 100 frames
max_t_deque = deque(maxlen=100)
min_t_deque = deque(maxlen=100)
mean_t_deque = deque(maxlen=100)

TEMP_TRESHOLD = 3

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
    #cv2.imshow('pi_frame0', pi_frame)

    #pts1 = np.float32([[75, 0], [160, 0], [67, 207], [180, 220]])
    #pts2 = np.float32([[0, 0], [THERMAL_HEIGHT, 0], [0, 340], [THERMAL_HEIGHT, 340]])
    pts1 = np.float32([[50, 58 - 5], [172, 55 - 5], [45, 215 + 5], [190, 225 + 5]])
    pts2 = np.float32([[0, 0], [THERMAL_HEIGHT, 0], [0, THERMAL_WIDTH], [THERMAL_HEIGHT, THERMAL_WIDTH]])

    # I want to double the resolution of input image, double points

    M = cv2.getPerspectiveTransform(pts1, pts2)
    pi_frame = cv2.warpPerspective(pi_frame, M, (THERMAL_HEIGHT, THERMAL_WIDTH), flags=cv2.INTER_NEAREST)

    #pyramid mean shift filtering
    #pi_frame = cv2.pyrMeanShiftFiltering(pi_frame, 3, 3)
    #pi_frame = cv2.bilateralFilter(pi_frame, 15, 30, 90)

    pi_frame = cv2.GaussianBlur(pi_frame, (5, 5), 0)

    return pi_frame

def detect_soups(pi_frame_foreground):
    contours, hierarchy = cv2.findContours(pi_frame_foreground.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pi_frame_foreground = np.zeros((THERMAL_WIDTH, THERMAL_HEIGHT), dtype=np.uint8)
    soups_rects = []
    for cnt in contours:
        hull = cv2.convexHull(cnt)
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)

        cv2.drawContours(pi_frame_foreground, [cnt], 0, (64, 64, 64), -1)

        if area > 3000 and w > 80 and h > w and y > 0 and y + h < THERMAL_WIDTH:
            soup_area_deque.append(area)
            cv2.drawContours(pi_frame_foreground, [hull], 0, (255, 255, 255), -1)
            cv2.rectangle(pi_frame_foreground, (x, y), (x + w, y + h), (255, 255, 255), 2)

            # make bounding box a bit smaller (about 20%), remember that numbers must be integers
            x += int(w * 0.05)
            y += int(h * 0.05)
            w = int(w * 0.9)
            h = int(h * 0.9)

            soups_rects.append((x, y, w, h))

    return pi_frame_foreground, soups_rects

def preprocess_thermal_frame(frame_usb):
    RAW_THERMAL_WIDTH = THERMAL_WIDTH
    RAW_THERMAL_HEIGHT = THERMAL_HEIGHT + 4


    dt = np.dtype(np.uint16)
    frame = frame_usb.view(dtype=dt)
    frame = frame.reshape(RAW_THERMAL_HEIGHT, RAW_THERMAL_WIDTH)
    frame_raw = frame
    f_visible = frame_raw[:frame_raw.shape[0] - 4,...]
    meta      = frame_raw[frame_raw.shape[0] - 4:,...]

    device_strings = device_info(meta)


    information, temperature_LUT_C = info(meta, device_strings, THERMAL_HEIGHT, THERMAL_WIDTH)

    frame_64 = temperature_LUT_C[f_visible]
    frame_32 = frame_64.astype(np.float32)

    # rotate frame 90 degrees clockwise
    frame_32 = cv2.rotate(frame_32, cv2.ROTATE_90_CLOCKWISE)



    return frame_32



pi_frames_deque = deque(maxlen=10*25)
usb_frames_deque = deque(maxlen=10*25)
TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
DIR_NAME = 'thermal_images_' + TIMESTAMP

def add_frames_to_buffer(pi_frame, usb_frame):
    pi_frames_deque.append(pi_frame)
    usb_frames_deque.append(usb_frame)

def save_frame_buffers_to_disk():
    dir_buffer = 'thermal_images_' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    #create copy of the deques, so we can continue adding frames to the deques while we are saving them to disk
    pi_frames_deque_copy = pi_frames_deque.copy()
    usb_frames_deque_copy = usb_frames_deque.copy()

    os.makedirs(dir_buffer, exist_ok=True)
    for i, (pi_frame, usb_frame) in enumerate(zip(pi_frames_deque_copy, usb_frames_deque_copy)):
        cv2.imwrite(f'{dir_buffer}/frame_{i:04d}.png', pi_frame)
        with open(f'{dir_buffer}/frame_{i:04d}.raw', 'wb') as f:
            f.write(usb_frame)
    shutil.make_archive(dir_buffer, 'zip', dir_buffer)
    shutil.rmtree(dir_buffer)

def save_frame_buffers_to_disk_on_separate_thread():
    t = threading.Thread(target=save_frame_buffers_to_disk)
    t.start()

power_led = OutputDevice(19, active_high=True, initial_value=True)
app_led = OutputDevice(13, active_high=True, initial_value=False)
soup_led = OutputDevice(6, active_high=True, initial_value=False)

relay_pin = OutputDevice(26, active_high=False)
relay_pin.off()


trigger_delay = 2.2
trigger_duration = 0.4
fire = False
last_fire_frames  = []


currently_ongoing_upload_timestamps_seconds = []
def upload_to_server(line, min_t, avg_t, delta_t, max_t, min_t_zscore, q_min_t, q_delta_t, current_datetime, image_thermal, image_color):
    # with timezone ISO 8601, with time zone as UTC offset
    current_timestamp = current_datetime.strftime("%Y-%m-%dT%H:%M:%S%z")
    short_timestamp = current_datetime.strftime("%H-%M-%S")

    # check if there is already an upload in progress based on short_timestamp
    if short_timestamp in currently_ongoing_upload_timestamps_seconds:
        return

    # add short_timestamp to currently_ongoing_upload_timestamps_seconds
    currently_ongoing_upload_timestamps_seconds.append(short_timestamp)

    form_data = {
        "line": line,
        "min_t": min_t,
        "avg_t": avg_t,
        "delta_t": delta_t,
        "max_t": max_t,
        "min_t_zscore": min_t_zscore,
        "q_min_t": q_min_t,
        "q_delta_t": q_delta_t,
        "timestamp": current_timestamp,
    }

    files = {
        "image_thermal": ('image_thermal.png', open(image_thermal, "rb"), "image/png"),
        "image_color": ('image_color.png', open(image_color, "rb"), "image/png"),
    }

    def send_request():
        response = requests.post("https://altin-admin.vercel.app/api/event", data=form_data, files=files)
        if response.status_code == 200:
            print("Request successful!", response.text)
        else:
            print("Request failed.")

    thread = threading.Thread(target=send_request)
    thread.finished = lambda: currently_ongoing_upload_timestamps_seconds.remove(short_timestamp)
    thread.start()

def fire_trigger(soup_min_t, soup_min_t_z_score, soup_min_t_deque ):

    def fire_trigger_on_thread():
        time.sleep(trigger_delay)
        global fire
        if fire:
            return

        fire = True
        if trigger_duration > 0:
            relay_pin.on()
            time.sleep(trigger_duration)
            relay_pin.off()
        fire = False

    t = threading.Thread(target=fire_trigger_on_thread)
    t.start()


    today_dir = 'X' + datetime.datetime.now().strftime("%Y-%m-%d")
    os.makedirs(today_dir, exist_ok=True)
    current_datetime = datetime.datetime.now(timezone)
    current_timestamp = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    short_timestamp = current_datetime.strftime("%H-%M-%S")
    global last_fire_frames
    last_fire_frames.append(cv2.applyColorMap(cv2.normalize(last_thermal_frame, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U), cv2.COLORMAP_JET))
    print(f'🔥🔥🔥🔥🔥🔥🔥🔥 min_t: {soup_min_t:.2f}, soup_avg_t {soup_avg_t:.2f},   soup_min_t_z_score: {soup_min_t_z_score:.2f} [{short_timestamp}]', )

    thermal_picture_path = f'{today_dir}/frame_{short_timestamp}_{str(int(soup_min_t_z_score))}.png'

    cv2.imwrite(thermal_picture_path, cv2.rotate(cv2.applyColorMap(cv2.normalize(soup_frame, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U), cv2.COLORMAP_JET), cv2.ROTATE_90_CLOCKWISE))

    with open(f'{today_dir}/frame_{short_timestamp}_{str(int(soup_min_t_z_score))}.txt', 'w') as f:
        f.write(f'queue_min_t: {np.nanmean(soup_min_t_deque):.2f}, queue_delta_t {np.nanmean(soup_delta_t_deque):.2f}, last_min_t_z_score: {last_soup_min_t_z_score:.2f}\n')
        f.write(f'soup_min_t: {soup_min_t:.2f}, soup_delta_t {soup_delta_t:.2f}, soup_min_t_z_score: {soup_min_t_z_score:.2f}, soup_avg_t: {soup_avg_t:.2f}\n')

    upload_to_server(2, soup_min_t, soup_avg_t, soup_delta_t, soup_max_t, soup_min_t_z_score, np.nanmean(soup_min_t_deque), np.nanmean(soup_delta_t_deque), current_datetime, thermal_picture_path, thermal_picture_path)



def draw_settings():
    frame = np.zeros((THERMAL_WIDTH, THERMAL_HEIGHT, 3), dtype=np.uint8)

    def drawButton(frame, x, y, width, height, text):
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 3)
        cv2.putText(frame, text, (x + width // 2 - 14, y + height // 2 + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    third_width = THERMAL_HEIGHT // 3
    cv2.putText(frame, 'ZPOZDENI', (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    drawButton(frame, 0, 40, third_width, third_width, '-')
    drawButton(frame, third_width, 40, third_width, third_width, str(round(trigger_delay, 1)))
    drawButton(frame, third_width * 2, 40, third_width, third_width, '+')

    cv2.putText(frame, 'TRVANI', (0, 40 + third_width + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    drawButton(frame, 0, 40 + third_width + 40, third_width, third_width, '-')
    drawButton(frame, third_width, 40 + third_width + 40, third_width, third_width, str(round(trigger_duration, 1)))
    drawButton(frame, third_width * 2, 40 + third_width + 40, third_width, third_width, '+')

    return frame

def draw_mosaic():
    frame = np.zeros((THERMAL_WIDTH, THERMAL_HEIGHT, 3), dtype=np.uint8)

    # get last 4 frames from last_fire_frames, resize them and paste them into frame in 2x2 grid
    for i, last_fire_frame in enumerate(last_fire_frames[-4:]):
        last_fire_frame = cv2.resize(last_fire_frame, (THERMAL_HEIGHT // 2, THERMAL_WIDTH // 2))
        frame[i // 2 * (THERMAL_WIDTH // 2):i // 2 * (THERMAL_WIDTH // 2) + (THERMAL_WIDTH // 2), i % 2 * (THERMAL_HEIGHT // 2):i % 2 * (THERMAL_HEIGHT // 2) + (THERMAL_HEIGHT // 2)] = last_fire_frame


    return frame

debug_screen_state = 0

def onMouseClick(event, x, y, flags, param):
    def didHitButton(x, y, width, height, mouse_x, mouse_y):
        return x <= mouse_x <= x + width and y <= mouse_y <= y + height

    global trigger_delay, trigger_duration

    if event == cv2.EVENT_LBUTTONDOWN:
        print(f'x: {x}, y: {y}\n')

        # if did hit left half
        if x < THERMAL_HEIGHT:
            global debug_screen_state
            debug_screen_state += 1
            if debug_screen_state > 3:
                debug_screen_state = 0

        third_width = THERMAL_HEIGHT // 3
        x_offset = THERMAL_HEIGHT
        if didHitButton(x_offset + 0, 40, third_width, third_width, x, y):
            trigger_delay -= 0.1
            if trigger_delay < 0:
                trigger_delay = 0
        if didHitButton(x_offset + third_width * 2, 40, third_width, third_width, x, y):
            trigger_delay += 0.1
        if didHitButton(x_offset + 0, 40 + third_width + 40, third_width, third_width, x, y):
            trigger_duration -= 0.1
            if trigger_duration < 0:
                trigger_duration = 0
        if didHitButton(x_offset + third_width * 2, 40 + third_width + 40, third_width, third_width, x, y):
            trigger_duration += 0.1
cv2.setMouseCallback('thermal_picture_colored', onMouseClick)
last_soup_thermal_frame = np.zeros((THERMAL_WIDTH, THERMAL_HEIGHT, 3), dtype=np.uint8)
last_thermal_frame = None
while 1:
    app_led.off()
    soup_led.off()
    frame_count += 1

    pi_frame = picam2.capture_array("main")
    frame_usb = thermalcamera.capture_array()
    add_frames_to_buffer(pi_frame, frame_usb)

    pi_frame = preprocess_pi_frame(pi_frame)

    pi_frame_fg = bgsub.apply(pi_frame)
    #pi_frame_mask[pi_frame_mask < 250] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    pi_frame_fg = cv2.dilate(pi_frame_fg, kernel, iterations=2)
    pi_frame_fg = cv2.erode(pi_frame_fg, kernel, iterations=2)

    pi_frame_mask, soups_rects = detect_soups(pi_frame_fg)

    thermal_frame = preprocess_thermal_frame(frame_usb)

    # DETECTING CALIBRATION EVENT
    # check if thermal_frame changed since last frame
    if np.array_equal(last_thermal_frame, thermal_frame):
        print('✋✋✋✋✋✋✋✋', end='\r')
        #continue
    last_thermal_frame = thermal_frame.copy()


    # check if last_soup_timestamp is older than 10 seconds
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
    for x, y, w, h in soups_rects:
        soup_frame = thermal_frame[y:y+h, x:x+w].copy()

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

        if((soup_min_t_z_score > 9999 or soup_min_t - np.nanmean(soup_min_t_deque) > TEMP_TRESHOLD) and len(soup_min_t_deque) > DEQUE_LEN / 2):
            fire_trigger(soup_min_t, soup_min_t_z_score, np.nanmean(soup_min_t_deque))

        last_soup_min_t_z_score = soup_min_t_z_score

        thermal_frame_without_soups[y:y+h, x:x+w] = np.nan

        # paste soup_frame into last_soup_thermal_frame
        last_soup_thermal_frame = np.zeros((THERMAL_WIDTH, THERMAL_HEIGHT, 3), dtype=np.uint8)
        last_soup_thermal_frame[y:y+h, x:x+w] = cv2.applyColorMap(cv2.normalize(np.clip(soup_frame, 20, 30), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U), cv2.COLORMAP_JET)

        last_soup_timestamp = datetime.datetime.now()




    # calculate mean temperature of thermal_frame but filter out 0 values
    mean_t_deque.append(np.nanmean(thermal_frame_without_soups))








    # overlay frame on thermal image
    thermal_picture_colored = cv2.applyColorMap(cv2.normalize(thermal_frame, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U), cv2.COLORMAP_JET)








    if (len(soup_min_t_deque) > DEQUE_LEN / 2):
        soup_min_t_deque_mean = np.nanmean(soup_min_t_deque)
        thermal_picture_colored = cv2.applyColorMap(cv2.normalize(np.clip(thermal_frame, 20, 30), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U), cv2.COLORMAP_JET)
        # dim down colors
        #thermal_picture_colored = cv2.addWeighted(thermal_picture_colored, 0.5, thermal_picture_colored, 0, 0)
        # set areas with np.nanmean(soup_min_t_deque) > 4 to 2, np.nanmean(soup_min_t_deque) < 4 to 0, else 1
        thermal_picture_bin = np.zeros(thermal_frame.shape, dtype=np.uint8)
        thermal_picture_bin[thermal_frame < soup_min_t_deque_mean - TEMP_TRESHOLD] = 255
        thermal_picture_bin[thermal_frame > soup_min_t_deque_mean + TEMP_TRESHOLD] = 0
        thermal_picture_bin[(thermal_frame >= soup_min_t_deque_mean - TEMP_TRESHOLD) & (thermal_frame <= soup_min_t_deque_mean + TEMP_TRESHOLD)] = 255

        # find in thermal_frame lowest temperature point
        min_t = np.nanmin(thermal_frame)
        min_t_pos = np.unravel_index(np.nanargmin(thermal_frame), thermal_frame.shape)

        # draw circle in the lowest temperature point on thermal_picture_colored with white color
        cv2.circle(thermal_picture_colored, (min_t_pos[1], min_t_pos[0]), 8, (255, 255, 255), 1)

    cv2.circle(thermal_picture_colored, (16, 16), 8, (0, 255, 255), 1)
    if len(soup_min_t_deque) > DEQUE_LEN / 2:
        cv2.circle(thermal_picture_colored, (16, 16), 8, (0, 255, 255), -1)


    # if last_soup_thermal_frame is not empty, toggle app_led
    cv2.circle(thermal_picture_colored, (16, 16 + 20), 8, (0, 255, 0), 1)
    if len(soups_rects) > 0:
        app_led.on()
        cv2.circle(thermal_picture_colored, (16, 16 + 20), 8, (0, 255, 0), -1)

    # draw red circle in top right corner to indicate fire detection
    cv2.circle(thermal_picture_colored, (16, 16 + 40), 8, (0, 0, 255), 1)
    if fire:
        cv2.circle(thermal_picture_colored, (16, 16 + 40), 8, (0, 0, 255), -1)
        soup_led.on()

    # choose based on debug_screen_state which frame to show
    left_image = np.zeros((THERMAL_WIDTH, THERMAL_HEIGHT, 3), dtype=np.uint8)
    right_image = np.zeros((THERMAL_WIDTH, THERMAL_HEIGHT, 3), dtype=np.uint8)
    if debug_screen_state == 0:
        left_image = thermal_picture_colored
        if last_soup_thermal_frame is not None:
            right_image = draw_mosaic()
    if debug_screen_state == 1:
        left_image = pi_frame
        right_image = draw_settings()
    if debug_screen_state == 2:
        left_image = cv2.cvtColor(pi_frame_fg, cv2.COLOR_GRAY2RGB)
    if debug_screen_state == 3:
        left_image = cv2.cvtColor(pi_frame_mask, cv2.COLOR_GRAY2RGB)
        right_image = last_soup_thermal_frame






    # calculate FPS

    # set timezone to Europe/Prague so datetime.now() returns correct time
    now = time.time()
    fps = 1 / (now - last_time)
    last_time = now
    print(f'fps: {fps:.2f}, soup_min_t: {np.nanmean(soup_min_t_deque):.2f}, soup_avg_t {np.nanmean(soup_avg_t_deque):.2f}', f'🍲 soup_min_t_z_score: {last_soup_min_t_z_score:.2f}' if len(soups_rects) > 0 else '  ', end='\r')


    # create doublewidth frame thermal_picture_colored
    double_width_frame = np.zeros((THERMAL_WIDTH, THERMAL_HEIGHT * 2, 3), dtype=np.uint8)
    double_width_frame[:, :THERMAL_HEIGHT, :] = left_image
    double_width_frame[:, THERMAL_HEIGHT:, :] = right_image
    # print fps on the frame
    cv2.putText(double_width_frame, f'{fps:.0f}', (6, double_width_frame.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if fps > 20 else (0, 0, 255), 2)
    # print last soup_min_t with avg mean of soupmint deque on the frame
    last_soup_min_t = soup_min_t_deque[-1] if len(soup_min_t_deque) > 0 else 0
    cv2.putText(double_width_frame, f'{last_soup_min_t:.2f} / {np.nanmean(soup_min_t_deque):.2f}', (6, double_width_frame.shape[0] - 6 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


    cv2.imshow('thermal_picture_colored', double_width_frame)




    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

picam2.stop()
relay_pin.off()
cv2.destroyAllWindows()

#save_frame_buffers_to_disk()
