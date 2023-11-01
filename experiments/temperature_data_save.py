import time
import uuid
from collections import deque

import cv2
import numpy as np
from skimage.feature import hog
from skimage.filters import threshold_multiotsu
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt


CAM_WIDTH = 256
CAM_HEIGHT = 384


vid = cv2.VideoCapture(0)
vid.set(cv2.CAP_PROP_CONVERT_RGB, 0)

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('thermal_output.avi', fourcc, 20.0, (CAM_WIDTH, CAM_HEIGHT//2), isColor=False)



while True:
    ret, frame = vid.read()

    frame_mid_pos = int(len(frame) / 2)
    picture_data = frame[0:frame_mid_pos]
    thermal_data = frame[frame_mid_pos:]
    yuv_picture = np.frombuffer(picture_data, dtype=np.uint8).reshape((CAM_HEIGHT // 2, CAM_WIDTH, 2))
    rgb_picture = cv2.cvtColor(yuv_picture, cv2.COLOR_YUV2RGB_YUY2)
    rgb_picture = cv2.rotate(rgb_picture, cv2.ROTATE_90_COUNTERCLOCKWISE)
    thermal_picture_16 = np.frombuffer(thermal_data, dtype=np.uint16).reshape((CAM_HEIGHT // 2, CAM_WIDTH))

    # Write the thermal_picture_16 frame to video
    out.write(thermal_picture_16)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release the video objects
vid.release()
out.release()