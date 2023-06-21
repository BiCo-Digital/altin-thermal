
import cv2
import numpy as np


P2Pro_resolution = (256, 384)
P2Pro_fps = 25.0

vid = cv2.VideoCapture(0)
vid.set(cv2.CAP_PROP_CONVERT_RGB, 0)


# Create a named window
cv2.namedWindow("Thermal Frame", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Thermal Frame", 800, 600)


while vid.isOpened():
    empty, frame = vid.read()

    # split video frame (top is pseudo color, bottom is temperature data)
    frame_mid_pos = int(len(frame) / 2)
    picture_data = frame[0:frame_mid_pos]
    thermal_data = frame[frame_mid_pos:]

    # convert buffers to numpy arrays
    yuv_picture = np.frombuffer(picture_data, dtype=np.uint8).reshape((P2Pro_resolution[1] // 2, P2Pro_resolution[0], 2))
    rgb_picture = cv2.cvtColor(yuv_picture, cv2.COLOR_YUV2RGB_YUY2)
    thermal_picture_16 = np.frombuffer(thermal_data, dtype=np.uint16).reshape((P2Pro_resolution[1] // 2, P2Pro_resolution[0]))
    thermal_picture_8 = cv2.normalize(thermal_picture_16, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    colored_image = cv2.applyColorMap(thermal_picture_8, cv2.COLORMAP_INFERNO)



    cv2.imshow("Thermal Frame", colored_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
print('done')