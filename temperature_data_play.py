import os

import cv2
import numpy as np
import matplotlib.pyplot as plt


CAM_WIDTH = 256
CAM_HEIGHT = 384
LEFT_GUIDE_X = 55
RIGHT_GUIDE_X = 140
BOTTOM_GUIDE_Y = 220

# Open the video file
cap = cv2.VideoCapture('2023-09-14_13-22-09_thermal_u16.mkv', apiPreference=cv2.CAP_FFMPEG)


# load thermal images from folder in png as 16 bit grayscale
thermal_images = []
filenames = os.listdir('/Users/matejnevlud/github/altin-thermal/orkla1')
# filenames are in format frame_0000.png, frame_0001.png, ...
# sort them by frame number
filenames = [filename for filename in filenames if filename.endswith('.png')]
filenames.sort(key=lambda x: int(x[6:-4]))

for filename in filenames:
    thermal_images.append(cv2.imread(os.path.join('thermal_images', filename), cv2.IMREAD_ANYDEPTH))



# named window for displaying video and 2 trackbars
cv2.namedWindow('Thermal Video')
cv2.namedWindow('Thermal Frame')

# create trackbars for adjusting the temperature range
cv2.createTrackbar('A', 'Thermal Video', 1, 400, lambda x: None)
cv2.createTrackbar('B', 'Thermal Video', 1, 400, lambda x: None)

#os.makedirs('thermal_images', exist_ok=True)

frame_count = 0
while True:
    # Read a frame from the video file
    frame = thermal_images[frame_count]
    #ret, frame =  cap.read()





    # rotate 90 degrees clockwise
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    frame_count += 1



    ## use (x >> 2) / 16 - 273 to convert to celsius
    frame = frame >> 2
    frame = frame / 16
    frame = frame - 273.15
    frame = frame.astype(np.float32)




    min_t = np.min(frame)
    mean_t = np.mean(frame)
    max_t = np.max(frame)
    print(min_t, mean_t, max_t)

    # crop to guides
    frame = frame[:, LEFT_GUIDE_X:RIGHT_GUIDE_X]
    frame = frame[:BOTTOM_GUIDE_Y, :]

    min_crop_t = np.min(frame)
    mean_crop_t = np.mean(frame)
    max_crop_t = np.max(frame)

    cold_frame = frame.copy()
    cold_frame = np.clip(cold_frame, min_t, mean_t)
    cold_frame = cv2.normalize(cold_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cold_frame = cv2.applyColorMap(cold_frame, cv2.COLORMAP_JET)

    cv2.imshow('Thermal Fr', cold_frame)
    cv2.waitKey(0)
    continue


    # the frame is in propietary format, holding data in kelvin, using equation t = x / 64 - 273
    # we can convert it to celsius

    #frame -= 25 # adjust for the room temperature

    #frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    #frame[frame > 50] = 255


    cold_frame = frame.copy()
    cold_frame = np.clip(cold_frame, min_t, mean_t)
    plt.imshow(cold_frame, cmap='hot', interpolation='nearest')
    plt.show()
    cold_frame = cv2.normalize(cold_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cold_frame = cv2.bitwise_not(cold_frame)
    cold_frame = cv2.bilateralFilter(cold_frame, 15, 30, 30)
    # otsu thresholding
    _, cold_frame = cv2.threshold(cold_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cold_frame = cv2.morphologyEx(cold_frame, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)





    frame_normalized = ((frame - 25) / (35 - 25) * 255).astype(np.uint8)
    frame_normalized[frame < 25] = 0
    frame_normalized[frame > 35] = 255
    frame = frame_normalized
    frame = cv2.bitwise_not(frame)


    # Display the frame
    cv2.imshow('Thermal Video', cold_frame)
    cv2.imshow('Thermal Frame', frame)

    # Wait for 30 milliseconds before moving to the next frame
    # Adjust the value to control the playback speed
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
