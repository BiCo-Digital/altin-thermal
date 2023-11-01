import os

import cv2
import numpy as np
import matplotlib.pyplot as plt


FILEDIR = 'thermal_images_10_20'

CAM_WIDTH = 256
CAM_HEIGHT = 384
LEFT_GUIDE_X = 55
RIGHT_GUIDE_X = 140
BOTTOM_GUIDE_Y = 220

# Open the video file
cap = cv2.VideoCapture('video.mkv', apiPreference=cv2.CAP_FFMPEG)

# load thermal images from folder in png as 16 bit grayscale
thermal_images = []
filenames = os.listdir(FILEDIR)
# filenames are in format frame_0000.png, frame_0001.png, ...
# sort them by frame number
filenames.sort(key=lambda x: int(x[6:-4]))

for filename in filenames:
    thermal_images.append(cv2.imread(os.path.join(FILEDIR, filename), cv2.IMREAD_ANYDEPTH))



# named window for displaying video and 2 trackbars
cv2.namedWindow('Thermal Video')
cv2.namedWindow('Thermal Frame')

# create trackbars for adjusting the temperature range
cv2.createTrackbar('A', 'Thermal Video', 1, 400, lambda x: None)
cv2.createTrackbar('B', 'Thermal Video', 1, 400, lambda x: None)


def draw_histogram(hist_values):
    # Create a blank image (white background)
    height, width = 600, 800
    hist_img = np.ones((height, width)) * 255

    # Find the maximum frequency to normalize the histogram bars
    max_freq = np.max(hist_values)

    # Get the width of each bin
    bin_width = width // len(hist_values)

    # Draw the histogram bars on the blank image
    for i, val in enumerate(hist_values):
        # Normalize the bar height
        bar_height = int((val / max_freq) * height)

        # Define the coordinates of the top left and bottom right corners of the rectangle representing the bar
        pt1 = (i * bin_width, height - bar_height)
        pt2 = ((i + 1) * bin_width, height)

        # Set the color to white for the bars and red for the isolated region
        color = (0) if left_crossing_index <= i <= peak_index else (200)

        # Draw the rectangle (bar) on the image
        cv2.rectangle(hist_img, pt1, pt2, color, -1)

    # Display the image using OpenCV
    cv2.imshow('Histogram', hist_img)



frame_count = 0
while True:
    # Read a frame from the video file
    frame = thermal_images[frame_count]
    # rotate 90 degrees clockwise
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    frame_count += 1



    ## use (x >> 2) / 16 - 273 to convert to celsius
    frame = frame >> 2
    frame = frame / 16
    frame = frame - 273
    frame = frame.astype(np.float32)



    min_t = np.min(frame)
    mean_t = np.mean(frame)
    mod_t = np.median(frame)
    modus_t = np.mod(frame, 1)
    max_t = np.max(frame)
    print(min_t, mean_t, max_t)


    frame = np.clip(frame, min_t, mean_t)
    #frame = cv2.bilateralFilter(frame, 5, 5, 5)
    #frame = cv2.GaussianBlur(frame, (7, 7), 0)
    frame = cv2.medianBlur(frame, 5)
    # sharpen image
    #frame = cv2.filter2D(frame, -1, np.array([[-1, -1, -1], [-1, 5, -1], [-1, -1, -1]]))


    # iterate over rows and color each pixel in row with minimum temperature in row
    for row in range(frame.shape[0]):
        min_row_t = np.min(frame[row])
        #frame[row] = min_row_t


    plt.imshow(frame, cmap='gray', interpolation='nearest')
    plt.show()
    break

    # histogram calc plt
    hist_values, hist_bins = np.histogram(frame, bins=100, range=(min_t, 49))

    peak_index = np.argmax(hist_values)
    peak_crop_t = hist_bins[peak_index]

    # Find the half maximum value
    half_maximum_value = hist_values[peak_index] / 2

    # Find the two points where the histogram values cross the half maximum value
    left_crossing_index = np.where(hist_values[:peak_index] < half_maximum_value)[0][-1]
    right_crossing_index = np.where(hist_values[peak_index:] < half_maximum_value)[0]
    right_crossing_index = right_crossing_index[0] + peak_index if right_crossing_index.size > 0 else len(hist_values) - 1

    left_crossing_t = hist_bins[left_crossing_index]


    # use plt to show histogram
    plt.plot(hist_bins[:-1], hist_values)
    plt.axvline(x=left_crossing_t, color='r')
    plt.axvline(x=peak_crop_t, color='g')
    plt.show()







    # the crop is in propietary format, holding data in kelvin, using equation t = x / 64 - 273
    # we can convert it to celsius

    #crop -= 25 # adjust for the room temperature

    #crop = cv2.normalize(crop, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    #crop[crop > 50] = 255


    cold_crop = crop.copy()
    cold_crop = np.clip(cold_crop, min_t, left_crossing_t)
    cold_crop = cv2.normalize(cold_crop, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cold_crop = cv2.bitwise_not(cold_crop)
    #cold_crop = cv2.bilateralFilter(cold_crop, 15, 30, 30)
    cold_crop = cv2.GaussianBlur(cold_crop, (11, 11), 0)


    # treshold using triangle method
    #_, cold_crop = cv2.threshold(cold_crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
    #_, cold_crop = cv2.threshold(cold_crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cold_crop = cv2.morphologyEx(cold_crop, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

    # find contours
    contours, _ = cv2.findContours(cold_crop, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #for c in contours:
        #cv2.drawContours(cold_crop, [c], -1, (255, 255, 255), -1)










    crop = np.clip(crop, mean_crop_t, max_crop_t)

    crop = cv2.normalize(crop, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)


    # Display the frame
    cv2.imshow('Cold Frame', cold_crop)
    cv2.imshow('Thermal Frame', frame)

    # Wait for 30 milliseconds before moving to the next frame
    # Adjust the value to control the playback speed
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break


# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
