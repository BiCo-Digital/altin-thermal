import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
from past.builtins import raw_input

LEFT_GUIDE_X = 10
RIGHT_GUIDE_X = 140
BOTTOM_GUIDE_Y = 220

DIR_NAME = "/Users/matejnevlud/github/altin-thermal/thermal_images_2023-09-18_07-41-20"
thermal_images = []
filenames = os.listdir(DIR_NAME)
# filenames are in format frame_0000.png, frame_0001.png, ...
# sort them by frame number
filenames = [filename for filename in filenames if filename.endswith('.png')]
filenames.sort(key=lambda x: int(x[6:-4]))

for filename in filenames:
    thermal_images.append(cv2.imread(os.path.join(DIR_NAME, filename), cv2.IMREAD_ANYDEPTH))

# use plt with ion to display the image
plt.ion()
plt.show()

# create figure and axes with 2 plots, one in 3D
#fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 5))
fig = plt.figure(figsize=(15, 9))
ax = fig.add_subplot(121)
ax2 = fig.add_subplot(122, projection='3d')





frame_count = 0
while 1:
    frame = thermal_images[frame_count]
    frame_count += 1

    # rotate 90 degrees clockwise
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    frame = frame >> 2
    frame = frame / 16
    frame = frame - 273.15
    frame = frame.astype(np.float32)

    frame = cv2.GaussianBlur(frame, (7, 7), 0)

    #frame = frame[:, LEFT_GUIDE_X:RIGHT_GUIDE_X]
    #frame = frame[:BOTTOM_GUIDE_Y, :]

    min_t = np.min(frame)
    mean_t = np.mean(frame)
    max_t = np.max(frame)

    hist_values, hist_bins = np.histogram(frame, bins=100)
    peak_index = np.argmax(hist_values)
    modus_t = hist_bins[peak_index]


    # get only the values Above hot 50 deg
    hot_frame = frame.copy()
    hot_frame[hot_frame < 50] = 0
    hot_frame[hot_frame >= 50] = 255
    contours, hierarchy = cv2.findContours(hot_frame.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hot_frame = np.zeros_like(hot_frame)
    hot_lines = []
    for contour in contours:
        [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
        bbox = cv2.boundingRect(contour)
        # get the line equation y = ax + b
        a = vy / vx
        b = y - a * x
        # get the line points
        x1 = bbox[0]
        y1 = a * x1 + b
        x2 = bbox[0] + bbox[2]
        y2 = a * x2 + b
        hot_lines.append(((x1, y1), (x2, y2)))


    # iterate over hotlines in pairs, using 4 corner points crop the image
    soup_candidates = []
    for i in range(len(hot_lines) - 1):
        p0, p1 = hot_lines[i]
        p2, p3 = hot_lines[i + 1]

        # crop the frame using the 4 points and transform it to 100x100
        src = np.array([p0, p1, p2, p3], dtype=np.float32)
        dst = np.array([[0, 0], [100, 0], [0, 100], [100, 100]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        cropped_frame = cv2.warpPerspective(frame, M, (100, 100))
        cropped_frame = np.clip(cropped_frame, min_t, mean_t)

        continue
        mean_crop_t = np.mean(cropped_frame)
        std_crop_t = np.std(cropped_frame)
        var_crop_t = np.var(cropped_frame)
        min_crop_t = np.min(cropped_frame)
        max_crop_t = np.max(cropped_frame)
        mean_to_min_t = mean_t - min_crop_t

        print(mean_crop_t, std_crop_t, var_crop_t, mean_to_min_t)
        #cropped_frame = (cropped_frame - min_t) / (max_t - min_t) * 255
        #cropped_frame = cropped_frame.astype(np.uint8)

        ax.cla()
        ax.imshow(cropped_frame, cmap='jet')


        X, Y = np.meshgrid(np.arange(0, cropped_frame.shape[1], 1), np.arange(0, cropped_frame.shape[0], 1))
        Z = cropped_frame
        ax2.cla()
        ax2.plot_surface(X, Y, Z, cmap='jet', antialiased=False)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')

        # print the mean temperature, std and var in the top left corner
        ax2.text2D(0.05, 0.95, f"mean: {mean_crop_t:.2f}\nmin_crop_t: {min_crop_t:.2f}\nmean - min: {mean_to_min_t:.2f}", transform=ax2.transAxes)
        plt.waitforbuttonpress()


    # crop the frame using min x and max x of the hot lines
    min_x = np.min([np.min([hot_line[0][0], hot_line[1][0]]) for hot_line in hot_lines])
    max_x = np.max([np.max([hot_line[0][0], hot_line[1][0]]) for hot_line in hot_lines])

    # instead of cropping the frame, set all the values outside the hot lines (min_x, max_x) to the closest value inside the hot lines (min_x, max_x)
    #soups_frame = frame[:, min_x:max_x]
    soups_frame = frame.copy()
    soups_frame = np.clip(soups_frame, min_t, 50)
    for x in range(soups_frame.shape[1]):
        if x < min_x:
            soups_frame[:, x] = soups_frame[:, min_x]
        elif x > max_x:
            soups_frame[:, x] = soups_frame[:, max_x]












    # Find the half maximum value
    half_maximum_value = hist_values[peak_index] / 3

    # Find the two points where the histogram values cross the half maximum value
    left_crossing_index = np.where(hist_values[:peak_index] < half_maximum_value)[0][-1]
    right_crossing_index = np.where(hist_values[peak_index:] < half_maximum_value)[0]
    right_crossing_index = right_crossing_index[0] + peak_index if right_crossing_index.size > 0 else len(hist_values) - 1

    right_crossing_t = hist_bins[right_crossing_index]
    left_crossing_t = hist_bins[left_crossing_index]



    ax.cla()
    ax.imshow(soups_frame, cmap='jet')






    frame = soups_frame
    frame = np.clip(frame, min_t, 50)

    X = np.arange(0, frame.shape[1], 1)
    Y = np.arange(0, frame.shape[0], 1)
    X, Y = np.meshgrid(X, Y)
    Z = frame

    ax2.cla()
    ax2.plot_surface(X, Y, Z, cmap='jet', antialiased=False)

    #for hot_line in hot_lines:
        #ax2.plot([hot_line[0][0], hot_line[1][0]], [hot_line[0][1], hot_line[1][1]], zs=50, color='r', linewidth=3)








    #if plt.waitforbuttonpress() == False:
    #    break
    plt.pause(0.01)
    plt.draw()
    plt.gcf().canvas.mpl_connect('key_press_event', lambda event: [exit(0) if event.key == 'escape' or event.key == 'q' else None])
    #plt.cla()
    #plt.waitforbuttonpress()




