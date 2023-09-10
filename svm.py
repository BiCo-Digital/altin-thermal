import os
import pickle
from collections import deque

import itertools
import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
from skimage.exposure import exposure
from skimage.feature import local_binary_pattern, hog
from skimage.filters import threshold_multiotsu
from sklearn import svm
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
# create named window and move it - 400px in x dir
cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
# show window
cv2.imshow('Frame', np.zeros((256, 192, 3), np.uint8))
cv2.moveWindow("Frame", 20, 0)

# resize window
cv2.resizeWindow("Frame", 800, 600)


LEFT_GUIDE_X = 50
RIGHT_GUIDE_X = 140
BOTTOM_GUIDE_Y = 220

# sliders
cv2.createTrackbar('A', 'Frame', 1, 100, lambda x: None)
cv2.createTrackbar('B', 'Frame', 1, 100, lambda x: None)

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        # if contains debug, skip
        if 'debug' in filename:
            continue
        img = cv2.imread(os.path.join(folder,filename), cv2.IMREAD_COLOR)
        if img is not None:
            images.append(img)
    return images

images_in_folder = load_images_from_folder('fire/fireeee')
index = 0


# create deque of 3 elements
hot_samples = []
cold_samples = []

clf_loaded = None
with open('svm_model.pkl', 'rb') as f:
    clf_loaded = pickle.load(f)

cold_svm = None
#with open('svm_model_cold.pkl', 'rb') as f:
#    cold_svm = pickle.load(f)

# Capture the frame from the video source while 1
cap = cv2.VideoCapture('output8_clean.mp4')
while (cap.isOpened()):
    ret, frame = cap.read()

    index += 1
    if ret == True:

        # flip frame
        # frame = cv2.flip(frame, 0)

        #rotate frame 90 degrees
        #frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # crop frame
        # frame = frame[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]

        # resize to 192x256
        # frame = cv2.resize(frame, (192, 256), interpolation=cv2.INTER_AREA)



        # crop to guides
        original_frame = frame.copy()
        frame = frame[:, LEFT_GUIDE_X:RIGHT_GUIDE_X]
        frame = frame[:BOTTOM_GUIDE_Y, :]

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)




        frameRGB = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        #cv2.imshow('clahe', clahe)

        try:
            hot_mask = cv2.bilateralFilter(frame, 15, 30, 30)
            # reduce linespace to 0 - 10
            hot_mask[hot_mask < 200] = 0
            hot_mask[hot_mask >= 200] = 255
            # erode filate
            hot_mask = cv2.erode(hot_mask, np.ones((1, 7), np.uint8), iterations=1)
            hot_mask = cv2.dilate(hot_mask, np.ones((1, 7), np.uint8), iterations=1)

            contours, _ = cv2.findContours(hot_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            hot_mask = np.zeros(frame.shape, np.uint8)
            for c in contours:
                hull = cv2.convexHull(c)
                area = cv2.contourArea(hull)
                rect = cv2.minAreaRect(c)
                box = np.int0(cv2.boxPoints(rect))
                x = box[0][0]

                w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
                if h > w:
                    w, h = h, w

                # predict with iso forest if it fit

                if area > 100 and w > 1 and h > 1:
                    cv2.drawContours(hot_mask, [hull], 0, (255, 255, 255), -1)
                    cv2.drawContours(frameRGB, [hull], 0, (0, 0, 255), 2)
                    hot_samples.append([w, h])

                # predict with svm
                if clf_loaded is not None:
                    prediction = clf_loaded.predict([[w, h]])
                    if prediction == 1:
                        cv2.drawContours(frameRGB, [hull], 0, (0, 255, 0), 2)


                    #x, y, w, h = cv2.boundingRect(hull)
                    #cv2.rectangle(frameRGB, (x, y), (x + w, y + h), (0, 0, 255), 2)


            # Apply linear contrast stretching
            frame = cv2.convertScaleAbs(frame, alpha=-2.1, beta=50)
            #frame = np.array(255 * (frame / 255) ** A / 10, dtype='uint8')
            #frameRGB = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

            cold_mask = cv2.bilateralFilter(frame, 15, 30, 30)
            #cold_mask = cv2.medianBlur(cold_mask, 15)
            #mean = cold_mask.copy()
            cold_mask = cv2.bitwise_not(cold_mask)
            bilateral = cold_mask.copy()


            cold_mask[cold_mask < 200] = 0
            cold_mask[cold_mask >= 200] = 255

            contours, _ = cv2.findContours(cold_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cold_mask = np.zeros(frame.shape, np.uint8)

            for c in contours:
                approx = cv2.approxPolyDP(c, 0.058 * cv2.arcLength(c, True), True)
                hull = cv2.convexHull(approx)
                area = cv2.contourArea(approx)
                bbox = cv2.boundingRect(approx)

                w = bbox[2]
                h = bbox[3]
                aspect_ratio = float(w) / float(h)
                x = bbox[0]


                if area > 250:
                    cv2.drawContours(cold_mask, [hull], 0, (255, 255, 255), -1)
                    cv2.drawContours(frameRGB, [hull], 0, (255, 0, 0), 2)
                    cold_samples.append([x, aspect_ratio])


            #TODO: DEBUG
            if True:
                # for debugging, show hot_mask and cold_mask on same window
                hot_mask = cv2.cvtColor(hot_mask, cv2.COLOR_GRAY2BGR)
                cold_mask = cv2.cvtColor(cold_mask, cv2.COLOR_GRAY2BGR)
                bilateral = cv2.cvtColor(bilateral, cv2.COLOR_GRAY2BGR)
                #mean = cv2.cvtColor(mean, cv2.COLOR_GRAY2BGR)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                original_frame[:frameRGB.shape[0], LEFT_GUIDE_X:LEFT_GUIDE_X + frameRGB.shape[1]] = frameRGB

                frame = np.concatenate((frameRGB, hot_mask, cold_mask, bilateral), axis=1)

                # create big canvas 800x600 and put all images on it
                canvas = np.zeros((300, 600, 3), np.uint8)
                canvas[:frame.shape[0], :frame.shape[1]] = frame
                canvas[:original_frame.shape[0], 400:400 + original_frame.shape[1]] = original_frame

                frame = canvas



        except Exception as e:
            print(e)
            pass



        cv2.imshow('Frame', frame)


        # print number of frame
        print(cap.get(cv2.CAP_PROP_POS_FRAMES) // 25 , cap.get(cv2.CAP_PROP_FRAME_COUNT) // 25,
              hot_samples[-1] if len(hot_samples) > 0 else None,
                cold_samples[-1] if len(cold_samples) > 0 else None
              )

        continue
        if cv2.waitKey(20) & 0xFF == ord(' '):
            while True:
                if cv2.waitKey(5) & 0xFF == ord(' '):
                    break
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break

    # Break the loop
    else:
        break

cap.release()



# Step 1: Data Preparation
# Replace with your actual data, use 90% of your data for training, 10% for testing
data = np.array(cold_samples)
test = data[int(0.9 * len(data)):]
data = data[:int(0.9 * len(data))]

#importdo svm
import sklearn.svm as svm

#do outlier detection using fast svm
clf = svm.OneClassSVM(nu=0.0002, kernel="rbf", gamma=0.03)
clf.fit(data)

#predict
y_pred_train = clf.predict(data)


#if 2D
if data[0].shape[0] == 2:
    plt.scatter(data[y_pred_train == 1, 0],data[y_pred_train == 1, 1], c='b', alpha=0.3)
    plt.scatter(data[y_pred_train == -1, 0],data[y_pred_train == -1, 1], c='r', alpha=0.3)
    # draw boundaries of decision region from svm, data is in range 0 - 100
    xx, yy = np.meshgrid(np.linspace(10, 40, 500), np.linspace(0, 8, 500))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='green')

    plt.show()

elif data[0].shape[0] == 3:

    #if 3D
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(data[y_pred_train == 1, 0], data[y_pred_train == 1, 1], data[y_pred_train == 1, 2], c='b', alpha=0.3)
    ax.scatter(data[y_pred_train == -1, 0], data[y_pred_train == -1, 1], data[y_pred_train == -1, 2], c='r', alpha=0.3)



    plt.show()



# save svm to file using pickle
with open('svm_model_cold.pkl', 'wb') as f:
    pickle.dump(clf, f)





