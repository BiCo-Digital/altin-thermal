import cv2
import time
import numpy as np


# Create named fullscreen window
cv2.namedWindow('window', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('window', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.moveWindow('window', -1, -1)

P2Pro_resolution = (256, 384)
P2Pro_fps = 25.0
cap = cv2.VideoCapture(0)  # Using default camera
cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)  # Get raw data from the camera

if not cap.isOpened():
    print("Error: Couldn't open the camera.")
    exit()

frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Couldn't read a frame.")
        break


    # Calculate FPS every 100 frames for better accuracy
    frame_count += 1
    if frame_count % 10 == 0:
        end_time = time.time()
        elapsed_time = end_time - start_time
        fps = frame_count / elapsed_time

        print(f"Real-time FPS: {fps:.2f}")

        # Reset for the next set of 100 frames
        frame_count = 0
        start_time = time.time()


    # Rotate the frame 90 degrees clockwise
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    # Split the frame into two parts
    frame_mid_pos = int(len(frame) / 2)
    picture_data = frame[0:frame_mid_pos]
    thermal_data = frame[frame_mid_pos:]










    # Display the frame (optional)
    cv2.imshow('window', picture_data)

    # Break out of the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
