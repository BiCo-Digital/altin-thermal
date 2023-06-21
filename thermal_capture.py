import cv2

CAP_AVFOUNDATION = 1200
CAP_DSHOW = 700
CAP_ANY = 0
# OpenCV video capture
cap = cv2.VideoCapture(0)

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
print('w: {}, h: {}, fps: {}'.format(w, h, fps))

cap.set(cv2.CAP_PROP_FPS, 25)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 256)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 384)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y','U','Y','V'))

while True:
    # Read frame from video capture
    ret, frame = cap.read()

    if not ret:
        continue

    cv2.imshow('Frame', frame)
    continue

    # Crop bottom half of the frame
    height, width, _ = frame.shape
    cropped_frame = frame[int(height/2):height, :]

    # Convert pixel format to yuyv422
    yuyv422_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2YUV_Y422)

    # Display grayscale version of the frame
    gray_frame = cv2.cvtColor(yuyv422_frame, cv2.COLOR_YUV2GRAY_10)
    cv2.imshow('Grayscale', gray_frame)

    if cv2.waitKey(1) == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()