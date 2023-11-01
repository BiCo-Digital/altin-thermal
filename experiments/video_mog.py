import cv2
import numpy as np


# Load video capture from file
cap = cv2.VideoCapture('thermal.mp4')




#fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
fgbg = cv2.bgsegm.createBackgroundSubtractorLSBP()
fgbg = cv2.bgsegm.createBackgroundSubtractorGSOC()
fgbg = cv2.bgsegm.createBackgroundSubtractorGMG(20, 0.7)
fgbg = cv2.createBackgroundSubtractorMOG2(100, 40)
fgbg = cv2.bgsegm.createBackgroundSubtractorCNT(2, True)


def display(window_name, image, grid_pos = (0, 0)):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, image)
    cv2.resizeWindow(window_name, 384, 512)
    cv2.moveWindow(window_name, 384 * grid_pos[0], (512 + 30) * grid_pos[1])

while(cap.isOpened()):
    ret, frame = cap.read()
    display('RGB Frame', frame, (0, 0))

    # resize to 192x256
    frame = cv2.resize(frame, (192, 256))


    # convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # gaussian blur
    gauss = cv2.GaussianBlur(gray, (3, 3), 0)
    display('Gaussian Blur', gauss, (1, 0))

    # threshold only hot areas using adaptive thresholding
    _, mask = cv2.threshold(gauss, 180, 255, cv2.THRESH_BINARY)
    display('Threshold', mask, (2, 0))

    # extract horizontal lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    display('Morphology', mask, (3, 0))

    # find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    display('Contours', mask, (0, 1))


    # get bounding boxes in a list
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    centers = [(int(x + w / 2), int(y + h / 2)) for (x, y, w, h) in bounding_boxes]
    for (x, y, w, h) in bounding_boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0))
    for (x, y) in centers:
        cv2.circle(frame, (x, y), 3, (0, 0, 0))

    # remove outliers using distance from center of frame, calculate mean and std
    middle = frame.shape[1] / 2
    distances = [abs(x - middle) for (x, y) in centers]
    mean = np.mean(distances)
    std = np.std(distances)

    # remove outliers only if std is high enough
    if std > 3:
        bounding_boxes = [bounding_boxes[i] for i in range(len(bounding_boxes)) if distances[i] < mean + 2 * std]

    # sort bounding boxes by y coordinate
    bounding_boxes.sort(key=lambda contour: contour[1])

    # join bounding boxes that are overlaping or close to each other on y axis
    for i in range(len(bounding_boxes) - 1):
        if i + 1 >= len(bounding_boxes):
            break

        (x1, y1, w1, h1) = bounding_boxes[i]
        (x2, y2, w2, h2) = bounding_boxes[i + 1]


        # if bounding boxes are close to each other on y axis
        if abs(y1 - y2) < 10:
            # join them together
            min_x = min(x1, x2)
            min_y = min(y1, y2)
            max_x = max(x1 + w1, x2 + w2)
            max_y = max(y1 + h1, y2 + h2)
            bounding_boxes[i] = (min_x, min_y, max_x - min_x, max_y - min_y)

            # remove second bounding box
            bounding_boxes.pop(i + 1)



    # draw bounding boxes
    for (x, y, w, h) in bounding_boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
    display('Bounding bnpxe', frame, (1, 1))


    # create list of bounding boxes pairs between each other
    bounding_boxes_pairs = [(bounding_boxes[i], bounding_boxes[i + 1]) for i in range(len(bounding_boxes) - 1)]

    # draw bounding boxes pairs, switch between red and blue
    big_boxes = []
    for i in range(len(bounding_boxes_pairs)):
        (x1, y1, w1, h1) = bounding_boxes_pairs[i][0]
        (x2, y2, w2, h2) = bounding_boxes_pairs[i][1]

        # create bounding box between two bounding boxes
        big_box = (min(x1, x2), min(y1, y2) + h1, max(x1 + w1, x2 + w2), y2)
        big_box_area = (big_box[2] - big_box[0]) * (big_box[3] - big_box[1])

        if big_box_area < 0:
            continue
        big_boxes.append(big_box)



    def my_otsu(gray_images):
        # Výpočet histogramu pro všechny obrázky
        histogram = np.zeros((256,))
        for image in gray_images:
            if image.size > 0: # Přidáno zkontrolování, jestli je obrázek prázdný
                hist = cv2.calcHist([image], [0], None, [256], [0,256])
                histogram += hist.flatten()

        # Normalizace histogramu
        histogram /= histogram.sum()

        # Výpočet kumulativní sumy a střední hodnoty
        cumulative_sum = np.cumsum(histogram)
        cumulative_mean = np.cumsum(histogram * np.arange(256))

        # Výpočet celkové střední hodnoty
        global_mean = cumulative_mean[-1]

        # Výpočet mezi třídní variance
        between_class_variance = (global_mean * cumulative_sum - cumulative_mean)**2 / (cumulative_sum * (1 - cumulative_sum) + 1e-10) # Přidáno 1e-10, aby se zabránilo dělení nulou

        # Ošetření případů, kdy je variance NaN nebo inf
        between_class_variance[np.isnan(between_class_variance)] = 0
        between_class_variance[np.isinf(between_class_variance)] = 0

        # Nalezení prahu
        optimal_threshold = np.argmax(between_class_variance)

        # Binarizace všech obrázků
        binary_images = []
        for image in gray_images:
            if image.size > 0: # Přidáno zkontrolování, jestli je obrázek prázdný
                _, binary_image = cv2.threshold(image, optimal_threshold, 255, cv2.THRESH_BINARY)
                binary_image = cv2.erode(binary_image, np.ones((3, 3)), iterations=1)
                binary_image = cv2.dilate(binary_image, np.ones((3, 3)), iterations=1)
                binary_images.append(binary_image)

        return binary_images


    # extract picture data from big boxes and paste them into new image
    big_boxes_masks = []
    for i in range(len(big_boxes)):
        (x1, y1, x2, y2) = big_boxes[i]
        big_boxes_masks.append(gauss[y1:y2, x1:x2])

    # Binarizace všech velkých boxů
    big_boxes_masks = my_otsu(big_boxes_masks)

    # TODO DEBUG Vykreslení všech velkých boxů do původního frame
    gray_copy = gray.copy()
    for i in range(len(big_boxes)):
        (x1, y1, x2, y2) = big_boxes[i]
        gray_copy[y1:y2, x1:x2] = big_boxes_masks[i]
    display('Bounding Boxes Pairs', gray_copy, (2, 1))

    # TODO DEBUG Find max width of bounding boxes
    max_width = max([image.shape[1] for image in big_boxes_masks])
    sum_height = sum([image.shape[0] for image in big_boxes_masks])
    debug_box_frame = np.zeros((sum_height, max_width), dtype=np.uint8)
    last_x = 0
    for bb in big_boxes_masks:
        bb_pos = big_boxes[big_boxes_masks.index(bb)]
        debug_box_frame[last_x:bb.shape[0] + last_x, 0:bb.shape[1]] = gray[bb_pos[1]:bb_pos[3], bb_pos[0]:bb_pos[2]]
        last_x += bb.shape[0]
    display('Debug Box Frame', debug_box_frame, (3, 1))


    # calculate mean and std using cv2.meanStdDev
    big_boxes_average = []
    big_boxes_std = []
    for i in range(len(big_boxes_masks)):
        mean, std = cv2.meanStdDev(big_boxes_masks[i])
        big_boxes_average.append(mean[0][0])
        big_boxes_std.append(std[0][0])

    # TODO DEBUG Vykreslení průměrných hodnot do původního frame
    gray_copy = gray.copy()
    for i in range(len(big_boxes)):
        (x1, y1, x2, y2) = big_boxes[i]
        cv2.putText(gray_copy, str(round(big_boxes_std[i], 2)), (x1, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255)
        cv2.putText(gray_copy, str(round(big_boxes_average[i], 2)), (0, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255)
    display('Bounding Boxes Pairs', gray_copy, (2, 1))















    # Find cold dark rectangles, their position is between hot rectangles pair



















    k = cv2.waitKey(0) & 0xff
    # if q
    if k == ord('q'):
        break

    # if space
    if k == ord('c'):
        cv2.waitKey(5000)

cap.release()
cv2.destroyAllWindows()

