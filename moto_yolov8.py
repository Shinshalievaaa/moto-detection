import cv2
import numpy as np

import time
import sys

from ultralytics import YOLO

# define some parameters
CONFIDENCE = 0.5
font_scale = 1
font_scale2 = 10
thickness = 1
labels = open("data/coco.names").read().strip().split("\n")
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

# loading the YOLOv8 model with the default weight file
model = YOLO("yolov8n.pt")

# read the file from the command line
video_file = sys.argv[1]
cap = cv2.VideoCapture(video_file)
_, image = cap.read()
h, w = image.shape[:2]
#fourcc = cv2.VideoWriter_fourcc(*"XVID") #сжатие видеофайла и сохранение его в виде кодеки Xvid
#out = cv2.VideoWriter("output.avi", fourcc, 20.0, (w, h))
while True:
    _, image = cap.read()
    
    start = time.perf_counter()
    results = model.predict(image, conf=CONFIDENCE)[0]
    time_took = time.perf_counter() - start
    #print("Time took:", time_took)

    # loop over the detections циклически перебирать обнаружения
    count_moto = 0
    count_moto_l = 0
    count_moto_r = 0
    for data in results.boxes.data.tolist():
        # get the bounding box coordinates, confidence, and class id получить координаты прямоугольника, достоверность и идентификатор класса
        xmin, ymin, xmax, ymax, confidence, class_id = data
        # converting the coordinates and the class id to integers
        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)
        xaverage = (xmin+xmax)/2
        class_id = int(class_id)
        print('xmin ',xmin)
        print('ymin ',ymin)
        print('xmax ',xmax)
        print('ymax ',ymax)

        if class_id == 3:
            
            count_moto  += 1
            if xaverage < 900:
                count_moto_l  += 1
            else:
                count_moto_r  += 1
            # draw a bounding box rectangle and label on the image нарисуйте прямоугольник ограничивающей рамки и метку на изображении
            color = [int(c) for c in colors[count_moto]]#class_id]]
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=color, thickness=thickness)
            text = f"{labels[class_id]} #{count_moto}"
            # calculate text width & height to draw the transparent boxes as background of the text
            (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=thickness)[0]
            text_offset_x = xmin
            text_offset_y = ymin - 5
            box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height))
            try:
                overlay = image.copy()
            except:
                break
            cv2.rectangle(overlay, box_coords[0], box_coords[1], color=color, thickness=cv2.FILLED)
            # add opacity (transparency to the box) добавить непрозрачность (прозрачность коробки)
            image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
            # now put the text (label: confidence %)
            cv2.putText(image, text, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=font_scale, color=(0, 0, 0), thickness=thickness)

    # end time to compute the fps
    end = time.perf_counter()
    # calculate the frame per second and draw it on the frame
    fps = f"FPS: {1 / (end - start):.2f}"
    #cv2.putText(image, fps, (50, 50),
    #            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 6)
    #out.write(image)
    text_left   = f"On the left: {count_moto_l}"
    text_center = f"Total motorcycles: {count_moto}"
    text_right   = f"On the right: {count_moto_r}"
    image[25:55, 50:295] = 119, 201, 105
    cv2.putText(image, text_left, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=font_scale, color=(0, 0, 0), thickness=thickness)
    image[25:55, 750:1100] = 119, 201, 105
    cv2.putText(image, text_center, (750, 50), cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=font_scale, color=(0, 0, 0), thickness=thickness)
    image[25:55, 1600:1870] = 119, 201, 105
    cv2.putText(image, text_right, (1600, 50), cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=font_scale, color=(0, 0, 0), thickness=thickness)
    cv2.imshow("image", image)
    
    if ord("q") == cv2.waitKey(1):
        break


cap.release()
cv2.destroyAllWindows()