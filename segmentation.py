import sys
import cv2
import numpy as np

width, height = 800, 600
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

font = cv2.FONT_HERSHEY_SIMPLEX
put_text_color = (18, 0, 255)
put_text_pos = (60, 50)

lower_thresh1 = 127
upper_thresh1 = 255

while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    # cv2.rectangle(img, (60, 60), (300, 300), (255, 255, 2), 4)  # outer most rectangle
    # crop_img = img[70:300, 70:300]
    # crop_img_2 = img[70:300, 70:300]
    img = cv2.rectangle(img, (425, 100), (625, 300), (0, 255, 0), thickness=2, lineType=8, shift=0)
    crop_img = img[102:298, 427:623]

    grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    value = (35, 35)
    blurred = cv2.GaussianBlur(grey, value, 0)
    _, thresh1 = cv2.threshold(blurred, lower_thresh1, upper_thresh1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    cv2.imshow("original", img)
    cv2.imshow("thresh1", thresh1)

    cv2.waitKey(10)
