#!/usr/bin/env python3

import cv2 as cv

vcap = cv.VideoCapture("rtsp://192.168.1.10/color")

count = 0
while True:
    ret, frame = vcap.read()
    cv.imshow("test_mode", frame)
    
    cv.imwrite("images/frame%d.jpg" % count, frame)
    
    if count > 3:
        break

    count += 1



