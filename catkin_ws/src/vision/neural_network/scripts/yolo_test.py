#!/usr/bin/env python
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import cv2

def main():
    cap = cv2.VideoCapture(0)
    model = YOLO()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        results = model.predict(frame)
        img_result = results[0].plot()
        color_list = [] # fill with color in (R, G, B)
        cv2.imshow('My Video', img_result)
        if cv2.waitKey(10) & 0xFF == 27:
            break

if __name__ == "__main__":
    main()
