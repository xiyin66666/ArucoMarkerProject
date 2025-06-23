import cv2
import cv2.aruco as aruco
import numpy as np
import os
import ArucoModule as arm

cap = cv2.VideoCapture(0)
augDics = arm.loadAugImages("Markers")

while True:
    success, img = cap.read()
    arucoFind = arm.findArucoMarker(img)

    # Loop through all the markers and augment each one
    if len(arucoFind[0]) != 0:
        for bbox,id in zip(arucoFind[0],arucoFind[1]):
            if int(id) in augDics.keys():
                img = arm.augmentAruco(bbox,id,img,augDics[int(id)])

    cv2.imshow('Image', img)
    cv2.waitKey(1)