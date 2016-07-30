#!/usr/local/env python
# coding=utf-8
#
# Author: Archer Reilly
# File: FaceDetector.py
# Desc: 检测人脸
# Date: 30/Jul/2016

from SimpleCV import Camera, Display

cam = Camera()
disp = Display(cam.getImage().size())

while disp.isNotDone():
    img = cam.getImage()

    # Look for a face
    faces = img.findHaarFeatures('face.xml')

    if faces is not None and len(faces) > 0:
        # Get the largest face
        faces = faces.sortArea()
        bigFace = faces[-1]

        # Draw a green box around the face
        bigFace.draw()

    img.save(disp)
