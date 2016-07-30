#!/usr/local/env python
# coding=utf-8
#
# Author: Archer Reilly
# Date: 30/Jul/2016
# File: TrackCircle.py
# Desc: 跟踪黑色的圈圈

from SimpleCV import Camera, Color, Display

cam = Camera()
disp = Display()

previous_ball_xy = None
previous_ball_size = 100

while disp.isNotDone:
    print 'looping'
    img = cam.getImage()

    dist = img.colorDistance(Color.BLACK).dilate(2)
    segmented = dist.binarize()
    blobs = segmented.findBlobs(minsize=2000)

    if blobs:
        circles = blobs.filter([b.isCircle(0.2) for b in blobs])
        if circles:
            if previous_ball_xy:
                fcircles = circles.filter([c.radius() > (previous_ball_size * 0.5) for c in circles])
                distances = [int(c.distanceFrom(previous_ball_xy)) for c in circles]
                nearest = blobs[distances.index(min(distances))]

                img.drawCircle((nearest.x, nearest.y), nearest.radius(), Color.RED, thickness=4)
                previous_ball_xy = (nearest.x, nearest.y)
                previous_ball_size = nearest.radius()
            else:
                previous_ball_xy = (circles[-1].x, circles[-1].y)
                previous_ball_size = circles[-1].radius()
        else:
            img.drawText("No circles found")

    img.save(disp)
