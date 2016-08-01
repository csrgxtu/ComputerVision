#!/usr/local/env python
# coding=utf-8
#
# Author: Archer Reilly
# Desc: 按照颜色找出物体blob
# File: FindLines.py
# Date: 30/July/2016
#
from SimpleCV import Color, Image

# img = Image('/home/archer/Downloads/Chapter 8/mandms-dark.png')
img = Image('/home/archer/Downloads/1185391864.jpg')

# blue_distance = img.colorDistance(Color.BLUE).invert()
# blue_distance = img.colorDistance(Color.BLACK).invert()

# blobs = blue_distance.findBlobs(minsize=15)
lines = img.findLines(threshold=20)
lines.draw(Color.RED, width=3)
img.save('res.png')
img.show()

# blobs.draw(color=Color.RED, width=3)
#
# blue_distance.show()
#
# img.addDrawingLayer(blue_distance.dl())
# img.save('res.png')
# img.show()
