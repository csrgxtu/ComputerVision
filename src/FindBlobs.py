#!/usr/local/env python
# coding=utf-8
#
# Author: Archer Reilly
# Desc: 按照颜色找出物体blob
# File: FindBlobs.py
# Date: 30/July/2016
#
from SimpleCV import Color, Image

img = Image('/home/archer/Downloads/Chapter 8/mandms.png')

blue_distance = img.colorDistance(Color.BLUE).invert()

blobs = blue_distance.findBlobs()

blobs.draw(color=Color.PUCE, width=3)

blue_distance.show()

img.addDrawingLayer(blue_distance.dl())
img.save('res.png')
img.show()
