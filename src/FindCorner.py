#!/usr/local/env python
# coding=utf-8
#
# Author: Archer Reilly
# Desc: 按照颜色找出物体blob
# File: FindBlobs.py
# Date: 30/July/2016
#
from SimpleCV import Color, Image

# img = Image('/home/archer/Downloads/Chapter 8/corners.png')
img = Image('/home/archer/Downloads/1335212990.jpg')

# corners = img.findCorners(maxnum=9, mindistance=10)
corners = img.findCorners()
corners.draw(Color.RED, width=3)
img.save('res.png')
img.show()
