#!/usr/local/env python
# coding=utf-8
#
# Author: Archer Reilly
# Desc: 将你直接显示到背景图片前面
# File: GreenScreen.py
# Date: 30/July/2016
#
from SimpleCV import Camera, Color, Image, Display

cam = Camera()
backgroud = Image('/home/archer/Downloads/Chapter 6/weather.png')

disp = Display()

while not disp.isDone():
    img = cam.getImage()
    img = img.flipHorizontal()

    bgcolor = img.getPixel(10, 10)
    dist = img.colorDistance(bgcolor)
    mask = dist.binarize(50)

    foreground = img - mask

    backgroud = backgroud - mask.invert()

    combined = backgroud + foreground
    combined.save(disp)
