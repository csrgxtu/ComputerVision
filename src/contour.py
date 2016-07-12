#!/usr/bin/env python
# coding=utf-8
# Author: Archer Reilly
# File: hist.py
# Desc: show how to use contour
#
# Produced By BR
from PIL import Image
from pylab import *

# read image 2 array
im = array(Image.open('../data/data/empire.jpg').convert('L'))

# create a new figure
figure()
hist(im.flatten(), 128)
show()
