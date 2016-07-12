#!/usr/bin/env python
# coding=utf-8
# Author: Archer Reilly
# File: contour.py
# Desc: show how to use contour
#
# Produced By BR
from PIL import Image
from pylab import *

# read image 2 array
im = array(Image.open('../data/data/empire.jpg').convert('L'))

# create a new figure
figure()

# dont use colors
gray()

# show contour with origin upper left corner
contour(im, origin='image')

axis('equal')
axis('off')

show()
