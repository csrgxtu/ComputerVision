#!/usr/bin/env python

from imtools import *
from PIL import Image


im = array(Image.open('/home/archer/Downloads/355605471.jpg').convert('L'))
im2,cdf = histeq(im)
pil_im = Image.fromarray(im2)
pil_im.show()
