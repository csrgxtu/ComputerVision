import os
from PIL import *
from pylab import *
from numpy import *

def get_imlist(path):
    """return a list of filenames for all images in a dir"""
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]

def imresize(im, sz):
    """ resize an image array using PIL """
    pil_im = PIL.fromarray(uint(im))

    return array(pil_im.resize(sz))

def histeq(im, nbr_bins=256):
    """Histogram equalization of a grayscale image. """
    # get image Histogram
    imhist, bins = histogram(im.flatten(), nbr_bins, normed=True)
    cdf = imhist.cumsum() #cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    im2 = interp(im.flatten(),bins[:-1],cdf)

    return im2.reshape(im.shape), cdf
