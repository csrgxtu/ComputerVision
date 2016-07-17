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

def pca(X):
    """ Principal Component Analysis
        input: X, matrix with traning data stored as flattened arrays in rows
        return: projection matrix (with important dimensions first), variance
        and mean."""

    # get dimensions
    num_data, dim = X.shape

    # center data
    mean_X = X.mean(axis=0)
    X = X - mean_X

    if dim > num_data:
        # pca - compact trick used
        M = dot(X, X.T) # convariance matrix
        e, EV = linalg.eigh(M) # eigenvalues and eigenvectors
        tmp = dot(X.T, EV) # this is the compact trick
        V = tmp[::-1] # reverse since last eigenvectors are the ones we want
        S = sqrt(e)[::-1] # reverse since eigenvalues are in increasing order
        for i in range(V.shape[1]):
            V[:,i] /= S
    else:
        # pca - svd used
        U, S, V = linalg.svd(X)
        V = V[:num_data] # only makes sense to return the first num_data

    # return the projection matrix, the variance and the mean
    return V, S, mean_X

def denoise(im, U_init, tolerance=0.1, tau=0.125, tv_weigth=100):
    """ an implementation of the Rudin-Osher-Fatemi(ROF) denoising model
        using the numberical procedure presented in eq(11) A. chambolle(2005).
        Input: noise input image (grayscale), initial guess for U, weigth of
        the TV-regularizing term, steplengtyh, """
    m, n = im.shape # size of noisy image

    # initialize
    U = U_init
    Px = im # x-component to the dual field
    Py = im # y-component to the dual field
    error = 1

    while (error > tolerance):
        Uold = U

        # gradient of primal variable
        GradUx = roll(U, -1, axis = 1) - U # x-component of U's gradient
        GradUy = roll(u, -1, axis = 0) - U # y-component of U's gradient

        # update the dual variable
        PxNew = Px + (tau / tv_weigth) * GradUx
        PyNew = Py + (tau / tv_weigth) * GradUy
        NormNew = maximum(1, sqrt(PxNew ** 2 + PyNew ** 2))

        Px = PxNew / NormNew # update of x-component
        Py = PyNew / NormNew # update of y-component

        # update the primal variable
        RxPx = roll(Px, 1, axis=1) # right x-translation of x-component
        RyPy = roll(Py, 1, axis=0) # right y-translation of y-component

        DivP = (Px - RxPx) + (Py - RyPy) # divergence of the dual field

        U = im + tv_weigth * DivP # update of the primal variable

        # update of error
        error = linalg.norm(U - Uold) / sqrt(n * m)

        return U, im-U

from scipy.ndimage import filters

def compute_harris_response(im, sigma=3):
    """ compute the Harris corner detector response function
        for each pixel in a graylevel image. """

    # derivatives
    imx = zeros(im.shape)
    filters.gaussian_filter(im, (sigma, sigma), (0, 1), imx)
    imy = zeros(im.shape)
    filters.gaussian_filter(im, (sigma, sigma), (1, 0), imy)

    # compute components of the harris matrix
    Wxx = filters.gaussian_filter(imx * imx, sigma)
    Wxy = filters.gaussian_filter(imx * imy, sigma)
    Wyy = filters.gaussian_filter(imy * imy, sigma)

    # determinant and trace
    Wdet = Wxx * Wyy - Wxy ** 2
    Wtr = Wxx + Wyy

    return Wdet / Wtr

def get_harris_points(harrisim, min_dist=10, threshhold=0.1):
    """ return corners from a harris response image, min_dist is the minimum
        number of pixels separating corners and image boundary. """

    # find top corner condidates above a threshhold
    corner_threshhold = harrisim.max() * threshhold
    harrisim_t = (harrisim > corner_threshhold) * 1

    # get coordinates of condidates
    coords = array(harrisim_t.nonzero()).T

    # .. and their values
    candidate_values = [harrisim[c[0], c[1]] for c in coords]

    # sort andidates
    index = argsort(candidate_values)

    # store allowed point locations in array
    allowed_locations = zeros(harrisim.shape)
    allowed_locations[min_dist:-min_dist, min_dist:-min_dist] = 1

    # select the best points taking min_distance into account
    filtered_coords = []
    for i in index:
        if allowed_locations[coords[i, 0], coords[i, 1]] == 1:
            filtered_coords.append(coords[i])
            allowed_locations[(coords[i,0] - min_dist):(coords[i, 0] + min_dist), (coords[i, 1] - min_dist):(coords[i, 1] + min_dist)] = 0

    return filtered_coords

def plot_harris_points(image, filtered_coords):
    """ plots corners found in image"""
    figure()
    gray()
    imshow(image)
    plot([p[1] for p in filtered_coords], [p[0] for p in filtered_coords], '*')
    axis('off')
    show()
