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

def normalize(points):
    """ Normalize a collection of points in haomogeneous coordinates so that
        last row = 1. """
    for row in points:
        row /= points[-1]

    return points

def make_homog(points):
    """ convert a set of points (dim*n array) to haomogeneous coordinates"""
    return vstack((points, ones((1, points, shape[1]))))

def H_from_points(fp, tp):
    """ find homography H, such that fp is mapped to tp using the linear
        DLT method, points are conditioned automatically"""

    if fp.shape != tp.shape:
        raise RuntimeError('number of points do not match')

    # condition points (important for numberical reasons)
    # -- from points --
    m = mean(fp[:2], axis=1)
    maxstd = max(std(fp[:2], axis=1)) + 1e-9
    C1 = diag([1/maxstd, 1/maxstd, 1])
    C1[0][2] = -m[0] / maxstd
    C1[1][2] = -m[1] / maxstd
    fp = dot(C1, fp)

    # -- to points --
    m = mean(tp[:2], axis=1)
    maxstd = max(std(tp[:2], axis=1)) + 1e-9
    C2 = diag([1/maxstd, 1/maxstd, 1])
    C2[0][2] = -m[0] / maxstd
    C2[1][2] = -m[1] / maxstd
    tp = dot(C2, tp)

    # create matrix for linear method, 2 rows for each correspondence pair
    nbr_correspondences = fp.shape[1]
    A = zeros((2 * nbr_correspondences, 9))
    for i in range(nbr_correspondences):
        A[2 * i] = [-fp[0][i], -fp[1][i], -1, 0, 0, 0, tp[0][i] * fp[0][i], tp[0][i] * fp[1][i], tp[0][i]]
        A[2 * i + 1] = [0, 0, 0, -fp[0][i], -fp[1][i], -1, tp[1][i] * fp[0][i], tp[1][i] * fp[1][i], tp[1][i]]

    U, S, V = linalg.svd(A)
    H = V[8].reshape((3, 3))

    # decondition
    H = dot(linalg.inv(C2), dot(H, C1))

    # normalize and return
    return H / H[2, 2]

def Haffine_from_points(fp, tp):
    """ find H, affine transformation, such that tp is affine transf of fp"""

    if fp.shape != tp.shape:
        raise RuntimeError('number of points do not match')

    # condition points (important for numberical reasons)
    # -- from points --
    m = mean(fp[:2], axis=1)
    maxstd = max(std(fp[:2], axis=1)) + 1e-9
    C1 = diag([1/maxstd, 1/maxstd, 1])
    C1[0][2] = -m[0] / maxstd
    C1[1][2] = -m[1] / maxstd
    fp_cond = dot(C1, fp)

    # -- to points --
    m = mean(tp[:2], axis=1)
    C2 = C1.copy() # must use same scaling for both point sets
    C2 = diag([1/maxstd, 1/maxstd, 1])
    C2[0][2] = -m[0] / maxstd
    C2[1][2] = -m[1] / maxstd
    tp_cond = dot(C2, tp)

    # conditioned points have mean zero, so translations is zeros
    A = concatenate((fp_cond[:2], tp_cond[:2]), axis = 0)
    U, S, V = linalg.svd(A.T)

    # create B and C matrices as Hartley-Zisserman (2:nd ed) p 130
    tmp = V[:2].T
    B = tmp[:2]
    C = tmp[2:4]

    tmp2 = concatenate((dot(C, linalg.pinv(B)), zeros((2, 1))), axis=1)
    H = vstack((tmp2, [0, 0, 1]))

    # decondition
    H = dot(linalg.inv(C2), dot(H, C1))

    return H / H[2,2]

def image_in_image(im1, im2, tp):
    """ put im1 in im2 with an affine transformation such that corners are
    a   as close to tp as possible. tp are homogeneous and counterclockwise
    from top left."""

    # points to warp from
    m, n = im1.shape[:2]
    fp = array([[0, m, m, 0], [0, 0, n, n], [1, 1, 1, 1]])

    ＃ compute affine transform and apply
    H = Haffine_from_points(tp, fp)
    im1_t = ndimage.affine_transform(im1, H[:2, :2], (H[0, 2], H[1, 2]), im2.shape[:2])
    alpha = (im1_t > 0)

    return (1 - alpha) * im2 + alpha * im1_t

def alpha_for_triangle(points, m, n):
    """ create alpha map of size (m, n) for a triangle with corners defined by points
        given in normalized homogeneous coordinates)"""

    alpha = zeros((m, n))
    for i in range(min(points[0]), max(points[0])):
        for j in range(min(points[1]), max(points[1])):
            x = linalg.solve(points, [i, j, 1])
            if min(x) > 0: # all coefficients positive
                alpha[i, j] = 1

    return alpha
