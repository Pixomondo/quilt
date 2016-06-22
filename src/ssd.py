#! /usr/bin/env python

# import builtin modules
from __future__ import division

# import 3rd party modules
import numpy as np
from numpy import power, sqrt, floor, ceil
from scipy import ndimage


def ssd(X, Y):
    """
    Computes the sum of squared distances between X and Y for each possible
    overlap of Y on X. Y is thus smaller than X

    Inputs:
       X - larger image
       Y - smaller image

    Outputs:
       Each pixel of Z contains the ssd for Y overlaid on X at that pixel
    """

    mask = np.ones((Y.shape[0], Y.shape[1]))

    # for every channel
    for k in range(0, X.shape[2]):
        A = X[:, :, k]
        B = Y[:, :, k]

        # conv2(image, mask) is the same as filter2(rot90(mask,2), image)
        a2 = filter2(mask, power(A, 2), 'valid')
        b2 = sum(sum(power(B, 2)))
        ab = filter2(B, A, 'valid') * 2

        r = (a2 - ab) + b2
        r[r < 0] = 0  # approximation errors

        if k == 0:
            result = sqrt(r)
        else:
            result = result + sqrt(r)

    # print '\nssd: X:', X.shape, 'Y:', Y.shape, 'RES:', result.shape

    return result


def filter2(mask, image, mode='same'):
    # conv2(image, mask) = filter2(rot90(mask,2), image)
    # filter2(mask, image) = conv2(image, rot90(mask, -2)))
    mask = np.rot90(mask, -2)
    return conv2(image, mask, mode=mode)


def conv2(x, y, mode='same'):
    """
    Emulate the function conv2 from Mathworks.

    Usage:
    z = conv2(x,y,mode='same')

    TODO:
     - Support other modes than 'same' (see conv2.m)
    """

    if not (mode == 'same' or mode == 'valid'):
        raise Exception("Mode not supported")

    # Add singleton dimensions
    if len(x.shape) < len(y.shape):
        dim = x.shape
        for i in range(len(x.shape), len(y.shape)):
            dim = (1,) + dim
        x = x.reshape(dim)
    elif len(y.shape) < len(x.shape):
        dim = y.shape
        for i in range(len(y.shape), len(x.shape)):
            dim = (1,) + dim
        y = y.reshape(dim)

    origin = (0, 0)

    z = ndimage.filters.convolve(x, y, mode='constant', origin=origin)

    if mode == 'valid':
        # reduce the result to just valid values
        i = y.shape[0]
        j = y.shape[1]
        z = z[floor((i - 1) / 2): (- ceil((i - 1) / 2) or None),
              floor((j - 1) / 2): (- ceil((j - 1) / 2) or None)]
    return z

