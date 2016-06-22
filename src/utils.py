#! /usr/bin/env python

# import 3rd party modules
import numpy as np


def filter_img(img_a, img_b, mask):
    res = np.zeros(img_a.shape)
    for i in range(0, 3):
        res[:, :, i] = np.multiply(img_a[:, :, i], (mask == 1)) + \
                       np.multiply(img_b[:, :, i], (mask == 0))
    return res


def gray2rgb(im):
    # we can use dstack and an array copy
    # this has to be slow, we create an array with
    # 3x the data we need and truncate afterwards
    return np.asarray(np.dstack((im, im, im)))


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def sub2ind(array_shape, rows, cols):
    ind = rows*array_shape[1] + cols
    ind[ind < 0] = -1
    ind[ind >= array_shape[0]*array_shape[1]] = -1
    return ind


def ind2sub(array_shape, ind):
    ind[ind < 0] = -1
    ind[ind >= array_shape[0]*array_shape[1]] = -1
    rows = (ind.astype('int') / array_shape[1])
    cols = ind  # array_shape[1]
    return rows, cols


