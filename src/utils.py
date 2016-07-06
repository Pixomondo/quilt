#! /usr/bin/env python

from copy import deepcopy

# import 3rd party modules
import numpy as np
from numpy import multiply
from scipy.misc import toimage


def filter_img(img_a, img_b, mask):
    res = np.zeros(img_a.shape)
    for i in range(0, 3):
        res[:, :, i] = multiply(img_a[:, :, i], (mask == 1)) + \
                       multiply(img_b[:, :, i], (mask == 0))
    return res


def gray2rgb(im):
    if len(im.shape) == 3:
        return im
    if len(im.shape) == 2:
        return np.asarray(np.dstack((im, im, im)))
    raise ValueError('Input image must be 2 or 3 dimensional')


def rgb2gray(im):
    if len(im.shape) == 2:
        return im
    if len(im.shape) == 3:
        r, g, b = im[:, :, 0], im[:, :, 1], im[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        if im.dtype is np.dtype('uint8'):
            gray = np.uint8(gray)
        return gray
    raise ValueError('Input mask must be 2 or 3 dimensional')


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


def im2double(im):
    if im.dtype == np.dtype('float64'):
        return im
    if im.dtype == np.dtype('uint8'):
        return im.astype('float') / 255.0
    if im.dtype == np.dtype('uint16'):
        return im.astype('float') / 65535.0
    raise ValueError("im2double: unsupported image type. Found: {0}".format(
                             im.dtype))


def show(img, title=None):
    if not isinstance(img, list):
        img = [img]
    for i in img:
        im = deepcopy(i)
        if im.dtype == np.dtype('uint8'):
            im[im > 255] = 255
            im[im < 0] = 0
            toimage(im).show(title)
        elif im.dtype == np.dtype('float64'):
            im[im > 1] = 1
            im[im < 0] = 0
            toimage(np.uint8(im2double(im)*255)).show(title)
        else:
            raise ValueError("show: unsupported image type: found {0}".format(
                             im.dtype))


def save(img, path):
    # handle singles or stacks
    if isinstance(img, list):
        if not isinstance(path, list) or not len(img) == len(path):
            raise ValueError('if images are in stack, so should be paths')
    else:
        img = [img]
        path = [path]

    # save it
    for idx, im in enumerate(img):
        print 'saving', path[idx]
        if im.dtype == np.dtype('uint8'):
            toimage(im).save(path[idx])
        elif im.dtype == np.dtype('float64'):
            toimage(np.uint8(im2double(im) * 255)).save(path[idx])
        else:
            raise ValueError("show: unsupported image type: found {0}".format(
                             im.dtype))
