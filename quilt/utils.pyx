#! /usr/bin/env python
"""
Utility functions.
"""

# import builtin modules
from copy import deepcopy

# import 3rd party modules
cimport cython
import numpy as np
cimport numpy as np
from numpy import multiply
from numpy cimport ndarray, float_t

from cpython cimport bool
from PIL import Image


def filter_img(img_a, img_b, mask):
    """
    Combines two RGB images according to the binary weight mask mask.
    Where the mask is white the first image will be copied, where the mask is
    black the second image will be copied.
    The images and the mask must have the same dimensions.

    Args:
        img_a: first image
        img_b: second image
        mask: binary mask

    Returns:
        an image with pixels from img_a (where mask = 1), and pixels from img_b
        (where mask = 0)
    """
    if not img_a.shape[0:2] == img_b.shape[0:2] == mask.shape:
        raise ValueError('Matrices dimensions mismatch: got {0}, {1} and {2}'.
                         format(img_a.shape, img_b.shape, mask.shape))
    res = np.zeros(img_a.shape)
    for i in range(0, 3):
        res[:, :, i] = multiply(img_a[:, :, i], (mask == 1)) + \
                       multiply(img_b[:, :, i], (mask == 0))
    return res


cpdef imresize(img,  int height=0, int width=0, float scale=0):
    """
    Resize an image based on new height and width or a scale value
    Args:
        img: image (PIL.Image)
        height: new desired height
        width: new desired width
        scale: scale value

    Returns: resized image
    """
    # if it is a matrix: turn it into image first
    cdef bool is_matrix = isinstance(img, np.ndarray)
    img = matrix2img(img) if is_matrix else img

    # find the scale value if only on dimension is given
    if height and not width:
        scale = (height / float(img.shape[1]))
    if width and not height:
        scale = (width / float(img.shape[0]))

    height = height or int(float(img.size[1]) * float(scale))
    width = width or int(float(img.size[0]) * float(scale))
    result = img.resize((width, height), Image.ANTIALIAS)
    return img2matrix(result) if is_matrix else result

def img2matrix(img):
    """
    Converts the input image into a matrix of float values in the range [0, 1].
    """
    if isinstance(img, np.ndarray):
        return img
    matrix = img.convert(img.mode)
    matrix = np.array(matrix)
    matrix = im2double(matrix.astype(float))
    return matrix

cpdef matrix2img(ndarray matrix, bool adjust_values=True):
    """
    Converts an image into a matrix.
    """
    if adjust_values and np.max(matrix) <= 1:
        matrix *= 255
    matrix = np.uint8(matrix)
    if matrix.ndim == 3 and matrix.shape[2] == 3:
        return Image.fromarray(matrix, 'RGB')
    return Image.fromarray(matrix)


def gray2rgb(ndarray im):
    """
    Convert a gray scale image (1 channel) to RGB (3 channels)
    Args:
        im: image or matrix to be converted

    Returns:
        RGB image
    """
    if im.ndim == 3 and im.shape[2] == 3:
        return im
    if im.ndim == 2:
        return np.asarray(np.dstack((im, im, im)))
    raise ValueError('Input image must be 2 or 3 dimensional')


def rgb2gray(im):
    """
    Convert a RGB image (3 channels) to a gray-scale image (1 channel)
    Args:
        im: image to convert

    Returns:
        gray-scale image
    """
    if im.ndim == 2:
        return im
    if im.ndim == 3 and im.shape[2] == 3:
        r, g, b = im[:, :, 0], im[:, :, 1], im[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        if im.dtype is np.dtype('uint8'):
            gray = np.uint8(gray)
        return gray
    raise ValueError('Input mask must be 2 or 3 dimensional')


cpdef ndarray im2double(ndarray im):
    """
    Convert an uint8 image to float.
    Args:
        im: image to convert

    Returns:
        image with float values
    """
    if im.dtype == np.dtype('float64'):
        if np.max(im) > 1:
            return im / 255.0
        return im
    if im.dtype == np.dtype('uint8'):
        return im.astype('float') / 255.0
    raise ValueError("im2double: unsupported image type. Found: {0}".format(
                             im.dtype))


def show(img):
    """
    Display image. Can display multiple images.
    Args:
        img: image (or list of images) to display
        title: title of the image/s
    """
    if not isinstance(img, list):
        img = [img]
    for i in img:
        im = deepcopy(i)
        if im.dtype == np.dtype('uint8'):
            # clamp values to [0, 255]
            im[im > 255] = 255
            im[im < 0] = 0
            matrix2img(im).show()
        elif im.dtype == np.dtype('float64'):
            # clamp values to [0, 1]
            im[im > 1] = 1
            im[im < 0] = 0
            matrix2img(im).show()
        else:
            raise ValueError("show: unsupported image type: found {0}".format(
                             im.dtype))


def save(img, path):
    """
    Save image to file. Can also save multiple images.
    Args:
        img: image (or list of images) in matrix form
        path: path (or list of paths) to file
    """
    # handle singles or stacks
    if isinstance(img, list):
        if not isinstance(path, list):
            path = [path]
        if not len(img) == len(path):
            raise ValueError('if images are in stack, so should be paths')
    else:
        img = [img]
        path = [path]

    # save it
    for idx, image in enumerate(img):
        im = deepcopy(image)
        print 'saving', path[idx]
        if im.dtype == np.dtype('uint8'):
            matrix2img(im).save(path[idx])
        elif im.dtype == np.dtype('float64'):
            matrix2img(im).save(path[idx])
        else:
            raise ValueError("show: unsupported image type: found {0}".format(
                             im.dtype))