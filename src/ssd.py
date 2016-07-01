#! /usr/bin/env python

# import builtin modules
from __future__ import division

# import 3rd party modules
from numpy import einsum
from numpy.lib.stride_tricks import as_strided


def ssd(img, patch):
    result = 0
    for k in range(0, img.shape[2]):
        result = result + sumsqdiff3(img[:, :, k], patch[:, :, k])
    return result


def sumsqdiff3(img, patch):
    """
    We want to calculate the difference between patch and img.
    So we want a matrix M where:
        M(i, j) = sum( (img[i.., j..]-patch)^2 ) =
        sum(img[i.., j..]^2) + sum(patch^2) - 2*img[i.., j..]*patch
    (squared diff so negative and positive values don't cancel each other).
    This multiply-then-reduce-with-a-sum operations can be expressed through
    Einstein summation (einsum), with a substantial improvement in both
    performance and memory use.

    To improve the performance, img can be represented as strided (y).
    """

    patch_size = patch.shape
    # stride is the step through the memory to fast access data
    y = as_strided(img,
                   shape=(img.shape[0] - patch_size[0] + 1,
                          img.shape[1] - patch_size[1] + 1,) + patch_size,
                   strides=img.strides * 2)
    ssd = einsum('ijkl,kl->ij', y, patch)       # sum(a*b)
    ssd *= - 2
    ssd += einsum('ijkl, ijkl->ij', y, y)       # sum(a**2)
    ssd += einsum('ij, ij', patch, patch)       # sum(b**2)

    ssd[ssd < 0] = 0  # approximation errors
    return ssd

