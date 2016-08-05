#! /usr/bin/env python
"""
Sum Square Difference calculator
"""

# import 3rd party modules
cimport cython
import numpy as np
cimport numpy as np
from numpy import einsum
from numpy.lib.stride_tricks import as_strided
from numpy cimport ndarray, float_t



cpdef ssd(ndarray[float_t, ndim=3] img,
        ndarray[float_t, ndim=3] patch):
    """
    Calculates ssd between two rgb images
    """
    cdef ndarray result = None
    cdef int k

    for k in xrange(img.shape[2]):
        if result is None:
            result = sumsqdiff(img[:, :, k], patch[:, :, k])
        else:
            result = result + sumsqdiff(img[:, :, k], patch[:, :, k])

    return result


cpdef sumsqdiff(ndarray[float_t, ndim=2] img,
              ndarray[float_t, ndim=2] patch):
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
    cdef ndarray[float_t, ndim=2] ssd
    cdef ndarray[float_t, ndim=4] y

    # stride is the step through the memory to fast access data
    y = as_strided(img,
                   shape=(img.shape[0] - patch.shape[0] + 1,
                          img.shape[1] - patch.shape[1] + 1,)
                         + (patch.shape[0], patch.shape[1]),
                   strides=(img.strides[0], img.strides[1]) * 2)
    ssd = einsum('ijkl,kl->ij', y, patch)       # sum(a*b)
    ssd *= - 2
    ssd += einsum('ijkl, ijkl->ij', y, y)       # sum(a**2)
    ssd += einsum('ij, ij', patch, patch)       # sum(b**2)

    ssd[ssd < 0] = 0  # approximation errors
    return ssd

