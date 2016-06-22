#! /usr/bin/env python
"""
def Y = imagequilt(X, tilesize, num_tiles, overlap, err)
Performs the Efros/Freeman Image quilting algorithm on the input

Inputs
  X:  The source image to be used in synthesis
  tilesize:   the dimensions of each square tile.  Should divide size(X) evenly
  num_tiles:  The number of tiles to be placed in the output image, in each dimension
  overlap: The amount of overlap to allow between pixels (def: 1/6 tilesize)
  err: used when computing list of compatible tiles (def: 0.1)
"""

# import builtin modules
from __future__ import division

# import 3rd party modules
import numpy as np
from numpy import power, ceil, multiply
from scipy.misc import toimage

# import internal modules
from ssd import ssd
from mincut import mincut
from quilt.src.utils import gray2rgb, filter_img, rgb2gray


def quilt(X, tilesize, num_tiles, overlap=None, err=0.002, niter=3):

    print '\n ----------- QUILT ------------'

    # --- set parameters ---

    # adjust image
    if np.amax(X) > 1:
        X = X / 255
    if len(X.shape) == 2:
        X = gray2rgb(X)
    elif not len(X.shape) == 3:
        raise ValueError('Input image must be 2 or 3 dimensional')

    # adjust tile size
    tilesize = long(tilesize)

    # adjust overlap
    if not overlap:
        overlap = int(round(tilesize / 6))
    elif overlap >= tilesize:
        raise ValueError('Overlap must be less than tilesize')
    print 'overlap:', overlap, 'tile size:', tilesize
    simple = 0

    # adjust num tiles
    if not isinstance(num_tiles, list):
        num_tiles = [num_tiles, num_tiles]
    elif len(num_tiles) == 1:
        num_tiles = [num_tiles[0], num_tiles[0]]

    # prepare result
    res_size = [num_tiles[0]*tilesize - (num_tiles[0]-1)*overlap,
                num_tiles[1]*tilesize - (num_tiles[1]-1)*overlap]
    Y = np.zeros((res_size[0], res_size[1], 3))

    # ---- compute ----
    for n in xrange(niter):
        for i in xrange(num_tiles[0]):

            startI = i * tilesize - i * overlap
            endI = startI + tilesize

            for j in xrange(num_tiles[1]):

                startJ = j*tilesize - j * overlap
                endJ = startJ + tilesize

                # Determine the distances from each tile to the overlap region
                # This will eventually be replaced with convolutions
                distances = calc_distance(X, Y[startI:endI, startJ:endJ, :],
                                          tilesize=tilesize, overlap=overlap,
                                          coord=[i, j])

                # Find the best candidates for the match
                best = np.min(distances)
                candidates = np.where(distances <= (1+err)*best)

                # choose a random best
                choice = ceil(np.random.rand()*(len(candidates[0])-1))
                sub = [candidates[0][choice], candidates[1][choice]]

                # print 'Picked tile ({0}, {1}) out of {2} candidates. Best ' \
                #       'error={3}\n'.format(sub[0], sub[1], len(candidates[0]),
                #        best)

                # If we do the simple quilting (no cut), just copy image
                if simple:

                    Y[startI:endI, startJ:endJ, :] = \
                        X[sub[0]:sub[0]+tilesize, sub[1]:sub[1]+tilesize, :]
                else:
                    Y[startI:endI, startJ:endJ, :] = \
                        sew(X[sub[0]:sub[0]+tilesize, sub[1]:sub[1]+tilesize],
                            Y[startI:endI, startJ:endJ, :],
                            coord=[i, j], overlap=overlap)

                # print Y[:, :, 0]
                # print '_______________________\n'
            # if i/2 == 0:
            #     toimage(Y).show()

        # toimage(Y).show('FINAL: tile {0} over {1}'.format(tilesize, overlap))
    return Y


def calc_distance(img, patch, tilesize=None, overlap=None, coord=None):

    # print '\n-- distance --'
    [i, j] = coord

    distances = np.zeros((img.shape[0] - tilesize, img.shape[1] - tilesize))

    # Compute the distances from the src to the left overlap region
    if j > 0:
        distances = ssd(img, patch[:, :overlap, :])
        distances = distances[:, 0: -tilesize+overlap]

    # Compute the distance from the source to top overlap region
    if i > 0:

        z = ssd(img, patch[: overlap, :, :])
        z = z[0: -tilesize+overlap, :]
        if j > 0:
            distances = distances + z
        else:
            distances = z

    # If both are greater, compute the distance of the overlap
    if i > 0 and j > 0:
        z = ssd(img, patch[:overlap, :overlap, :])
        z = z[0: -tilesize+overlap, 0: -tilesize+overlap]
        distances = distances + z

    return distances


def sew(src, dst, coord=None, overlap=None):

    [i, j] = coord
    if i == 0 and j == 0:
        return src

    # Initialize the mask to all ones
    mask = np.ones((src.shape[0], src.shape[1]))

    # We have a left overlap
    if j > 0:

        # difference
        diff = power(rgb2gray(src[:, :overlap] - dst[:, :overlap]), 2)
        # min-cut
        cut = mincut(diff, 0)
        # find the mask
        mask[:, :overlap] = (cut >= 0)

    # We have a top overlap
    if i > 0:

        # difference
        diff = power(rgb2gray(src[:overlap, :] - dst[:overlap, :]), 2)
        # min-cut
        cut = mincut(diff, 1)
        # find the mask
        mask[:overlap, :] = multiply(mask[:overlap, :], cut >= 0)

    # Write to the destination using the mask
    return filter_img(src, dst, mask)
