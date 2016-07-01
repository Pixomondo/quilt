#! /usr/bin/env python
"""
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
from copy import deepcopy
from multiprocessing import cpu_count, Queue, Pool, JoinableQueue
import sys
sys.path.insert(0, r"C:\dev\textures\Lib\site-packages")

# import 3rd party modules
import numpy as np
from numpy import power, ceil, multiply, where, rot90, floor, zeros
from numpy.random import rand

# import internal modules
from ssd import ssd
from mincut import mincut
from utils import gray2rgb, filter_img, rgb2gray, im2double, show, save, sub2ind


def unwrap_self(*arg, **kwarg):
    return Quilt.optimized_compute_small(*arg, **kwarg)


class Quilt:

    def __init__(self, X, Y=None, Xmask=None, Ymask=None,
                 rotations=0, tilesize=None, overlap=None, num_tiles=None,
                 big_tilesize=500, big_overlap=50,
                 err=0.002, niter=3, result_path=None):

        # tile size and overlap
        if overlap >= tilesize:
            raise ValueError('Overlap must be less than the tile size')
        self.tilesize = long(tilesize)
        self.overlap = overlap or int(round(self.tilesize / 5))
        print 'overlap:', overlap, 'tile size:', self.tilesize

        # ------ images ------
        # src
        X, self.X = self._set_src(X, rotate=rotations)
        print 'X: len:', len(X), '- dim:', self.X[0].shape
        self.Xmask = self._set_src_mask(Xmask, reference=X, rotate=rotations)
        print 'Xmask:', Xmask is None and 'None' or self.Xmask.shape

        # dst
        self.num_tiles = self.set_num_tiles(num_tiles)
        print 'num tiles:', self.num_tiles

        # size of Y: depends of the number of tiles or the size of the Y-mask
        if num_tiles:
            # y_size = num * size - (num-1) * overlap
            y_size = [self.num_tiles[0] * self.tilesize -
                      (self.num_tiles[0] - 1) * self.overlap,
                      self.num_tiles[1] * self.tilesize -
                      (self.num_tiles[1] - 1) * self.overlap]
        elif Ymask is not None:
            y_size = [Ymask.shape[0], Ymask.shape[1]]
        else:
            raise ValueError("Please provide the num_tiles or Ymask")
        self.Y = self._set_dst(Y, y_size)
        print 'Y: len:', len(self.Y), '- dim:', self.Y[0].shape
        self.Ymask = self._set_dst_mask(Ymask, y_size)

        # ----- big tiles -------------------
        if big_overlap >= big_tilesize:
            raise ValueError('Overlap must be less than the tile size')
        self.big_tilesize = min(long(big_tilesize), min(X.shape[0:2]) - 1)
        self.big_overlap = min(big_overlap, int(self.big_tilesize / 3))

        # ------ independent parameters ------

        # error in choosing best patches
        if err < 0:
            raise ValueError('Error must be a positive number')
        self.err = err

        # number of iterations
        if niter < 1:
            raise ValueError('Number of iterations must be at least 1')
        self.niter = niter

        self.result_path = result_path

    def _set_src(self, img, rotate=0):

        # img can be a single image or a list of images
        if not isinstance(img, list):
            img = [img]

        reference = None
        for i in xrange(len(img)):
            # the size of the chained img must be equal to the size of the first
            if i and not img[i].shape[0:2] == img[0].shape[0:2]:
                raise ValueError('Chained images must have the same size. Got '
                                 '{0} and {1}'.format(img[i].shape[0:2],
                                                      img[0].shape[0:2]))
            # dimensions
            img[i] = gray2rgb(img[i])
            # values
            img[i] = im2double(img[i])
            if not i:
                reference = img[i]
            # rotation
            img[i] = self.create_rotations(img[i], rotate)

        return reference, img

    def _set_src_mask(self, mask, reference=None, rotate=0):

        # no mask
        if mask is None:
            print 'Xmask: None'
            return

        # coherent with reference
        reference = reference or self.X
        if not mask.shape[0:2] == reference.shape[0:2]:
            raise ValueError('X and Xmask cannot have different sizes:'
                             'X={0} Xmask={1}'.format(reference.shape,
                                                      mask.shape))
        # size
        mask = rgb2gray(mask)
        if not rotate and not mask.shape == reference.shape[0:2]:
            raise ValueError('X and Xmask cannot have different sizes')

        # values
        # if there is a mask on the input: remove all the points leading to
        # patches with an overlap on that area. i.e.: remove the area and
        # anything a tile-size distant on the left or top.
        # to remove an area: replace it with inf value so that it won't be
        # chose computing the min.
        mask = im2double(mask)
        mask[mask < 0.7] = np.inf
        mask[mask < 1] = 0
        # rotate
        mask = self.create_rotations(mask, rotate)

        # expand the values
        for i in xrange(mask.shape[0] - self.tilesize):
            for j in xrange(mask.shape[1] - self.tilesize):
                tile = mask[i:i + self.tilesize, j:j + self.tilesize]
                if np.any(tile == np.inf):
                    mask[i, j] = np.inf

        mask = mask[:-self.tilesize + 1, :-self.tilesize + 1]

        # show
        temp = deepcopy(mask)
        temp[temp > 0] = 1
        show(temp)

        print 'Xmask:', mask.shape, ', values:', np.unique(mask)
        return mask

    def _set_dst(self, img, size):
        if img is None:
            img = np.zeros((size[0], size[1], 3))
        # we will have as many output layers as we got as input
        res = [deepcopy(img) for i in xrange(len(self.X))]
        return res

    def _set_dst_mask(self, mask, size):

        if mask is None:
            print 'Ymask: None'
            return

        # channels
        if len(mask.shape) == 3:
            mask = rgb2gray(mask)
        elif not len(mask.shape) == 1:
            raise ValueError('Input mask must be 2 or 3 dimensional')

        # size
        if mask.shape[0] >= size[0]:
            mask = mask[:size[0], :]
        if mask.shape[1] >= size[1]:
            mask = mask[:, size[1]]

        # values
        mask[mask < 0.01] = 0
        mask[mask > 0] = 1

        print 'Ymask:', mask.shape
        return mask

    @classmethod
    def create_rotations(cls, img, amount):

        if not amount:
            return img
        if amount not in [2, 4]:
            raise ValueError('Rotation must be None, 2 or 4. Got {0}'.format(
                amount))

        # could be better ...
        third_dim = len(img.shape) == 3
        img = gray2rgb(img)

        rot = None
        if amount == 2:
            # create a bigger image with the two rotations
            rot = np.zeros((img.shape[0] * 2 + 1, img.shape[1], 3))
            rot[:img.shape[0], :, :] = img
            rot[img.shape[0] + 1:, :, :] = rot90(img, 2)

        if amount == 4:
            # create a bigger image with the four rotations
            rot = np.zeros((max(img.shape[1], img.shape[0] * 2 + 1),
                            img.shape[1] + img.shape[0] * 2 + 2, 3))
            rot[:img.shape[0], :img.shape[1], :] = img
            rot[img.shape[0] + 1: img.shape[0] * 2 + 1, :img.shape[1], :] = \
                rot90(img, 2)
            rot[:img.shape[1], img.shape[1] + 1:img.shape[1] + 1 + img.shape[0],
            :] = \
                rot90(img, 1)
            rot[:img.shape[1], img.shape[0] + img.shape[1] + 2:, :] = \
                rot90(img, 3)

        if not third_dim:
            rot = rot[:, :, 0]
        return rot

    def set_overlap(self, overlap):
        if overlap >= self.tilesize:
            raise ValueError('Overlap must be less than tilesize')
        self.overlap = overlap

    def set_tilesize(self, size):
        if self.overlap >= size:
            raise ValueError('Overlap must be less than tilesize')
        self.tilesize = size

    def set_num_tiles(self, num):
        # make it a list with 2 elements
        if not isinstance(num, list):
            num = [num, num]
        elif len(num) == 1:
            num = [num[0], num[0]]
        return num

    def calc_num_tiles(self, img_size=None, tile_size=None, overlap=None):
        img_size = img_size or self.Y[0].shape
        tile_size = tile_size or self.tilesize
        overlap = overlap or self.overlap
        num_tiles = [
            np.int(ceil((img_size[0] - overlap) / (tile_size - overlap))),
            np.int(ceil((img_size[1] - overlap) / (tile_size - overlap)))]
        return num_tiles

    def get_result(self):
        return self.Y

    def compute(self):

        print '\nCOMPUTING ...'

        for n in xrange(self.niter):
            for i in xrange(self.num_tiles[0]):

                startI = i * self.tilesize - i * self.overlap
                endI = startI + self.tilesize

                for j in xrange(self.num_tiles[1]):

                    startJ = j * self.tilesize - j * self.overlap
                    endJ = startJ + self.tilesize

                    if self.Ymask and not np.any(self.Ymask[startI:endI,
                                                            startJ:endJ]):
                        continue

                    # Dist from each tile to the overlap region
                    patch = self.Y[0][startI:endI, startJ:endJ, :]
                    distances = self.distance(patch, coord=[i, j])

                    # Find the best candidates for the match
                    best = np.min(distances)
                    candidates = where(distances <= (1+self.err)*best)

                    # choose a random best
                    choice = ceil(rand()*(len(candidates[0])-1))
                    sub = [candidates[0][choice], candidates[1][choice]]

                    # sew them together
                    x_patch = self.X[0][sub[0]:sub[0]+self.tilesize,
                                        sub[1]:sub[1]+self.tilesize]
                    y_patch = self.Y[0][startI:endI, startJ:endJ, :]
                    mask = self.compute_mask(x_patch, y_patch, coord=[i, j])
                    for idx, x in enumerate(self.X):
                        x_patch = x[sub[0]:sub[0]+self.tilesize,
                                    sub[1]:sub[1]+self.tilesize]
                        y_patch = self.Y[idx][startI:endI, startJ:endJ, :]

                        self.Y[idx][startI:endI, startJ:endJ, :] = \
                            filter_img(x_patch, y_patch, mask)

                # if i % 5 == 0:
                #     print 'row', i, '/', self.num_tiles[0]
                #     if self.result_path:
                #         save(self.Y, self.result_path)
            # print 'iter', n, '/', self.niter
            show(self.Y[0], title='FINAL: tile {0} over {1}'
                                  ''.format(self.tilesize, self.overlap))
            if self.result_path:
                save(self.Y, self.result_path)

    def distance(self, patch, coord=None, tilesize=None, overlap=None):

        [i, j] = coord
        tilesize = tilesize or self.tilesize
        if len(tilesize) == 1:
            tilesize = [tilesize, tilesize]
        overlap = overlap or self.overlap
        
        distances = np.zeros((self.X[0].shape[0] - tilesize[0],
                              self.X[0].shape[1] - tilesize[1]))

        # Compute the distances from the src to the left overlap region
        if j > 0:
            distances = ssd(self.X[0], patch[:, :overlap, :])
            distances = distances[:, 0: -tilesize[1] + overlap]

        # Compute the distance from the source to top overlap region
        if i > 0:

            z = ssd(self.X[0], patch[: overlap, :, :])
            z = z[0: -tilesize[0] + overlap, :]
            if j > 0:
                distances = distances + z
            else:
                distances = z

        # If both are greater, compute the distance of the overlap
        if i > 0 and j > 0:
            z = ssd(self.X[0], patch[:overlap, :overlap, :])
            z = z[0: -tilesize[0] + overlap, 0: -tilesize[1] + overlap]
            distances = distances + z

        # if there is a mask on the input: remove all the points leading to
        # patches with an overlap on that area. i.e.: remove the area and
        # anything a tile-size distant on the left or top
        # to remove an area: replace it with inf value so that it won't be
        # chose computing the min.
        if self.Xmask is not None:
            distances = distances + self.Xmask[:distances.shape[0],
                                               :distances.shape[1]]

        return distances

    def compute_mask(self, src, dst, coord=None, overlap=None):

        [i, j] = coord
        overlap = overlap or self.overlap

        # Initialize the mask to all ones
        mask = np.ones((src.shape[0], src.shape[1]))

        # if it is the first one, or if src==dst, or is dst is empty
        if (i == 0 and j == 0) or np.all(src == dst) or not np.any(dst):
            return mask

        # We have a left overlap
        if j > 0:

            # difference
            diff = power(rgb2gray(src[:, :overlap] - dst[:, :overlap]), 2)
            # min-cut
            cut = mincut(diff, 0)
            # find the mask
            mask[:, :overlap] = cut

        # We have a top overlap
        if i > 0:

            # difference
            diff = power(rgb2gray(src[:overlap, :] - dst[:overlap, :]), 2)
            # min-cut
            cut = mincut(diff, 1)
            # find the mask
            mask[:overlap, :] = multiply(mask[:overlap, :], cut)

        # Write to the destination using the mask
        return mask

    # #########################################################################
    # #########################################################################
    # ################## MULTIPROCESS OPTIMIZATION ############################
    # #########################################################################

    def optimized_compute_big(self):
        """
        process 1: big tiles
         for each of the tile: process n

        """
        print '\nPYMAXFLOW MULTIPROCESSING COMPUTING ...'

        big_num_tiles = self.calc_num_tiles(tile_size=self.big_tilesize,
                                            overlap=self.big_overlap)

        # prepare the pool
        n_proc = min(big_num_tiles[0]*big_num_tiles[1], cpu_count()-2)
        out_queue = JoinableQueue()
        in_queue = Queue()
        pool = Pool(n_proc, unwrap_self, (self, in_queue, out_queue,))
        print 'preparing', n_proc, 'processes'

        for i in xrange(big_num_tiles[0]):

            startI = i * self.big_tilesize - i * self.big_overlap
            endI = min(self.Y[0].shape[0], startI + self.big_tilesize)
            sizeI = endI - startI
            if sizeI <= self.overlap:
                continue

            for j in xrange(big_num_tiles[1]):

                startJ = j * self.big_tilesize - j * self.big_overlap
                endJ = min(self.Y[0].shape[1], startJ + self.big_tilesize)
                sizeJ = endJ - startJ
                if sizeJ <= self.overlap:
                    continue

                if self.Ymask and not np.any(self.Ymask[startI:endI,
                                             startJ:endJ]):
                    continue

                dst_patches = [self.Y[l][startI:endI, startJ:endJ, :]
                               for l in xrange(len(self.Y))]
                res_patches = self._optimized_compute(
                               dst_patches, [sizeI, sizeJ], [i, j])
                for idx, res in enumerate(res_patches):
                    self.Y[idx][startI:endI, startJ:endJ, :] = res

                # make a process start in this big tile
                _img = [y[startI:endI, startJ:endJ, :] for y in self.Y]
                _mask = self.Ymask is not None and self.Ymask[startI:endI,
                                                        startJ:endJ, :] or None
                _id = (startI, startJ)
                in_queue.put({'dst': _img, 'mask': _mask, 'id': _id})
                # print _id, 'launched '

        # wait for all the children
        print 'master finished'
        show(self.Y[0])
        pool.close()
        # print 'closed, queue:', in_queue.qsize()
        out_queue.join()
        # print 'all children finished'

        # get the results
        results = sorted([out_queue.get() for _ in xrange(n_proc)])
        # sew them together
        for idx, res in results:
            # calculate the mask
            base_patch = self.Y[0][idx[0]:idx[0] + self.big_tilesize,
                                   idx[1]:idx[1] + self.big_tilesize]
            new_patch = res[0]
            mask_patch = self.compute_mask(base_patch, new_patch, coord=idx,
                                           overlap=self.big_overlap)
            # apply the mask to each layer
            for i, y in enumerate(self.Y):
                base_patch = y[idx[0]:idx[0] + self.big_tilesize,
                               idx[1]:idx[1] + self.big_tilesize]
                new_patch = res[i]
                self.Y[i][idx[0]:idx[0]+self.big_tilesize,
                          idx[1]:idx[1]+self.big_tilesize, :] = \
                    filter_img(new_patch, base_patch, mask_patch)

        show(self.Y[0], title='FINAL')
        if self.result_path:
            save(self.Y, self.result_path)

    def optimized_compute_small(self, in_queue, out_queue):

        item = in_queue.get(True)
        dst = item['dst']
        mask = item['mask']
        identifier = item['id']

        num_tiles = self.calc_num_tiles(img_size=dst[0].shape[0:2])
        # print identifier, 'started'

        for i in xrange(num_tiles[0]):
            startI = i * self.tilesize - i * self.overlap
            endI = min(dst[0].shape[0], startI + self.tilesize)
            sizeI = endI-startI
            if sizeI <= self.overlap:
                continue

            for j in xrange(num_tiles[1]):
                startJ = j * self.tilesize - j * self.overlap
                endJ = min(dst[0].shape[1], startJ + self.tilesize)
                sizeJ = endJ - startJ
                if sizeJ <= self.overlap:
                    continue

                if mask is not None and not np.any(mask[startI:endI, startJ:endJ]):
                    continue

                # if it is the first patch: instead of taking a random one,
                # take it from the background image (computed by the master)
                if i == 0 and j == 0:
                    continue

                dst_patches = [dst[l][startI:endI, startJ:endJ, :] for l in xrange(len(dst))]
                res_patches = self._optimized_compute(dst_patches,
                                                      [sizeI, sizeJ], [i, j])

                for idx, res in enumerate(res_patches):
                    dst[idx][startI:endI, startJ:endJ, :] = res

        # if self.result_path:
        #     save(dst, self.result_path)

        # show(dst[0], title='FINAL {0}'.format(identifier))
        out_queue.put([identifier, dst])
        out_queue.task_done()
        # print identifier, ': finished, left', in_queue.qsize()

    def _optimized_compute(self, y_patches, tilesize, coord):
        """
        Given a patch, calculates the resulting one.
          1) finds the best matching patch
          2) sews the patches together along the min-cut

        Args:
            y_patches: stack of the patch to be matched
            tilesize: patch size
            coord: tile coordinates in the destination image

        Returns:
            Matching patch from the src image, sewed with the input one
        """
        # Dist from each tile to the overlap region
        distances = self.distance(y_patches[0], coord=coord, tilesize=tilesize)

        # Find the best candidates for the match
        best = np.min(distances)
        candidates = where(distances <= (1 + self.err) * best)

        # choose a random best
        choice = ceil(rand() * (len(candidates[0]) - 1))
        sub = [candidates[0][choice], candidates[1][choice]]

        # sew them together
        x_patch = self.X[0][sub[0]:sub[0] + tilesize[0],
                            sub[1]:sub[1] + tilesize[1]]
        patch_mask = self.compute_mask(x_patch, y_patches[0], coord=coord)
        result = [zeros(layer.shape) for layer in y_patches]
        for idx, x in enumerate(self.X):
            x_patch = x[sub[0]:sub[0] + tilesize[0], sub[1]:sub[1]+tilesize[1]]
            result[idx] = filter_img(x_patch, y_patches[idx], patch_mask)
        return result
