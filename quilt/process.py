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
import time

# import 3rd party modules
import numpy as np
from numpy import power, ceil, multiply, where, rot90, inf, zeros, ones
from numpy import fliplr, flipud
from numpy.random import rand

# import internal modules
from ssd import ssd
from mincut import mincut
from utils import gray2rgb, filter_img, rgb2gray, im2double, imresize
from utils import show, save


def unwrap_self(*arg, **kwarg):
    return Quilt.optimized_compute_small(*arg, **kwarg)


class Quilt:

    debug = True

    def __init__(self, X, output_size=None,
                 input_mask=None, cut_mask=None,
                 tilesize=30, overlap=10, big_tilesize=500, big_overlap=200,
                 rotations=0, flip=(0, 0),
                 error=0.002, constraint_start=False, cores=None,
                 result_path=None, niter=1):
        """
        Initialize Quilt class. Sets all the necessary parameters.

        Args:
            X: source image.
               Can be a list of images, where the first one is the master one.
               E.g. [color, bump, spec]. The result will be a stack with the
               same number of images.

            output_size: [height, width] desired size of the output image.

            input_mask: black and white image to mask the regions which not
                    wanted to appear in the output.
                      - white: remove
                      - black: pass
                   If present must have the same size of X.
            cut_mask: black and white image to describe the shape of the output
                      image.
                      - white: to be filled with texture
                      - black: remains empty

            tilesize: size of the tiles used to perform the computation.
            overlap: size of the overlap.
                     If present, has to be < tilesize. Default: tilesize / 5.
            big_tilesize: size of the tiles to use in the multiprocess
                          computation: used in the first process to divide the
                          image in big tiles, each of which is the assigned to
                          a different process.
                          Relevant only if optimized_compute is called.
            big_overlap: size of the overlap associated to the big tile.
                         If present, must be < big_tilesize.
                         Default: big_tilesize / 3.
                         Relevant only if optimized_compute is called.

            rotations: number of rotations of 90 degrees to apply to the source
                       texture in order to increase the variety of patches.
                       Accepted values: 0, 2, 4.
            flip: tuple of two booleans indicating the flip to apply to the
                  source texture in order to increase the variety of patches.
                  In the form: (flip_vertical, flip_horizontal)

            error: amount of error accepted when selecting the best matching
                   patches.
            result_path: path for the temporary results.
            constraint_start: flag to define a constraint on the first patch
                              (top left corner) of the dst image:
                               - False: random
                               - True: pick the first patch of the src image
            cores: number of available cores
            niter: number of iterations to perform in the single process
                   computation. Relevant only if compute is called.
        """

        # tile size and overlap
        if overlap >= tilesize:
            raise ValueError('Overlap must be less than the tile size')
        self.tilesize = long(tilesize)
        self.overlap = overlap or int(round(self.tilesize / 5))
        print 'overlap:', overlap, 'tile size:', self.tilesize

        # ------ images ------
        # src
        print 'rot:', rotations
        print 'flip:', flip
        X, self.X = self._set_src(X, rotate=rotations, flip=flip)
        print 'X: len:', len(X), '- dim:', self.X[0].shape
        self.Xmask = self._set_src_mask(input_mask, self.tilesize, reference=X,
                                        rotate=rotations, flip=flip)
        print 'Xmask:', self.Xmask is None and 'None' or self.Xmask.shape

        # dst
        if output_size:
            output_size = output_size or 2 * X.shape[0:2]
        elif cut_mask is not None:
            output_size = [cut_mask.shape[0], cut_mask.shape[1]]
        else:
            raise ValueError("Please provide the output_size or cut_mask")
        self.num_tiles = self.calc_num_tiles(output_size)
        print 'num tiles:', self.num_tiles
        self.Y = self._set_dst(output_size)
        print 'Y: len:', len(self.Y), '- dim:', self.Y[0].shape
        self.Ymask = self._set_cut_mask(cut_mask, output_size)

        # ----- big tiles -------------------
        if big_overlap >= big_tilesize:
            raise ValueError('Overlap must be less than the tile size')
        self.big_tilesize = min(long(big_tilesize), min(X.shape[0:2]) - 1)
        self.big_overlap = min(big_overlap, int(self.big_tilesize / 3))
        self.Xmask_big = self._set_src_mask(input_mask, self.big_tilesize,
                                            reference=X,
                                            rotate=rotations, flip=flip)
        print 'Big overlap:', self.big_overlap, 'tile size:', self.big_tilesize

        # ------ independent parameters ------

        # error in choosing best patches
        if error < 0:
            raise ValueError('Error must be a positive number')
        self.err = error
        self.constraint_start = constraint_start
        self.result_path = result_path
        self.cores = cores or cpu_count()-2
        self.preview = False
        self.niter = niter

    def _set_src(self, img, rotate=0, flip=None):
        """
        Manages source image/s:
            - turns it into a stack of images of the same size
            - images are set to float values in range [0, 1]
            - rotated images are added if required
        Args:
            img: source image/s: can be a single image or a stack of images of
                 the same size
            rotate: number of 90 degrees rotations to apply to each image in
                    the stack.
            flip: list of two booleans for [flip_vertical, flip_horizontal]

        Returns:
            - reference image (the first image of the stack with no rotations)
            - stack of the source images
        """
        # stack of images
        if not isinstance(img, list):
            img = [img]

        reference = None
        for i in xrange(len(img)):
            # images in the stack must have the same size
            if i and not img[i].shape[0:2] == img[0].shape[0:2]:
                raise ValueError('Chained images must have the same size. Got '
                                 '{0} and {1}'.format(img[i].shape[0:2],
                                                      img[0].shape[0:2]))
            # 3 channel images
            img[i] = gray2rgb(img[i])
            # float values in range [0, 1]
            img[i] = im2double(img[i])
            if not i:
                reference = img[i]
        # rotation
        img = [self.create_rotations(i, rotate) for i in img]
        img = [self.create_flip(i, flip) for i in img]

        return reference, img

    def _set_src_mask(self, mask, tilesize, reference=None,
                      rotate=0, flip=(0, 0)):
        """
        Manages the mask of the source image (if present):
            - checks the mask has the same size of the source image
            - turns it into a 1-channel image with values in {0, infinite}
            - expands masked areas:
              for every masked (= white) pixel: all the pixels tilesize-distant
              from it are masked. In this way all the tiles containing that
              pixel are removed. This is used in the calculation of the
              convolution between a patch and the source image.

        Args:
            mask: input mask
            reference: master source image without rotation
            rotate: number of 90 degrees rotations to be applied to the mask
            flip: tuple of two flag controlling flip transformations

        Returns:
            mask
        """
        # no mask
        if mask is None:
            if not rotate and flip == (0, 0):
                print 'Xmask: None'
                return
            # create the mask to remove the lines between the rotations
            mask = ones(reference.shape[0:2])

        # coherent with reference
        if reference is None:
            reference = self.X
        if not mask.shape[0:2] == reference.shape[0:2]:
            raise ValueError('X and Xmask cannot have different sizes:'
                             'X={0} Xmask={1}'.format(reference.shape,
                                                      mask.shape))
        # one channel image
        mask = rgb2gray(mask)

        # rotate
        mask = self.create_rotations(mask, rotate)
        mask = self.create_flip(mask, flip)

        # values in {0, inf}
        mask = im2double(mask)
        mask[mask < 0.7] = inf
        mask[mask <= 1] = 0

        # expand masked values:
        # if there is a mask on the input: remove all the points leading to
        # patches with an overlap on that area. i.e.: mask the area and
        # anything a tile-size distant on the left or top.
        # The mask will be summed to the convolution result. Doing so, when
        # searching the min of the convolution, masked areas will not be
        # considered since their value is infinite.

        # change overlap?
        for i in xrange(mask.shape[0] - self.overlap + 1):
            for j in xrange(mask.shape[1] - self.overlap + 1):
                # get the tile starting from the pixel
                tile = mask[i:i + tilesize, j:j + tilesize]
                # if it contains a masked value: also the pixel generating the
                # tile has to be masked
                if np.any(tile == inf):
                    mask[i, j] = inf

        print 'Xmask:', mask.shape, ', values:', np.unique(mask)
        return mask

    def _set_dst(self, size):
        """
        Manages the destination image, if given.
            - if no image is given, a zero-image is created according to size
            - the image is turned into a stack of images. The size of the stack
              is equal to the size of the stack of the source image.
        Args:
            size: image size

        Returns:
            stack of destination images.
        """
        # zero-value image
        img = zeros((size[0], size[1], 3))
        # stack of images
        res = [deepcopy(img) for _ in xrange(len(self.X))]
        return res

    def _set_cut_mask(self, mask, size):
        """
        Manages the cut mask for the destination image, if present.
         The mask is turned into a binary image of the same size of the
         destination image.

        Args:
            mask: mask of the destination image
            size: size of the destination image

        Returns:
            mask of the destination image
        """
        if mask is None:
            print 'Ymask: None'
            return

        # 1 channel
        if len(mask.shape) == 3:
            mask = rgb2gray(mask)
        elif not len(mask.shape) == 2:
            raise ValueError('Input mask must be 2 or 3 dimensional')

        # resize it according to size
        mask = im2double(imresize(mask, size=size))

        # binary values
        mask[mask < 0.01] = 0
        mask[mask > 0] = 1

        print 'Ymask:', mask.shape
        return mask

    @classmethod
    def create_rotations(cls, img, amount):
        """
        Generates the required rotations of the input image and builds an image
        containing the input image and its rotations (the remaining space is
        left zero). The final image is so composed:
                          _______
        If amount == 2:  | rot0  |
                         |_______|
                         |rot180 |
                         |_______|
                          _______ _____ _____
        If amount == 4:  | rot0  |     |     |
                         |_______| rot | rot |
                         |rot180 | 90  | -90 |
                         |_______|_____|_____|
        Args:
            img: image to be rotated
            amount: amount of rotation of 90 degrees.
                    E.g.: amount = 2 --> rotation = 90*2 = 180
                    Accepted values: {0, 2, 4}
        Returns:
            image composed of the input one and it rotations
        """
        # check the amount
        if not amount:
            return img
        if amount not in [2, 4]:
            raise ValueError('Rotation must be None, 2 or 4. Got {0}'.format(
                amount))

        # turn the input image into a 3-channel image
        third_dim = len(img.shape) == 3
        img = gray2rgb(img)

        rot = None
        if amount == 2:
            rot = zeros((img.shape[0] * 2 + 1, img.shape[1], 3))
            rot[:img.shape[0], :, :] = img
            rot[img.shape[0] + 1:, :, :] = rot90(img, 2)

        if amount == 4:
            # set so that height > width
            if img.shape[0] > img.shape[1]:
                img = rot90(img, 1)

            rot = zeros((max(img.shape[1], img.shape[0] * 2 + 1),
                         img.shape[1] + img.shape[0] * 2 + 2, 3))
            rot[:img.shape[0], :img.shape[1], :] = img
            rot[img.shape[0] + 1: img.shape[0] * 2 + 1, :img.shape[1], :] = \
                rot90(img, 2)
            rot[:img.shape[1], img.shape[1] + 1:img.shape[1] + 1 + img.shape[0],
                :] = rot90(img, 1)
            rot[:img.shape[1], img.shape[0] + img.shape[1] + 2:, :] = \
                rot90(img, 3)

        # if the input image had 1 channel only, also the result will do
        if not third_dim:
            rot = rot[:, :, 0]

        show(rot) if cls.debug else None

        return rot

    @classmethod
    def create_flip(cls, img, amount=(0, 0)):
        """
        Generates the required flips of the input image and builds an image
        containing the input image and its flips.
        The final image is so composed:
                              _______
        If amount = [0, 1]:  | img   |
                             |_______|
                             | flipV |
                             |_______|
                              _______ _______
        If amount = [1, 0]:  | img   | flipH |
                             |_______|_______|

                              _______ _______
        If amount = [1, 1]:  | img   | flipH |
                             |_______|_______|
                             | flipV |flipHV |
                             |_______|_______|
        Args:
            img: image to be rotated
            amount:
        Returns:
            image composed of the input one and it rotations
        """
        # check the amount
        if not amount or amount == (0, 0):
            return img

        # turn the input image into a 3-channel image
        third_dim = len(img.shape) == 3
        img = gray2rgb(img)

        # vertical
        if amount[0]:
            flip = zeros((img.shape[0] * 2 + 1, img.shape[1], 3))
            flip[:img.shape[0], :, :] = img
            flip[img.shape[0] + 1:, :, :] = flipud(img)
            img = flip

        # horizontal
        if amount[1]:
            flip = zeros((img.shape[0], img.shape[1] * 2 + 1, 3))
            flip[:, :img.shape[1], :] = img
            flip[:, img.shape[1] + 1:, :] = fliplr(img)
            img = flip

        # if the input image had 1 channel only, also the result will do
        if not third_dim:
            img = img[:, :, 0]

        show(img) if cls.debug else None

        return img

    @classmethod
    def val2tuple(cls, val):
        """
        Turns a value in a list [value, value].
        If the value is a list of one element, it appends the element again.

        Args:
            val: value to be transformed

        Returns:
            a list with two values
        """
        if not isinstance(val, list):
            val = [val, val]
        elif len(val) == 1:
            val = [val[0], val[0]]
        return val

    def calc_num_tiles(self, img_size=None, tile_size=None, overlap=None):
        """
        Calculates the number of tiles in an image. If the resulting number is
        not int, its ceil is returned.

        Args:
            img_size: size of the image to be filled with tiles.
                      Default: size of the destination image
            tile_size: size of the tile
                       Default: self.tilesize
            overlap: size of the overlap
                     Default: self.overlap

        Returns:
            the number of tiles that fits in the image.
        """
        img_size = img_size or self.Y[0].shape
        tile_size = tile_size or self.tilesize
        overlap = overlap or self.overlap
        num_tiles = [
            np.int(ceil((img_size[0] - overlap) / (tile_size - overlap))),
            np.int(ceil((img_size[1] - overlap) / (tile_size - overlap)))]
        return num_tiles

    def get_result(self):
        """
        Returns:
            result of quilting.
        """
        return self.Y

    def compute(self):
        """
        Single, traditional, single process computation.
        """
        print '\nCOMPUTING ...'

        for n in xrange(self.niter):
            for i in xrange(self.num_tiles[0]):

                startI = i * self.tilesize - i * self.overlap
                endI = min(self.Y[0].shape[0], startI + self.tilesize)
                sizeI = endI - startI
                if sizeI <= self.overlap:
                    continue

                for j in xrange(self.num_tiles[1]):

                    startJ = j * self.tilesize - j * self.overlap
                    endJ = min(self.Y[0].shape[1], startJ + self.tilesize)
                    sizeJ = endJ - startJ
                    if sizeJ <= self.overlap:
                        continue

                    # skip if this patch is not meant to be filled
                    if self.Ymask and not np.any(self.Ymask[startI:endI,
                                                            startJ:endJ]):
                        continue

                    # Dist from each tile to the overlap region
                    y_patches = [y[startI:endI, startJ:endJ, :] for y in self.Y]
                    res_patches, track = self._compute_patch(
                        y_patches, [sizeI, sizeJ], (i, j), mask=self.Xmask,
                        constraint_start=self.constraint_start)
                    for idx, res in enumerate(res_patches):
                        self.Y[idx][startI:endI, startJ:endJ, :] = res

                # if i % 5 == 0:
                #     print 'row', i, '/', self.num_tiles[0]
                #     if self.result_path:
                #         save(self.Y, self.result_path)
            # print 'iter', n, '/', self.niter
            show(self.Y[0]) if self.debug else None
            if self.result_path:
                save(self.Y[0], self.result_path)

    def distance(self, patch, coord=None, tilesize=None, overlap=None,
                 mask=None):
        """
        Calculates the distance between a the overlap regions of a given patch
        and the source. Considered overlaps: left and top of the patch.
        Optionally, it sums the mask of the source image to the result.

        Args:
            patch: patch whose distance from the source we want to compute.
            coord: coordinates of the patch in the destination image.
                    The coordinates are expressed in tile-space (i.e. number of
                    tiles, not of pixels)
            tilesize: size of the tile. Default: self.tilesize
            overlap: size of the overlap. Default: self.overlap
            mask: mask to apply
                  flag to specify if to use or not the source mask to remove
                      unwanted areas from the difference.

        Returns:
            matrix containing the distance between the overlaps of the patch
            and the master source image. The size of the distance matrix is
            equal to the size of the source image, minus the size of the tile.
        """
        [i, j] = coord
        tilesize = self.val2tuple(tilesize or self.tilesize)
        overlap = overlap or self.overlap

        distances = zeros((self.X[0].shape[0] - tilesize[0],
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

        # if there is a mask of the source: sum it to the distance matrix. In
        # this way masked pixels appear to have a big distance value, while
        # unmasked pixels keep their real distance value.
        mask = mask if mask is not None else self.Xmask
        if mask is not None:
            distances = distances+mask[:distances.shape[0], :distances.shape[1]]
        return distances

    def calc_patch_mask(self, src, dst, coord=None, overlap=None):
        """
        Calculates the blend mask between two patches according to the min cut
        in their top and left overlapping areas.
        It computes the min-cuts of the difference between the two patches both
        in their top and left overlapping areas. Then it uses them to build the
        blend mask of the patch.
                     ___________
                    |/\_/\_/\_/_| min-cut in the top overlap
            min-cut |\|         |
             in the |/|  PATCH  |
               left |\|         |
            overlap |_|_________|


        Args:
            src: "background" patch
            dst: "foreground" patch
            coord: coordinate in tile-size
            overlap: depth of the overlap regions

        Returns:
            blend mask: 0 where background, 1 where foreground
        """
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

    def optimized_compute(self, preview=False, tracked=None):
        """
        First process: it computes the quilt algorithm with big tiles,
        manages child processes and them combines the results.

         1) creates the child processes (number defined according to the
            available cores and the number of big tiles in the image)
         2) computes quilting with big tiles
         3) every time a tile is computed (and sewed with the image), it is put
            in a queue
        process 1: big tiles
         for each of the tile: process n

        """
        self.preview = preview
        track = {}

        print '\nMULTIPROCESSING COMPUTING ...'

        big_num_tiles = self.calc_num_tiles(tile_size=self.big_tilesize,
                                            overlap=self.big_overlap)

        # prepare the pool
        n_proc = min(big_num_tiles[0]*big_num_tiles[1], self.cores)
        out_queue = Queue()
        in_queue = JoinableQueue()
        pool = Pool(n_proc, unwrap_self, (self, in_queue, out_queue,))
        print 'preparing', n_proc, 'processes', time.strftime("%H:%M:%S")

        if self.Ymask is not None:
            # zero values will become inf
            Ymask_rgb = gray2rgb(self.Ymask)
            # use the mask as a draft of the dst img so that boundaries are
            # respected
            self.Y[0] = deepcopy(Ymask_rgb)

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

                dst_patches = [y[startI:endI, startJ:endJ, :] for y in self.Y]
                # for the big tiles don't consider the mask, since it would
                # remove most of the image because the tiles are so big
                res_patches, track_patch = self._compute_patch(
                    dst_patches, [sizeI, sizeJ], (i, j), mask=self.Xmask_big,
                    constraint_start=self.constraint_start, err=0.8,
                    tracked=None if not tracked else
                    tracked.get((startI, startJ))[0])
                track[(startI, startJ)] = track_patch.values()

                # add the mask on top
                if self.Ymask is not None:
                    res_patches = [r*Ymask_rgb[startI:endI, startJ:endJ]
                                   for r in res_patches]
                for idx, res in enumerate(res_patches):
                    self.Y[idx][startI:endI, startJ:endJ, :] = res

                # make a process start in this big tile
                _img = [y[startI:endI, startJ:endJ, :] for y in self.Y]
                _mask = self.Ymask[startI:endI, startJ:endJ] \
                    if self.Ymask is not None else None
                _id = (startI, startJ)
                _tracked = None if not tracked else \
                    tracked.get((startI, startJ))[1]
                in_queue.put({'dst': _img, 'mask': _mask, 'id': _id,
                              'tracked': _tracked})
                # print _id, 'launched '

        # wait for all the children
        print 'master finished:', time.strftime("%H:%M:%S")
        show(self.Y[0]) if self.debug else None
        pool.close()
        print 'closed, in queue:', in_queue.qsize(), 'out:', out_queue.qsize()
        in_queue.join()
        print 'all children finished', time.strftime("%H:%M:%S")

        # get the results
        results = sorted([out_queue.get() for _ in xrange(big_num_tiles[0]*big_num_tiles[1])])

        if self.preview:
            print 'preview ...'
            # do not sew them
            for idx, res, trk in results:
                # store tracking data
                track[idx].append(trk)
                # layers
                for i in xrange(len(self.Y)):
                    self.Y[i][idx[0]:idx[0] + self.big_tilesize,
                              idx[1]:idx[1] + self.big_tilesize, :] = res[i]

        else:
            print 'not preview ...'
            # sew them together
            for idx, res, trk in results:
                # store tracking data
                track[idx].append(trk)

                # calculate the mask
                base_patch = self.Y[0][idx[0]:idx[0] + self.big_tilesize,
                                       idx[1]:idx[1] + self.big_tilesize]
                new_patch = res[0]
                mask_patch = self.calc_patch_mask(base_patch, new_patch,
                                                  coord=idx,
                                                  overlap=self.big_overlap)
                # apply the mask to each layer
                for i, y in enumerate(self.Y):
                    base_patch = y[idx[0]:idx[0] + self.big_tilesize,
                                   idx[1]:idx[1] + self.big_tilesize]
                    new_patch = res[i]
                    self.Y[i][idx[0]:idx[0]+self.big_tilesize,
                              idx[1]:idx[1]+self.big_tilesize, :] = \
                        filter_img(new_patch, base_patch, mask_patch)

        # apply the mask again
        if self.Ymask is not None:
            self.Y = [r * Ymask_rgb for r in self.Y]

        show(self.Y[0]) if self.debug else None
        if self.result_path:
            save(self.Y[0], self.result_path)
            print 'saving', self.result_path

        return track

    def optimized_compute_small(self, in_queue, out_queue):

        for item in iter(in_queue.get, 'STOP'):

            dst = item['dst']
            mask = item['mask']
            identifier = item['id']
            tracked = item['tracked']
            track = {}

            num_tiles = self.calc_num_tiles(img_size=dst[0].shape[0:2])
            print identifier, 'started'

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

                    if mask is not None and not np.any(mask[startI:endI,
                                                       startJ:endJ]):
                        continue

                    # if it is the first patch: instead of taking a random one,
                    # take it from the background image (computed by the master)
                    if i == 0 and j == 0:
                        continue

                    dst_patches = [d[startI:endI, startJ:endJ, :] for d in dst]
                    res_patches, track_patch = self._compute_patch(
                           dst_patches, [sizeI, sizeJ], (i, j), mask=self.Xmask,
                           tracked=None if not tracked else tracked.get((i, j)))
                    # store the tracking data
                    track.update(track_patch)
                    # manage the layers
                    for idx, res in enumerate(res_patches):
                        dst[idx][startI:endI, startJ:endJ, :] = res

            # show(dst[0], title='FINAL {0}'.format(identifier))
            out_queue.put([identifier, dst, track])
            print identifier, ': finished, left', in_queue.qsize()
            in_queue.task_done()

    def _compute_patch(self, y_patches, tilesize, coord, mask=None,
                       constraint_start=False, tracked=None, err=None):
        """
        Given a patch, calculates the resulting one.
          1) finds the best matching patch
          2) sews the patches together along the min-cut

        Args:
            y_patches: stack of the patch to be matched
            tilesize: patch size
            coord: tile coordinates in the destination image
            use_mask: if to use Xmask to remove unwanted parts of the image

        Returns:
            Matching patch from the src image, sewed with the input one
        """

        if tracked:
            # we already know what to choose
            sub = tracked

        else:
            err = err or self.err
            # Find the best candidates for the match
            if coord == (0, 0) and constraint_start:
                # the first one of dst should be the first one of src
                sub = [0, 0]
            else:
                # Dist from each tile to the overlap region
                distances = self.distance(y_patches[0], coord=coord,
                                          tilesize=tilesize, mask=mask)
                best = np.min(distances)
                candidates = where(distances <= (1 + err) * best)

                # choose a random best
                choice = ceil(rand() * (len(candidates[0]) - 1))
                sub = [candidates[0][choice], candidates[1][choice]]

        x_patch = self.X[0][sub[0]:sub[0] + tilesize[0],
                            sub[1]:sub[1] + tilesize[1]]
        # dictionary to take track of the choices:
        track = {coord: sub}

        # preview: do not sew the results together
        if self.preview:
            result = []
            for idx, x in enumerate(self.X):
                x_patch = x[sub[0]:sub[0] + tilesize[0],
                            sub[1]:sub[1] + tilesize[1]]
                result.append(x_patch)
            return result, track

        # sew them together
        patch_mask = self.calc_patch_mask(x_patch, y_patches[0], coord=coord)
        result = [zeros(layer.shape) for layer in y_patches]
        for idx, x in enumerate(self.X):
            x_patch = x[sub[0]:sub[0] + tilesize[0], sub[1]:sub[1]+tilesize[1]]
            result[idx] = filter_img(x_patch, y_patches[idx], patch_mask)

        return result, track
