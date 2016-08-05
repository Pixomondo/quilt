# import builtin modules
import unittest
import os
from glob import glob
import shutil

# import 3rd party modules
import numpy as np
from numpy import zeros
from numpy.testing import assert_array_equal
from PIL import Image

# import internal modules
from quilt.process import Quilt
from quilt.utils import gray2rgb, im2double, imresize, img2matrix


class TestConstructor(unittest.TestCase):

    x = np.array([[9, 9, 1, 8, 5, 7, 4, 1, 8, 10],
                  [9, 8, 7, 4, 6, 3, 5, 7, 4, 1],
                  [4, 8, 8, 0, 2, 4, 6, 3, 0, 7],
                  [5, 7, 9, 1, 3, 0, 2, 9, 1, 8],
                  [6, 3, 5, 2, 9, 1, 8, 5, 2, 4],
                  [7, 4, 6, 3, 0, 2, 9, 6, 3, 5],
                  [3, 5, 2, 9, 1, 8, 0, 2, 9, 6],
                  [9, 6, 3, 5, 7, 9, 1, 8, 5, 2]])
    x = np.uint8(x)

    def test_src_size(self):
        """
        Test the source image is turned to rgb float and is not reshaped.
        """
        q = Quilt(self.x, output_size=self.x.shape)

        # there is just one image in the stack
        self.assertEqual(1, len(q.X))
        # rgb
        assert_array_equal((self.x.shape[0], self.x.shape[1], 3), q.X[0].shape)
        # float
        self.assertEqual('float', q.X[0].dtype)

        expected = gray2rgb(im2double(self.x))
        assert_array_equal(expected, q.X[0])

    def test_stack(self):
        """
        Test the the source images are edited consistently and a same number of
        destination images is prepared.
        """
        q = Quilt([self.x, self.x, self.x], output_size=(30, 20))

        # there are 3 images in the stacks
        expected_x = gray2rgb(im2double(self.x))
        expected_y = zeros((30, 20, 3))
        self.assertEqual(3, len(q.X))
        self.assertEqual(3, len(q.Y))

        for i in xrange(3):
            # src
            assert_array_equal((self.x.shape[0], self.x.shape[1], 3),
                               q.X[i].shape)
            self.assertEqual('float', q.X[i].dtype)
            assert_array_equal(expected_x, q.X[i])

            # dst
            assert_array_equal((30, 20, 3), q.Y[i].shape)
            assert_array_equal(expected_y, q.Y[i])

    def test_src_mask_mismatch(self):
        """
        Test failure if the given src image and the input mask have different
        sizes.
        """
        self.assertRaises(ValueError, Quilt, self.x, input_mask=zeros((50, 40)),
                          output_size=self.x.shape)

    def test_src_stack_mismatch(self):
        """
        Test failure if the given src images have different sizes.
        """
        self.assertRaises(ValueError, Quilt, [self.x, zeros((10, 100, 3))],
                          output_size=self.x.shape)

    def test_dst_img(self):
        """
        Test the destination image initializes as a zeros matrix of the required
        size. Test the cut_mask is also resized.
        """
        q = Quilt(self.x, output_size=(100, 200), cut_mask=zeros((100, 300)))

        # one layer in the stack
        self.assertEqual(1, len(q.Y))
        # rgb of the required size
        assert_array_equal((100, 200, 3), q.Y[0].shape)
        # all zeros
        expected = zeros((100, 200, 3))
        assert_array_equal(expected, q.Y[0])
        expected = zeros((100, 200))
        assert_array_equal(expected, q.Ymask)

    def test_dst_img_mask(self):
        """
        Test the destination image initializes as a zeros matrix of size of the
        cut_mask if no other size is specified.
        """
        q = Quilt(self.x, cut_mask=zeros((100, 200)))

        # one layer in the stack
        self.assertEqual(1, len(q.Y))
        # rgb of the required size
        assert_array_equal((100, 200, 3), q.Y[0].shape)
        # all zeros
        expected = zeros((100, 200, 3))
        assert_array_equal(expected, q.Y[0])

    def test_tiles_fail(self):
        """
        Test failure if tilesize < overlap
        """
        self.assertRaises(ValueError, Quilt, self.x, output_size=self.x.shape,
                          tilesize=10, overlap=20)

    def test_tiles_big_fail(self):
        """
        Test failure if Big_tilesize < big_overlap
        """
        self.assertRaises(ValueError, Quilt, self.x, output_size=self.x.shape,
                          big_tilesize=10, big_overlap=20)

    def test_big_tiles_over(self):
        """
        Test big_tilesize and big_overlap are adjusted on the src image size
        """
        q = Quilt(self.x, output_size=self.x.shape, big_tilesize=200,
                  big_overlap=100)
        # tilesize
        expected = self.x.shape[0] - 1
        self.assertEqual(expected, q.big_tilesize)
        # overlap
        expected = int(q.big_tilesize/3)
        self.assertEqual(expected, q.big_overlap)


class TestImages(unittest.TestCase):

    root = os.path.abspath(os.path.join(os.getcwd(), '..'))
    test_folder = os.path.join(root, r"data\test")
    src_paths = os.path.join(root, r"data\test\src*.jpg")
    src_path = os.path.join(root, r"data\test\src.jpg")
    imask_path = os.path.join(root, r"data\test\imask.jpg")
    cmask_path = os.path.join(root, r"data\test\cmask.jpg")
    temp_folder = os.path.join(test_folder, 'test_dst')

    @classmethod
    def setUpClass(cls):
        # destination folder
        if not os.path.isdir(cls.temp_folder):
            os.makedirs(cls.temp_folder)

        # load images
        _size = (60, 60)
        cls.src = [img2matrix(Image.open(p)) for p in glob(cls.src_paths)]
        cls.src = [imresize(s, *_size) for s in cls.src]
        cls.imask = img2matrix(imresize(Image.open(cls.imask_path), *_size))
        cls.cmask = img2matrix(Image.open(cls.cmask_path))
        cls.result = os.path.join(cls.temp_folder, 'result.png')

        cls.dst_size = (50, 20, 3)

        cls.q = Quilt(cls.src, output_size=cls.dst_size,
                      input_mask=cls.imask, cut_mask=cls.cmask,
                      tilesize=10, overlap=3, big_tilesize=20, big_overlap=5,
                      rotations=0, flip=(0, 0),
                      error=0.02, constraint_start=True, cores=1,
                      result_path=cls.result)
        cls.q.compute()
        cls.results = cls.q.get_result()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_folder)

    def test_result_layers(self):
        """
        Test src and dst have the same number of layers
        """
        expected = len(self.src)
        self.assertEqual(expected, len(self.results))

    def test_result_size(self):
        """
        Test result images have the required size
        """
        expected = self.dst_size
        for r in self.results:
            assert_array_equal(expected, r.shape)

    def test_result_values(self):
        """
        Test result images contains float values in the interval [0, 1]
        """
        for r in self.results:
            self.assertEqual('float', r.dtype)
            self.assertTrue(np.min(r) >= 0)
            self.assertTrue(np.max(r) <= 1)

    def test_imask(self):
        """
        Test black regions in the imask do not appear in the output. In this
        example, imask masks red pixels, so we expect to find no red pixels in
        the first layer of the result image
        """
        result = self.results[0]
        # round to be more robust
        result = np.ceil(result * 10)
        red = [10, 0, 0]
        indices = np.where(np.all(result == red, axis=-1))
        self.assertFalse(indices[0] and indices[1])

    def test_cmask(self):
        """
        Test black regions in the cmask are not covered with texture in the
        output.
        """
        result = self.results[0]
        cmask = imresize(self.cmask, *self.dst_size)
        result[cmask > 0] = 0
        assert_array_equal([0], np.unique(result))

    def test_saved_output(self):
        """
        Test output size
        """
        self.assertTrue(os.path.isfile(self.result))

        result = img2matrix(Image.open(self.result))
        # test size
        assert_array_equal(self.dst_size, result.shape)


class TestMatrices(unittest.TestCase):

    eye = np.eye(8) * 0.5 + 0.5
    chessboard = np.asarray([[1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
                             [1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
                             [0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
                             [0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
                             [1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
                             [1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
                             [0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
                             [0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
                             [1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
                             [1, 1, 0, 0, 1, 1, 0, 0, 1, 1]])

    def test_eye(self):
        """
        Test compute method on the known eye matrix. This method is more
        accurate but it is slower.
        Test the final result is a composition of eye matrices:
            for every pixel P, check its neighborhood as following:
                - if P = 1:  1 .5 .5      - if P = .5: .5  *  *
                            .5  P .5                    *  P  *
                            .5 .5  1                    *  * .5
        """
        q = Quilt(self.eye, output_size=(10, 10), tilesize=4, overlap=2,
                  error=0, constraint_start=True)
        q.compute()
        result = q.get_result()[0]

        # if there is a 1, its neighborhood must be:   1 .5 .5
        #                                             .5  1 .5
        #                                             .5 .5  1
        self.assertEqual((10, 10, 3), result.shape)
        assert_array_equal(result[:, :, 0], result[:, :, 1], result[:, :, 2])

        result = result[:, :, 0]
        expected = np.asarray([[1, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1]])
        for i in range(1, result.shape[0] - 1):
            for j in range(1, result.shape[1] - 1):
                self.assertIn(result[i, j], [1, 0.5])
                if result[i, j] == 1:
                    np.testing.assert_array_equal(
                        expected, result[i - 1:i + 2, j - 1:j + 2])
                else:
                    try:
                        np.testing.assert_array_equal(0.5, result[i - 1, j - 1])
                        np.testing.assert_array_equal(0.5, result[i + 1, j + 1])
                    except AssertionError as err:
                        print err

    def test_chessboard(self):
        """
        Test compute method on the chessboard matrix.
        Test the final result is still a chessboard matrix with tiles of the
        same size as the ones in the input.
        """
        q = Quilt(np.float64(self.chessboard), output_size=(16, 16), tilesize=4,
                  overlap=2, error=0, constraint_start=True)
        q.compute()
        result = q.get_result()[0]

        # check the dimension
        self.assertEqual((16, 16, 3), result.shape)
        assert_array_equal(result[:, :, 0], result[:, :, 1], result[:, :, 2])

        # check the values
        result = result[:, :, 0]
        expected = np.asarray([[1, 1, 0, 0],
                               [1, 1, 0, 0],
                               [0, 0, 1, 1],
                               [0, 0, 1, 1]])
        for i in xrange(0, result.shape[0] - 4, 4):
            for j in xrange(0, result.shape[1] - 4, 4):
                patch = result[i:i+4, j:j+4]
                assert_array_equal(expected, patch)

    @unittest.skip("Cython not working with multiprocessing")
    def test_chessboard_multiproc(self):
        """
        Test optimized_compute method on the chessboard matrix.
        Test the final result is still a chessboard matrix with tiles of the
        same size as the ones in the input.
        """
        q = Quilt(np.float64(self.chessboard), output_size=(16, 16), tilesize=4,
                  overlap=2, error=0, big_tilesize=8, big_overlap=3,
                  constraint_start=True)
        q.optimized_compute()
        result = q.get_result()[0]

        # check the dimension
        self.assertEqual((16, 16, 3), result.shape)
        assert_array_equal(result[:, :, 0], result[:, :, 1], result[:, :, 2])

        # check the values
        result = result[:, :, 0]
        expected = np.asarray([[1, 1, 0, 0],
                               [1, 1, 0, 0],
                               [0, 0, 1, 1],
                               [0, 0, 1, 1]])
        for i in xrange(0, result.shape[0] - 4, 4):
            for j in xrange(0, result.shape[1] - 4, 4):
                patch = result[i:i + 4, j:j + 4]
                assert_array_equal(expected, patch)


class TestMethods(unittest.TestCase):
    a = np.asarray([[0, 1, 2, 3],
                    [5, 6, 7, 8],
                    [4, 9, 2, 3]])

    def test_rotation2(self):
        """
        Test create_rotations with 180 rotation only.
        """
        result = Quilt.create_rotations(self.a, 2)
        expected = np.asarray([[0, 1, 2, 3],
                               [5, 6, 7, 8],
                               [4, 9, 2, 3],
                               [0, 0, 0, 0],
                               [3, 2, 9, 4],
                               [8, 7, 6, 5],
                               [3, 2, 1, 0]])
        np.testing.assert_array_equal(expected, result)

    def test_rotation4(self):
        """
        Test create_rotations with 4 rotations (every 90 degrees)
        """
        result = Quilt.create_rotations(self.a, 4)
        expected = np.asarray([[0, 1, 2, 3, 0, 3, 8, 3, 0, 4, 5, 0],
                               [5, 6, 7, 8, 0, 2, 7, 2, 0, 9, 6, 1],
                               [4, 9, 2, 3, 0, 1, 6, 9, 0, 2, 7, 2],
                               [0, 0, 0, 0, 0, 0, 5, 4, 0, 3, 8, 3],
                               [3, 2, 9, 4, 0, 0, 0, 0, 0, 0, 0, 0],
                               [8, 7, 6, 5, 0, 0, 0, 0, 0, 0, 0, 0],
                               [3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        np.testing.assert_array_equal(expected, result)

    def test_rotation4_invert(self):
        """
        Test create_rotations with 4 rotations (every 90 degrees). Test it with
        a matrix that has height>width
        """
        a = np.rot90(self.a, -1)
        result = Quilt.create_rotations(a, 4)
        expected = np.asarray([[0, 1, 2, 3, 0, 3, 8, 3, 0, 4, 5, 0],
                               [5, 6, 7, 8, 0, 2, 7, 2, 0, 9, 6, 1],
                               [4, 9, 2, 3, 0, 1, 6, 9, 0, 2, 7, 2],
                               [0, 0, 0, 0, 0, 0, 5, 4, 0, 3, 8, 3],
                               [3, 2, 9, 4, 0, 0, 0, 0, 0, 0, 0, 0],
                               [8, 7, 6, 5, 0, 0, 0, 0, 0, 0, 0, 0],
                               [3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        np.testing.assert_array_equal(expected, result)

    def test_rotation_fail(self):
        """
        Test create_rotations with amount not in [0, 2, 4]
        """
        self.assertRaises(ValueError, Quilt.create_rotations, self.a, 1)

    def test_flipV(self):
        """
        Test create_flip with vertical flipping.
        """
        result = Quilt.create_flip(self.a, (1, 0))
        expected = np.asarray([[0, 1, 2, 3],
                               [5, 6, 7, 8],
                               [4, 9, 2, 3],
                               [0, 0, 0, 0],
                               [4, 9, 2, 3],
                               [5, 6, 7, 8],
                               [0, 1, 2, 3]])
        np.testing.assert_array_equal(expected, result)

    def test_flipH(self):
        """
        Test create_flip with horizontal flipping.
        """
        result = Quilt.create_flip(self.a, (0, 1))
        expected = np.asarray([[0, 1, 2, 3, 0, 3, 2, 1, 0],
                               [5, 6, 7, 8, 0, 8, 7, 6, 5],
                               [4, 9, 2, 3, 0, 3, 2, 9, 4]])
        np.testing.assert_array_equal(expected, result)

    def test_flipVH(self):
        """
        Test create_flip with both the flip dimensions activated.
        """
        result = Quilt.create_flip(self.a, (1, 1))
        expected = np.asarray([[0, 1, 2, 3, 0, 3, 2, 1, 0],
                               [5, 6, 7, 8, 0, 8, 7, 6, 5],
                               [4, 9, 2, 3, 0, 3, 2, 9, 4],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [4, 9, 2, 3, 0, 3, 2, 9, 4],
                               [5, 6, 7, 8, 0, 8, 7, 6, 5],
                               [0, 1, 2, 3, 0, 3, 2, 1, 0]])
        np.testing.assert_array_equal(expected, result)

    def test_val2tuple(self):
        """
        Test val2tuple creates the same tuple from different inputs
        """
        expected = [10, 10]
        result1 = Quilt.val2tuple(10)
        result2 = Quilt.val2tuple([10])
        result3 = Quilt.val2tuple([10, 10])
        self.assertEqual(expected, result1)
        self.assertEqual(expected, result2)
        self.assertEqual(expected, result3)

    def test_distance_submatrix(self):
        """
        Test distance computation. Test a submatrix is found inside a bigger
        matrix.
        """
        img = np.array([[64, 2, 3, 61, 60, 6, 7, 57],
                        [9, 55, 54, 12, 13, 51, 50, 16],
                        [17, 47, 46, 20, 21, 43, 42, 24],
                        [40, 26, 27, 37, 1, 1, 1, 33],
                        [32, 34, 35, 29, 1, 1, 1, 25],
                        [41, 23, 22, 44, 1, 1, 1, 48],
                        [49, 15, 14, 52, 53, 11, 10, 56]])
        img = np.uint8(gray2rgb(img))
        patch = np.array([[26, 27, 37],
                          [34, 35, 29],
                          [23, 22, 44]])
        patch = im2double(np.uint8(gray2rgb(patch)))

        q = Quilt(img, output_size=(20, 20))
        q.tilesize = 3
        result = q.distance(patch, overlap=2, coord=(1, 1))
        expected = zeros((5, 6))
        self.assertEqual(expected.shape, result.shape)

        # check where the min is
        arg_min = np.where(result == np.min(result))
        expected = np.asarray([[3], [1]])
        assert_array_equal(expected, arg_min)

    def test_distance_candidates(self):
        """
        Test that all the candidates are good choices.
        """
        # eye matrix
        img = -np.eye(10) * 2 + 1
        img = np.asarray(np.dstack((img, img, img)))
        # parch is similar to a submatrix of img
        patch = np.asarray([[1,  1, 1, 1.],
                            [1,  1, 1, 1.],
                            [-1, 1, 0, 0.],
                            [1, -1, 0, 0.]])
        patch = im2double(gray2rgb(patch))

        # compute distance
        q = Quilt(img, output_size=(20, 20))
        q.tilesize = 4
        distances = q.distance(patch, overlap=2, coord=(2, 2))
        best = np.min(distances)
        candidates = np.where(distances <= best)

        # submatrix of img similar to patch
        expected = np.asarray([[1,  1, 1, 1],
                               [1,  1, 1, 1],
                               [-1, 1, 1, 1],
                               [1, -1, 1, 1]])
        for i in range(len(candidates[0])):
            sub = [candidates[0][i], candidates[1][i]]
            result = img[sub[0]:sub[0] + 4, sub[1]:sub[1] + 4, 1]
            assert_array_equal(expected, result)

