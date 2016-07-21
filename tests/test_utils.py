# import builtin modules
import unittest

# import 3rd party modules
import numpy as np
from numpy.testing import assert_array_equal

# import internal modules
from quilt.utils import filter_img, gray2rgb, rgb2gray, im2double


class MockImage(object):
    def show(self): pass


class TestUtils(unittest.TestCase):

    def test_filter_img(self):
        """
        Test filter_img. Given two images and a mask, test the result is correct
        """
        a = np.array([[30, 39, 48,  1],
                      [38, 47,  7,  9],
                      [46,  6,  8, 17],
                      [5,  14, 16, 25]])
        a = gray2rgb(a)
        b = np.array([[9,  18, 27, 29],
                      [17, 26, 35, 37],
                      [25, 34, 36, 45],
                      [33, 42, 44, 4]])
        b = gray2rgb(b)
        m = np.array([[0, 0, 1, 1],
                      [0, 1, 1, 1],
                      [1, 1, 1, 1],
                      [0, 0, 0, 1]])
        result = filter_img(a, b, m)
        expected = np.asarray([[9,  18, 48,  1],
                               [17, 47,  7,  9],
                               [46,  6,  8, 17],
                               [33, 42, 44, 25]])
        expected = gray2rgb(expected)

        # size
        self.assertEqual(expected.shape, result.shape)
        # values
        np.testing.assert_array_equal(expected, result)

    def test_filter_img_fail(self):
        """
        Test failure of filter_img when images do not have the same size.
        """
        # text one image is different
        a = np.ones((10, 10, 3))
        b = np.zeros((10, 5, 3))
        m = np.ones((10, 10))
        self.assertRaises(ValueError, filter_img, a, b, m)

        # text the mask is different
        b = np.zeros((10, 10, 3))
        m = np.ones((10, 5))
        self.assertRaises(ValueError, filter_img, a, b, m)

    def test_gray2rgb_gray(self):
        """
        Test gray2rgb behaviour when given a mono-channel matrix.
        """
        a = np.array([[0, 9, 8],
                      [8, 7, 9]])
        result = gray2rgb(a)

        # size
        expected_size = (a.shape[0], a.shape[1], 3)
        assert_array_equal(expected_size, result.shape)

        # values are from input
        expected_values = np.unique(a)
        assert_array_equal(expected_values, np.unique(result))

        # test with matrix
        expected = np.array([[[0, 0, 0], [9, 9, 9], [8, 8, 8]],
                             [[8, 8, 8], [7, 7, 7], [9, 9, 9]]])
        assert_array_equal(expected, result)

    def test_gray2rgb_rgb(self):
        """
        Test gray2rgb behaviour when given a 3-channels matrix.
        """
        a = np.array([[[0, 0, 0], [9, 9, 9], [8, 8, 8]],
                      [[8, 8, 8], [7, 7, 7], [9, 9, 9]]])
        result = gray2rgb(a)

        expected = a
        assert_array_equal(expected, result)

    def test_gray2rgb_fail(self):
        """
        Test gray2rgb fails when given a 4channels matrix.
        """
        a = np.array([[[0, 0, 0, 0], [9, 9, 9, 9], [8, 8, 8, 8]],
                      [[8, 8, 8, 8], [7, 7, 7, 7], [9, 9, 9, 9]]])
        self.assertRaises(ValueError, gray2rgb, a)

    def test_rgb2gray_gray(self):
        """
        Test rgb2gray behaviour when given a 3-channel matrix.
        """
        a = np.array([[0, 9, 8],
                      [8, 7, 9]])
        result = rgb2gray(a)

        # size
        expected_size = a.shape
        assert_array_equal(expected_size, result.shape)

        # test with matrix
        expected = a
        assert_array_equal(expected, np.round(result))

    def test_rgb2gray_rgb(self):
        """
        Test rgb2gray behaviour when given a 3-channel matrix.
        """
        a = np.array([[[0, 0, 0], [9, 9, 9], [8, 8, 8]],
                     [[8, 8, 8], [7, 7, 7], [9, 9, 9]]])
        result = rgb2gray(a)

        # size
        expected_size = (a.shape[0], a.shape[1])
        assert_array_equal(expected_size, result.shape)

        # values
        self.assertEqual('float', result.dtype)

        # test with matrix
        expected = np.array([[0, 9, 8],
                             [8, 7, 9]])
        assert_array_equal(expected, np.round(result))

    def test_rgb2gray_fail(self):
        """
        Test gray2rgb fails when given a 4channels matrix.
        """
        a = np.array([[[0, 0, 0, 0], [9, 9, 9, 9], [8, 8, 8, 8]],
                      [[8, 8, 8, 8], [7, 7, 7, 7], [9, 9, 9, 9]]])
        self.assertRaises(ValueError, rgb2gray, a)

    def test_im2double_uint8(self):
        """
        Test im2double behaviour when given a matrix with uint8 values.
        """
        float_a = np.array([[0, 0.5, 0.8],
                            [1, 0.3, 0.7]])
        int_a = np.uint8(float_a*255)
        result = im2double(int_a)

        self.assertEqual('float', result.dtype)

        # round at the first decimal
        expected = float_a*10
        assert_array_equal(expected, np.round(result*10))

    def test_im2double_fail(self):
        """
        Test im2double failure when given a matrix whose values are not
        float and not uint8.
        """
        float_a = np.array([[0, 0.5, 0.8],
                            [1, 0.3, 0.7]])
        int_a = np.uint16(float_a * 255)
        self.assertRaises(ValueError, im2double, int_a)

    # @mock.patch('utils.toimage')
    # def test_show(self, mk_toimage):
    #
    #     a = np.array([[0, 0.5, 0.8],
    #                   [1, 0.3, 0.7]])
    #     a = gray2rgb(a)
    #     utils.show(a)
    #     mk_toimage.assert_called_once_with(mock.ANY)
    #     mk_toimage.show.assert_called_once_with(mock.ANY)

