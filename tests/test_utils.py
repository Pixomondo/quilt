# import builtin modules
import unittest

# import 3rd party modules
import numpy as np
from numpy.testing import assert_array_equal
from PIL import Image
import mock

# import internal modules
from quilt import utils
from quilt.utils import filter_img, gray2rgb, rgb2gray, im2double, imresize
from quilt.utils import matrix2img, img2matrix, show, save


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
        expected = float_a
        assert_array_equal(expected, np.around(result, decimals=1))

    def test_im2double_fail(self):
        """
        Test im2double failure when given a matrix whose values are not
        float and not uint8.
        """
        float_a = np.array([[0, 0.5, 0.8],
                            [1, 0.3, 0.7]])
        int_a = np.uint16(float_a * 255)
        self.assertRaises(ValueError, im2double, int_a)

    def test_imresize_2Dmatrix_base(self):
        """
        Test imresize basic behaviour with input image as matrix.
        Test the result is consistent if no scale is applied.
        """
        img = np.asarray([[0., 1., .2, .3, .4],
                          [.5, .8, .7, .8, .9],
                          [1., .0, .2, .3, .4],
                          [.5, .7, .9, .9, .8]])
        # scale
        result = np.around(imresize(img, scale=1), decimals=1)
        self.assertTrue(isinstance(result, np.ndarray))
        assert_array_equal(img, result)

        # size
        result = np.around(imresize(img, [4, 5]), decimals=1)
        self.assertTrue(isinstance(result, np.ndarray))
        assert_array_equal(img, result)

        # height only
        result = np.around(imresize(img, height=4), decimals=1)
        self.assertTrue(isinstance(result, np.ndarray))
        assert_array_equal(img, result)

        # width only
        result = np.around(imresize(img, width=5), decimals=1)
        self.assertTrue(isinstance(result, np.ndarray))
        assert_array_equal(img, result)

        # height and width
        result = np.around(imresize(img, height=4, width=5), decimals=1)
        self.assertTrue(isinstance(result, np.ndarray))
        assert_array_equal(img, result)

    def test_imresize_3Dmatrix_base(self):
        """
        Test imresize basic behaviour with input image as matrix.
        Test the result is consistent if no scale is applied.
        """
        img = np.asarray([[0., 1., .2, .3, .4],
                          [.5, .8, .7, .8, .9],
                          [1., .0, .2, .3, .4],
                          [.5, .7, .9, .9, .8]])
        img = gray2rgb(img)

        # scale
        result = np.around(imresize(img, scale=1), decimals=1)
        self.assertTrue(isinstance(result, np.ndarray))
        assert_array_equal(img, result)

        # size
        result = np.around(imresize(img, [4, 5]), decimals=1)
        self.assertTrue(isinstance(result, np.ndarray))
        assert_array_equal(img, result)

        # height only
        result = np.around(imresize(img, height=4), decimals=1)
        self.assertTrue(isinstance(result, np.ndarray))
        assert_array_equal(img, result)

        # width only
        result = np.around(imresize(img, width=5), decimals=1)
        self.assertTrue(isinstance(result, np.ndarray))
        assert_array_equal(img, result)

        # height and width
        result = np.around(imresize(img, height=4, width=5), decimals=1)
        self.assertTrue(isinstance(result, np.ndarray))
        assert_array_equal(img, result)

    def test_imresize_2Dimg_base(self):
        """
        Test imresize basic behaviour with input image as Image.
        Test the result is consistent if no scale is applied.
        """
        img = np.asarray([[0., 1., .2, .3, .4],
                          [.5, .8, .7, .8, .9],
                          [1., .0, .2, .3, .4],
                          [.5, .7, .9, .9, .8]])
        img = Image.fromarray(img)

        # scale
        result = imresize(img, scale=1)
        self.assertTrue(isinstance(result, Image.Image))
        assert_array_equal(img, result)

        # size
        result = imresize(img, [4, 5])
        self.assertTrue(isinstance(result, Image.Image))
        assert_array_equal(img, result)

        # height only
        result = imresize(img, height=4)
        self.assertTrue(isinstance(result, Image.Image))
        assert_array_equal(img, result)

        # width only
        result = imresize(img, width=5)
        self.assertTrue(isinstance(result, Image.Image))
        assert_array_equal(img, result)

        # height and width
        result = imresize(img, height=4, width=5)
        self.assertTrue(isinstance(result, Image.Image))
        assert_array_equal(img, result)

    def test_imresize_3Dimg_base(self):
        """
        Test imresize basic behaviour with input image as Image.
        Test the result is consistent if no scale is applied.
        """
        img = np.asarray([[0., 1., .2, .3, .4],
                          [.5, .8, .7, .8, .9],
                          [1., .0, .2, .3, .4],
                          [.5, .7, .9, .9, .8]])
        img = Image.fromarray(gray2rgb(img), 'RGB')

        # scale
        result = imresize(img, scale=1)
        self.assertTrue(isinstance(result, Image.Image))
        assert_array_equal(img, result)

        # size
        result = imresize(img, [4, 5])
        self.assertTrue(isinstance(result, Image.Image))
        assert_array_equal(img, result)

        # height only
        result = imresize(img, height=4)
        self.assertTrue(isinstance(result, Image.Image))
        assert_array_equal(img, result)

        # width only
        result = imresize(img, width=5)
        self.assertTrue(isinstance(result, Image.Image))
        assert_array_equal(img, result)

        # height and width
        result = imresize(img, height=4, width=5)
        self.assertTrue(isinstance(result, Image.Image))
        assert_array_equal(img, result)

    def test_imresize_matrix_scale(self):
        """
        Test imresize result when a different scale is required.
        """
        img = np.asarray([[0., 1., .2, .3, .4],
                          [.5, .8, .7, .8, .9],
                          [1., .0, .2, .3, .4],
                          [.5, .7, .9, .9, .8]])
        # bigger
        result = imresize(img, scale=1.4)
        self.assertTrue(isinstance(result, np.ndarray))
        expected_size = (5, 7)
        assert_array_equal(expected_size, result.shape)

        # smaller
        result = imresize(img, scale=0.8)
        self.assertTrue(isinstance(result, np.ndarray))
        expected_size = (3, 4)
        assert_array_equal(expected_size, result.shape)

    def test_imresize_img_scale(self):
        """
        Test imresize result when a different scale is required.
        """
        img = np.asarray([[0., 1., .2, .3, .4],
                          [.5, .8, .7, .8, .9],
                          [1., .0, .2, .3, .4],
                          [.5, .7, .9, .9, .8]])
        img = Image.fromarray(gray2rgb(img), 'RGB')

        # bigger
        result = imresize(img, scale=1.4)
        self.assertTrue(isinstance(result, Image.Image))
        expected_size = (7, 5)
        assert_array_equal(expected_size, result.size)

        # smaller
        result = imresize(img, scale=0.8)
        self.assertTrue(isinstance(result, Image.Image))
        expected_size = (4, 3)
        assert_array_equal(expected_size, result.size)

    def test_imresize_matrix_size(self):
        """
        Test imresize result when a different size is required.
        """
        img = np.asarray([[0., 1., .2, .3, .4],
                          [.5, .8, .7, .8, .9],
                          [1., .0, .2, .3, .4],
                          [.5, .7, .9, .9, .8]])
        # bigger
        size = [5, 9]
        result = imresize(img, size)
        self.assertTrue(isinstance(result, np.ndarray))
        assert_array_equal(size, result.shape)
        result = imresize(img, height=size[0], width=size[1])
        assert_array_equal(size, result.shape)

        # smaller
        size = [3, 2]
        result = imresize(img, size)
        self.assertTrue(isinstance(result, np.ndarray))
        assert_array_equal(size, result.shape)
        result = imresize(img, height=size[0], width=size[1])
        assert_array_equal(size, result.shape)

        # other
        size = [10, 3]
        result = imresize(img, size)
        self.assertTrue(isinstance(result, np.ndarray))
        assert_array_equal(size, result.shape)
        result = imresize(img, height=size[0], width=size[1])
        assert_array_equal(size, result.shape)

    def test_imresize_img_size(self):
        """
        Test imresize result when a different size is required.
        """
        img = np.asarray([[0., 1., .2, .3, .4],
                          [.5, .8, .7, .8, .9],
                          [1., .0, .2, .3, .4],
                          [.5, .7, .9, .9, .8]])
        img = Image.fromarray(gray2rgb(img), 'RGB')

        # bigger
        size = [5, 9]
        result = imresize(img, size)
        self.assertTrue(isinstance(result, Image.Image))
        assert_array_equal(size[::-1], result.size)
        result = imresize(img, height=size[0], width=size[1])
        assert_array_equal(size[::-1], result.size)

        # smaller
        size = [3, 2]
        result = imresize(img, size)
        assert_array_equal(size[::-1], result.size)
        self.assertTrue(isinstance(result, Image.Image))
        result = imresize(img, height=size[0], width=size[1])
        assert_array_equal(size[::-1], result.size)

        # other
        size = [10, 3]
        result = imresize(img, size)
        assert_array_equal(size[::-1], result.size)
        self.assertTrue(isinstance(result, Image.Image))
        result = imresize(img, height=size[0], width=size[1])
        assert_array_equal(size[::-1], result.size)

    def test_matrix2img_2Dmatrix(self):
        """
        Test matrix2img with a 2d matrix as input.
        """
        matrix = np.asarray([[0., 1., .2, .3, .4],
                             [.5, .8, .7, .8, .9],
                             [1., .0, .2, .3, .4],
                             [.5, .7, .9, .9, .8]])
        result = matrix2img(matrix)
        # type
        self.assertTrue(isinstance(result, Image.Image))
        # dimensions
        assert_array_equal(matrix.shape[::-1], result.size)
        # values
        expected = np.uint8(matrix*255)
        assert_array_equal(expected, np.asarray(result))

    def test_matrix2img_3Dmatrix(self):
        """
        Test matrix2img with a 3d matrix as input.
        """
        matrix = np.asarray([[0., 1., .2, .3, .4],
                             [.5, .8, .7, .8, .9],
                             [1., .0, .2, .3, .4],
                             [.5, .7, .9, .9, .8]])
        matrix = gray2rgb(matrix)
        result = matrix2img(matrix)
        # type
        self.assertTrue(isinstance(result, Image.Image))
        # dimensions
        assert_array_equal(matrix.shape[0:2][::-1], result.size)
        # values
        expected = np.uint8(matrix * 255)
        assert_array_equal(expected, np.asarray(result))

    def test_matrix2img_2Dimg(self):
        """
        Test matrix2img with a gray image as input.
        """
        matrix = np.asarray([[0., 1., .2, .3, .4],
                             [.5, .8, .7, .8, .9],
                             [1., .0, .2, .3, .4],
                             [.5, .7, .9, .9, .8]])
        img = Image.fromarray(matrix)
        result = matrix2img(img)
        # type
        self.assertTrue(isinstance(result, Image.Image))
        # dimensions
        assert_array_equal(img.size, result.size)
        # values
        assert_array_equal(img, result)

    def test_matrix2img_3Dimg(self):
        """
        Test matrix2img with an RGB image as input.
        """
        matrix = np.asarray([[0., 1., .2, .3, .4],
                             [.5, .8, .7, .8, .9],
                             [1., .0, .2, .3, .4],
                             [.5, .7, .9, .9, .8]])
        img = Image.fromarray(gray2rgb(matrix), 'RGB')
        result = matrix2img(img)
        # type
        self.assertTrue(isinstance(result, Image.Image))
        # dimensions
        assert_array_equal(img.size, result.size)
        # values
        assert_array_equal(img, result)

    def test_img2matrix_2Dmatrix(self):
        """
        Test img2matrix with a 2d matrix as input.
        """
        matrix = np.asarray([[0., 1., .2, .3, .4],
                             [.5, .8, .7, .8, .9],
                             [1., .0, .2, .3, .4],
                             [.5, .7, .9, .9, .8]])
        result = img2matrix(matrix)
        # type
        self.assertTrue(isinstance(result, np.ndarray))
        # dimensions
        assert_array_equal(matrix.shape, result.shape)
        # values
        assert_array_equal(matrix, result)

    def test_img2matrix_3Dmatrix(self):
        """
        Test img2matrix with a 3d matrix as input.
        """
        matrix = np.asarray([[0., 1., .2, .3, .4],
                             [.5, .8, .7, .8, .9],
                             [1., .0, .2, .3, .4],
                             [.5, .7, .9, .9, .8]])
        matrix = gray2rgb(matrix)
        result = img2matrix(matrix)
        # type
        self.assertTrue(isinstance(result, np.ndarray))
        # dimensions
        assert_array_equal(matrix.shape, result.shape)
        # values
        assert_array_equal(matrix, result)

    def test_img2matrix_2Dimg(self):
        """
        Test img2matrix with a gray image as input.
        """
        matrix = np.asarray([[0., 1., .2, .3, .4],
                             [.5, .8, .7, .8, .9],
                             [1., .0, .2, .3, .4],
                             [.5, .7, .9, .9, .8]])
        img = matrix2img(matrix)
        result = img2matrix(img)
        # type
        self.assertTrue(isinstance(result, np.ndarray))
        # dimensions
        assert_array_equal(matrix.shape, result.shape)
        # values
        assert_array_equal(matrix, np.around(result, decimals=1))

    def test_img2matrix_3Dimg(self):
        """
        Test img2matrix with an RGB image as input.
        """
        matrix = np.asarray([[.3, 1., .2, .3, .4],
                             [.5, .8, .7, .8, .9],
                             [1., .0, .2, .3, .4],
                             [.5, .7, .9, .9, .8]])
        matrix = gray2rgb(matrix)
        img = matrix2img(matrix)
        result = img2matrix(img)
        # type
        self.assertTrue(isinstance(result, np.ndarray))
        # dimensions
        assert_array_equal(matrix.shape, result.shape)
        # values
        assert_array_equal(matrix, np.around(result, decimals=1))

    @mock.patch.object(utils.Image, '_show')
    def test_show_single_uint(self, mk_show):
        """
        Test show calls _show method in Image just once.
        Test with a single uint8 image as input.
        """
        a = np.array([[0, 0.5, 0.8],
                      [1, 0.3, 0.7]])
        show(np.uint8(gray2rgb(a)))
        mk_show.assert_called_once_with(mock.ANY, command=None, title=None)

    @mock.patch.object(utils.Image, '_show')
    def test_show_single_float(self, mk_show):
        """
        Test show calls _show method in Image just once with the right args.
        Test with a single float64 image as input.
        """
        a = np.array([[0, 0.5, 0.8],
                      [1, 0.3, 0.7]])
        a = matrix2img(gray2rgb(a))
        show(a)
        mk_show.assert_called_once_with(a, command=None, title=None)

    @mock.patch.object(utils.Image, '_show')
    def test_show_list(self, mk_show):
        """
        Test show calls _show method in Image just once with the right args.
        Test with a list of images as input.
        """
        a = matrix2img(gray2rgb(np.array([[0, 0.5, 0.8],
                                          [1, 0.3, 0.7]])))
        b = matrix2img(np.array([[0, 0.5, 0.8],
                                 [1, 0.3, 0.7],
                                 [1, 0.3, 0.7]]))
        c = matrix2img(gray2rgb(np.array([[0.5, 0.8],
                                          [0.3, 0.7],
                                          [0.3, 0.7]])))
        show([a, b, c])
        expected = [mock.call(a, command=None, title=None),
                    mock.call(b, command=None, title=None),
                    mock.call(c, command=None, title=None)]
        mk_show.assert_has_calls(expected)

    def test_show_fail(self):
        """
        Test show fails when the matrix type is not supported.
        """
        a = np.int16(np.array([[0, 0.5, 0.8],
                               [1, 0.3, 0.7]]))
        self.assertRaises(ValueError, show, a)

    def test_save_single_img(self):
        """
        Test save calls the save() method of the input image once and with the
        given path.
        """
        a = matrix2img(gray2rgb(np.array([[0, 0.5, 0.8],
                                          [1, 0.3, 0.7]])))
        path = 'custom/path'

        with mock.patch.object(a, 'save') as mk_save:
            save(a, path)
            mk_save.assert_called_once_with(path)

    def test_save_list_img(self):
        """
        Test save calls the save() method of each input images with the realtive
        paths.
        """
        a = matrix2img(gray2rgb(np.array([[0, 0.5, 0.8],
                                          [1, 0.3, 0.7]])))
        b = matrix2img(np.array([[0, 0.5, 0.8],
                                 [1, 0.3, 0.7],
                                 [1, 0.3, 0.7]]))
        c = matrix2img(gray2rgb(np.array([[0.5, 0.8],
                                          [0.3, 0.7],
                                          [0.3, 0.7]])))
        a_path = 'custom/path/a.jpg'
        b_path = 'custom/path/b.jpg'
        c_path = 'custom/path/c.jpg'

        with mock.patch.object(a, 'save') as mka:
            with mock.patch.object(b, 'save') as mkb:
                with mock.patch.object(c, 'save') as mkc:
                    save([a, b, c], [a_path, b_path, c_path])
                    mka.assert_called_once_with(a_path)
                    mkb.assert_called_once_with(b_path)
                    mkc.assert_called_once_with(c_path)

    @mock.patch.object(utils.Image.Image, 'save')
    def test_save_single_matrix(self, mk_save):
        """
        Test save calls the save() method of each input images with the realtive
        paths.
        """
        a = gray2rgb(np.array([[0, 0.5, 0.8],
                               [1, 0.3, 0.7]]))
        path = 'custom/path/a.jpg'
        save(a, path)
        mk_save.assert_called_once_with(path)

    def test_save_fail(self):
        """
        Test save fails when the matrix type is not supported.
        """
        a = np.int16(np.array([[0, 0.5, 0.8],
                               [1, 0.3, 0.7]]))
        self.assertRaises(ValueError, save, a, 'path')


