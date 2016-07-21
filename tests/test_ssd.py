#! /usr/bin/env python
"""
Test ssd.py
"""

# import builtin modules
import unittest

# import 3rd party modules
import numpy as np
import mock

# import internal modules
from quilt import ssd


class TestSsd(unittest.TestCase):
    img = np.array([[92, 99, 1, 8, 15, 67, 74, 51, 58, 40],
                    [98, 80, 7, 14, 16, 73, 55, 57, 64, 41],
                    [4, 81, 88, 20, 22, 54, 56, 63, 70, 47],
                    [85, 87, 19, 21, 3, 60, 62, 69, 71, 28],
                    [86, 93, 25, 2, 9, 61, 68, 75, 52, 34],
                    [17, 24, 76, 83, 90, 42, 49, 26, 33, 65],
                    [23, 5, 82, 89, 91, 48, 30, 32, 39, 66],
                    [79, 6, 13, 95, 97, 29, 31, 38, 45, 72],
                    [10, 12, 94, 96, 78, 35, 37, 44, 46, 53]])
    img3 = np.asarray(np.dstack((img, img, img)))
    patch = np.array([[0, 8, 1],
                      [1, 1, 3]])
    patch3 = np.asarray(np.dstack((patch, patch, patch)))
    a = np.array([[64, 2, 3, 61, 60, 6, 7, 57],
                  [9, 55, 54, 12, 13, 51, 50, 16],
                  [17, 47, 46, 20, 21, 43, 42, 24],
                  [40, 26, 27, 37, 1, 1, 1, 33],
                  [32, 34, 35, 29, 1, 1, 1, 25],
                  [41, 23, 22, 44, 1, 1, 1, 48],
                  [49, 15, 14, 52, 53, 11, 10, 56]])

    def test_output_values(self):
        """
        Test the resulting matrix contains positive floats only.
        """
        result = ssd.ssd(self.img.astype('float'), self.patch.astype('float'))
        self.assertEqual(type(result), np.ndarray)
        self.assertEqual(result.dtype, 'float')
        self.assertTrue(np.all(result >= 0))

    def test_example(self):
        """
        Test computation of the difference between two matrices with and example
        """
        expected = np.array([
            [97233, 48891,   1713,  29289, 51444, 67083, 61041, 45513],
            [85374, 62484,  25803,  26541, 40368, 60324, 62259, 55578],
            [82866, 64098,  27159,  21174, 37773, 61866, 71673, 60234],
            [89889, 51393,   3441,  22128, 42969, 73239, 74493, 55491],
            [63933, 64479,  61929,  59313, 57291, 51933, 46629, 42429],
            [38733, 77691, 124113,  99489, 67794, 23883, 20541, 35313],
            [39924, 65634, 117603, 107241, 64518, 19974, 20859, 42828],
            [44616, 78948, 120759, 101124, 56973, 20616, 26373, 42684]])

        result = np.floor(ssd.ssd(self.img3, self.patch3))
        self.assertEqual(expected.shape, result.shape)
        np.testing.assert_array_equal(expected, result)

    def test_ssd_3channels(self):
        """
        Test ssd function calls sumsqdiff three times if input matrix has tree
        channels, and that it sums the results together.
        """
        temp_res = np.asarray([[0,   1,  2,  3,  4],
                               [5,   6,  7,  8,  9],
                               [10, 11, 12, 13, 14],
                               [15, 16, 17, 18, 19]])

        with mock.patch('quilt.ssd.sumsqdiff', return_value=temp_res) as mk:
            result = ssd.ssd(self.img3, self.patch3)
            self.assertEqual(3, len(mk.mock_calls))

            for c in mk.mock_calls:
                # sumsqdiff called with 2 args
                self.assertEqual(2, len(c[1]))
                # check the args are correct
                np.testing.assert_array_equal(self.img, c[1][0])
                np.testing.assert_array_equal(self.patch, c[1][1])

        expected = temp_res*3
        np.testing.assert_array_equal(expected, result)

    def test_ssd_1channel(self):
        """
        Test ssd function calls sumsqdiff once if input matrix has only one
        channel, and that it sums the results together.
        """
        temp_res = np.asarray([[0, 1, 2, 3, 4],
                               [5, 6, 7, 8, 9],
                               [10, 11, 12, 13, 14],
                               [15, 16, 17, 18, 19]])

        with mock.patch('quilt.ssd.sumsqdiff', return_value=temp_res) as mk:
            result = ssd.ssd(self.img, self.patch)
            self.assertEqual(1, len(mk.mock_calls))

            c = mk.mock_calls[0]
            # sumsqdiff called with 2 args
            self.assertEqual(2, len(c[1]))
            # check the args are correct
            np.testing.assert_array_equal(self.img, c[1][0])
            np.testing.assert_array_equal(self.patch, c[1][1])

        np.testing.assert_array_equal(temp_res, result)

    def test_ssd_fail(self):
        """
        Test ssd function raises ValueError when the sizes of the input values
        are inconsistent.
        """
        self.assertRaises(ValueError, ssd.ssd, self.img3, self.patch)
