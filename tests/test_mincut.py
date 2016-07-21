#! /usr/bin/env python
"""
Test of mincut.py
"""

# import builtin modules
import unittest

# import 3rd party modules
import numpy as np

# import internal modules
from quilt.mincut import mincut


class TestMinCut(unittest.TestCase):

    def test_values(self):
        """
        Test the returned matrix contains only 0 and 1 values.
        """
        a = np.array([[30, 39, 48, 1, 10, 19, 28],
                      [38, 47, 7, 9, 18, 27, 29],
                      [46, 6, 8, 17, 26, 35, 37],
                      [5, 14, 16, 25, 34, 36, 45],
                      [13, 15, 24, 33, 42, 44, 4]])
        result = mincut(a)
        np.testing.assert_array_equal([0, 1], np.unique(result))

    def test_size(self):
        """
        Test the returned matrix has the same shape of the input one
        """
        a = np.array([[30, 39, 48, 1, 10, 19, 28],
                      [38, 47, 7, 9, 18, 27, 29],
                      [46, 6, 8, 17, 26, 35, 37],
                      [5, 14, 16, 25, 34, 36, 45],
                      [13, 15, 24, 33, 42, 44, 4]])
        result = mincut(a)
        np.testing.assert_array_equal(a.shape, result.shape)

    def test_small(self):
        """
        Test expected output with a small matrix.
        """
        a = np.array([[7, 8, 9, 0, 0, 6],
                      [5, 9, 0, 4, 3, 3],
                      [8, 1, 3, 0, 6, 6]])
        result = mincut(a)
        expected = np.asarray([[0, 0, 0, 0, 0, 1],
                               [0, 0, 0, 1, 1, 1],
                               [0, 0, 0, 0, 1, 1]])
        self.assertEqual(expected.shape, result.shape)
        np.testing.assert_array_equal(expected, result)

    def test_big(self):
        """
        Test expected output with a big matrix.
        """
        a = np.array([[30, 39, 48, 1, 10, 19, 28],
                      [38, 47, 7, 9, 18, 27, 29],
                      [46, 6, 8, 17, 26, 35, 37],
                      [5, 14, 16, 25, 34, 36, 45],
                      [13, 15, 24, 33, 42, 44, 4]])
        result = mincut(a)
        expected = np.asarray([[0, 0, 0, 0, 1, 1, 1],
                               [0, 0, 0, 1, 1, 1, 1],
                               [0, 0, 1, 1, 1, 1, 1],
                               [0, 1, 1, 1, 1, 1, 1],
                               [0, 1, 1, 1, 1, 1, 1]])
        self.assertEqual(expected.shape, result.shape)
        np.testing.assert_array_equal(expected, result)

    def test_transpose(self):
        """
        Test expected output with inverted matrix direction.
        """
        a = np.array([[30, 39, 48, 1, 10, 19, 28],
                      [38, 47, 7, 9, 18, 27, 29],
                      [46, 6, 8, 17, 26, 35, 37],
                      [5, 14, 16, 25, 34, 36, 45],
                      [13, 15, 24, 33, 42, 44, 4]])
        result = mincut(a, direction=1)
        expected = np.asarray([[0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 1, 1, 1, 1],
                               [0, 0, 1, 1, 1, 1, 1],
                               [0, 1, 1, 1, 1, 1, 1],
                               [1, 1, 1, 1, 1, 1, 1]])
        self.assertEqual(expected.shape, result.shape)
        np.testing.assert_array_equal(expected, result)

if __name__ == '__main__':
    unittest.main()
