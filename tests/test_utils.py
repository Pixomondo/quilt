# import builtin modules
import unittest

# import 3rd party modules
import numpy as np

# import internal modules
from utils import filter_img, gray2rgb


class TestUtils(unittest.TestCase):

    def test_gray2rgb(self):
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
        self.assertEqual(expected.shape, result.shape)
        np.testing.assert_array_equal(expected, result)

if __name__ == '__main__':
    unittest.main()
