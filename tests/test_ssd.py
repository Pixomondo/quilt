# import builtin modules
import unittest

# import 3rd party modules
import numpy as np

# import internal modules
import ssd


class TestSsd(unittest.TestCase):
    a = np.array([[64, 2, 3, 61, 60, 6, 7, 57],
                  [9, 55, 54, 12, 13, 51, 50, 16],
                  [17, 47, 46, 20, 21, 43, 42, 24],
                  [40, 26, 27, 37, 1, 1, 1, 33],
                  [32, 34, 35, 29, 1, 1, 1, 25],
                  [41, 23, 22, 44, 1, 1, 1, 48],
                  [49, 15, 14, 52, 53, 11, 10, 56]])

    def test_conv2_same_submatrix(self):
        k = np.array([[26, 27],
                      [34,  35],
                      [23,  22]])
        expected = np.array([[3981, 3061, 3949, 4837, 3981, 3125, 3949, 2427],
                             [5320, 6339, 5529, 4670, 5332, 6031, 5517, 2462],
                             [5412, 7065, 5445, 3002, 3709, 5258, 4639, 2083],
                             [5487, 5748, 5374, 3061, 1573, 2034, 3310, 2358],
                             [5459, 4766, 5394, 3100,  167,  167, 2941, 2897],
                             [5416, 3875, 5433, 5017, 1831,  671, 3990, 3742],
                             [3656, 2013, 3754, 4613, 2274,  770, 3380, 3016]])
        result = ssd.conv2(self.a, k, mode='same')
        self.assertEqual(expected.shape, result.shape)
        np.testing.assert_array_equal(expected, result)

    def test_conv2_same(self):
        k = np.array([[6],
                      [4],
                      [3]])
        expected = np.array([[310, 338, 336, 316, 318, 330, 328, 324],
                             [330, 508, 501, 351, 358, 480, 473, 379],
                             [335, 509, 508, 338, 129, 331, 324, 342],
                             [403, 449, 456, 382, 73,  139, 136, 354],
                             [494, 352, 353, 491, 13,  13,  13,  487],
                             [554, 284, 277, 575, 325, 73,  67,  603],
                             [319, 129, 122, 340, 215,  47,  43, 368]])
        result = ssd.conv2(self.a, k, mode='same')
        self.assertEqual(expected.shape, result.shape)
        np.testing.assert_array_equal(expected, result)

    def test_conv2_valid(self):
        k = np.array([[6],
                      [4],
                      [3]])
        expected = np.array([[330, 508, 501, 351, 358, 480, 473, 379],
                             [335, 509, 508, 338, 129, 331, 324, 342],
                             [403, 449, 456, 382, 73, 139, 136, 354],
                             [494, 352, 353, 491, 13, 13, 13, 487],
                             [554, 284, 277, 575, 325, 73, 67, 603]])
        result = ssd.conv2(self.a, k, mode='valid')
        self.assertEqual(expected.shape, result.shape)
        np.testing.assert_array_equal(expected, result)

    def test_conv2_valid_submatrix(self):
        k = np.array([[26, 27],
                      [34, 35],
                      [23, 22]])
        expected = np.array([[5320, 6339, 5529, 4670, 5332,  6031, 5517],
                             [5412, 7065, 5445, 3002, 3709, 5258, 4639],
                             [5487, 5748, 5374, 3061, 1573, 2034, 3310],
                             [5459, 4766, 5394, 3100,  167,  167, 2941],
                             [5416, 3875, 5433, 5017, 1831,  671, 3990]])

        result = ssd.conv2(self.a, k, mode='valid')
        self.assertEqual(expected.shape, result.shape)
        np.testing.assert_array_equal(expected, result)

    def test_filter2_same(self):
        k = np.array([[6],
                      [4],
                      [3]])
        expected = np.array([[283, 173, 174, 280, 279, 177, 178, 276],
                             [471, 373, 372, 474, 475, 369, 368, 478],
                             [242, 596, 589, 263, 165, 481, 471, 291],
                             [358, 488, 489, 355, 133, 265, 259, 351],
                             [491, 361, 368, 470,  13,  13,  13, 442],
                             [503, 341, 340, 506, 169,  43,  40, 510],
                             [442, 198, 188, 472, 218,  50,  46, 512]])

        result = ssd.filter2(k, self.a, mode='same')
        self.assertEqual(expected.shape, result.shape)
        np.testing.assert_array_equal(expected, result)

    def test_filter2_same_submatrix(self):
        k = np.array([[26, 27],
                      [34, 35],
                      [23, 22]])
        expected = np.array([[3663, 2626, 3743, 4736, 3671, 2722, 3735, 2306],
                             [5374, 5986, 5479, 4991, 5378, 5742, 5475, 2578],
                             [5434, 7288, 5427, 2951, 3979, 5653, 4749, 1991],
                             [5465, 5845, 5392, 3069, 1821, 2366, 3502, 2321],
                             [5469, 4799, 5380, 3044,  167,  167, 2905, 2812],
                             [5406, 4034, 5447, 4674, 1583,  595, 3877, 3570],
                             [3878, 2192, 4056, 4794, 2240,  777, 3622, 3152]])

        result = ssd.filter2(k, self.a, mode='same')
        self.assertEqual(expected.shape, result.shape)
        np.testing.assert_array_equal(expected, result)

    def test_filter2_valid(self):
        k = np.array([[6],
                      [4],
                      [3]])
        expected = np.array([[471, 373, 372, 474, 475, 369, 368, 478],
                             [242, 596, 589, 263, 165, 481, 471, 291],
                             [358, 488, 489, 355, 133, 265, 259, 351],
                             [491, 361, 368, 470,  13,  13,  13, 442],
                             [503, 341, 340, 506, 169,  43,  40, 510]])

        result = ssd.filter2(k, self.a, mode='valid')
        self.assertEqual(expected.shape, result.shape)
        np.testing.assert_array_equal(expected, result)

    def test_filter2_valid_submatrix(self):
        k = np.array([[26, 27],
                      [34, 35],
                      [23, 22]])
        expected = np.array([[5374, 5986, 5479, 4991, 5378, 5742, 5475],
                             [5434, 7288, 5427, 2951, 3979, 5653, 4749],
                             [5465, 5845, 5392, 3069, 1821, 2366, 3502],
                             [5469, 4799, 5380, 3044,  167,  167, 2905],
                             [5406, 4034, 5447, 4674, 1583,  595, 3877]])

        result = ssd.filter2(k, self.a, mode='valid')
        self.assertEqual(expected.shape, result.shape)
        np.testing.assert_array_equal(expected, result)

    def test_ssd(self):

        x = np.array([[92,  99,   1,   8,  15,  67,  74,  51,  58,  40],
                      [98,  80,   7,  14,  16,  73,  55,  57,  64,  41],
                      [4,   81,  88,  20,  22,  54,  56,  63,  70,  47],
                      [85,  87,  19,  21,   3,  60,  62,  69,  71,  28],
                      [86,  93,  25,   2,   9,  61,  68,  75,  52,  34],
                      [17,  24,  76,  83,  90,  42,  49,  26,  33,  65],
                      [23,   5,  82,  89,  91,  48,  30,  32,  39,  66],
                      [79,   6,  13,  95,  97,  29,  31,  38,  45,  72],
                      [10,  12,  94,  96,  78,  35,  37,  44,  46,  53]])
        x = np.asarray(np.dstack((x, x, x)))
        y = np.array([[0, 8, 1],
                      [1, 1, 3]])
        y = np.asarray(np.dstack((y, y, y)))
        expected = np.array([
            [97233, 48891,   1713,  29289, 51444, 67083, 61041, 45513],
            [85374, 62484,  25803,  26541, 40368, 60324, 62259, 55578],
            [82866, 64098,  27159,  21174, 37773, 61866, 71673, 60234],
            [89889, 51393,   3441,  22128, 42969, 73239, 74493, 55491],
            [63933, 64479,  61929,  59313, 57291, 51933, 46629, 42429],
            [38733, 77691, 124113,  99489, 67794, 23883, 20541, 35313],
            [39924, 65634, 117603, 107241, 64518, 19974, 20859, 42828],
            [44616, 78948, 120759, 101124, 56973, 20616, 26373, 42684]])

        result = np.floor(ssd.ssd(x, y))
        self.assertEqual(expected.shape, result.shape)
        np.testing.assert_array_equal(expected, result)


if __name__ == '__main__':
    unittest.main()
