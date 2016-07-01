# import builtin modules
import unittest

# import 3rd party modules
import numpy as np

# import internal modules
from old.imagequilt import calc_distance, quilt


class TestDistance(unittest.TestCase):

    img = np.array([[64, 2, 3, 61, 60, 6, 7, 57],
                    [9, 55, 54, 12, 13, 51, 50, 16],
                    [17, 47, 46, 20, 21, 43, 42, 24],
                    [40, 26, 27, 37, 1, 1, 1, 33],
                    [32, 34, 35, 29, 1, 1, 1, 25],
                    [41, 23, 22, 44, 1, 1, 1, 48],
                    [49, 15, 14, 52, 53, 11, 10, 56]])
    img = np.asarray(np.dstack((img, img, img)))

    patch = np.array([[26, 27, 37],
                      [34, 35, 29],
                      [23, 22, 44]])
    patch = np.asarray(np.dstack((patch, patch, patch)))

    def test_submatrix_start(self):
        result = calc_distance(self.img, self.patch,
                               tilesize=3, overlap=1, coord=[0, 0])
        expected = np.zeros((4, 5))
        self.assertEqual(expected.shape, result.shape)
        np.testing.assert_array_equal(expected, result)

    def test_submatrix(self):
        """
        Test the submatrix 'patch' is found in the matrix 'img'
        """
        distances = calc_distance(self.img, self.patch,
                                  tilesize=3, overlap=1, coord=[1, 1])

        result_val = np.min(distances)
        result_idx = np.where(distances == result_val)

        expected_val = 0.0
        expected_idx = (np.array([3]), np.array([1]))

        self.assertEqual(expected_val, result_val)
        np.testing.assert_array_equal(expected_idx, result_idx)
        
    def test_patch(self):
        """
        Test that all the candidates are good choices
        """
        img = -np.eye(10)*2 + 1
        img = np.asarray(np.dstack((img, img, img)))

        patch = [[1,   1,  1,  1.],
                 [1,   1,  1,  1.],
                 [-1,  1,  0,  0.],
                 [1,  -1,  0,  0.]]
        patch = np.asarray(np.dstack((patch, patch, patch)))

        distances = calc_distance(img, patch, tilesize=4, overlap=2, coord=[2, 2])
        best = np.min(distances)
        candidates = np.where(distances <= best)

        expected = np.asarray([[1,  1, 1, 1],
                               [1,  1, 1, 1],
                               [-1, 1, 1, 1],
                               [1, -1, 1, 1]])

        for i in range(len(candidates[0])):
            sub = [candidates[0][i], candidates[1][i]]
            result = img[sub[0]:sub[0]+4, sub[1]:sub[1]+4, 1]
            np.testing.assert_array_equal(expected, result)


class TestQuilt(unittest.TestCase):

    def test_eye(self):

        img = np.eye(10)*0.5 + 0.5
        result = quilt(img, tilesize=4, num_tiles=3, overlap=2, err=0)

        # if there is a 1, its neighborhood must be:  1 .5 .5
        #                                             .5 1 .5
        #                                             .5 .5 1
        self.assertEqual((8, 8, 3), result.shape)
        np.testing.assert_array_equal(result[:, :, 0], result[:, :, 1], result[:, :, 2])

        result = result[:, :, 0]
        expected = np.asarray([[1, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1]])
        for i in range(1, result.shape[0]-1):
            for j in range(1, result.shape[1]-1):
                self.assertIn(result[i, j], [1, 0.5])
                if result[i, j] == 1:
                    np.testing.assert_array_equal(expected,
                                                  result[i-1:i+2, j-1:j+2])

                else:
                    np.testing.assert_array_equal(0.5, result[i-1, j-1])
                    np.testing.assert_array_equal(0.5, result[i+1, j+1])


if __name__ == '__main__':
    unittest.main()
