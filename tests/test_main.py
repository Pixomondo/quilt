#! /usr/bin/env python
"""
Test quilt command line interface.
"""

# import builtin modules
import unittest
import traceback
import os
from multiprocessing import cpu_count
import shutil
from copy import deepcopy

# import 3rd party modules
from click.testing import CliRunner
import mock
import numpy as np
from PIL import Image

# import internal modules
import quilt
from quilt.main_cmd import cli
from quilt.process import Quilt
from quilt.utils import img2matrix, imresize


class Cli(unittest.TestCase):

    # data needed in all the test classes
    root = os.path.abspath(os.path.join(os.getcwd(), '..'))
    test_folder = os.path.join(root, r'data\test')
    src_glob = os.path.join(test_folder, 'src*.jpg')
    src_paths = [os.path.join(test_folder, 'src.jpg'),
                 os.path.join(test_folder, 'src_layer1.jpg'),
                 os.path.join(test_folder, 'src_layer2.jpg')]
    imask_path = os.path.join(test_folder, 'imask.jpg')
    cmask_path = os.path.join(test_folder, 'cmask.jpg')
    dst_folder = os.path.join(test_folder, 'dst')
    src = [Image.open(i) for i in src_paths]
    src_mtx = [img2matrix(s) for s in src]

    @classmethod
    def setUpClass(cls):
        """
        Prepare the basic environment to run cli tests.
        """
        cls.runner = CliRunner()
        Quilt.debug = False

    def invoke(self, args, expected_code=0):
        """
        Invoke quilt with args and test that the return code is as expected.
        """
        result = self.runner.invoke(cli, args)
        try:
            self.assertEqual(expected_code, result.exit_code)
            return result
        except AttributeError:
            print 'Test failed due to:'
            print result.output
            raise
        except Exception:
            print 'Test errored due to:'
            print result.output
            print result.exception
            traceback.print_tb(result.exc_info[-1])
            raise


class TestCli(Cli):

    @mock.patch('quilt.main_cmd.save', return_value=None)
    @mock.patch('quilt.main_cmd.Quilt')
    def test_defaults(self, mk_quilt, mk_save):
        """
        Test set of arguments with which Quilt class is called.
        Test the default values of those arguments.
        """
        self.invoke([self.src_glob])

        kwargs = {
            'input_mask': None,
            'cut_mask': None,
            'tilesize': 30,
            'overlap': 10,
            'big_tilesize': 500,
            'big_overlap': 100,
            'error': 0.2,
            'constraint_start': False,
            'cores': cpu_count()-2,
            'rotations': 0,
            'output_size': self.src[0].size[0:2][::-1],
            'result_path': os.path.join(self.test_folder, 'result_temp.png'),
        }
        np.testing.assert_array_equal(self.src_mtx, mk_quilt.call_args[0][0])
        np.testing.assert_array_equal(kwargs, mk_quilt.call_args[0][1:])

    @mock.patch('quilt.main_cmd.save', return_value=None)
    @mock.patch('quilt.main_cmd.Quilt')
    def test_input_scale(self, mk_quilt, mk_save):
        """
        Test the input image is resize before it is passed to Quilt.
        """
        self.invoke([self.src_glob, '-iscale', 0.8])
        src = [imresize(i, scale=0.8) for i in self.src_mtx]
        np.testing.assert_array_equal(src, mk_quilt.call_args[0][0])

    @mock.patch('quilt.main_cmd.save', return_value=None)
    @mock.patch('quilt.main_cmd.Quilt')
    def test_custom(self, mk_quilt, mk_save):
        """
        Test set of arguments with which Quilt class is called when all the
        command line options are set.
        """
        self.invoke([self.src_paths[0],
                     '-imask', self.imask_path,
                     '-iscale', 0.5,
                     '-dst', os.path.join(self.test_folder, 'dst.jpg'),
                     '-owidth', 10,
                     '-oheight', 20,
                     '-cmask', self.cmask_path,
                     '-tile', 5,
                     '-over', 2,
                     '-btile', 10,
                     '-bover', 4,
                     '--constraint_start', True,
                     '--cores', 1,
                     '-err', 0.1,
                     '-rot', 4
                     ])
        src = imresize(deepcopy(self.src_mtx[0]), scale=0.5)

        kwargs = {
            'input_mask':
                img2matrix(imresize(Image.open(self.imask_path), scale=0.5)),
            'cut_mask': img2matrix(Image.open(self.cmask_path)),
            'tilesize': 5,
            'overlap': 2,
            'big_tilesize': 10,
            'big_overlap': 4,
            'error': 0.1,
            'constraint_start': True,
            'cores': 1,
            'rotations': 4,
            'flip': (False, False),
            'output_size': [20, 10],
            'result_path': os.path.join(self.test_folder, 'dst_temp.jpg'),
        }

        np.testing.assert_array_equal([src], mk_quilt.call_args[0][0])
        np.testing.assert_array_equal(sorted(kwargs),
                                      sorted(mk_quilt.call_args[1]))

    def test_fail_src(self):
        """
        Test failure if src is not a vail path.
        """
        bad_src = os.path.join(self.test_folder, 'bad_src*.jpg')
        self.invoke([bad_src], expected_code=2)

    def test_fail_dst_path(self):
        """
        Test failure if the destination path is not valid.
        """
        # it could be a path to a non existing folder
        bad_dst = os.path.join(self.test_folder, 'bad_folder')
        self.invoke([self.src_glob, '-dst', bad_dst], expected_code=2)

        # it could be a path to a file in a non existing folder
        bad_dst = os.path.join(self.test_folder, 'bad_folder', 'test.jpg')
        self.invoke([self.src_glob, '-dst', bad_dst], expected_code=2)


class TestComputation(Cli):

    def setUp(self):
        """
        Create temporal destination folder
        """
        if not os.path.isdir(self.dst_folder):
            os.mkdir(self.dst_folder)

    def tearDown(self):
        """"""
        shutil.rmtree(self.dst_folder)

    @mock.patch.object(quilt.main_cmd.Quilt, 'compute')
    def test_single_proc(self, mk_compute):
        """
        Test Quilt.compute is called if a single process computation is required
        """
        # set up
        self.invoke([self.src_glob, '-iscale', 0.2, '-multiproc', False,
                     '-dst', self.dst_folder])

        # it called compute instead of optimized_compute
        mk_compute.assert_called_once_with()

    @mock.patch.object(quilt.main_cmd.Quilt, 'optimized_compute')
    def test_multi_proc(self, mk_optimized):
        """
        Test Quilt.optimized_compute is called if a multi process computation
        is required.
        """
        self.invoke([self.src_glob, '-iscale', 0.2, '-dst', self.dst_folder])

        # it called compute instead of optimized_compute
        mk_optimized.assert_called_once_with()

    @mock.patch.object(quilt.main_cmd.Quilt, 'compute')
    def test_multi_results(self, mk_compute):
        """
        Test the result images saved to disk when a stack of images is given as
        source. Test location, number and names.
        """
        self.invoke([self.src_glob, '-iscale', 0.2, '-multiproc', False,
                     '-dst', self.dst_folder])

        # check the results
        results = os.listdir(self.dst_folder)
        # right number of results
        self.assertEqual(len(self.src), len(results))
        # right names
        expected = ['src_result_layer0.png', 'src_result_layer1.png',
                    'src_result_layer2.png']
        self.assertEqual(sorted(expected), sorted(results))

    @mock.patch.object(quilt.main_cmd.Quilt, 'compute')
    def test_single_result(self, mk_compute):
        """
        Test the result image saved to disk when a single image is given as
        source. Test location, number and names.
        """
        self.invoke([self.src_paths[0], '-iscale', 0.2, '-multiproc', False,
                     '-dst', self.dst_folder])

        # check the results
        results = os.listdir(self.dst_folder)
        # right number of results
        self.assertEqual(1, len(results))
        # right names
        expected = ['src_result.png']
        self.assertEqual(sorted(expected), sorted(results))

