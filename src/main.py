#! /usr/bin/env python
"""
Launch image quilt.
"""

# import builtin modules
from __future__ import division
import time
import sys
import os
sys.path.insert(0, r"C:\dev\textures\Lib\site-packages")

# import 3rd party modules
import numpy as np
import OpenImageIO as oiio
from PIL import Image
from scipy.misc import toimage

# import internal modules
from quilt import Quilt
from utils import save, show

if __name__ == '__main__':

    root = r"C:\dev\textures\quilt"

    data = os.path.join(root, 'data')
    results = os.path.join(root, 'results')

    # basename = "bricks+20_d100"
    basename = '2_stone-58_b010'
    # basename = 'ASP_sq_004_2048x2048_Ex'

    chain = False

    ext = 'png'
    input_path = [os.path.join(data, '.'.join([basename, ext]))]
    # input_path.append(os.path.join(data, '.'.join(['test_layer1', 'png'])))

    result_path = [os.path.join(results, '.'.join([basename, 'png']))]
    # result_path.append(os.path.join(results, '.'.join(['test_layer1', 'png'])))

    temp_result_path = [os.path.join(results, '_temp.'.join([basename, 'png']))]
    # temp_result_path.append(os.path.join(results, '_temp.'.join(['test_layer1', 'png'])))

    xmask_path = os.path.join(data, '{0}_xmask.{1}'.format(basename, ext))
    ymask_path = os.path.join(data, '{0}_ymask.{1}'.format(basename, ext))

    x_chain_paths = []

    # ########## X ##############
    matrix = []

    # final_size = [1500, 4000]  # bricks
    final_size = [500, 500]

    for path in input_path:
        print 'layer:', path
        img = Image.open(path)

        # resize
        basewidth = 800
        wpercent = (basewidth/float(img.size[0]))
        hsize = int((float(img.size[1])*float(wpercent)))
        img = img.resize((basewidth, hsize), Image.ANTIALIAS)

        print 'SIZE:', img.size
        toimage(img).show()

        # to matrix
        m = np.asarray(img)
        m.astype(float)
        matrix.append(m)

    # ########## num tiles ##############
    tile_size = 60
    overlap = 25
    # express final_size in tiles
    num_tiles = [np.int(np.floor((final_size[0]-overlap)/(tile_size-overlap))),
                 np.int(np.floor((final_size[1]-overlap) / (tile_size-overlap)))]

    # ########## Masks ##############
    # x mask
    print 'xmask path:', xmask_path
    x_mask = None
    if os.path.isfile(xmask_path):
        x_mask = Image.open(xmask_path)
        x_mask = x_mask.resize(img.size, Image.ANTIALIAS)
        x_mask = np.asarray(x_mask)

    # y mask
    y_mask = None
    if os.path.isfile(ymask_path):
        y_mask = Image.open(ymask_path)
        y_mask = np.asarray(y_mask)

    # ------------- quilt --------------

    qs = time.time()
    print 'Quilt started at {0}'.format(time.strftime("%H:%M:%S"))
    q = Quilt(matrix,
              Xmask=x_mask,
              tilesize=tile_size, num_tiles=num_tiles, overlap=overlap,
              big_tilesize=300,
              niter=1, err=0.1, result_path=temp_result_path,
              rotations=0)
    # q.compute()
    q.optimized_compute_big()
    # result = q.get_result()
    # show(result, title='result')
    # save(result, result_path)
    print 'Quilt took {0:0.6f} minutes'.format((time.time()-qs)/60)
    print 'End {0}'.format(time.strftime("%H:%M:%S"))