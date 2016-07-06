#! /usr/bin/env python
"""
Launch image quilt.
"""

# import builtin modules
from __future__ import division
import time
import sys
import os
from pprint import pprint
sys.path.insert(0, r"C:\dev\textures\Lib\site-packages")

# import 3rd party modules
import numpy as np
from numpy import floor
import OpenImageIO as oiio
from PIL import Image
from scipy.misc import toimage
import yaml

# import internal modules
from quilt import Quilt
from utils import save, show


def set_paths(base, root=None, source=None, destination=None, tile_size=None,
              overlap=None, **kwargs):

    data = os.path.join(root, 'data')
    res = os.path.join(root, 'results')
    ext = source['extension']

    # source
    if not source['paths'][0]:
        source['paths'] = [os.path.join(data, '.'.join([base, ext]))]

    # destination
    if not destination['paths'][0]:
        destination['paths'] = [os.path.join(res, base + '.png')]
    # dest size
    size = [destination['size']['height'], destination['size']['width']]
    destination['num_tiles'] = [
        np.int(floor((size[0] - overlap) / (tile_size - overlap))),
        np.int(floor((size[1] - overlap) / (tile_size - overlap)))]

    # other layers
    if not len(source['paths']) == len(destination['paths']):
        raise ValueError('number of layes in src and dst must be the same')

    # destination temp
    destination['temp'] = '_temp'.join(os.path.splitext(
                                                       destination['paths'][0]))

    # masks
    source['mask'] = source['mask'] or os.path.join(
                                              data, '_xmask.'.join([base, ext]))
    destination['mask'] = destination['mask'] or os.path.join(
                                               res, '_ymask.'.join([base, ext]))

    print 'SOURCE'
    pprint(source)
    print 'DESTINATION'
    pprint(destination)
    print

    return source, destination


def load_data(src, dst, src_width=0):
    matrix = []

    # SOURCE
    for path in src['paths']:
        img = Image.open(path)

        # resize
        basewidth = src['new_width'] or src_width
        wpercent = (basewidth / float(img.size[0]))
        hsize = int((float(img.size[1]) * float(wpercent)))
        img = img.resize((basewidth, hsize), Image.ANTIALIAS)

        toimage(img).show()

        # to matrix
        m = np.asarray(img)
        m.astype(float)
        matrix.append(m)

    # MASKS
    # x mask
    x_mask = None
    if os.path.isfile(src['mask']):
        x_mask = Image.open(src['mask'])
        x_mask = x_mask.resize(img.size, Image.ANTIALIAS)
        x_mask = np.asarray(x_mask)

    # y mask
    y_mask = None
    if os.path.isfile(dst['mask']):
        y_mask = Image.open(dst['mask'])
        y_mask = np.asarray(y_mask)

    return matrix, x_mask, y_mask


if __name__ == '__main__':

    # basename = "bricks+20_d100"
    # basename = 'TexturesCom_BrickOldRounded0281_3'
    # basename = 'TexturesCom_Roads0059_1'
    basename = 'MarbleTiles0054_1'

    config_yaml = 1

    if config_yaml:
        yaml_file = os.path.join(r'C:\dev\textures\quilt\yaml_input',
                                 basename+'.yaml')
        with open(yaml_file, 'r') as stream:
            config = yaml.load(stream)
        src, dst = set_paths(basename, **config)

        matrix, x_mask, y_mask = load_data(src=src, dst=dst)

    else:
        root = r"C:\dev\textures\quilt"
        config = {}
        config['tile_size'] = 50
        config['overlap'] = 30
        config['big_tile_size'] = 300
        config['rotations'] = 0
        config['error'] = 0.5
        config['cores'] = None
        final_size = [800, 300]
        src_width = 300
        source, dst = set_paths(basename, root,
                                dst={'size': {'height': final_size[0],
                                              'width': final_size[1]}},
                                tile_size=config['tile_size'],
                                overlap=config['overlap'])

        matrix, x_mask, y_mask = load_data(source, dst, src_width=src_width)

    #
    # ------------- quilt --------------

    qs = time.time()
    print 'Quilt started at {0}'.format(time.strftime("%H:%M:%S"))
    q = Quilt(matrix,
              Xmask=x_mask, Ymask=y_mask,
              tilesize=config['tile_size'],  overlap=config['overlap'],
              num_tiles=dst['num_tiles'],
              big_tilesize=config['big_tile_size'],
              err=config['error'],
              result_path=dst['temp'],
              rotations=config['rotations'],
              constraint_start=config['constraint_start'],
              cores=config['cores'])
    q.optimized_compute_big()

    result = q.get_result()
    show(result, title='result')
    save(result, dst['paths'])

    t = (time.time()-qs)/60
    print 'Quilt took {0:0.6f} minutes'.format(t)
    print 'End {0}'.format(time.strftime("%H:%M:%S"))
