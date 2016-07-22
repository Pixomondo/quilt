#! /usr/bin/env python
"""
Quilt command line interface
"""

# import builtin modules
from __future__ import division
import time
import os
from os.path import split, splitext, join, isdir
from pprint import pprint
import glob
from multiprocessing import cpu_count

# import 3rd party modules
from PIL import Image
import click
from click import BadParameter

# import internal modules
from process import Quilt
from utils import save, show, imresize, img2matrix


def set_src_path(ctx, _, path):
    """
    Solves source path: looks for all the matching files. If nothing is found,
    errors out
    Args:
        ctx (click.Context): the current context
        _: required by click callbacks but ignored
        path: source path/s

    Returns:
        list or string instance of the resolved path/s
    """
    path = os.path.abspath(path)
    paths = sorted(glob.glob(path))
    if not paths:
        raise click.BadParameter('Source file/s could not be found',
                                 param_hint='source')
    return paths


@click.command(
    name='quilt',
    context_settings={'help_option_names': ['-h', '--help']})
@click.argument(
    'src',
    type=click.Path(),
    callback=set_src_path)
@click.option(
    '-imask', '--input_mask',
    type=click.Path(exists=True, dir_okay=False),
    callback=lambda ctx, _, x: Image.open(x) if x else None,
    help="Mask on the input to remove image areas with unwanted content")
@click.option(
    '-iscale', '--input_scale',
    type=click.FLOAT,
    default=1,
    help="Scale to apply to the input image before computing")
@click.option(
    '-dst', '--destination',
    type=click.Path(),
    help='Path to the result folder')
@click.option(
    '-owidth', '--output_width',
    type=click.INT,
    help='Width of the result image')
@click.option(
    '-oheight', '--output_height',
    type=click.INT,
    help='Height of the result image')
@click.option(
    '-cmask', '--cut_mask',
    type=click.Path(exists=True, dir_okay=False),
    callback=lambda ctx, _, x: img2matrix(Image.open(x)) if x else None,
    help='Mask for the output image')
@click.option(
    '-tile', '--tilesize',
    type=click.INT,
    default=30,
    help='Size of the small tiles')
@click.option(
    '-over', '--overlap',
    type=click.INT,
    default=10,
    help='Size of the small overlap')
@click.option(
    '-btile', '--big_tilesize',
    type=click.INT,
    default=500,
    help='Size of the big tiles')
@click.option(
    '-bover', '--big_overlap',
    type=click.INT,
    default=100,
    help='Size of the big overlap')
@click.option(
    '-err', '--error',
    type=click.FLOAT,
    default=0.2,
    help='Error accepted in selecting similar tiles')
@click.option(
    '--constraint_start',
    type=click.BOOL,
    default=False,
    help='The first tile in dst is equal the one in src')
@click.option(
    '--cores',
    type=click.INT,
    default=cpu_count()-2,
    help='Number of available cores')
@click.option(
    '-multiproc', '--multiprocess',
    type=click.BOOL,
    default=True,
    help='Multiprocess (faster) or single process computation')
@click.option(
    '-rot', '--rotations',
    type=click.IntRange(0, 4),
    default=0,
    help='Number of 90 degrees rotations to apply to src')
@click.option(
    '--debug/--no-debug',
    default=False,
    help='Enable/Disable debug messages and image display')
def cli(src, **kwargs):

    pprint(src)
    pprint(kwargs)

    # turn paths into matrices. We need to do it here because we need iscale val
    src_matrices = []
    for p in src:
        img = Image.open(p)
        # resize
        img = imresize(img, scale=kwargs['input_scale'])
        src_matrices.append(img2matrix(img))
    if kwargs['input_mask'] is not None:
        kwargs['input_mask'] = imresize(kwargs['input_mask'],
                                        scale=kwargs['input_scale'])
        kwargs['input_mask'] = img2matrix(kwargs['input_mask'])

    kwargs.pop('input_scale')
    hgt = kwargs.pop('output_height') or src_matrices[0].shape[0]
    wdt = kwargs.pop('output_width') or src_matrices[0].shape[1]
    kwargs['output_size'] = [hgt, wdt]

    # destination paths
    dst = kwargs.pop('destination')
    # for every case: create 'path_form' which contains: a form to derive the
    # paths of the temporal result and, if necessary, the paths of the
    # additional layers
    if not dst:
        prefix, ext = splitext(src[0])
        path_form = '{1}_result{0}{2}'.format('{0}', prefix, ext)
    else:
        msg = 'Location of the destination file could not be found: {0}'
        # file
        if splitext(dst)[1]:
            # check if the location exists
            result_dir = split(dst)[0]
            if not isdir(result_dir):
                raise BadParameter(msg.format(result_dir), param_hint='dst')
            # path form: contains the single path name and the path form in case
            # there are multiple layers
            path_form = '{1}{0}{2}'.format('{0}', *splitext(dst))

        # directory
        else:
            if not isdir(dst):
                raise BadParameter(msg.format(dst), param_hint='dst')
            base_name = splitext(split(src[0])[1])[0]
            path_form = join(dst, '{1}_result{0}.png'.format('{0}', base_name))

    # master result
    if len(src) == 1:
        result_path = path_form.format('')
    else:
        # there are multiple layers: every output file has the suffix _layer
        result_path = [path_form.format('_layer' + str(i))
                       for i in xrange(len(src))]

    # derive the temp path to pass to Quilt
    kwargs['result_path'] = path_form.format('_temp')
    launch_quilt(result_path, src_matrices, **kwargs)


def launch_quilt(final_path, src, **kwargs):
    print kwargs['debug']

    multi_proc = kwargs.pop('multiprocess')
    debug = kwargs.pop('debug')

    # ------------- quilt --------------

    qs = time.time()
    print 'Quilt started at {0}'.format(time.strftime("%H:%M:%S"))

    # compute quilting

    Quilt.debug = debug
    q = Quilt(src, **kwargs)
    if multi_proc:
        q.optimized_compute()
    else:
        q.compute()

    # get the result
    result = q.get_result()
    if debug:
        show(result)
    save(result, final_path)

    # track = None
    # if config['preview']:
    #     print '\n- PREVIEW -'
    #     track = q.optimized_compute_big(preview=True)
    #     track = q.optimized_compute_big(preview=True, tracked=track)
    # if config['full']:
    #     print '\n- FULL -'
    #     track = q.optimized_compute_big(preview=False, tracked=track)

    t = (time.time()-qs)/60
    print 'Quilt took {0:0.6f} minutes'.format(t)
    print 'End {0}'.format(time.strftime("%H:%M:%S"))


if __name__ == '__main__':
    cli()