#! /usr/bin/env python
"""
Quilt command line interface
"""

# import builtin modules
from __future__ import division
import time
import os
from os.path import split, splitext, join, isdir
import glob
from multiprocessing import cpu_count
from collections import namedtuple
import logging
import multiprocessing
import threading
import traceback
import sys

# import 3rd party modules
from PIL import Image
import click
from click import BadParameter

# import internal modules
from quilt.process import Quilt
from quilt.utils import save, show, imresize, img2matrix


# styles for click logging
_Style = namedtuple('_Style', 'fg bg bold dim underline blink reverse reset')
_style = _Style(None, None, True, None, None, None, None, True)
_STYLES = {
    logging.CRITICAL: _style._replace(fg='red', blink=True),
    logging.ERROR: _style._replace(fg='red', underline=True),
    logging.WARNING: _style._replace(fg='yellow'),
    logging.INFO: _style._replace(fg='green'),
    logging.DEBUG: _style._replace(fg='cyan'),
    logging.NOTSET: _style._replace(fg='magenta')}


class ClickHandler(logging.Handler):
    """
    Logging handler that prints colorful messages with click.echo.
    """
    def __init__(self, level=logging.NOTSET, err=True):
        super(ClickHandler, self).__init__(level=level)
        self.err = err

    def emit(self, record):
        try:
            msg = self.format(record)
            try:
                self.acquire()
                click.echo(message=msg, err=self.err)
            finally:
                self.release()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)

    def createLock(self):
        self.lock = multiprocessing.RLock()


class QueuedClickHandler(logging.Handler):
    """
    Multi-thread logging handler to handle logging with multi-processing (used
    in Quilt). When called, the message is stored in a queue which is then read
    by the worker and the message is passed to ClickHandler.
    """

    def __init__(self, level=logging.NOTSET, err=True):
        super(QueuedClickHandler, self).__init__(level=level)
        # handler that does the actual log message emission
        self._handler = ClickHandler(level=level, err=err)
        # queue to hold actual emitted messages
        self.queue = multiprocessing.JoinableQueue(-1)
        # worker to empty the queue
        worker = threading.Thread(target=self.receive)
        worker.daemon = True
        worker.start()

    def receive(self):
        while True:
            try:
                record = self.queue.get()
                self._handler.emit(record)
                self.queue.task_done()
            except (KeyboardInterrupt, SystemExit):
                raise
            except EOFError:
                break
            except:
                traceback.print_exc(file=sys.stderr)

    def setFormatter(self, fmt):
        logging.Handler.setFormatter(self, fmt)
        self._handler.setFormatter(fmt)

    def setLevel(self, lvl):
        logging.Handler.setLevel(self, lvl)
        self._handler.setLevel(lvl)

    def _format_record(self, record):
        if record.args:
            record.msg = record.msg % record.args
            record.args = None
        if record.exc_info:
            dummy = self.format(record)
            record.exc_info = None
        return record

    def emit(self, record):
        try:
            self.queue.put(self._format_record(record))
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)

    def close(self):
        self.queue.join()
        self._handler.close()
        logging.Handler.close(self)

    def createLock(self):
        self.lock = multiprocessing.RLock()


class ClickFormatter(logging.Formatter):
    """
    Logging formatter that associated different log levels to different Click
    styles.
    """
    def __init__(self, fmt=None, datefmt=None):
        super(ClickFormatter, self).__init__(fmt=fmt, datefmt=datefmt)

    def format(self, record):
        return click.style(
            super(ClickFormatter, self).format(record),
                  **_STYLES[record.levelno]._asdict())


# create logger
h = QueuedClickHandler()
h.setFormatter(ClickFormatter())
log = logging.getLogger('quilt')
log.addHandler(h)
logging.root.setLevel(0)


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
    '-i', '--input_mask',
    type=click.Path(exists=True, dir_okay=False),
    callback=lambda ctx, _, x: Image.open(x) if x else None,
    help="Mask on the input to remove image areas with unwanted content")
@click.option(
    '-s', '--input_scale',
    type=click.FLOAT,
    default=1,
    help="Scale to apply to the input image before computing")
@click.option(
    '-d', '--destination',
    type=click.Path(),
    help='Path to the result folder')
@click.option(
    '-w', '--output_width',
    type=click.INT,
    help='Width of the result image')
@click.option(
    '-h', '--output_height',
    type=click.INT,
    help='Height of the result image')
@click.option(
    '-c', '--cut_mask',
    type=click.Path(exists=True, dir_okay=False),
    callback=lambda ctx, _, x: img2matrix(Image.open(x)) if x else None,
    help='Mask for the output image')
@click.option(
    '-t', '--tilesize',
    type=click.INT,
    default=30,
    help='Size of the small tiles')
@click.option(
    '-o', '--overlap',
    type=click.INT,
    default=10,
    help='Size of the small overlap')
@click.option(
    '--big_tilesize',
    type=click.INT,
    default=500,
    help='Size of the big tiles')
@click.option(
    '--big_overlap',
    type=click.INT,
    default=100,
    help='Size of the big overlap')
@click.option(
    '-e', '--error',
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
    '-m', '--multiprocess',
    type=click.BOOL,
    default=True,
    help='Multiprocess (faster) or single process computation')
@click.option(
    '-r', '--rotations',
    type=click.IntRange(0, 4),
    default=0,
    help='Number of 90 degrees rotations to apply to src')
@click.option(
    '-f', '--flip',
    type=tuple,
    default=(False, False),
    help='Tuple of booleans for (flip_vertical, flip_horizontal) to apply to '
         'src')
@click.option(
    '--debug/--no-debug',
    default=False,
    help='Enable/Disable debug messages and image display')
def cli(src, **kwargs):
    """
    Parses the arguments for quilt processing.
    """

    # set log to debug if necessary
    debug = kwargs.pop('debug')
    log.setLevel(0 if debug else 20)

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
    hgt = kwargs.pop('output_height') or src_matrices[0].shape[0]*2
    wdt = kwargs.pop('output_width') or src_matrices[0].shape[1]*2
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
    _launch_quilt(debug, result_path, src_matrices, **kwargs)


def _launch_quilt(debug, final_path, src, **kwargs):
    """
    Launches quilt process with the parsed arguments.
    """

    multi_proc = kwargs.pop('multiprocess')

    # ------------- quilt --------------

    qs = time.time()
    log.info('Quilt started at {0}'.format(time.strftime("%H:%M:%S")))

    try:
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

    except ValueError as err:
        log.error(err.message)
        return

    t = (time.time()-qs)/60
    log.debug('Quilt took {0:0.6f} minutes'.format(t))
    log.info('End {0}'.format(time.strftime("%H:%M:%S")))


if __name__ == '__main__':
    cli()
