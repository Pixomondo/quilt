#! /usr/bin/env python
"""
Launch image quilt.
"""

# import builtin modules
import time
import sys

sys.path.insert(0, r"C:\dev\textures\Lib\site-packages")

# import 3rd party modules
import numpy as np
import OpenImageIO as oiio
from PIL import Image
from scipy.misc import toimage

# import internal modules
from imagequilt import quilt


path = r"C:\dev\textures\quilt\data\bricks2.jpg"
path = r"X:\freenx_transfer\john.nelson.bak\Reasources\txt\Arroway - Edition one and a half\2_stone-14_d100.jpg"

# Read a camera raw, crop and write out to a tiff
buf = oiio.ImageBuf(path)
spec = buf.spec()
roi = oiio.ROI(buf.roi)

# divide in channels
r = roi
r.chbegin, r.chend = 0, 1
g = roi
g.chbegin, g.chend = 1, 2
b = roi
b.chbegin, b.chend = 0, 1


# pixels = buf.get_pixels(oiio.FLOAT)
#
# print 'sizes: ', spec.height, spec.width, spec.nchannels
# print 'npixels: ', spec.image_pixels()
#
# matrix = np.array(pixels)
# print 'pixels:', matrix.shape
# matrix = np.reshape(matrix, (spec.height, spec.width, spec.nchannels))
#
# print 'matrix:', matrix.shape
# print np.amax(matrix)
#
# matrix *= 255
# matrix = matrix.astype(int)

#####
img = Image.open(path)

# resize
basewidth = 300
wpercent = (basewidth/float(img.size[0]))
hsize = int((float(img.size[1])*float(wpercent)))
img = img.resize((basewidth, hsize), Image.ANTIALIAS)

# make it square
dim = min(img.size[0], img.size[1])
imm = img.crop((0, 0, dim, dim))

print 'SIZE:', imm.size
toimage(imm).show()

# to matrix
matrix = np.asarray(imm)
matrix.astype(float)
print np.amax(matrix)
####


# quilt
qs = time.time()
print 'Quilt started at {0:0.6f}'.format(qs)
quilt(matrix, 25, [15, 30], niter=1, overlap=10)
print 'Quilt took {0:0.6f} seconds'.format((time.time()-qs)/60)
