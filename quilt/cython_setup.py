try:
    from setuptools import setup
    from setuptools import Extension
except ImportError:    
    from distutils.core import setup
    from distutils.extension import Extension
    
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy

extensions = [Extension("process", ["process.pyx"],
                        include_dirs=[numpy.get_include()]),
              Extension("ssd", ["ssd.pyx"],
                        include_dirs=[numpy.get_include()]),
              Extension("utils", ["utils.pyx"],
                        include_dirs=[numpy.get_include()])
              ]

setup(
    name='quilt',
    cmdclass={'build_ext': build_ext},
    ext_modules=cythonize(extensions)
)

