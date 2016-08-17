#! /usr/bin/env python
# import builtin modules
from setuptools import setup
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name='quilt',
    version='0.0.1',

    description='Texture synthesis through Quilting implementation',
    long_description='Quilt is a python tool to synthesize a texture ' \
                     'starting from an input one. Instead of patching the ' \
                     'input texture and then manually removing seams and '  \
                     'repetitions, this tool automatically breaks the '  \
                     'texture into small tiles and seamlessly recombines '  \
                     'them in the output image. With this process the '  \
                     'original texture structure is preserved and repetitions' \
                     ' hard to notice.',

    # The project's main homepage.
    url='https://github.com/Pixomondo/quilt',

    # Author details
    author='Rachele Bellini',

    # Choose your license
    license='GPL',

    classifiers=[
        'Development Status :: 3 - Alpha',
        # who this project is intended for
        'Intended Audience :: VFX Artists',
        'Topic :: Image Processing :: Texture Synthesis',
        'License :: GPL License',
        'Natural Language :: English',
        'Operating System :: Windows',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering :: Image Processing'
    ],

    keywords='texture synthesis quilt',

    packages=['quilt'],

    # run-time dependencies (will be installed by pip when the project is
    # installed)
    install_requires=['colorama',
                      'Click',
                      'numpy',
                      'Pillow',
                      'PyMaxflow'],

    # additional groups of dependencies.
    extras_require={},
    # data files included in the packages that need to be installed
    package_data={},
    # data files outside of the packages
    data_files=[],

    # entry points
    entry_points={
        'console_scripts': [
            'quilt=quilt.main_cmd:cli',
        ],
    },
)

