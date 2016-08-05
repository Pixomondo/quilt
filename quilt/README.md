Quilt Source Code
=================

A brief explanation of the source code structure.

Cython:

In this branch I added Cython tools. The process speed does not actually
 change since most of the computation is already done through numpy, 
 which is already quite optimized.

The Cython code is stored in the .pyx files. Once compiled, the files .c 
 and .pyd are created.
The script cython_setup.py is used to compile the Cython code. 
 Command line to compile:
    >> python cython_setup.py build_ext -i
 
Some changes in the code related to Cython:
- declaration of variables at the beginning of the functions/methods
- declaration of attributed at class level
- declaration of the types of the arguments
- decorators to perform division in c and remove bound checks
- type of declaration of functions:
    def (python), cpdef (both python and c), cdef (c)

Finally, I skipped the tests that used the Quilt multiprocessing
  computation because it seems not to work with Cython.