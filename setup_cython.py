'''Compile .pyx to .c*.

Notes
-----

Developers: to compile Cython files:

.. code-block:: python

    python3 setup_cython.py build_ext --inplace
'''

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

import numpy as np

extensions = [
    Extension(
        'pygrappa.cgrappa',
        ['src/cgrappa.pyx', 'src/get_sampling_patterns.cpp'],
        include_dirs=['src/', np.get_include()]),
    # Extension(
    #     'pygrappa.idft2d',
    #     ['src/dft.pyx'],
    #     include_dirs=[np.get_include()])
]

setup(
    ext_modules=cythonize(extensions),
)
