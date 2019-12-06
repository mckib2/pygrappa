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
    Extension(
        'pygrappa.grog_powers',
        ['src/grog_powers.pyx', 'src/_grog_powers_template.cpp'],
        include_dirs=['src/', np.get_include()],
        extra_compile_args=["-std=c++14"],
        extra_link_args=["-std=c++14"]),
    Extension(
        'pygrappa.grog_gridding',
        ['src/grog_gridding.pyx'],
        include_dirs=['src/', np.get_include()]),
]

setup(
    ext_modules=cythonize(extensions),
)
