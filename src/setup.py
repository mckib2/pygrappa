'''Setup script for CGRAPPA.'''

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

import numpy as np

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=cythonize([Extension(
        'cgrappa',
        sources=['cgrappa.pyx', 'get_sampling_patterns.cpp'],
        include_dirs=[np.get_include()])])
)
