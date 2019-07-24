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
        sources=['cgrappa.pyx', 'grappa_in_c.cpp'],
        include_dirs=[np.get_include()])])
)

# from Cython.Build import cythonize
#
# setup(name='Hello world app',
#       ext_modules=cythonize('pyx_cgrappa.pyx'))
