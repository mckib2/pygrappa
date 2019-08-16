'''Setup.py

Notes
-----
Developers: to build C++ code:

.. code-block:: python

    python3 setup.py build_ext --inplace
'''

# import os
from setuptools import setup, Extension, find_packages
import numpy as np

# os.environ['CC'] = 'gcc'

setup(
    name='pygrappa',
    version='0.3.8',
    author='Nicholas McKibben',
    author_email='nicholas.bgp@gmail.com',
    packages=find_packages(),
    scripts=[],
    url='https://github.com/mckib2/pygrappa',
    license='GPLv3',
    description=(
        'GeneRalized Autocalibrating Partially Parallel '
        'Acquisitions.'),
    long_description=open('README.rst').read(),
    keywords=(
        'mri grappa parallel-imaging image-reconstruction python'),
    install_requires=[
        "numpy>=1.16.4",
        "matplotlib>=2.2.4",
        "phantominator>=0.1.2",
        "scikit-image>=0.14.3",
        "tqdm>=4.32.2",
    ],
    python_requires='>=3.5',

    # And now for Cython generated files...
    ext_modules=[Extension(
        "pygrappa.cgrappa",
        ["src/cgrappa.cpp", "src/get_sampling_patterns.cpp"],
        include_dirs=['src/', np.get_include()])]
)
