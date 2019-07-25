'''Setup.py'''

# from distutils.core import setup
# from distutils.extension import Extension
# from setuptools import find_packages, Extension
# from Cython.Distutils import build_ext
# from Cython.Build import cythonize

from setuptools import setup, Extension, find_packages


import numpy as np

setup(
    name='pygrappa',
    version='0.3.4',
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
        "Cython>=0.28.5"
    ],
    python_requires='>=3.5',

    # # And now for some Cython...
    # cmdclass={'build_ext': build_ext},
    # ext_modules=cythonize([Extension(
    #     'pygrappa.cgrappa',
    #     sources=[
    #         'src/cgrappa.pyx',
    #         'src/get_sampling_patterns.cpp'],
    #     include_dirs=[np.get_include()])]),
    ext_modules=[Extension(
        "pygrappa.cgrappa",
        ["src/cgrappa.cpp", "src/get_sampling_patterns.cpp"],
        include_dirs=['src/', np.get_include()])]
)
