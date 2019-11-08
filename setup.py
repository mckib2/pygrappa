'''Setup.py

Notes
-----
Developers: to build C++ code:

.. code-block:: python

    python3 setup.py build_ext --inplace
'''

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext as _build_ext

class build_ext(_build_ext):
    '''Subclass build_ext to bootstrap numpy.'''
    def finalize_options(self):
        _build_ext.finalize_options(self)

        # Prevent numpy from thinking it's still in its setup process
        import numpy as np
        self.include_dirs.append(np.get_include())

setup(
    name='pygrappa',
    version='0.14.0',
    author='Nicholas McKibben',
    author_email='nicholas.bgp@gmail.com',
    packages=find_packages(),
    scripts=[],
    url='https://github.com/mckib2/pygrappa',
    license='GPLv3',
    description=(
        'GeneRalized Autocalibrating Partially Parallel '
        'Acquisitions.'),
    long_description=open('README.rst', encoding='utf-8').read(),
    keywords=(
        'mri grappa parallel-imaging image-reconstruction python '
        'tgrappa slice-grappa sms split-slice-grappa vc-grappa '
        'igrappa hp-grappa segmented-grappa grappa-operator '
        'through-time-grappa pars grog'),
    install_requires=[
        "numpy>=1.17.0",
        "scipy>=1.3.1",
        "matplotlib>=3.1.1",
        "phantominator>=0.4.3",
        "scikit-image>=0.15.0",
        "tqdm>=4.33.0",
    ],
    cmdclass={'build_ext': build_ext},
    setup_requires=['numpy'],
    python_requires='>=3.5',

    # And now for Cython generated files...
    ext_modules=[
        Extension(
            "pygrappa.cgrappa",
            ["src/cgrappa.cpp", "src/get_sampling_patterns.cpp"],
            include_dirs=['src/']),
        Extension(
            "pygrappa.grog_powers",
            ["src/grog_powers.cpp"],
            include_dirs=['src/']),
        Extension(
            "pygrappa.grog_gridding",
            ["src/grog_gridding.cpp"],
            include_dirs=['src/']),
        ]
)
