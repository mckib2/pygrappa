'''Setup.py'''

from distutils.core import setup
from setuptools import find_packages

setup(
    name='pygrappa',
    version='0.0.3',
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
        "numpy>=1.16.2",
        "matplotlib>=3.0.3",
        "phantominator>=0.0.7",
        "scikit-image>=0.15.0",
    ],
    python_requires='>=3.6',
)
