'''Setup.py'''

from distutils.core import setup
from setuptools import find_packages

setup(
    name='pygrappa',
    version='0.2.1',
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
        "tqdm>=4.32.2"
    ],
    python_requires='>=3.5',
)
