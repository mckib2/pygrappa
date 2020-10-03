'''Python GRAPPA image reconstruction.'''

import subprocess
from distutils.spawn import find_executable
from setuptools import find_packages
from numpy.distutils.core import setup

from setup_helpers import get_build_ext_override

VERSION = '0.24.0'


def pre_build_hook(build_ext, ext):
    from scipy._build_utils.compiler_helper import get_cxx_std_flag
    std_flag = get_cxx_std_flag(build_ext._cxx_compiler)
    if std_flag is not None:
        ext.extra_compile_args.append(std_flag)


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    from numpy import get_include
    config = Configuration('pygrappa', parent_package, top_path)
    config.version = VERSION

    DEFINE_MACROS = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]

    # Run Cython on files if we have it
    if find_executable('cython') is not None:
        print('Running cython...')
        subprocess.call(['cython -3 --cplus src/*.pyx'], shell=True)

    # GRAPPA helpers
    config.add_extension(
        'train_kernels',
        sources=[
            'src/train_kernels.cpp',
        ],
        include_dirs=['src/', get_include()],
        language='c++',
        define_macros=DEFINE_MACROS,
    )

    # GRAPPA with some C components
    config.add_extension(
        'cgrappa',
        sources=[
            'src/cgrappa.cpp',
            'src/get_sampling_patterns.cpp',
        ],
        include_dirs=['src/'],
        language='c++',
        define_macros=DEFINE_MACROS,
    )

    # GROG powers
    ext = config.add_extension(
        'grog_powers',
        sources=[
            'src/grog_powers.cpp',
            'src/_grog_powers_template.cpp',
        ],
        include_dirs=['src/'],
        language='c++',
        define_macros=DEFINE_MACROS,
    )
    ext._pre_build_hook = pre_build_hook

    # GROG
    ext = config.add_extension(
        'grog_gridding',
        sources=[
            'src/grog_gridding.cpp',
        ],
        include_dirs=['src/'],
        language='c++',
        define_macros=DEFINE_MACROS,
    )
    ext._pre_build_hook = pre_build_hook

    return config


setup(
    author='Nicholas McKibben',
    author_email='nicholas.bgp@gmail.com',
    url='https://github.com/mckib2/pygrappa',
    license='GPLv3',
    description='GeneRalized Autocalibrating Partially Parallel Acquisitions.',
    long_description=open('README.rst', encoding='utf-8').read(),
    packages=find_packages(),
    keywords=[
        'mri grappa parallel-imaging image-reconstruction python '
        'tgrappa slice-grappa sms split-slice-grappa vc-grappa '
        'igrappa hp-grappa segmented-grappa grappa-operator '
        'through-time-grappa pars grog nonlinear-grappa g-factor'
        'sense', 'cg-sense'],
    install_requires=open('requirements.txt').read().split(),
    setup_requires=['numpy', 'scipy'],
    python_requires='>=3.5',
    cmdclass={'build_ext': get_build_ext_override()},
    **configuration(top_path='').todict(),
)
