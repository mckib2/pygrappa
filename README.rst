About
=====

GRAPPA is a popular parallel imaging reconstruction algorithm.
Unfortunately there aren't a lot of easy to use Python implementations
available, so I decided to release this simple one.

Included in the `pygrappa` module are the following:

- `grappa()`
- `tgrappa()`


Installation
============

This package is developed in Ubuntu 18.04 using Python 3.6.8.  That's
not to say it won't work on other things.  You should submit an issue
when it doesn't work like it says it should.  The whole idea was to
have an easy to use, pip-install-able GRAPPA module, so let's try to
do that.

In general, it's a good idea to work inside virtual environments.  I
create and activate mine like this:

.. code-block:: bash

    python3 -m venv ~/Documents/venvs/pygrappa
    source ~/Documents/venvs/pygrappa/bin/activate

More information can be found in the venv documentation:
https://docs.python.org/3/library/venv.html

Installation under a Unix-based platform should then be as easy as:

.. code-block:: bash

    pip install pygrappa

Windows 10 Installation
=======================

If you are using Windows, then, first of all: sorry.  This is not
ideal, but I understand that it might not be your fault.  I will
assume you are trying to get pygrappa installed on Windows 10. I
will further assume that you are using Python 3.7 64-bit build.
We will need a C++ compiler to install pygrappa, so officially
you should have "Microsoft Visual C++ Build Tools" installed. I
haven't tried this, but it should work with VS build tools
installed.

However, if you are not able to install the build tools, we can
do it using the MinGW compiler instead.  It'll be a little more
involved than a simple `pip install`, but that's what you get for
choosing Windows.

Steps:

- Download 64-bit fork of MinGW from https://sourceforge.net/projects/mingw-w64/
- Follow this guide: https://github.com/orlp/dev-on-windows/wiki/Installing-GCC--&-MSYS2
- Now you should be able to use gcc/g++/etc. from CMD-line
- Modify cygwinccompiler.py similar to tgalal/yowsup#2494 but using the version number `1916`:

.. code-block:: python

    def get_msvcr():
        """Include the appropriate MSVC runtime library if Python was built
        with MSVC 7.0 or later.
        """
        msc_pos = sys.version.find('MSC v.')
        if msc_pos != -1:
            msc_ver = sys.version[msc_pos+6:msc_pos+10]
            if msc_ver == '1300':
                # MSVC 7.0
                return ['msvcr70']
            elif msc_ver == '1310':
                # MSVC 7.1
                return ['msvcr71']
            elif msc_ver == '1400':
                # VS2005 / MSVC 8.0
                return ['msvcr80']
            elif msc_ver == '1500':
                # VS2008 / MSVC 9.0
                return ['msvcr90']
            elif msc_ver == '1600':
                # VS2010 / MSVC 10.0
                return ['msvcr100']
            elif msc_ver == '1916':
                # Visual Studio 2015 / Visual C++ 14.0
                return ['vcruntime140']    
            else:
                raise ValueError("Unknown MS Compiler version %s " % msc_ver)

- now run the command:

.. code-block:: python

    pip install --global-option build_ext --global-option --compiler=mingw32 --global-option -DMS_WIN64 pygrappa

Hopefully this works for you.  Refer to
https://github.com/mckib2/pygrappa/issues/17
for a more detailed discussion.

Usage
=====

`pygrappa.grappa()` is called with undersampled k-space data and
calibration data (usually a fully sampled portion of the center of
k-space).  The unsampled points in k-space should be exactly 0:

.. code-block:: python

    from pygrappa import grappa

    sx, sy, ncoils = kspace.shape[:]
    cx, cy, ncoils = calib.shape[:]
    res = grappa(kspace, calib, kernel_size=(5, 5), coil_axis=-1)
    sx, sy, ncoils = res.shape[:]

If calibration data is in the k-space data, simply extract it (make
sure to call the ndarray.copy() method, may break if using reference
to the original k-space data):

.. code-block:: python

    from pygrappa import grappa

    sx, sy, ncoils = kspace.shape[:] # center 20 lines are ACS
    ctr, pd = int(sy/2), 10
    calib = kspace[:, ctr-pd:ctr+pad, :].copy()
    res = grappa(kspace, calib, kernel_size=(5, 5), coil_axis=-1)
    sx, sy, ncoils = res.shape[:]

A very similar GRAPPA implementation with the same interface can be
called like so:

.. code-block:: python

    from pygrappa import cgrappa
    res = cgrappa(kspace, calib, kernel_size=(5, 5), coil_axis=-1)

This function uses much of the same code as the Python grappa()
implementation, but has certain parts written in C++ and all compiled
using Cython.  It runs about twice as fast but is considered
experimental.  It will probably become the default GRAPPA
implementation in future releases.

TGRAPPA does not require calibration data and can be called as:

.. code-block:: python

    from pygrappa import tgrappa
    res = tgrappa(
        kspace, calib_size=(20, 20), kernel_size=(5, 5),
        coil_axis=-2, time_axis=-1)

Calibration region size and kernel size must be provided.  The
calibration regions will be constructed in a greedy manner: once
enough time frames have been consumed to create an entire ACS, GRAPPA
will be run.  TGRAPPA uses the `cgrappa` implementation for its
speed.

Also see the `examples` module.  It has several scripts showing basic
usage.  Docstrings are also a great resource -- check them out for all
possible arguments and usage info.
