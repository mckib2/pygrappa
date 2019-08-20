Installation
============

.. code-block:: bash

    pip install pygrappa

Usage
=====

The function is called with undersampled k-space data and calibration
data (usually a fully sampled portion of the center of k-space).  The
unsampled points in k-space should be exactly 0:

.. code-block:: python

    from pygrappa import grappa

    sx, sy, ncoils = kspace.shape[:]
    cx, cy, ncoils = calib.shape[:]
    res = grappa(kspace, calib, kernel_size=(5, 5), coil_axis=-1)
    sx, sy, ncoils = res.shape[:]

If calibration data is in the k-space data, simply extract it:

.. code-block:: python

    from pygrappa import grappa

    sx, sy, ncoils = kspace.shape[:] # center 20 lines are ACS
    ctr, pd = int(sy/2), 10
    calib = kspace[:, ctr-pd:ctr+pad, :].copy()
    res = grappa(kspace, calib, kernel_size=(5, 5), coil_axis=-1)
    sx, sy, ncoils = res.shape[:]

Also see the `examples` module.

A very similar experimental GRAPPA implementation with the same
interface can be called:

.. code-block:: python

    from pygrappa import cgrappa
    res = cgrappa(kspace, calib, kernel_size=(5, 5), coil_axis=-1)

This function uses much of the same code as the Python grappa()
implementation, but has certain parts written in C++ and all compiled
using Cython.  It runs about twice as fast but is considered
experimental.

About
=====

GRAPPA is a popular parallel imaging reconstruction algorithm.
Unfortunately there aren't a lot of easy to use Python implementations
available, so I decided to release this simple one.
