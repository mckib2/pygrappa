About
=====

GRAPPA is a popular parallel imaging reconstruction algorithm.
Unfortunately there aren't a lot of easy to use Python implementations
available, so I decided to release this simple one.

Included in the `pygrappa` module are the following:

- GRAPPA: `grappa()`
- VC-GRAPPA: `vcgrappa()`
- iGRAPPA: `igrappa()`
- hp-GRAPA: `hpgrappa()`
- TGRAPPA: `tgrappa()`
- Slice-GRAPPA: `slicegrappa()`
- Split-Slice-GRAPPA: `splitslicegrappa()`

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

See INSTALLATION.rst for more info on installing under Windows.

Usage
=====

See the `examples` module.  It has several scripts showing basic
usage.  Docstrings are also a great resource -- check them out for all
possible arguments and usage info.

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

`vcgrappa()` is a VC-GRAPPA implementation that simply constructs
conjugate virtual coils, appends them to the coil dimension, and
passes everything through to `cgrappa()`.  The function signature
is identical to `pygrappa.grappa()`.

`igrappa()` is an Iterative-GRAPPA implementation that can be called
as follows:

.. code-block:: python

    from pygrappa import igrappa
    res = igrappa(kspace, calib, kernel_size=(5, 5))

    # You can also provide the reference kspace to get the MSE at
    # each iteration, showing you the performance.  Regularization
    # parameter k (as described in paper) can also be provided:
    res, mse = igrappa(kspace, calib, k=0.6, ref=ref_kspace)

`igrappa()` makes calls to `cgrappa()` on the back end.

`hpgrappa()` implements the High-Pass GRAPPA (hp-GRAPPA) algorithm.
It requires FOV to construct an appropriate high pass filter.  It can
be called as:

.. code-block:: python

    from pygrappa import hpgrappa
    res = hpgrappa(kspace, caliv, fov=(FOV_x, FOV_y))

TGRAPPA does not require calibration data and can be called as:

.. code-block:: python

    from pygrappa import tgrappa

    sx, sy, ncoils, nt = kspace.shape[:]
    res = tgrappa(
        kspace, calib_size=(20, 20), kernel_size=(5, 5),
        coil_axis=-2, time_axis=-1)

Calibration region size and kernel size must be provided.  The
calibration regions will be constructed in a greedy manner: once
enough time frames have been consumed to create an entire ACS, GRAPPA
will be run.  TGRAPPA uses the `cgrappa` implementation for its
speed.

`slicegrappa()` is a Slice-GRAPPA implementation that can be called
like:

.. code-block:: python

    from pygrappa import slicegrappa

    sx, sy, ncoils, nt = kspace.shape[:]
    sx, sy, ncoils, sl = calib.shape[:]
    res = slicegrappa(kspace, calib, kernel_size=(5, 5), prior='sim')

`kspace` is assumed to SMS-like with multiple collapsed slices and
multiple time frames that each need to be separated.  `calib` are the
individual slices' kspace data at the same size/resolution.  `prior`
tells the Slice-GRAPPA algorithm how to construct the sources, that
is, how to solve T = S W, where T are the targets (calibration data),
S are the sources, and W are GRAPPA weights. `prior='sim'` creates
S by simulating the SMS acquisition, i.e., S = sum(calib, slice_axis).
`prior='kspace'` uses the first time frame from the `kspace` data,
i.e., S = kspace[1st time frame].  The result is an array containing
all target slices for all time frames in `kspace`.

Similarly, Split-Slice-GRAPPA can be called like so:

.. code-block:: python

    from pygrappa import splitslicegrappa as ssgrappa

    sx, sy, ncoils, nt = kspace.shape[:]
    sx, sy, ncoils, sl = calib.shape[:]
    res = ssgrappa(kspace, calib, kernel_size=(5, 5))

    # Note that pygrappa.splitslicegrappa is an alias for
    # pygrappa.slicegrappa(split=True), so it can also be called
    # like this:
    from pygrappa import slicegrappa
    res = slicegrappa(kspace, calib, kernel_size=(5, 5), split=True)
