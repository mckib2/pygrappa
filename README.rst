About
=====

GRAPPA is a popular parallel imaging reconstruction algorithm.
Unfortunately there aren't a lot of easy to use Python
implementations of it or its many variants available, so I decided to
release this simple package.

Included in the `pygrappa` module are the following:

- GRAPPA [1]_: `grappa()`
- VC-GRAPPA [2]_: `vcgrappa()`
- iGRAPPA [3]_: `igrappa()`
- hp-GRAPPA [4]_: `hpgrappa()`
- Segmented-GRAPPA [5]_: `seggrappa()`
- TGRAPPA [6]_: `tgrappa()`
- Slice-GRAPPA [7]_: `slicegrappa()`
- Split-Slice-GRAPPA [8]_: `splitslicegrappa()`
- GRAPPA operator [9]_: `grappaop()`

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

`pygrappa.grappa()` implements GRAPPA ([1]_) for arbitrarily
sampled Cartesian datasets.  It is called with undersampled k-space
data and calibration data (usually a fully sampled portion of the
center of k-space).  The unsampled points in k-space should be
exactly 0:

.. code-block:: python

    from pygrappa import grappa

    # These next two lines are to show you the sizes of kspace and
    # calib -- you need to bring your own data.  It doesn't matter
    # where the coil dimension is, you just need to let 'grappa' know
    # when you call it by providing the 'coil_axis' argument
    sx, sy, ncoils = kspace.shape[:]
    cx, cy, ncoils = calib.shape[:]

    # Here's the actual reconstruction
    res = grappa(kspace, calib, kernel_size=(5, 5), coil_axis=-1)

    # Here's the resulting shape of the reconstruction.  The coil
    # axis will end up in the same place you provided it in
    sx, sy, ncoils = res.shape[:]

If calibration data is in the k-space data, simply extract it (make
sure to call the ndarray.copy() method, may break if using reference
to the original k-space data):

.. code-block:: python

    from pygrappa import grappa

    sx, sy, ncoils = kspace.shape[:] # center 20 lines are ACS
    ctr, pd = int(sy/2), 10
    calib = kspace[:, ctr-pd:ctr+pad, :].copy() # call copy()!

    # coil_axis=-1 is default, so if coil dimension is last we don't
    # need to explicity provide it
    res = grappa(kspace, calib, kernel_size=(5, 5))
    sx, sy, ncoils = res.shape[:]

A very similar GRAPPA implementation with the same interface can be
called like so:

.. code-block:: python

    from pygrappa import cgrappa
    res = cgrappa(kspace, calib, kernel_size=(5, 5), coil_axis=-1)

This function uses much of the same code as the Python grappa()
implementation, but has certain parts written in C++ and all compiled
using Cython.  It runs about twice as fast.  It will probably become
the default GRAPPA implementation in future releases.

`vcgrappa()` is a VC-GRAPPA ([2]_) implementation that simply
constructs conjugate virtual coils, appends them to the coil
dimension, and passes everything through to `cgrappa()`.  The
function signature is identical to `pygrappa.grappa()`.

`igrappa()` is an Iterative-GRAPPA ([3]_) implementation that can be
called as follows:

.. code-block:: python

    from pygrappa import igrappa
    res = igrappa(kspace, calib, kernel_size=(5, 5))

    # You can also provide the reference kspace to get the MSE at
    # each iteration, showing you the performance.  Regularization
    # parameter k (as described in paper) can also be provided:
    res, mse = igrappa(kspace, calib, k=0.6, ref=ref_kspace)

`igrappa()` makes calls to `cgrappa()` on the back end.

`hpgrappa()` implements the High-Pass GRAPPA (hp-GRAPPA) algorithm
([4]_). It requires FOV to construct an appropriate high pass filter.
It can be called as:

.. code-block:: python

    from pygrappa import hpgrappa
    res = hpgrappa(kspace, calib, fov=(FOV_x, FOV_y))

`seggrappa()` is a generalized Segmented GRAPPA implementation ([5]_).
It is supplied a list of calibration regions, `cgrappa` is run for
each, and all the reconstructions are averaged together to yield the
final image.  It can be called with all the normal `cgrappa`
arguments:

.. code-block:: python

    from pygrappa import seggrappa

    cx1, cy1, ncoil = calib1.shape[:]
    cx2, cy2, ncoil = calib2.shape[:]
    res = seggrappa(kspace, [calib1, calib2])

TGRAPPA is a Temporal GRAPPA implementation ([6]_) and does not
require calibration data.  It can be called as:

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

`slicegrappa()` is a Slice-GRAPPA ([7]_) implementation that can be
called like:

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

Similarly, Split-Slice-GRAPPA ([8]_) can be called like so:

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

`grappaop` returns two unit GRAPPA operators ([9]_, [10]_) found from
a 2D calibration dataset:

.. code-block:: python

    from pygrappa import grappaop

    sx, sy, ncoils = calib.shape[:]
    Gx, Gy = grappaop(calib, coil_axis=-1)

See the examples to see how to use the GRAPPA operators to
reconstruct datasets.

References
==========
.. [1] Griswold, Mark A., et al. "Generalized autocalibrating
       partially parallel acquisitions (GRAPPA)." Magnetic
       Resonance in Medicine: An Official Journal of the
       International Society for Magnetic Resonance in Medicine
       47.6 (2002): 1202-1210.
.. [2] Blaimer, Martin, et al. "Virtual coil concept for improved
       parallel MRI employing conjugate symmetric signals."
       Magnetic Resonance in Medicine: An Official Journal of the
       International Society for Magnetic Resonance in Medicine
       61.1 (2009): 93-102.
.. [3] Zhao, Tiejun, and Xiaoping Hu. "Iterative GRAPPA (iGRAPPA)
       for improved parallel imaging reconstruction." Magnetic
       Resonance in Medicine: An Official Journal of the
       International Society for Magnetic Resonance in Medicine
       59.4 (2008): 903-907.
.. [4] Huang, Feng, et al. "High‐pass GRAPPA: An image support
       reduction technique for improved partially parallel
       imaging." Magnetic Resonance in Medicine: An Official
       Journal of the International Society for Magnetic
       Resonance in Medicine 59.3 (2008): 642-649.
.. [5] Park, Jaeseok, et al. "Artifact and noise suppression in
       GRAPPA imaging using improved k‐space coil calibration and
       variable density sampling." Magnetic Resonance in
       Medicine: An Official Journal of the International Society
       for Magnetic Resonance in Medicine 53.1 (2005): 186-193.
.. [6] Breuer, Felix A., et al. "Dynamic autocalibrated parallel
       imaging using temporal GRAPPA (TGRAPPA)." Magnetic
       Resonance in Medicine: An Official Journal of the
       International Society for Magnetic Resonance in Medicine
       53.4 (2005): 981-985.
.. [7] Setsompop, Kawin, et al. "Blipped‐controlled aliasing in
       parallel imaging for simultaneous multislice echo planar
       imaging with reduced g‐factor penalty." Magnetic resonance
       in medicine 67.5 (2012): 1210-1224.
.. [8] Cauley, Stephen F., et al. "Interslice leakage artifact
       reduction technique for simultaneous multislice
       acquisitions." Magnetic resonance in medicine 72.1 (2014):
       93-102.
.. [9] Griswold, Mark A., et al. "Parallel magnetic resonance
       imaging using the GRAPPA operator formalism." Magnetic
       resonance in medicine 54.6 (2005): 1553-1556.
.. [10] Blaimer, Martin, et al. "2D‐GRAPPA‐operator for faster 3D
        parallel MRI." Magnetic Resonance in Medicine: An Official
        Journal of the International Society for Magnetic Resonance
        in Medicine 56.6 (2006): 1359-1364.
