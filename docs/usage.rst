
.. usage:

Usage
======

.. note::

   These should probably be moved into docstrings for each method.

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

For reconstructions with more than 2 dimensions, there is a
generalized multidimensional implementation called `mdgrappa()` that
can be called as follows:

.. code-block:: python

    from pygrappa import mdgrappa
    res = mdgrappa(kspace, calib, kernel_size=(5, 5, 5)) # e.g., 3D

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
a 2D Cartesian calibration dataset:

.. code-block:: python

    from pygrappa import grappaop

    sx, sy, ncoils = calib.shape[:]
    Gx, Gy = grappaop(calib, coil_axis=-1)

See the examples to see how to use the GRAPPA operators to
reconstruct datasets.

Similarly, `radialgrappaop()` returns two unit GRAPPA operators [13]_
found from a radial calibration dataset:

.. code-block:: python

    from pygrappa import radialgrappaop
    sx, nr = kx.shape[:] # sx: number of samples along each spoke
    sx, nr = ky.shape[:] # nr: number of rays/spokes
    sx, nr, nc = k.shape[:] # nc is number of coils

    Gx, Gy = radialgrappaop(kx, ky, k)

For large number of coils, warnings will appear about matrix
logarithms and exponents, but I think it should be fine.

`ttgrappa` implements the through-time GRAPPA algorithm ([11]_).
It accepts arbitrary k-space sampling locations and measurements
along with corresponding fully sampled calibration data.  The kernel
is specified by the number of points desired, not a tuple as is
usually the case:

.. code-block:: python

    from pygrappa import ttgrappa

    # kx, ky are both 1D arrays describing the points (kx, ky)
    # sampled in kspace.  kspace is a matrix with two dimensions:
    # (meas., coil) corresponding to the measurements takes at each
    # (kx, ky) from each coil.  (cx, cy) and calib are similarly
    # supplied.  kernel_size is the number of nearest neighbors used
    # for the least squares fit.  25 corresponds to a kernel size of
    # (5, 5) for Cartesian GRAPPA:

    res = ttgrappa(kx, ky, kspace, cx, cy, calib, kernel_size=25)

PARS [12]_ is an older parallel imaging algorithm, but it checks out.
It can be called like so:

.. code-block:: python

    from pygrappa import pars

    # Notice we provide the image domain coil sensitivity maps: sens
    res = pars(kx, ky, kspace, sens, kernel_radius=.8, coil_axis=-1)

    # You can use kernel_size instead of kernel_radius, but it seems
    # that kernel_radius gives better reconstructions.

In general, PARS is slower in this Python implementation because
the size of the kernels change from target point to target point,
so we have to loop over every single one.  Notice that `pars` returns
the image domain reconstruction on the Cartesian grid, not
interpolated k-space as most methods in this package do.

GROG [14]_ is called with trajectory information and unit GRAPPA
operators Gx and Gy:

.. code-block:: python

    from pygrappa import grog

    # (N, M) is the resolution of the desired Cartesian grid
    res = grog(kx, ky, k, N, M, Gx, Gy)

    # Precomputations of fractional matrix powers can be accelerated
    # using a prime factorization technique submitted to ISMRM 2020:
    res = grog(kx, ky, k, N, M, Gx, Gy, use_primefac=True)

See `examples.basic_radialgrappaop.py` for usage example.

Esoterically, forward and inverse gridding are supported out of the
box with this implementation of GROG, i.e.,
non-Cartesian -> Cartesian can be reversed.  It's not perfect and
I've never heard of anyone doing this via GROG, but check out
`examples.inverse_grog` for more info.

NL-GRAPPA uses machine learning feature augmentation to reduce model-
based reconstruction error [15]_.  It's implementation is based on
the original script, so its function signature looks different than
normal.  Please see example for better understanding of arguments.
It can be called like so:

.. code-block:: python

    from pygrappa import nlgrappa_matlab
    res = nlgrappa_matlab(
        kspace_u, R, pe_loc, calib, acs_line_loc, num_block,
        num_column, times_comp)

You might need to play around with the arguments to get good images.
The implementation is pretty much a straight mapping of the original
MATLAB script to Python, so performance is not going to be very
good compared to the other GRAPPA implementations in this package.

There was Python implementation in previous versions of pygrappa,
but it never worked correctly and raises an exception now if you
try to call it.

g-factor maps show geometry factor and a general sense of how well
parallel imaging techniques like GRAPPA will work.  Coil sensitivities
must be known for to use this function as well as integer
acceleration factors in x and y:

.. code-block:: python

    from pygrappa import gfactor
    g = gfactor(sens, Rx, Ry)

SENSE implements the algorithm described in [16]_ for unwrapping
aliased images along a single axis.  Coil sensitivity maps must be
provided.  Coil images may be provided in image domain or k-space
with the approprite flag:

.. code-block:: python

    from pygrappa import sense1d
    res = sense1d(im, sens, Rx=2, coil_axis=-1)

    # Or, kspace data for coil images may be provided:
    res = sense1d(kspace, sens, Rx=2, coil_axis=-1, imspace=False)

CG-SENSE implements a Cartesian version of the algorithm described
in [17]_.  It works for arbitrary undersampling of Cartesian datasets.
Undersampled k-space and coil sensitivity maps are provided:

.. code-block:: python

    from pygrappa import cgsense
    res = cgsense(kspace, sens, coil_axis=-1)

Although SENSE is more commonly known as an image domain parallel
imaging reconstruction technique, it is useful to include in this
package for comparison to kernel based and hybrid reconstructions.

References
==========

.. include:: references.rst
