'''Python implementation of Radial GRAPPA operator.'''

import numpy as np
from scipy.linalg import expm, logm


def radialgrappaop(
        kx, ky, k, nspokes=None, spoke_axis=-2, coil_axis=-1,
        spoke_axis_coord=-1, lamda=0.01, ret_lGtheta=False,
        traj_warn=True):
    '''Non-Cartesian Radial GRAPPA operator.

    Parameters
    ----------
    kx, ky: array_like
        k-space coordinates of kspace data, k.  kx and ky are 2D
        arrays containing (sx, nr) : (number of samples along ray,
        number of rays).
    k : array_like
        Complex kspace data corresponding to the measurements at
        locations kx, ky.  k has three dimensions: sx, nr, and coil.
    nspokes : int, optional
        Number of spokes.  Used when (kx, ky) and k are given with
        flattened sample and spoke axes, i.e., (sx*nr, nc).
    spoke_axis : int, optional
        Axis of k that contains the spoke data.  Not for kx, ky: see
        spoke_axis_coord to specify spoke axis for kx and ky.
    coil_axis : int, optional
        Axis of k that contains the coil data.
    spoke_axis_coord : int, optional
        Axis of kx and ky that hold the spoke data.
    lamda : float, optional
        Tikhonov regularization term used both for fitting Gtheta
        and log(Gx), log(Gy).
    ret_lGtheta : bool, optional
        Return log(Gtheta) instead of Gx, Gy.
    traj_warn : bool, optional
        Warn about potential inconsistencies in trajectory, e.g.,
        not shaped correctly.

    Returns
    -------
    Gx, Gy : array_like
        GRAPPA operators along the x and y axes.

    Raises
    ------
    AssertionError
        If kx and ky do not have spokes along spoke_axis_coord or if
        the standard deviation of distance between spoke points is
        greater than or equal to 1e-10.

    Notes
    -----
    Implements the radial training scheme for self calibrating GRAPPA
    operators in [1]_.  Too many coils could lead to instability of
    matrix exponents and logarithms -- use PCA or other suitable
    coil combination technique to reduce dimensionality if needed.

    References
    ----------
    .. [1] Seiberlich, Nicole, et al. "Self‐calibrating GRAPPA
           operator gridding for radial and spiral trajectories."
           Magnetic Resonance in Medicine: An Official Journal of the
           International Society for Magnetic Resonance in Medicine
           59.4 (2008): 930-935.
    '''

    # Move coils and spoke_axis to the back:
    if k.ndim == 2:
        # assume we only have a coil axis
        k = np.moveaxis(k, coil_axis, -1)
    else:
        k = np.moveaxis(k, (spoke_axis, coil_axis), (-2, -1))
    kx = np.moveaxis(kx, spoke_axis_coord, -1)
    ky = np.moveaxis(ky, spoke_axis_coord, -1)

    if k.ndim == 2 and nspokes is not None:
        nc = k.shape[-1]
        k = np.reshape(k, (-1, nspokes, nc))
    sx, nr, nc = k.shape[:]

    if kx.ndim == 1 and nspokes is not None:
        kx = np.reshape(kx, (sx, nr))
        ky = np.reshape(ky, (sx, nr))

    # We can do a sanity check to make sure we do indeed have rays.
    # We should have very little variation in dx, dy along each ray:
    if traj_warn:
        tol = 1e-5 if kx.dtype == np.float32 else 1e-10
        assert np.all(np.std(np.diff(kx, axis=0), axis=0) < tol)
        assert np.all(np.std(np.diff(ky, axis=0), axis=0) < tol)

    # We need sources (last source has no target!) and targets (first
    # target has no associated source!)
    S = k[:-1, ...]
    T = k[1:, ...]

    # We need a single GRAPPA operator to relate sources and
    # targets for each spoke.  We'll call it lGtheta.  Loop through
    # all rays -- maybe a way to do this without for loop?
    lGtheta = np.zeros((nr, nc, nc), dtype=k.dtype)
    for ii in range(nr):
        Sh = S[:, ii, :].conj().T
        ShS = Sh @ S[:, ii, :]
        ShT = Sh @ T[:, ii, :]
        lamda0 = lamda*np.linalg.norm(ShS)/ShS.shape[0]
        res = np.linalg.solve(
            ShS + lamda0*np.eye(ShS.shape[0]), ShT)
        lGtheta[ii, ...] = logm(res)

    # If the user only asked for the lGthetas, give them back!
    if ret_lGtheta:
        return lGtheta

    # Otherwise, we now need Gx, Gy.
    # Some implementations I've seen of this assume the same interval
    # always along a single ray, i.e.:
    # dx = kx[1, :] - kx[0, :]
    # dy = ky[1, :] - ky[0, :]
    # I'm going to assume they are similar and take the average:
    dx = np.mean(np.diff(kx, axis=0), axis=0)
    dy = np.mean(np.diff(ky, axis=0), axis=0)
    dxy = np.concatenate((dx[:, None], dy[:, None]), axis=1)

    # Let's solve this equation:
    #     lGtheta = dxy @ lGxy
    #     (nr, nc^2) = (nr, 2) @ (2, nc^2)
    #     dxy.T lGtheta = dxy.T @ dxy @ lGxy
    #     (dxy.T @ dxy)^-1 @ dxy.T lGtheta = lGxy
    lGtheta = np.reshape(lGtheta, (nr, nc**2), 'F')
    RtR = dxy.T @ dxy
    RtG = dxy.T @ lGtheta
    lamda0 = lamda*np.linalg.norm(RtR)/RtR.shape[0]
    res = np.linalg.solve(RtR + lamda*np.eye(RtR.shape[0]), RtG)
    lGx = np.reshape(res[0, :], (nc, nc))
    lGy = np.reshape(res[1, :], (nc, nc))

    # Take matrix exponential to get from (lGx, lGy) -> (Gx, Gy)
    # and we're done!
    return (expm(lGx), expm(lGy))


if __name__ == '__main__':
    pass
