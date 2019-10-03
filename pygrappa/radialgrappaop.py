'''Python implementation of Radial GRAPPA operator.'''

import numpy as np
from scipy.linalg import expm

def radialgrappaop(
        kx, ky, k, spoke_axis=-2, coil_axis=-1, spoke_axis_coord=-1,
        lamda=0.01, ret_Gtheta=False):
    '''Non-Cartesian Radial GRAPPA operator.

    Parameters
    ----------

    Returns
    -------
    Gx, Gy : array_like
        GRAPPA operators for both the x and y directions.

    References
    ----------
    .. [1] Seiberlich, Nicole, et al. "Self‐calibrating GRAPPA
           operator gridding for radial and spiral trajectories."
           Magnetic Resonance in Medicine: An Official Journal of the
           International Society for Magnetic Resonance in Medicine
           59.4 (2008): 930-935.
    '''

    # Move coils to the back, spoke axis next to the back
    k = np.moveaxis(k, (spoke_axis, coil_axis), (-2, -1))
    _sx, nr, nc = k.shape[:]
    kx = np.moveaxis(kx, spoke_axis_coord, -1)
    ky = np.moveaxis(ky, spoke_axis_coord, -1)

    # We need sources (last source has no target!) and targets (first
    # target has no associated source!)
    S = k[:-1, ...]
    T = k[1:, ...]

    # We need a single GRAPPA operator to relate sources and
    # targets for each spoke.  We'll call it Gtheta.  Loop through
    # all rays -- maybe a way to do this without for loop?
    Gtheta = np.zeros((nc, nc, nr), dtype=k.dtype)
    for ii in range(nr):
        Sh = S[:, ii, :].conj().T
        ShS = Sh @ S[:, ii, :]
        ShT = Sh@ T[:, ii, :]
        lamda = 0.01
        Gtheta[..., ii] = np.linalg.solve(
            ShS + lamda*np.eye(ShS.shape[0]), ShT)

        # We need the logarithm if we're not returning Gtheta
        if not ret_Gtheta:
            _E, V = np.linalg.eig(Gtheta[..., ii])
            Vi = np.linalg.pinv(V)
            Ap = Vi @ Gtheta[..., ii] @ V
            lAp = np.diag(np.log(np.diag(Ap))) # force diagonal
            Gtheta[..., ii] = V @ lAp @ Vi

    # If the user only asked for the Gthetas, give them back!
    if ret_Gtheta:
        return Gtheta

    # Otherwise, we now need Gx, Gy.  Here we're assuming that
    # the stepsize between samples along a single ray is the same.
    # I'm not sure we want to do that?  How would be get around that?
    dx = kx[1, :] - kx[0, :]
    dy = ky[1, :] - ky[0, :]
    dxy = np.concatenate((dx[None, :], dy[None, :]), axis=0)

    # We have the following:
    #     Gtheta = [lGx, lGy] @ dxy
    #     (nc, nc, nr) = (nc, nc, 2) @ (2, nr)
    # Let's reshape it so we can solve it using least squares solver:
    Gtheta = np.reshape(Gtheta, (nc**2, nr))

    # Now we have:
    #     Gtheta = lGxy @ dxy
    #     (nc^2, nr) = (nc^2, 2) @ (2, nr)
    #
    #     Gtheta @ dxy.T = lGxy @ dxy @ dxy.T
    #     Gtheta @ dxy.T @ (dxy @ dxy.T)^-1 = lGxy
    RRt = dxy @ dxy.T
    GRt = Gtheta @ dxy.T
    res = np.linalg.solve(RRt, GRt.T) # maybe conj()?
    lGx = np.reshape(res[0, :], (nc, nc))
    lGy = np.reshape(res[1, :], (nc, nc))

    # Take matrix exponential to get from (lGx, lGy) -> (Gx, Gy)
    # and we're done!
    return(expm(lGx), expm(lGy))
