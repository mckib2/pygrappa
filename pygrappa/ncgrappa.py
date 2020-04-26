'''Python implementation of Non-Cartesian GRAPPA.'''

import numpy as np
from scipy.spatial import cKDTree  # pylint: disable=E0611


def ncgrappa(kx, ky, k, cx, cy, calib, kernel_size, coil_axis=-1):
    '''Non-Cartesian GRAPPA.

    Parameters
    ----------
    kx, ky : array_like
        k-space coordinates of kspace data, k.  kx and ky are 1D
        arrays.
    k : array_like
        Complex kspace data corresponding the measurements at
        locations kx, ky.  k has two dimensions: data and coil.  The
        coil dimension will be assumed to be last unless coil_axis=0.
        Unsampled points should be exactly 0.
    calib : array_like

    kernel_size : float
        Radius of kernel.
    coil_axis : int, optional
        Dimension of calib holding coil data.

    Notes
    -----
    Implements to the algorithm described in [1]_.

    References
    ----------
    .. [1] Luo, Tianrui, et al. "A GRAPPA algorithm for arbitrary
           2D/3D non‐Cartesian sampling trajectories with rapid
           calibration." Magnetic resonance in medicine 82.3 (2019):
           1101-1112.
    '''

    # Assume k has coil at end unless user says it's upfront.  We
    # want to assume that coils are in the back of calib and k:
    # calib = np.moveaxis(calib, coil_axis, -1)
    # if coil_axis == 0:
    #     k = np.moveaxis(k, coil_axis, -1)

    # Find all sampled and unsampled points
    mask = np.abs(k[:, 0]) > 0
    idx_unsampled = np.argwhere(~mask).squeeze()
    idx_sampled = np.argwhere(mask).squeeze()

    # Identify all the constellations for calibration using the
    # sampled kspace points
    kxy = np.concatenate((kx[:, None], ky[:, None]), axis=-1)
    kdtree = cKDTree(kxy[idx_sampled, :])

    # For each un‐sampled k‐space point, query the kd‐tree with the
    # prescribed distance (i.e., GRAPPA kernel size)
    constellations = kdtree.query_ball_point(
        kxy[idx_unsampled, :], r=kernel_size)

    # # Look at one to make sure we're doing what we think we're doing
    # import matplotlib.pyplot as plt
    # c = 500
    # plt.scatter(
    #     kx[idx_sampled][constellations[c]],
    #     ky[idx_sampled][constellations[c]])
    # plt.plot(kx[idx_unsampled[c]], ky[idx_unsampled[c]], 'r.')
    # plt.show()

    # Make an interpolator for the calibration data
    from scipy.interpolate import CloughTocher2DInterpolator
    cxy = np.concatenate((cx[:, None], cy[:, None]), axis=-1)
    f = CloughTocher2DInterpolator(cxy, calib)

    # For each constellation, let's train weights and fill in a hole
    T = f([0, 0]).squeeze()
    for ii, con in enumerate(constellations):
        Txy = kxy[idx_unsampled[ii]]
        Sxy = kxy[idx_sampled][con]
        Pxy = Sxy - Txy

        S = f(Pxy)
        print(S.shape, T.shape)

        # T = W S
        # (8) = (1, 24) @ (24, 8)
        TSh = T @ S.conj().T
        SSh = S @ S.conj().T
        W = np.linalg.solve(SSh, TSh)
        print(T)
        print(W @ S)

        assert False

    # # Now we need to find all the unique constellations
    # P = dict()
    # for ii, con in enumerate(constellations):
    #     T = kxy[idx_unsampled[ii]]
    #     S = kxy[idx_sampled][con]
    #
    #     # Move everything to be relative to the target
    #     P0 = S - T

    #     # Try to find the existing constellation
    #     key = P0.tostring()
    #     if key in P:
    #         P[key].append(ii)
    #     else:
    #         P[key] = [ii]
    #
    #     if ii == 500:
    #         import matplotlib.pyplot as plt
    #         plt.scatter(S[:, 0], S[:, 1])
    #         plt.plot(T[0], T[1], 'r.')
    #         plt.show()
    #
    # keys = list(P.keys())
    # print(len(keys))
