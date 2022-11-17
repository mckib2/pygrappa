'''Python implementation of hp-GRAPPA.'''

import numpy as np

from pygrappa import mdgrappa


def hpgrappa(
        kspace, calib, fov, kernel_size=(5, 5), w=None, c=None,
        ret_filter=False, coil_axis=-1, lamda=0.01, silent=True):
    '''High-pass GRAPPA.

    Parameters
    ----------
    fov : tuple, (FOV_x, FOV_y)
        Field of view (in m).
    w : float, optional
        Filter parameter: determines the smoothness of the filter
        boundary.
    c : float, optional
        Filter parameter: sets the cutoff frequency.
    ret_filter : bool, optional
        Returns the high pass filter determined by (w, c).

    Notes
    -----
    If w and/or c are None, then the closest values listed in
    Table 1 from [1]_ will be used.

    F2 described by Equation [2] in [1]_ is used to generate the
    high pass filter.

    References
    ----------
    .. [1] Huang, Feng, et al. "High‐pass GRAPPA: An image support
           reduction technique for improved partially parallel
           imaging." Magnetic Resonance in Medicine: An Official
           Journal of the International Society for Magnetic
           Resonance in Medicine 59.3 (2008): 642-649.
    '''

    # Pass GRAPPA arguments forward
    grappa_args = {
        'kernel_size': kernel_size,
        'coil_axis': -1,
        'lamda': lamda,
    }

    # Put the coil dim in the back
    kspace = np.moveaxis(kspace, coil_axis, -1)
    calib = np.moveaxis(calib, coil_axis, -1)
    kx, ky, nc = kspace.shape[:]
    cx, cy, nc = calib.shape[:]
    kx2, ky2 = int(kx/2), int(ky/2)
    cx2, cy2 = int(cx/2), int(cy/2)

    # Save the original type
    tipe = kspace.dtype

    # Get filter parameters if None provided
    if w is None or c is None:
        _w, _c = _filter_parameters(nc, np.min([cx, cy]))
        if w is None:
            w = _w
        if c is None:
            c = _c

    # We'll need the filter, seeing as this is high-pass GRAPPA
    fov_x, fov_y = fov[:]
    kxx, kyy = np.meshgrid(
        kx*np.linspace(-1, 1, ky)/(fov_x*2),  # I think this gives
        ky*np.linspace(-1, 1, kx)/(fov_y*2))  # kspace FOV?
    F2 = (1 - 1/(1 + np.exp((np.sqrt(kxx**2 + kyy**2) - c)/w)) +
          1/(1 + np.exp((np.sqrt(kxx**2 + kyy**2) + c)/w)))

    # Apply the filter to both kspace and calibration data
    kspace_fil = kspace*F2[..., None]
    calib_fil = calib*F2[kx2-cx2:kx2+cx2, ky2-cy2:ky2+cy2, None]

    # Do regular old GRAPPA on filtered data
    res = mdgrappa(kspace_fil, calib_fil, **grappa_args)

    # Inverse filter
    res = res/F2[..., None]

    # Restore measured data
    mask = np.abs(kspace[..., 0]) > 0
    res[mask, :] = kspace[mask, :]
    res[kx2-cx2:kx2+cx2, ky2-cy2:ky2+cy2, :] = calib
    res = np.moveaxis(res, -1, coil_axis)

    # Return the filter if user asked for it
    if ret_filter:
        return (res.astype(tipe), F2)
    return res.astype(tipe)


def _filter_parameters(ncoils, num_acs_lines):
    '''Table 1: predefined filter parameters from [1]_.

    Parameters
    ----------
    ncoils : int
        Number of coil channels.
    num_acs_lines : {24, 32, 48, 64}
        Number of lines in the ACS region (i.e., number of PEs
        in [1]_).

    Returns
    -------
    (w, c) : tuple
        Filter parameters.
    '''

    LESS_THAN_8 = True
    MORE_THAN_8 = False
    lookup = {
        # key: (num_acs_lines, ncoils <= 8), value: (w, c)
        (24, LESS_THAN_8): (12, 24),
        (32, LESS_THAN_8): (10, 24),
        (48, LESS_THAN_8): (8, 24),
        (64, LESS_THAN_8): (6, 24),

        (24, MORE_THAN_8): (2, 12),
        (32, MORE_THAN_8): (2, 14),
        (48, MORE_THAN_8): (2, 18),
        (64, MORE_THAN_8): (2, 24)
    }

    # If num_acs_lines is not in {24, 32, 48, 64}, find the closest
    # one and use that:
    valid = np.array([24, 32, 48, 64])
    idx = np.argmin(np.abs(valid - num_acs_lines))
    num_acs_lines = valid[idx]

    return lookup[num_acs_lines, ncoils <= 8]


if __name__ == '__main__':
    pass
