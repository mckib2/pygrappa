'''Python implementation of hp-GRAPPA.'''

import numpy as np

def hpgrappa(kspace, calib, fov, coil_axis=-1):
    '''High-pass GRAPPA.

    Parameters
    ----------
    fov : tuple, (FOV_x, FOV_y)
        Field of view (in ?).

    References
    ----------
    .. [1] Huang, Feng, et al. "High‐pass GRAPPA: An image support
           reduction technique for improved partially parallel
           imaging." Magnetic Resonance in Medicine: An Official
           Journal of the International Society for Magnetic
           Resonance in Medicine 59.3 (2008): 642-649.
    '''

    # Put the coil dim in the back
    kspace = np.moveaxis(kspace, coil_axis, -1)
    calib = np.moveaxis(calib, coil_axis, -1)
    kx, ky, nc = kspace.shape[:]
    cx, cy, nc = calib.shape[:]

    # We'll need the filter, seeing as this is high-pass GRAPPA
    w, c = _filter_parameters(nc, np.min(cx, cy)) # min or max?
    fov_x, fov_y = fov[:]
    kxx, kyy = np.meshgrid(
        kx*np.linspace(-1, 1, kx)/(fov_x*2), # I think this gives
        ky*np.linspace(-1, 1, ky)/(fov_y*2)) # kspace FOV?
    F2 = (1 - 1/(1 + np.exp((np.sqrt(kxx**2 + kyy**2) - c)/w)) +
          1/(1 + np.exp((np.sqrt(kxx**2 + kyy**2) + c)/w)))


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
