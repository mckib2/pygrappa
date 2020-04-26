'''Calculate g-factor maps.'''

import numpy as np


def gfactor(coils, Rx, Ry, coil_axis=-1, tol=1e-6):
    '''Compute g-factor map for coil sensitities and accelerations.

    Parameters
    ----------
    C : array_like
        Array of coil sensitivities
    Ry : int,
        x acceleration
    Ry : int
        y acceleration
    coil_axis : int, optional
        Dimension holding coil data.
    tol : float, optional

    Returns
    -------
    g : array_like
        g-factor map

    Notes
    -----
    Adapted from John Pauly's MATLAB script found at [1]_.

    References
    ----------
    .. [1] https://web.stanford.edu/class/ee369c/restricted/
           Solutions/assignment_4_solns.pdf
    '''

    # Coils to da back
    coils = np.moveaxis(coils, coil_axis, -1)
    nx, ny, _nc = coils.shape[:]

    # Get a reference SOS image
    sos = np.sqrt(np.sum(np.abs(coils)**2, axis=-1))

    nrx = nx/Rx
    nry = ny/Ry
    g = np.zeros((nx, ny))
    for idx in np.ndindex((nx, ny)):
        ii, jj = idx[:]

        if sos[ii, jj] > tol:
            s = []
            for LXLY in np.ndindex((Rx, Ry)):
                LX, LY = LXLY[:]

                ndx = int(np.mod(ii + LX*nrx, nx))
                ndy = int(np.mod(jj + LY*nry, ny))
                CT = coils[ndx, ndy, :]
                if ((LX == 0) and (LY == 0)):
                    s.append(CT)
                elif sos[ndx, ndy] > tol:
                    s.append(CT)

            s = np.array(s).T
            scs = (s.conj().T @ s).real
            scsi = np.linalg.pinv(scs)
            g[ii, jj] = np.sqrt(scs[0, 0]*scsi[0, 0])

    return g


def gfactor_single_coil_R2(coil, Rx=2, Ry=1):
    '''Specific example of a single homogeneous coil, R=2.

    Parameters
    ----------
    coil : array_like
        Single coil sensitivity.
    Ry : int,
        x acceleration
    Ry : int
        y acceleration

    Returns
    -------
    g : array_like
        g-factor map

    Notes
    -----
    Analytical solution for a single, homogeneous coil with an
    undersampling factor of R=2.  Equation 11 in [2]_.

    Comparing head-to-head with pygrappa.gfactor(), this does
    produce different results.  I don't know which one is more
    correct...

    References
    ----------
    .. [2] Blaimer, Martin, et al. "Virtual coil concept for improved
           parallel MRI employing conjugate symmetric signals."
           Magnetic Resonance in Medicine: An Official Journal of the
           International Society for Magnetic Resonance in Medicine
           61.1 (2009): 93-102.
    '''

    assert coil.ndim == 2, 'Must be single coil!'
    assert (Rx == 2 and Ry == 1) or (Rx == 1 and Ry == 2), (
        'Only one of Rx, Ry can be 2!')

    mask = np.abs(coil) > 0
    if Rx == 2:
        shifted = np.fft.fftshift(np.angle(coil), axes=0)
    else:
        shifted = np.fft.fftshift(np.angle(coil), axes=1)

    return mask/np.sin(np.abs(np.angle(coil) - shifted))


if __name__ == '__main__':
    pass
