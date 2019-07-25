'''Python k-t BLAST implementation.'''

import numpy as np
from skimage.util import pad

def ktblast(kspace, calib, calib_win=None, freq_win=None,
            safety_margin=2, time_axis=-1):
    '''k-t BLAST.

    Parameters
    ----------
    kspace : array_like
        Undersampled k-space time frames to be reconstructed.
        Unsampled pixels should be exactly zero.
    calib : array_like
        Training stage data (calibration).  Time frames containing
        only the center of k-space.
    calib_win : array_like, optional
        2D window to apply to calibration k-space data before
        zero-padding and inverse Fourier transformation.
    freq_win : array_like, optional
        1D window to apply to x-f data to attenuate high temporal
        frequencies.
    safety_margin : float, optional
        Factor to multiply x-f data by.  Higher safety margins result
        in more image features being reconstructed at the expense of
        noise increase.  Default is 2 as suggested in [1]_.
    time_axis : int, optional
        Dimension corresponding to time.

    Notes
    -----
    Implements k-t BLAST algorithm described in [1]_.

    References
    ----------
    .. [1] Tsao, Jeffrey, Peter Boesiger, and Klaas P. Pruessmann.
           "k‐t BLAST and k‐t SENSE: dynamic MRI with high frame rate
           exploiting spatiotemporal correlations." Magnetic
           Resonance in Medicine: An Official Journal of the
           International Society for Magnetic Resonance in Medicine
           50.5 (2003): 1031-1042.
    '''

    # Move time dimension to the back so we know where it is
    kspace = np.moveaxis(kspace, time_axis, -1)
    calib = np.moveaxis(calib, time_axis, -1)

    # Get sizes of matrices
    kx, ky, kt = kspace.shape[:]
    cx, cy, ct = calib.shape[:]

    # Get windows with which to filter zero-padded calib data and
    # temporal frequencies
    if calib_win is None:
        calib_win = 1
    else:
        assert calib_win.ndim == 2, 'calib_win must be 2D!'
        calib_win = calib_win[..., None]
    if freq_win is None:
        freq_win = 1
    else:
        assert freq_win.ndim == 1, 'freq_win must be 1D!'
        freq_win = freq_win[None, :]

    # We need a baseline estimate, so let's temporally average the
    # kspace!
    kspace_avg = np.sum(kspace, axis=-1)
    kspace_avg /= np.sum(np.abs(kspace) > 0, axis=-1)

    assert False


    # In-plane inverse Fourier transform of calibration data
    # Zero-padding: adds zeros around calibration data to match size
    # of kspace.  If difference in size between calib and kspace is
    # odd, we can't evenly pad calib, so we arbitrarily choose to
    # throw the leftovers on the left-hand side of the tuple in pad().
    axes = (0, 1)
    px, py = (kx - cx), (ky - cy)
    px2, py2 = int(px/2), int(py/2)
    adjx, adjy = np.mod(px, 2), np.mod(py, 2)
    fac = np.sqrt(kx*ky)
    lowres = fac*np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(
        pad( #pylint: disable=E1102
            calib*calib_win,
            ((px2+adjx, px2), (py2+adjy, py2), (0, 0)),
            mode='constant'),
        axes=axes), axes=axes), axes=axes)

    # Now construct an x-t array for each column
    t_ctr = int(ct/2)
    for ii in range(ky):
        xt = lowres[:, ii, :]

        # Inverse Fourier transform along t to get x-f array
        xf = np.sqrt(ct)*np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(
            xt, axes=-1), axis=-1), axes=-1)

        # Set f=0 to zero
        xf[:, t_ctr] = 0

        # Filter in f to attenuate high temporal frequencies and
        # safety margin
        xf *= freq_win*safety_margin

        # Squared magnitude gives estimated squared deviation
        Mxf2_diag = np.abs(xf)**2

if __name__ == '__main__':
    pass
