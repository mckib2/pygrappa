'''Basic usage of k-t BLAST implementation.'''

import numpy as np
from phantominator import shepp_logan

from pygrappa import ktblast

if __name__ == '__main__':

    N = 128
    ph = shepp_logan(N)
    ph = (ph + 1j*ph)*N
    kt = 100
    ct = 40

    # Bring into k-space with the desired number of time frames
    axes = (0, 1)
    _kspace = 1/N*np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(
        ph, axes=axes), axes=axes), axes=axes)
    kspace = np.tile(_kspace[..., None], (1, 1, kt))

    # Simple intensity variation in time
    tt = np.linspace(0, 2*np.pi, kt+ct+1)[1:]
    kspace *= np.sin(tt[ct:]*2)[None, None, :]

    # crop 20 lines from the center of k-space for calibration
    pd = 10
    ctr = int(N/2)
    calib = np.tile(_kspace[ctr-pd:ctr+pd, :, None], (1, 1, ct))
    calib *= np.sin(tt[:ct]*2)[None, None, :]
    print(calib.shape)

    # Undersample kspace: R=4
    kspace[0::4, :, 0::4] = 0
    kspace[1::4, :, 1::4] = 0
    kspace[2::4, :, 2::4] = 0
    kspace[3::4, :, 3::4] = 0

    # Run k-t BLAST
    calib_win = np.hanning(calib.shape[0])[:, None] # PE direction
    freq_win = np.hanning(ct)
    ktblast(kspace, calib, calib_win=calib_win, freq_win=freq_win)
