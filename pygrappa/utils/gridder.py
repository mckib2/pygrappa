'''Simple gridding for non-Cartesian kspace.'''

import numpy as np
from scipy.interpolate import griddata


def gridder(
        kx, ky, k, sx, sy, coil_axis=-1, ifft=True, os=2,
        method='linear'):
    '''Helper function to grid non-Cartesian data.

    Parameters
    ----------
    kx, ky : array_like
        1D arrays of (kx, ky) coordinates cooresponding to
        measurements, k.
    k : array_like
        k-space measurements corresponding to spatial frequencies
        (kx, ky).
    sx, sy : int
        Size of gridded kspace.
    coil_axis : int, optional
        Dimension of k that holds the coil data.
    ifft : bool, optional
        Perform inverse FFT on gridded data and remove oversampling
        factor before returning.
    os : float, optional
        Oversampling factor for gridding.
    method : str, optional
        Strategy for interpolation used by
        scipy.interpolate.griddata().  See scipy docs for complete
        list of options.

    Returns
    -------
    imspace : array_like, optional
        If ifft=True.
    kspace : array_like, optional
        If ifft=False.
    '''

    # Move coil data to the back
    k = np.moveaxis(k, coil_axis, -1)

    yy, xx = np.meshgrid(
        np.linspace(np.min(kx), np.max(kx), sx*os),
        np.linspace(np.min(ky), np.max(ky), sy*os))
    grid_kspace = griddata((kx, ky), k, (xx, yy), method=method)

    if ifft:
        padx = int(sx*(os - 1)/2)
        pady = int(sy*(os - 1)/2)
        return np.fft.fftshift(np.fft.ifft2(
            np.fft.ifftshift(np.nan_to_num(grid_kspace), axes=(0, 1)),
            axes=(0, 1)), axes=(0, 1))[padx:-padx, pady:-pady, :]
    return grid_kspace


if __name__ == '__main__':
    pass
