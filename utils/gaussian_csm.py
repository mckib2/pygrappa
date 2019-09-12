'''Simple coil sensitivity maps.'''

import numpy as np
from scipy.stats import multivariate_normal

def gaussian_csm(sx, sy, ncoil, sigma=1):
    '''Make a 2D Gaussian walk in a circle for coil sensitivities.

    Parameters
    ----------
    sx, sy : int
        Height and width of coil images.
    ncoil : int
        Number of coils to be simulated.

    Returns
    -------
    csm : array_like
        Simulated coil sensitivity maps.
    '''

    X, Y = np.meshgrid(
        np.linspace(-1, 1, sx), np.linspace(-1, 1, sy))
    pos = np.stack((X[..., None], Y[..., None]), axis=-1)
    csm = np.zeros((sx, sy, ncoil))
    cov = [[sigma, 0], [0, sigma]]
    for ii in range(ncoil):
        mu = [np.cos(ii/ncoil*np.pi*2), np.sin(ii/ncoil*2*np.pi)]
        csm[..., ii] = multivariate_normal(mu, cov).pdf(pos)
    return csm + 1j*csm

if __name__ == '__main__':
    pass
