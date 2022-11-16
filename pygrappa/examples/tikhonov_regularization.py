'''Demonstrate the effect of Tikhonov regularization.'''

import numpy as np
import matplotlib.pyplot as plt
from phantominator import shepp_logan
try:
    from skimage.metrics import normalized_root_mse as compare_nrmse  # pylint: disable=E0611,E0401
except ImportError:
    from skimage.measure import compare_nrmse
from tqdm import tqdm

from pygrappa import cgrappa as grappa
from pygrappa.utils import gaussian_csm

if __name__ == '__main__':

    # Simple phantom
    N = 128
    ncoils = 5
    csm = gaussian_csm(N, N, ncoils)
    ph = shepp_logan(N)[..., None]*csm

    # Put into k-space
    ax = (0, 1)
    kspace = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(
        ph, axes=ax), axes=ax), axes=ax)
    kspace_orig = kspace.copy()

    # 20x20 ACS region
    pad = 10
    ctr = int(N/2)
    calib = kspace[ctr-pad:ctr+pad, ctr-pad:ctr+pad, :].copy()

    # R=2x2
    kspace[::2, 1::2, :] = 0
    kspace[1::2, ::2, :] = 0

    # Find Tikhonov param that minimizes NRMSE
    nlam = 20
    lamdas = np.linspace(1e-9, 5e-4, nlam)
    mse = np.zeros(lamdas.shape)
    akspace = np.abs(kspace_orig)
    for ii, lamda in tqdm(enumerate(lamdas), total=nlam, leave=False):
        recon = grappa(kspace, calib, lamda=lamda)
        mse[ii] = compare_nrmse(akspace, np.abs(recon))

    # Optimal param minimizes NRMSE
    idx = np.argmin(mse)

    # Take a look
    plt.plot(lamdas, mse)
    plt.plot(lamdas[idx], mse[idx], 'rx', label='Optimal lamda')
    plt.title('Tikhonov param vs NRMSE')
    plt.xlabel('lamda')
    plt.ylabel('NRMSE')
    plt.legend()
    plt.show()
