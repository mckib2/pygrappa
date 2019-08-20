'''Make sure that simple recon produce expected results.'''
import unittest

import numpy as np
from scipy.stats import multivariate_normal
from skimage.measure import compare_nrmse
from phantominator import shepp_logan
from pygrappa import grappa

class TestBasicRecon(unittest.TestCase):
    '''Numerical phantom reconstructions.'''

    def setUp(self):

        # Small phantoms for quick tests
        self.N = 32

        # Make a 2D Gaussian walk in a circle for coil sensitivities
        nc = 32
        X, Y = np.meshgrid(
            np.linspace(-1, 1, self.N), np.linspace(-1, 1, self.N))
        pos = np.stack((X[..., None], Y[..., None]), axis=-1)
        self.mps = np.zeros((self.N, self.N, nc))
        cov = [[1, 0], [0, 1]]
        for ii in range(nc):
            mu = [np.cos(ii/nc*np.pi*2), np.sin(ii/nc*2*np.pi)]
            self.mps[..., ii] = multivariate_normal(mu, cov).pdf(pos)

    def test_shepp_logan(self):
        '''The much-abused Shepp-Logan.'''

        ph = shepp_logan(self.N)
        ph /= np.max(ph.flatten())
        phs = ph[..., None]*self.mps

        kspace = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(
            phs, axes=(0, 1)), axes=(0, 1)), axes=(0, 1))
        kspace_u = np.array(np.zeros_like(kspace)) # wrap for pylint
        kspace_u[:, ::2, :] = kspace[:, ::2, :]
        ctr = int(self.N/2)
        pd = 5
        calib = kspace[:, ctr-pd:ctr+pd, :].copy()

        recon = grappa(kspace_u, calib, (5, 5), coil_axis=-1)
        recon = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(
            recon, axes=(0, 1)), axes=(0, 1)), axes=(0, 1))
        recon = np.abs(np.sqrt(np.sum(recon*np.conj(recon), axis=-1)))
        recon /= np.max(recon.flatten())

        # Make sure less than 4% NRMSE
        # print(compare_nrmse(ph, recon))
        self.assertTrue(compare_nrmse(ph, recon) < .04)

if __name__ == '__main__':
    unittest.main()
