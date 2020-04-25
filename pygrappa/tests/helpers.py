'''Functions to generate tests.'''

import unittest

import numpy as np
from phantominator import shepp_logan
from skimage.metrics import structural_similarity as ssim
from utils import gaussian_csm

def shepp_logan2d(M=64, N=64, nc=4, dtype=np.complex128):
    '''Make 2d phantom.'''
    ax = (0, 1)
    imspace = shepp_logan((M, N))
    coil_ims = imspace[..., None]*gaussian_csm(M, N, nc)
    coil_ims = coil_ims.astype(dtype)
    kspace = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(
        coil_ims, axes=ax), axes=ax, norm='ortho'), axes=ax)
    kspace = kspace.astype(dtype)
    imspace /= np.max(imspace.flatten()) # normalize
    return(imspace, coil_ims, kspace)

def calib2d(kspace, M=10, N=10):
    '''Extract a rectangle from the center of 2d kspace.'''
    ctr0, ctr1 = kspace.shape[0]//2, kspace.shape[1]//2
    M2, N2 = M//2, N//2
    return kspace[ctr0-M2:ctr0+M2, ctr1-N2:ctr1+N2, :].copy()

def no_undersampling(kspace):
    return kspace
def undersample_x2(kspace):
    kspace_u = kspace.copy()
    kspace_u[::2, ...] = 0
    return kspace_u
def undersample_y2(kspace):
    kspace_u = kspace.copy()
    kspace_u[:, ::2, ...] = 0
    return kspace_u
def undersample_x2_y2(kspace):
    kspace_u = kspace.copy()
    kspace_u[::2, 1::2, :] = 0
    kspace_u[1::2, ::2, :] = 0
    return kspace_u
def undersample_x3(kspace):
    kspace_u = kspace.copy()
    kspace_u[0::3, ...] = 0
    kspace_u[1::3, ...] = 0
    return kspace_u
def undersample_y3(kspace):
    kspace_u = kspace.copy()
    kspace_u[:, 0::3, ...] = 0
    kspace_u[:, 1::3, ...] = 0
    return kspace_u

def make_base_test_case_2d(grappa_fun, ssim_thresh=.92, extra_args=None):

    if extra_args is None:
        extra_args = dict()

    class TestBaseGRAPPA2D(unittest.TestCase):
        '''Tests that every GRAPPA method should handle.'''
        def setUp(self):
            pass

    funcname_template = 'test_recon_{phantom_fun}_M{M}_N{N}_nc{nc}_{calib_fun}_cM{cM}_cN{cN}_{undersampling_fun}_{type}'
    for phantom_fun in [shepp_logan2d]:
        for M in [32, 30]:#, 64, 63]: # try to keep these small for tests to run quickly
            for N in [32, 30]:#, 64, 63]:
                for nc in [4, 7]:
                    for calib_fun in [calib2d]:
                        for cM in [M, M//2, M//3, M//4]:
                            for cN in [N, N//2, N//3, N//4]:
                                for undersampling_fun in [
                                        no_undersampling,
                                        undersample_x2, undersample_y2, undersample_x2_y2,
                                        #undersample_x3, undersample_y3, # TODO: get R=3 working
                                ]:
                                    for tipe in [('complex64', np.complex64), ('complex128', np.complex128)]:

                                        # Only run if the dimensions are both even or odd
                                        @unittest.skipIf(M%2 ^ N%2, 'One odd dimension')
                                        def _test_fun(
                                                self,
                                                phantom_fun=phantom_fun,
                                                M=M,
                                                N=N,
                                                nc=nc,
                                                calib_fun=calib_fun,
                                                cM=cM,
                                                cN=cN,
                                                undersampling_fun=undersampling_fun,
                                                tipe=tipe,
                                        ):
                                            imspace, _coil_ims, kspace = phantom_fun(M=M, N=N, nc=nc, dtype=tipe[1])
                                            calib = calib_fun(kspace, cM, cN)
                                            kspace_u = undersampling_fun(kspace)
                                            recon = grappa_fun(kspace_u, calib, **extra_args)

                                            # Make sure types match
                                            self.assertEqual(recon.dtype, tipe[1])

                                            # Do SOS recon to compare reconstruction quality
                                            # via SSIM measure
                                            ax = (0, 1)
                                            recon = np.sqrt(np.sum(np.abs(
                                                np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(
                                                    recon, axes=ax), axes=ax, norm='ortho'), axes=ax))**2, axis=-1))
                                            recon /= np.max(recon.flatten())
                                            ssim0 = ssim(imspace, recon)
                                            self.assertTrue(ssim0 > ssim_thresh, 'ssim=%g' % ssim0)

                                        # Add test function to TestClass
                                        setattr(
                                            TestBaseGRAPPA2D,
                                            funcname_template.format(
                                                phantom_fun=phantom_fun.__name__,
                                                M=M, N=N, nc=nc,
                                                calib_fun=calib_fun.__name__,
                                                cM=cM, cN=cN,
                                                undersampling_fun=undersampling_fun.__name__,
                                                type=str(tipe[0])),
                                            _test_fun)

    return TestBaseGRAPPA2D
