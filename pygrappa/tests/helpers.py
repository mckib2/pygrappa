'''Functions to generate tests.'''

import unittest

import numpy as np
from phantominator import shepp_logan
try:
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    from skimage.measure import compare_ssim as ssim
from pygrappa.utils import gaussian_csm


def shepp_logan2d(M=64, N=64, nc=4, dtype=np.complex128):
    '''Make 2d phantom.'''
    ax = (0, 1)
    imspace = shepp_logan((M, N))
    mps = gaussian_csm(M, N, nc)
    coil_ims = imspace[..., None]*mps
    coil_ims = coil_ims.astype(dtype)
    kspace = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(
        coil_ims, axes=ax), axes=ax, norm='ortho'), axes=ax)
    kspace = kspace.astype(dtype)
    imspace = np.abs(imspace)
    imspace /= np.max(imspace.flatten())  # normalize
    return (imspace, coil_ims, kspace, mps)


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


def undersample_x3_y3(kspace):
    return undersample_x3(undersample_y3(kspace.copy()))


def make_base_test_case_2d(
        grappa_fun,
        ssim_thresh=.92,
        Ms=(32, 30),
        Ns=(32, 30),
        ncoils=(4, 7),
        cMs=None,
        cNs=None,
        types=None,
        extra_args=None,
        phantom_fun_args=None,
        use_R3=False,
        is_sense=False,
        output_kspace=True,
):
    '''Make a test based on combinations of input parameters.

    Parameters
    ----------
    grappa_fun : callable
        The GRAPPA-like reconstruction function to be tested.
    ssim_thresh : float, optional
        SSIM value required to pass the test.
    Ms, Ns : iterable of ints, optional
        kspace sizes along `x` and `y` respectively.
    ncoils : iterable of ints, optional
        Number of coils to use.
    cMs, cNs : iterable of floats, optional
        calib sizes along `x` and `y` respectively given as
        fractions of Ms and Ns.
    types : tuple, (str, dtype), optional
        A tuple describing a type for input kspace data.
        `types[ii][0]` has a string that the type will be refered
        to in the test name while `types[ii][1]` has the numpy
        dtype to use with `.astype(...)`. `complex64` and
        `complex128` are the default.
    extra_args : dict, optional
        Extra arguments to be passed to `grappa_fun`.  No extra
        arguments are passed by default.
    phantom_fun_args : dict, optional
        Extra arguments to be passed to `phantom_fun`.  No extra
        arguments are passed by default.
    use_R3 : bool, optional
        Use R=3 reduction factors.
    is_sense : bool, optional
        Whether ``grappa_fun`` is for a GRAPPA-like function or
        a SENSE-like function.
    output_kspace : bool, optional
        Whether ``grappa_fun`` returns kspace or imspace.
    '''

    if cMs is None:
        cMs = [1, 1/2, 1/3, 1/4]
    if cNs is None:
        cNs = [1, 1/2, 1/3, 1/4]
    if types is None:
        types = [('complex64', np.complex64), ('complex128', np.complex128)]
    if extra_args is None:
        extra_args = dict()
    if phantom_fun_args is None:
        phantom_fun_args = dict()

    # Undersampling factors
    undersampling_funs = [
        no_undersampling,
        undersample_x2,
        undersample_y2,
        undersample_x2_y2,
    ]
    if use_R3:
        undersampling_funs += [
            undersample_x3,
            undersample_y3,
            # undersample_x3_y3,
        ]

    class TestBaseGRAPPA2D(unittest.TestCase):
        '''Tests that every GRAPPA method should handle.'''
        def setUp(self):
            pass

    if not is_sense:
        funcname_template = 'test_recon_{phantom_fun}_M{M}_N{N}_nc{nc}_{calib_fun}_cM{cM}_cN{cN}_{undersampling_fun}_{type}'
    else:
        funcname_template = 'test_recon_{phantom_fun}_M{M}_N{N}_nc{nc}_{undersampling_fun}_{type}'
        calib_fun = [None]
        cM, cN = [None], [None]

    for phantom_fun in [shepp_logan2d]:
        for M in Ms:  # try to keep these small for tests to run quickly
            for N in Ns:
                for nc in ncoils:
                    for calib_fun in [calib2d]:
                        for cM in [int(frac*M) for frac in cMs]:
                            for cN in [int(frac*N) for frac in cNs]:
                                for undersampling_fun in undersampling_funs:
                                    for tipe in types:

                                        # Only run if the dimensions are both even or odd
                                        @unittest.skipIf((M % 2) ^ (N % 2), 'One odd dimension')
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
                                            # print(funcname_template.format(
                                            #     phantom_fun=phantom_fun.__name__,
                                            #     M=M, N=N, nc=nc,
                                            #     calib_fun=calib_fun.__name__,
                                            #     cM=cM, cN=cN,
                                            #     undersampling_fun=undersampling_fun.__name__,
                                            #     type=str(tipe[0])))
                                            imspace, _coil_ims, kspace, mps = phantom_fun(
                                                M=M, N=N, nc=nc, dtype=tipe[1],
                                                **phantom_fun_args)

                                            # Don't need calibration region if not GRAPPA
                                            if not is_sense:
                                                calib = calib_fun(kspace, cM, cN)

                                            # Undersample
                                            kspace_u = undersampling_fun(kspace)

                                            # Run with either grappa API or sense API
                                            if not is_sense:
                                                recon = grappa_fun(kspace_u, calib, **extra_args)
                                            else:
                                                recon = grappa_fun(kspace_u, mps, **extra_args)

                                            # Make sure types match
                                            self.assertEqual(recon.dtype, tipe[1])

                                            # Do SOS recon to compare reconstruction quality
                                            # via SSIM measure
                                            if output_kspace:
                                                ax = (0, 1)
                                                recon = np.sqrt(np.sum(np.abs(
                                                    np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(
                                                        recon, axes=ax), axes=ax, norm='ortho'), axes=ax))**2, axis=-1))
                                            else:
                                                recon = np.abs(recon).astype(imspace.dtype)
                                            recon /= np.max(recon.flatten())

                                            ssim0 = ssim(
                                                imspace, recon,
                                                data_range=imspace.flatten().max() - imspace.flatten().min())
                                            # print(ssim0)
                                            self.assertTrue(ssim0 > ssim_thresh, f'ssim={ssim0} ({ssim_thresh})')

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
