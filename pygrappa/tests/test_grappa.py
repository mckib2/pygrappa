'''Unit tests for GRAPPA.'''

import unittest

import numpy as np

from pygrappa import grappa
from .helpers import make_base_test_case_2d


class TestGRAPPA(make_base_test_case_2d(grappa, ssim_thresh=.92)):
    def test_reconstruct_large_sampling_pattern(self):
        # The interior pattern has 65 * 130 targets, exceeding 8192.
        kspace = np.ones((132, 132, 2), dtype=np.complex128)
        kspace[::2] = 0
        calib = np.ones((8, 8, 2), dtype=np.complex128)

        recon = grappa(kspace, calib, kernel_size=(3, 3))

        self.assertTrue(np.all(np.abs(recon[::2]) > 0))
        np.testing.assert_array_equal(recon[1::2], kspace[1::2])

    def test_reconstruct_single_target(self):
        kspace = np.ones((3, 3, 2), dtype=np.complex128)
        kspace[1, 1] = 0
        calib = np.ones((8, 8, 2), dtype=np.complex128)

        recon = grappa(kspace, calib, kernel_size=(3, 3))

        self.assertTrue(np.all(np.abs(recon[1, 1]) > 0))


if __name__ == '__main__':
    unittest.main()
