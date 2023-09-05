'''Unit tests for hp-GRAPPA.'''

import unittest

from pygrappa import hpgrappa
from .helpers import make_base_test_case_2d


class TesthpGRAPPA(make_base_test_case_2d(hpgrappa, ssim_thresh=.86, extra_args={'fov': (10e-2, 10e-2)})):
    # TODO: failing on some nonsymmetric cases; thresh could be higher

    @unittest.expectedFailure
    def test_recon_shepp_logan2d_M30_N32_nc7_calib2d_cM7_cN8_undersample_x2_complex128(self):
        super().test_recon_shepp_logan2d_M30_N32_nc7_calib2d_cM7_cN8_undersample_x2_complex128()

    @unittest.expectedFailure
    def test_recon_shepp_logan2d_M30_N32_nc7_calib2d_cM7_cN8_undersample_x2_complex64(self):
        super().test_recon_shepp_logan2d_M30_N32_nc7_calib2d_cM7_cN8_undersample_x2_complex64()


if __name__ == '__main__':
    unittest.main()
