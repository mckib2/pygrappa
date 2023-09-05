'''Unit tests for CG-SENSE.'''

import unittest

from pygrappa import cgsense
from .helpers import make_base_test_case_2d


class TestCGSENSE2d(make_base_test_case_2d(cgsense, ssim_thresh=0.93, use_R3=True, is_sense=True, output_kspace=False)):
    pass


if __name__ == '__main__':
    unittest.main()
