"""Unit tests for VC-GRAPPA."""

import unittest

from pygrappa import vcgrappa
from pygrappa.tests.helpers import make_base_test_case_2d


class TestVCGRAPPA(make_base_test_case_2d(
        vcgrappa,
        ssim_thresh=.90)):
    # TODO: adjust/improve ssim_thresh; right now less than mdgrappa
    pass


if __name__ == '__main__':
    unittest.main()
