'''Unit tests for CGRAPPA.'''

import unittest

import numpy as np
from pygrappa import cgrappa
from .helpers import make_base_test_case_2d


class TestCGRAPPA(make_base_test_case_2d(cgrappa, ssim_thresh=.92, types=[('complex128', np.complex128)])):
    pass


if __name__ == '__main__':
    unittest.main()
