'''Unit tests for GRAPPA.'''

import unittest

from pygrappa import grappa
from .helpers import make_base_test_case_2d


class TestGRAPPA(make_base_test_case_2d(grappa, ssim_thresh=.92)):
    pass


if __name__ == '__main__':
    unittest.main()
