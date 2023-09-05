'''Unit tests for multidimensional GRAPPA.'''

import unittest

from pygrappa import mdgrappa
from .helpers import make_base_test_case_2d


class TestMDGRAPPA(make_base_test_case_2d(mdgrappa, ssim_thresh=0.95)):
    pass


if __name__ == '__main__':
    unittest.main()
