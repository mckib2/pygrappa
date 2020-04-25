'''Unit tests for iGRAPPA.'''

import unittest

from pygrappa import igrappa
from .helpers import make_base_test_case_2d

class TestiGRAPPA(make_base_test_case_2d(igrappa, ssim_thresh=.92)):
    # TODO: threshold could probably be higher
    pass

if __name__ == '__main__':
    unittest.main()
