'''Unit tests for iGRAPPA.'''

import unittest

import numpy as np
from pygrappa import igrappa
from .helpers import make_base_test_case_2d


# This runs really slow, so reduce number of tests
class TestiGRAPPA(make_base_test_case_2d(
        igrappa,
        ssim_thresh=.92,
        Ms=[32, 30],
        Ns=[32],
        ncoils=[4],
        cMs=[1/3],
        cNs=[1/3],
        types=[('complex64', np.complex64)],
)):
    # TODO: threshold could probably be higher
    pass


if __name__ == '__main__':
    unittest.main()
