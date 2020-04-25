'''Unit tests for hp-GRAPPA.'''

import unittest

from pygrappa import hpgrappa
from .helpers import make_base_test_case_2d

class TesthpGRAPPA(make_base_test_case_2d(hpgrappa, ssim_thresh=.88, extra_args={'fov': (10e-2, 10e-2)})):
    # TODO: failing on some nonsymmetric cases; thresh could be higher
    pass

if __name__ == '__main__':
    unittest.main()
