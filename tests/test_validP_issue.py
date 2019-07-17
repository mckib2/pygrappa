'''Replicate issue.'''

import unittest

import numpy as np
from pygrappa import grappa

class TestValidP(unittest.TestCase):
    '''Bug replication.'''

    @unittest.skip('Bug fix only')
    def test_validP_issue(self):
        '''Replication.'''

        # Pull in data we know will break grappa
        pc = np.load('pc0.npy')
        print('Loaded PC0')
        _sx, _sy, sz, _sc = pc.shape[:]
        pd = 12
        ctr = int(pc.shape[1]/2)

        # I think it happens somewhere in this dataset: slice 35
        for ii in range(35, sz):
            calib = pc[:, ctr-pd:ctr+pd+1, ii, :].copy()
            _recon = grappa(
                pc[:, :, ii, :], calib, (5, 5), coil_axis=-1)
            print('done with slice %d' % ii)

if __name__ == '__main__':
    unittest.main()
