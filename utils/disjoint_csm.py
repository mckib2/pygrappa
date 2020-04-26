'''Too-good-to-be-real coil sensitivity maps.'''

import numpy as np


def disjoint_csm(sx, sy, ncoil):
    '''Make ncoil partitions of (sx, sy) box for coil sensitivities.

    Parameters
    ----------
    sx, sy : int
        Height and width of coil images.
    ncoil : int
        Number of coils to be simulated.

    Returns
    -------
    csm : array_like
        Simulated coil sensitivity maps.
    '''

    blocks = np.ones((sx, sy))
    blocks = np.array_split(blocks, ncoil, axis=0)
    csm = np.zeros((sx, sy, ncoil))
    idx = 0
    for ii in range(ncoil):
        sh = blocks[ii].shape[0]
        csm[idx:idx+sh, :, ii] = blocks[ii]
        idx += sh
    return csm


if __name__ == '__main__':
    pass
