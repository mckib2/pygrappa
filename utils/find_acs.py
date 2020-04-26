'''Automatically find the ACS given a set of undersampled kspace.'''

import numpy as np
from skimage.segmentation import flood_fill

from utils import findRectangle2d


def find_acs(kspace):
    '''Start at center of kspace and find largest hyper-rectangle.

    Parameters
    ----------
    kspace : N-D array
        Assume coil_axis is at -1 and currently unpadded.
    '''

    # import matplotlib.pyplot as plt

    # Flood fill from center
    mask = np.abs(kspace[..., 0]) > 0
    ctr = tuple([sh//2 for sh in kspace.shape[:-1]])
    if mask[ctr] == 0:
        raise ValueError('There is no sample at the center!')
    ACS_val = 2
    region = flood_fill(mask.astype(int), seed_point=ctr, new_value=ACS_val, connectivity=0) == ACS_val

    # plt.imshow(region)
    # plt.show()

    if region.ndim == 2:

        # Find a centered rectangle
        acs, top, bottom = findRectangle2d(region, mask, ctr)

        # plt.imshow(acs)
        # plt.show()

        nc = kspace.shape[-1]
        acs = np.tile(acs[..., None], (1, 1, nc))
        # I'm not sure what I'm doing wrong yet...
        for ii in range(3):
            try:
                calib = kspace[acs].reshape((bottom-top-ii, -1, nc))
                break
            except ValueError:
                pass
        else:
            raise ValueError()
    else:
        raise NotImplementedError()

    return calib


if __name__ == '__main__':
    pass
