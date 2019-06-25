'''Python GRAPPA implementation.'''

import numpy as np
from skimage.util import pad

def grappa(
        kspace, calib, kernel_size=(5, 5), coil_axis=-1, lamda=0.01,
        disp=False, memmap=False, memmap_filename='out.memmap'):
    '''Now in Python.'''

    # Put the coil dimension at the end
    kspace = np.moveaxis(kspace, coil_axis, -1)
    calib = np.moveaxis(calib, coil_axis, -1)

    # Get displays up and running if we need them
    if disp:
        import matplotlib.pyplot as plt

    # get number of coils
    sx, sy, nc = kspace.shape[:]

    # We need to get weights!  So let's find all the sources in the
    # calibration data
    pcalib = pad(calib)
