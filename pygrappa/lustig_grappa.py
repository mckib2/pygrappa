'''Reference GRAPPA implementation ported to python.'''

import numpy as np
from skimage.util import pad
from tqdm import trange

def lustig_grappa(
        kspace, calib, kernel_size=(5, 5), coil_axis=-1, lamda=0.01,
        disp=False, memmap=False, memmap_filename='out.memmap'):
    '''GeneRalized Autocalibrating Partially Parallel Acquisitions.

    Parameters
    ----------
    kspace : array_like
        2D multi-coil k-space data to reconstruct from.  Make sure
        that the missing entries have exact zeros in them.
    calib : array_like
        Calibration data (fully sampled k-space).
    kernel_size : tuple, optional
        size of the 2D GRAPPA kernel (kx, ky).
    coil_axis : int, optional
        Dimension holding coil data.  The other two dimensions should
        be sizes (kx, ky).
    lamda : float, optional
        Tikhonov regularization for the kernel calibration.
    disp : bool, optional
        Display images as they are reconstructed
    memmap : bool, optional
        Store data in Numpy memmaps.  Use when datasets are too large
        to store in memory.
    memmap_filename : str, optional
        Name of memmap to store results in.  File is only saved if
        memmap=True.

    Returns
    -------
    res : array_like
        k-space data where missing entries have been filled in.

    Notes
    -----
    Based on implementation of the GRAPPA algorithm [1]_ from Miki
    Lustig [2]_.

    This is a GRAPPA reconstruction algorithm that supports
    arbitrary Cartesian sampling. However, the implementation
    is highly inefficient because it uses for loops.

    This implementation is very similar to the GE ARC implementation.
    The reconstruction looks at a neighborhood of a point and
    does a calibration according to the neighborhood to synthesize
    the missing point. This is a k-space varying interpolation.
    A sampling configuration is stored in a list, and retrieved
    when needed to accelerate the reconstruction (a bit).

    If memmap=True, the results will be written to memmap_filename
    and nothing is returned from the function.  Currently all
    intermediate matrices are still stored in memory.

    References
    ----------
    .. [1] Griswold, Mark A., et al. "Generalized autocalibrating
           partially parallel acquisitions (GRAPPA)." Magnetic
           Resonance in Medicine: An Official Journal of the
           International Society for Magnetic Resonance in Medicine
           47.6 (2002): 1202-1210.
    .. [2] https://people.eecs.berkeley.edu/~mlustig/Software.html
    '''

    # Put the coil dimension at the end
    kspace = np.moveaxis(kspace, coil_axis, -1)
    calib = np.moveaxis(calib, coil_axis, -1)

    # Get displays up and running if we need them
    if disp:
        import matplotlib.pyplot as plt

    # get number of coils
    sx, sy, ncoils = kspace.shape[:]

    # If user asked for it, store result in memmap.  If not,
    # business as usual
    if memmap:
        res = np.memmap(
            memmap_filename, mode='w+', shape=kspace.shape,
            dtype=kspace.dtype)
    else:
        res = np.zeros(kspace.shape, dtype=kspace.dtype)

    # construct calibration matrix
    AtA = dat2AtA(calib, kernel_size)

    # reconstruct single coil images
    for nn in trange(ncoils, leave=False):
        res[..., nn] = ARC(
            kspace, AtA, kernel_size, nn, lamda)
        if disp:
            plt.imshow(
                np.abs(np.sqrt(sx*sy)*np.fft.fftshift(
                    np.fft.ifft2(np.fft.ifftshift(
                        res[..., nn])))), cmap='gray')
            plt.show()

    # Move the coil dimension back where the user had it
    res = np.moveaxis(res, -1, coil_axis)

    # Don't return anything for memmap, just close the files
    # Otherwise, return the reconstructed coil images
    if memmap:
        del res
        return None
    return res

def ARC(kspace, AtA, kernel_size, c, lamda):
    '''ARC.'''
    sx, sy, ncoils = kspace.shape[:]
    kx, ky = kernel_size[:]

    # Zero-pad data
    px = int(kx/2)
    py = int(ky/2)
    kspace = pad( #pylint: disable=E1102
        kspace, ((px, px), (py, py), (0, 0)), mode='constant')

    dummyK = np.zeros((kx, ky, ncoils))
    dummyK[int(kx/2), int(ky/2), c] = 1
    idxy = np.where(dummyK)
    res = np.zeros((sx, sy), dtype=kspace.dtype)

    MaxListLen = 100 # max number of kernels we'll store for lookup
    LIST = np.zeros((kx*ky*ncoils, MaxListLen), dtype=kspace.dtype)
    KEY = np.zeros((kx*ky*ncoils, MaxListLen))

    count = 0 # current index for the next kernel to store in LIST
    for xy in np.ndindex((sx, sy)):
        x, y = xy[:]

        tmp = kspace[x:x+kx, y:y+ky, :]
        pat = np.abs(tmp) > 0
        if pat[idxy]:
            # If we aquired this k-space sample, use it!
            res[x, y] = tmp[idxy].squeeze()
        else:
            # If we didn't aquire it, let's either look up the
            # kernel or compute a new one
            key = pat.flatten()

            # If we have a matching kernel, the key will exist
            # in our array of KEYs.  This little loop looks through
            # the list to find if we have the kernel already
            idx = 0
            for nn in range(1, KEY.shape[1]+1):
                if np.sum(key == KEY[:, nn-1]) == key.size:
                    idx = nn
                    break

            if idx == 0:
                # If we didn't find a matching kernel, compute one.
                # We'll only hold MaxListLen kernels in the lookup
                # at one time to save on memory and lookup time
                count += 1
                kernel = calibrate(
                    AtA, kernel_size, ncoils, c, lamda,
                    pat)[0].flatten()
                KEY[:, np.mod(count, MaxListLen)] = key
                LIST[:, np.mod(count, MaxListLen)] = kernel
            else:
                # If we found it, use it!
                kernel = LIST[:, idx-1]

            # Apply kernel weights to coil data for interpolation
            res[x, y] = np.sum(kernel*tmp.flatten())

    return res

def dat2AtA(data, kernel_size):
    '''Computes the calibration matrix from calibration data.
    '''

    tmp = im2row(data, kernel_size)
    tsx, tsy, tsz = tmp.shape[:]
    A = np.reshape(tmp, (tsx, tsy*tsz), order='F')
    return np.dot(A.T.conj(), A)

def im2row(im, win_shape):
    '''res = im2row(im, winSize)'''
    sx, sy, sz = im.shape[:]
    wx, wy = win_shape[:]
    sh = (sx-wx+1)*(sy-wy+1)
    res = np.zeros((sh, wx*wy, sz), dtype=im.dtype)

    count = 0
    for y in range(wy):
        for x in range(wx):
            #res[:, count, :] = np.reshape(
            #    im[x:sx-wx+x+1, y:sy-wy+y+1, :], (sh, sz), order='F')
            res[:, count, :] = np.reshape(
                im[x:sx-wx+x+1, y:sy-wy+y+1, :], (sh, sz))
            count += 1
    return res

def calibrate(AtA, kernel_size, ncoils, coil, lamda, sampling=None):
    '''Calibrate.
    '''

    kx, ky = kernel_size[:]

    if sampling is None:
        sampling = np.ones((kernel_size, ncoils))

    dummyK = np.zeros((kx, ky, ncoils))
    dummyK[int(kx/2), int(ky/2), coil] = 1

    # To match MATLAB output, use Fortran ordering and make sure
    # indices come out sorted
    idxY = np.where(dummyK)
    idxY_flat = np.sort(
        np.ravel_multi_index(idxY, dummyK.shape, order='F'))
    sampling[idxY] = 0
    idxA = np.where(sampling)
    idxA_flat = np.sort(
        np.ravel_multi_index(idxA, sampling.shape, order='F'))

    Aty = AtA[:, idxY_flat]
    Aty = Aty[idxA_flat]

    AtA0 = AtA[idxA_flat, :]
    AtA0 = AtA0[:, idxA_flat]

    kernel = np.zeros(sampling.size, dtype=AtA0.dtype)

    lamda = np.linalg.norm(AtA0)/AtA0.shape[0]*lamda

    rawkernel = np.linalg.inv(
        AtA0 + np.eye(AtA0.shape[0])*lamda).dot(Aty)
    kernel[idxA_flat] = rawkernel.squeeze()
    kernel = np.reshape(kernel, sampling.shape, order='F')

    return(kernel, rawkernel)

if __name__ == '__main__':
    pass
