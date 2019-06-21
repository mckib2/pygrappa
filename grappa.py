'''Reference GRAPPA implementation ported to python.'''

import numpy as np
from skimage.util import pad
from tqdm import trange

def grappa(kData, kCalib, kSize=(5, 5), lamda=0.01, disp=False):
    '''GRAPPA(kData,kCalib,kSize,lambda [, disp)

    Parameters
    ----------
    kData : array_like, [Size x, Size y, num coils]
        2D multi-coil k-space data to reconstruct from.  Make sure
        that the missing entries have exact zeros in them.
    kCalib : array_like
        Calibration data (fully sampled k-space)
    kSize : tuple, optional
        size of the 2D GRAPPA kernel (kx, ky)
    lamda : float, optional
        Tykhonov regularization for the kernel calibration.
    disp : bool, optional
        Display images as they are reconstructed

    Returns
    -------
    res : array_like
        k-space data where missing entries have been filled in.

    Examples
    --------
    Generate fake Sensitivity maps:
    >>> N = 128
    >>> ncoils = 4
    >>> xx = np.linspace(0, 1, N)
    >>> x, y = np.meshgrid(xx, xx)
    >>> sMaps = np.zeros((N, N, ncoils))
    >>> sMaps[..., 0] = x**2
    >>> sMaps[..., 1] = 1 - x**2
    >>> sMaps[..., 2] = y**2
    >>> sMaps[..., 3] = 1 - y**2

    Generate 4 coil phantom:
    >>> ph = np.load('phantom.npy')
    >>> imgs = ph[..., None]*sMaps
    >>> imgs = imgs.astype('complex')
    >>> DATA = fft2c(imgs)

    Crop 20x20 window from the center of k-space for calibration:
    >>> pd = 10
    >>> ctr = int(N/2)
    >>> kCalib = DATA[ctr-pd:ctr+pd, ctr-pd:ctr+pd, :].copy()

    Calibration kernel size:
    >>> kSize = (5, 5)

    Undersample by a factor of 2 in both x and y:
    >>> DATA[::2, 1::2, :] = 0
    >>> DATA[1::2, ::2, :] = 0

    Reconstruct:
    ... res = GRAPPA(DATA, kCalib, kSize, 0.01, False)

    Notes
    -----
    Based on implementation at [1]_.

    This is a GRAPPA reconstruction algorithm that supports
    arbitrary Cartesian sampling. However, the implementation
    is highly inefficient in Matlab because it uses for loops.
    This implementation is very similar to the GE ARC implementation.
    The reconstruction looks at a neighborhood of a point and
    does a calibration according to the neighborhood to synthesize
    the missing point. This is a k-space varying interpolation.
    A sampling configuration is stored in a list, and retrieved
    when needed to accelerate the reconstruction (a bit)

    References
    ----------
    .. [1] https://people.eecs.berkeley.edu/~mlustig/Software.html
    '''

    # Get displays up and running if we need them
    if disp:
        import matplotlib.pyplot as plt

    # get sizes
    _fe, _pe, coils = kData.shape[:]

    res = np.zeros(kData.shape, dtype=kData.dtype)
    AtA, _, _ = dat2AtA(kCalib, kSize) # build coil calibrating matrix

    # reconstruct single coil images
    for nn in trange(coils, leave=False):
        res[..., nn] = ARC(
            kData, AtA, kSize, nn, lamda)
        if disp:
            plt.imshow(np.abs(ifft2c(res[..., nn])))
            plt.show()

    return res

def ARC(kData, AtA, kSize, c, lamda):
    '''ARC.'''
    sx, sy, nCoil = kData.shape[:]

    px = int((kSize[0])/2)
    py = int((kSize[1])/2)
    kData = pad(kData, ((px, px), (py, py), (0, 0)), mode='constant') #pylint: disable=E1102

    dummyK = np.zeros((kSize[0], kSize[1], nCoil))
    dummyK[int((kSize[0])/2), int((kSize[1])/2), c] = 1
    idxy = np.where(dummyK)
    res = np.zeros((sx, sy), dtype=kData.dtype)

    MaxListLen = 100
    LIST = np.zeros(
        (kSize[0]*kSize[1]*nCoil, MaxListLen), dtype=kData.dtype)
    KEY = np.zeros((kSize[0]*kSize[1]*nCoil, MaxListLen))
    count = 0

    for xy in np.ndindex((sx, sy)):
        x, y = xy[:]

        tmp = kData[x:x+kSize[0], y:y+kSize[1], :]
        pat = np.abs(tmp) > 0
        if pat[idxy]:
            # If we aquired this k-space sample, use it!
            res[x, y] = tmp[idxy].squeeze()
        else:
            # If we didn't aquire it, let's either look up the
            # kernel or compute a new one
            key = pat.flatten('F')

            # If we have a matching kernel, the key will exist
            # in our array of KEYs
            idx = 0
            for nn in range(1, KEY.shape[1]+1):
                if np.sum(key == KEY[:, nn-1]) == key.size:
                    idx = nn
                    break

            # If we didn't find a matching kernel, compute one.
            # We'll only hold MaxListLen kernels in the lookup
            # at one time
            if idx == 0:
                count += 1
                kernel, _ = calibrate(
                    AtA, kSize, nCoil, c, lamda, pat)
                KEY[:, np.mod(
                    count, MaxListLen)] = key.flatten('F')
                LIST[:, np.mod(
                    count, MaxListLen)] = kernel.flatten('F')
            else:
                # If we found it, use it!
                kernel = LIST[:, idx-1]

            # Must be flattened in Fortran order because kernel
            # was stored in LIST like this
            res[x, y] = np.sum(
                kernel.flatten('F')*tmp.flatten('F'))

    return res

def dat2AtA(data, kSize):
    '''[AtA,A,kernel] = dat2AtA(data, kSize)

    Function computes the calibration matrix from calibration data.
    (c) Michael Lustig 2013
    '''

    _sx, _sy, nc = data.shape[:]

    tmp = im2row(data, kSize)
    tsx, tsy, tsz = tmp.shape[:]
    A = np.reshape(tmp, (tsx, tsy*tsz), order='F')

    AtA = np.dot(A.T.conj(), A)

    kernel = AtA.copy()
    kernel = np.reshape(
        kernel, (kSize[0], kSize[1], nc, kernel.shape[1]), order='F')

    return(AtA, A, kernel)

def im2row(im, winSize):
    '''res = im2row(im, winSize)'''
    sx, sy, sz = im.shape[:]

    res = np.zeros(
        ((sx-winSize[0]+1)*(sy-winSize[1]+1), np.prod(winSize), sz),
        dtype=im.dtype)
    count = 0
    for y in range(winSize[1]):
        for x in range(winSize[0]):
            res[:, count, :] = np.reshape(
                im[x:sx-winSize[0]+x+1, y:sy-winSize[1]+y+1, :],
                ((sx-winSize[0]+1)*(sy-winSize[1]+1), sz), order='F')
            count += 1
    return res

def fft2c(x):
    '''Forward 2D Fourier transform.'''
    S = x.shape
    fctr = S[0]*S[1]

    x = np.reshape(x, (S[0], S[1], int(np.prod(S[2:]))), 'F')

    res = np.zeros(x.shape, dtype=x.dtype)
    for n in range(x.shape[2]):
        res[:, :, n] = 1/np.sqrt(fctr)*np.fft.fftshift(np.fft.fft2(
            np.fft.ifftshift(x[:, :, n])))

    res = np.reshape(res, S, 'F')
    return res

def ifft2c(x):
    '''Inverse 2D Fourier transform.'''
    S = x.shape
    fctr = S[0]*S[1]

    x = np.reshape(x, (S[0], S[1], int(np.prod(S[2:]))), 'F')

    res = np.zeros(x.shape, dtype=x.dtype)
    for n in range(x.shape[2]):
        res[:, :, n] = np.sqrt(fctr)*np.fft.fftshift(np.fft.ifft2(
            np.fft.ifftshift(x[:, :, n])))

    res = np.reshape(res, S, 'F')
    return res

def calibrate(AtA, kSize, nCoil, coil, lamda, sampling=None):
    '''Calibrate.'''

    if sampling is None:
        sampling = np.ones((kSize, nCoil))

    dummyK = np.zeros((kSize[0], kSize[1], nCoil))
    dummyK[int((kSize[0])/2), int((kSize[1])/2), coil] = 1

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

    AtA = AtA[idxA_flat, :].copy()
    AtA = AtA[:, idxA_flat]

    kernel = np.zeros(sampling.size, dtype=AtA.dtype)

    lamda = np.linalg.norm(AtA)/AtA.shape[0]*lamda

    rawkernel = np.linalg.inv(
        AtA + np.eye(AtA.shape[0])*lamda).dot(Aty)
    kernel[idxA_flat] = rawkernel.squeeze()
    kernel = np.reshape(kernel, sampling.shape, order='F')

    return(kernel, rawkernel)

if __name__ == '__main__':

    # Generate fake Sensitivity maps
    N = 128
    ncoils = 4
    xx = np.linspace(0, 1, N)
    x, y = np.meshgrid(xx, xx)
    sMaps = np.zeros((N, N, ncoils))
    sMaps[..., 0] = x**2
    sMaps[..., 1] = 1 - x**2
    sMaps[..., 2] = y**2
    sMaps[..., 3] = 1 - y**2

    # generate 4 coil phantom
    # from scipy.io import loadmat
    # ph = loadmat('phantom.mat')['tmp']
    # np.save('phantom.npy', ph)
    ph = np.load('phantom.npy')
    imgs = ph[..., None]*sMaps
    imgs = imgs.astype('complex')
    DATA = fft2c(imgs)

    # crop 20x20 window from the center of k-space for calibration
    pd = 10
    ctr = int(N/2)
    kCalib = DATA[ctr-pd:ctr+pd, ctr-pd:ctr+pd, :].copy()

    # calibrate a kernel
    kSize = (5, 5)

    # undersample by a factor of 2 in both x and y
    DATA[::2, 1::2, :] = 0
    DATA[1::2, ::2, :] = 0

    # reconstruct:
    res = grappa(DATA, kCalib, kSize, 0.01, False)

    # Take a look
    from mr_utils import view
    view(res, fft=True)
