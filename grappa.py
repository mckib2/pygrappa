'''Reference GRAPPA implementation ported to python.'''

import numpy as np
from skimage.util import pad
from tqdm import trange

def grappa(
        kspace, calib, kernel_size=(5, 5), coil_axis=-1, lamda=0.01,
        disp=False):
    '''GeneRalized Autocalibrating Partially Parallel Acquisitions.

    Parameters
    ----------
    kspace : array_like
        2D multi-coil k-space data to reconstruct from.  Make sure
        that the missing entries have exact zeros in them.
    calib : array_like
        Calibration data (fully sampled k-space)
    kernel_size : tuple, optional
        size of the 2D GRAPPA kernel (kx, ky)
    coil_axis : int, optional
        Dimension holding coil data.  The other two dimensions should
        be (kx, ky).
    lamda : float, optional
        Tikhonov regularization for the kernel calibration.
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
    >>> res = GRAPPA(DATA, kCalib, kSize, -1, 0.01, False)

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

    # Put the coil dimension at the end
    kspace = np.moveaxis(kspace, coil_axis, -1)

    # Get displays up and running if we need them
    if disp:
        import matplotlib.pyplot as plt

    # get number of coils
    ncoils = kspace.shape[-1]
    res = np.zeros(kspace.shape, dtype=kspace.dtype)

    # build coil calibrating matrix
    AtA = dat2AtA(calib, kernel_size)[0]

    # reconstruct single coil images
    for nn in trange(ncoils, leave=False):
        res[..., nn] = ARC(
            kspace, AtA, kernel_size, nn, lamda)
        if disp:
            plt.imshow(np.abs(ifft2c(res[..., nn])))
            plt.show()

    # Move the coil dimension back where the user had it
    res = np.moveaxis(res, -1, coil_axis)
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
    '''[AtA, A, kernel] = dat2AtA(data, kSize)

    Function computes the calibration matrix from calibration data.
    (c) Michael Lustig 2013
    '''

    nc = data.shape[-1]
    kx, ky = kernel_size[:]

    tmp = im2row(data, kernel_size)
    tsx, tsy, tsz = tmp.shape[:]
    A = np.reshape(tmp, (tsx, tsy*tsz), order='F')

    AtA = np.dot(A.T.conj(), A)

    kernel = AtA.copy()
    kernel = np.reshape(
        kernel, (kx, ky, nc, kernel.shape[1]), order='F')

    return(AtA, A, kernel)

def im2row(im, win_shape):
    '''res = im2row(im, winSize)'''
    sx, sy, sz = im.shape[:]
    wx, wy = win_shape[:]
    sh = (sx-wx+1)*(sy-wy+1)
    res = np.zeros((sh, wx*wy, sz), dtype=im.dtype)

    count = 0
    for y in range(wy):
        for x in range(wx):
            res[:, count, :] = np.reshape(
                im[x:sx-wx+x+1, y:sy-wy+y+1, :], (sh, sz), order='F')
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
    res = grappa(
        DATA, kCalib, kSize, coil_axis=-1, lamda=0.01, disp=False)

    # Take a look
    import matplotlib.pyplot as plt
    res = np.abs(ifft2c(res))
    res0 = np.zeros((2*N, 2*N))
    kk = 0
    for idx in np.ndindex((2, 2)):
        ii, jj = idx[:]
        res0[ii*N:(ii+1)*N, jj*N:(jj+1)*N] = res[..., kk]
        kk += 1
    plt.imshow(res0, cmap='gray')
    plt.show()
