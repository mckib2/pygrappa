'''Least Squares solver for generator observations.

Notes
-----
Creating all overlapping patches of ACS can be memory intensive.  To
solve this, we'll try to use a solver that is tailored towards a
memory-lite worldview.  We still want good results, but we want them
quick and without crashing our machines.
'''

from itertools import zip_longest

import numpy as np
from tqdm import tqdm

def _grouper(iterable, n, fillvalue=None):
    '''Collect data into fixed-length chunks or blocks.'''
    args = [iter(iterable)]*n
    return zip_longest(*args, fillvalue=fillvalue)

def isgd(
        acs, mask, kernel_size, thrash=5, eta0=0.1, lamda=0.01,
        batch_size=50, coil_axis=-1):
    '''Implicit stochastic gradient descent for generators.

    Parameters
    ----------
    acs : array_like
        Holds the autocalibration kspace data.
    mask : array_like
        Boolean sampling mask with size kernel_size.
    kernel_size : tuple
        The size of the GRAPPA kernels.
    thrash : int, optional
        Number of times to cycle through all patches during training.
    eta0 : float, optional
        Starting learning rate.
    lamda : float, optional
        Tikhonov regularization factor.
    batch_size : int, optional
        Size of minibatch sizes.  Set to 1 for standard stochastic
        gradient descent.
    coil_axis : int, optional
        Dimension holding the coil data.

    Returns
    -------
    W : array_like
        Weights that turn sources into targets, i.e., S @ W = T.

    Notes
    -----
    Uses the implicit stochastic GD algorithm to train GRAPPA
    weights [1]_.  The patches are provided by a generator function
    for effeicient memory performance.

    Also implements minibatch and stepsize contraction as well as
    Tikhonov (L2) regularization.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Stochastic_gradient_descent
    '''

    # Coils to the back
    acs = np.moveaxis(acs, coil_axis, -1)

    # Scale the ACS
    acs = acs/np.linalg.norm(acs)

    # We will assume some things about the arguments...
    assert mask.shape == kernel_size
    assert mask.dtype == bool

    # adj = [(k0 % 2) for k0 in kernel_size]
    inner_sh = tuple(
        [n0 - k0 for n0, k0 in zip(acs.shape, kernel_size)])
    totinner = np.prod(inner_sh)

    def _next_patch():
        '''Extract the next multidimensional patch in random order.'''
        # for idx in np.ndindex(inner_sh):
        for flat_idx in np.random.choice(
                totinner, size=int(thrash*totinner)):
            idx = np.unravel_index(flat_idx, inner_sh)
            sl = tuple([
                slice(ii, ii+k0)
                for k0, ii in zip(kernel_size, idx)])
            yield acs[sl + (slice(None),)]

    nc = acs.shape[-1]
    W = np.zeros((np.sum(mask.flatten())*nc, nc), dtype=acs.dtype)
    ctr = tuple([k0//2 for k0 in kernel_size])
    eta = eta0
    # err = []
    for it, batch in enumerate(_grouper(_next_patch(), batch_size)):

        # Do the math:
        #     S W = T
        #     S : (batch_size, nx)
        #     W : (nx, nc)
        #     T : (batch_size, nc)

        # Do minibatches
        batch = [p for p in batch if p is not None]
        batch_size = len(batch)
        patches = np.array(batch)
        S = np.reshape(patches[:, mask], (batch_size, W.shape[0]))
        T = np.reshape(
            patches[(slice(None),) + ctr], (batch_size, nc))

        # Solve normal equation
        #     S^H S W = S^H T
        #         A W  = b
        #     S^H S : A : (nx, nx)
        #     S^H T : b : (nx, nc)
        A = S.conj().T @ S
        b = S.conj().T @ T

        # Gradient descent update with Tikhonov regularization:
        #     d/dW 1/2 || A W - b ||_2^2 + lamda/2 || W ||_2^2
        #     = A^H (AW - b) + lamda W
        r = A @ W - b
        dW = A.conj().T @ r
        dW /= np.linalg.norm(dW) # normalize update
        lamda0 = lamda*np.linalg.norm(A)/A.shape[0] # L2 reg factor
        W -= eta*(dW + lamda*W)

        # Contract stepsize
        eta = eta0/(1 + eta0*lamda0*it)

        # # Track error to look at
        # err.append(np.linalg.norm(r))
    # import matplotlib.pyplot as plt
    # plt.plot(err)
    # plt.show()

    # print(W)
    return W

def reconstructor(
        kspace, calib, kernel_size, weight_fun=isgd, coil_axis=-1,
        **kwargs):
    '''Reconstructs kspace using weights from provided function.

    '''

    # Move coils to the back
    kspace = np.moveaxis(kspace, coil_axis, -1)
    calib = np.moveaxis(calib, coil_axis, -1)
    orig_sh = kspace.shape[:]
    holes = np.abs(kspace[..., 0]) == 0

    # Pad the suckers
    k2 = [k0//2 for k0 in kernel_size]
    # adj = [k0 % 2 for k0 in kernel_size]
    pads = tuple([(k0//2, k0//2) for k0 in kernel_size] + [(0, 0)])
    kspace = np.pad(kspace, pads, mode='constant')
    calib = np.pad(calib, pads, mode='constant')

    # Find the holes and fill 'em up
    Ws = dict() # dictionary of mask : weight pairs
    res = np.empty(kspace.shape, dtype=kspace.dtype)
    for idx in tqdm(
            np.argwhere(holes),
            total=np.sum(holes.flatten()),
            leave=False):
        sl = tuple([
            slice(idx0, idx0+k0)
            for idx0, k0 in zip(idx, kernel_size)]) + (slice(None),)
        patch = kspace[sl]
        mask = np.abs(patch[..., 0]) > 0
        key = str(mask)
        if key not in Ws:
            W = weight_fun(
                calib,
                mask,
                kernel_size=kernel_size,
                coil_axis=-1,
                **kwargs)
            Ws[key] = W
        else:
            W = Ws[key]

        ctr = tuple([
            k20+idx0 for k20, idx0 in zip(k2, idx)]) + (slice(None),)
        S = patch[mask].flatten()[None, :]
        res[ctr] = S @ W

    res[np.abs(kspace) > 0] = kspace[np.abs(kspace) > 0]
    return res

if __name__ == '__main__':
    pass
