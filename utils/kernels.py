'''Machine learning kernel functions.'''

import numpy as np
from sklearn.preprocessing import PolynomialFeatures

def polynomial_kernel(X, degree, gamma, coef0):
    '''Computes polynomial kernel.

    Parameters
    ----------
    X : array_like of shape (sx, sy, nc)
        Features to map to high dimensional feature-space.
    degree : int
        Degree of polynomial.
    gamma : float, optional
        Coefficient of inner-product.
    coef0 : float, optional
        Bias term.
    '''

    sx, sy, nc = X.shape[:]

    # Get the mask
    mask = np.abs(X[..., 0]) > 0

    # We need the coil images
    ax = (0, 1)
    ims = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(
        X, axes=ax), axes=ax), axes=ax)

    # Build up a new coil set
    res = []

    # coil layer
    res.append(ims*np.sqrt(2))

    # 1s layer
    res.append(np.ones((sx, sy, 1)))

    # squared-coil layer
    # res.append(ims**2)

    for ii in range(nc):
        for jj in range(nc):
            if ii == jj:
                continue
            if np.abs(ii - jj) > 1:
                continue
            res.append(
                (ims[..., ii]*ims[..., jj]*np.sqrt(2))[..., None])

    res = np.concatenate(res, axis=-1)
    return np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(
        res, axes=ax), axes=ax), axes=ax)#*mask[..., None]

    # # Mag and phase?  How to do phase?
    # M = np.reshape(np.abs(X), (-1, nc))
    # poly = PolynomialFeatures(degree)
    # res = poly.fit_transform(M)
    # return np.reshape(res, (sx, sy, -1))

    # # Real/Imag?
    # R = np.reshape(X.real, (-1, nc))
    # I = np.reshape(X.imag, (-1, nc))
    # poly_r = PolynomialFeatures(degree)
    # poly_i = PolynomialFeatures(degree)
    # res_r = poly_r.fit_transform(R)
    # res_i = poly_i.fit_transform(I)
    # return np.reshape(res_r + 1j*res_i, (sx, sy, -1))
