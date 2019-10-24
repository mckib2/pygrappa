'''Machine learning kernel functions.'''

import numpy as np
# from sklearn.preprocessing import PolynomialFeatures

def polynomial_kernel(X, cross_term_neighbors=2):
    '''Computes polynomial kernel.

    Parameters
    ----------
    X : array_like of shape (sx, sy, nc)
        Features to map to high dimensional feature-space.

    '''

    _sx, _sy, nc = X.shape[:]

    # Build up a new coil set
    res = []

    # coil layer
    res.append(X*np.sqrt(2))

    # This produces a strong PSF overlay on the recon:
    # # 1s layer
    # res.append(np.ones((_sx, _sy, 1)))

    # squared-coil layer
    res.append(np.abs(X)**2)

    # Cross term layer
    for ii in range(nc):
        for jj in range(nc):
            if ii == jj:
                continue
            if np.abs(ii - jj) > cross_term_neighbors:
                continue
            res.append(
                (X[..., ii]*np.conj(X[..., jj])*np.sqrt(2))[
                    ..., None])

    return np.concatenate(res, axis=-1)
    #
    # # Real/Imag?
    # R = np.reshape(X.real, (-1, nc))
    # I = np.reshape(X.imag, (-1, nc))
    # poly_r = PolynomialFeatures(degree=2, include_bias=False)
    # poly_i = PolynomialFeatures(degree=2, include_bias=False)
    # res_r = poly_r.fit_transform(R)
    # res_i = poly_i.fit_transform(I)
    #
    # # Filter out cross term neighbors
    # print(np.sum(poly_r.powers_, axis=1))
    #
    # # print(poly_r.powers_)
    # return np.reshape(res_r + 1j*res_i, (_sx, _sy, -1))
