#include <vector>
#include <unordered_set>
#include <cmath>

template <typename T>
std::vector<std::unordered_set<T> > _grog_powers_template(
        const T tx[],
        const T ty[],
        const T kx[],
        const T ky[],
        std::vector<std::vector<int> > idx,
        const int precision
    ) {
    /* Find unique fractional matrix powers.

    Parameters
    ----------
    tx : const T[]
        Target x coordinates, 1d array.  Same size as idx.size().
    ty : const T[]
        Target y coordinates, 1d array.  Same size as idx.size().
    kx : const T[]
        Source x coordinates, 1d array.
    ky : const T[]
        Source x coordinates, 1d array.
    idx : vector<vector<int> >
        Indices of sources close to targets.
    precision : const int
        How many decimal points to round fractional matrix powers to.

    Returns
    -------
    retVal : vector<unordered_set<T> > with exactly 2 entries
        The fractional matrix powers for Gx (retVal[0]) and Gy
        (retVal[1]).

    Notes
    -----
    All calculations are done in the template function (this one).
    Cython cannot resolve template function types, so wrapper
    functions are provided for both float and double coordinate
    arrays.
    */

    std::unordered_set<T> dx, dy;
    T pval = std::pow(10.0, precision);
    int ii = 0;
    for (auto idx_list : idx) {
        for (auto idx0 : idx_list) {
            dx.insert(std::round((tx[ii] - kx[idx0])*pval)/pval);
            dy.insert(std::round((ty[ii] - ky[idx0])*pval)/pval);
        }
        ii += 1;
    }
    auto retVal = std::vector<std::unordered_set<T> > { dx, dy };
    return retVal;
}

// We also need simple wrappers for double and float versions to
// deal with Cython template limitations:
std::vector<std::unordered_set<double> > _grog_powers_double(
        const double tx[],
        const double ty[],
        const double kx[],
        const double ky[],
        std::vector<std::vector<int> > idx,
        const int precision
    ) {
    return _grog_powers_template(tx, ty, kx, ky, idx, precision);
}

std::vector<std::unordered_set<float> > _grog_powers_float(
        const float tx[],
        const float ty[],
        const float kx[],
        const float ky[],
        std::vector<std::vector<int> > idx,
        const int precision
    ) {
    return _grog_powers_template(tx, ty, kx, ky, idx, precision);
}
