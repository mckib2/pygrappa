#include <unordered_set>
#include <vector>

#ifndef _GROG_POWERS_TEMPLATE_H
#define _GROG_POWERS_TEMPLATE_H

// Split into two functions instead of just using template function
// because Cython can't resolve template functions!

std::vector<std::unordered_set<double> > _grog_powers_double(
        const double tx[],
        const double ty[],
        const double kx[],
        const double ky[],
        std::vector<std::vector<int> > idx,
        const int precision);

std::vector<std::unordered_set<float> > _grog_powers_float(
        const float tx[],
        const float ty[],
        const float kx[],
        const float ky[],
        std::vector<std::vector<int> > idx,
        const int precision);

#endif
