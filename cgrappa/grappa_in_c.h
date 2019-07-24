#include <complex>

#ifndef GRAPPA_IN_C_H
#define GRAPPA_IN_C_H

void grappa_in_c(
    std::complex<double> kspace[],
    int mask[],
    unsigned int kx,
    unsigned int ky,
    std::complex<double> calib[],
    unsigned int cx,
    unsigned int cy,
    unsigned int ncoil,
    unsigned int ksx,
    unsigned int ksy);

#endif
