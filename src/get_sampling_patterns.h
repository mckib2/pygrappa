#include <complex>
#include <vector>
#include <map>

#ifndef GET_SAMPLING_PATTERNS_H
#define GET_SAMPLING_PATTERNS_H

std::map<unsigned long long int, std::vector<unsigned int> > get_sampling_patterns(
    int mask[],
    unsigned int kx,
    unsigned int ky,
    unsigned int ksx,
    unsigned int ksy);

#endif
