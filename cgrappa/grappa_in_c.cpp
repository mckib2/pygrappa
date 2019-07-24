#include "grappa_in_c.h"
#include <complex>
#include <vector>
#include <map>
#include <iostream>
#include <bitset>

/* grappa_in_c: GRAPPA reconstruction.

    Parameters
    ----------
    kspace[kx, ky, ncoil] : complex<double>
        Complex 2D k-space coil data (total 3D).
    kx, ky : unsigned int
        Size of 2D k-space coil data array (and mask).
    calib[cx, cy, ncoil] : complex<double>
        Complex 2D calibration data (total 3D).
    cx, cy : unsigned int
        Size of 2D calibration data array.
    ncoil : unsigned int
        Number of coils (for kspace, mask, and calib arrays).
    ksx, ksy : unsigned int
        Size of kernel: (ksx, ksy).

*/
void grappa_in_c(
    std::complex<double> kspace[],
    int mask[],
    unsigned int kx, unsigned int ky,
    std::complex<double> calib[], unsigned int cx, unsigned int cy,
    unsigned int ncoil, unsigned int ksx, unsigned int ksy)
{

    // This is correct!
    unsigned int ii, jj;
    unsigned int ksx2, ksy2;
    ksx2 = ksx/2 + (ksx % 2);
    ksy2 = ksy/2 + (ksy % 2);
    for (ii = 0; ii < kx; ii++) {
        for (jj = 0; jj < ky; jj++) {
            fprintf(stderr, "%d ", mask[(ii + ksx2)*kx + (jj + ksy2)]);
        }
        fprintf(stderr, "\n");
    }


    // Iterate through all possible overlapping patches of the mask
    // to find unqiue sampling patterns.  Assume zero-padded edges.

    // // For each valid patch, find all sampling patterns
    // std::multimap<unsigned long long int, unsigned int> patterns;
    // unsigned int ii, jj, idx, ksx2, ksy2;
    // int xx, yy;
    // ksx2 = ksx/2;
    // ksy2 = ksy/2;
    // for (ii = 0; ii < kx; ii++) {
    //     for (jj = 0; jj < ky; jj++) {
    //
    //         // Find index of patch center (array has been
    //         // zero-padded, so add half the kernel size to each dim).
    //         // idx = (ii + ksx2)*kykx + (jj + ksy2)*ky;
    //         idx = ((ii + ksx2)*kx + (jj + ksy2))*ky;
    //
    //         // If the center is a hole, it's a valid patch
    //         if (mask[idx] == 0) {
    //             // Look at the entire patch: if it's all zeros, we
    //             // don't care about it.  Treat sum as a binary number
    //             // and flip the bits corresponding to filled locations
    //             // within the patch.  Then sum also tells us what the
    //             // mask is, since the binary representation maps
    //             // directly to sampled and unsampled pixels.
    //
    //             unsigned long long int sum; // this will break for inordinately large kernel sizes!
    //             unsigned int pos;
    //             sum = 0;
    //             pos = 0;
    //             for (xx = -ksx2; xx <= (int)ksx2; xx++) {
    //                 for (yy = -ksy2; yy <= (int)ksy2; yy++) {
    //
    //                     if (mask[((ii + ksx2 + xx)*kx + (jj + ksy2 + yy))*ky] > 0) {
    //                         std::cout << mask[((ii + ksx2 + xx)*kx + (jj + ksy2 + yy))*ky] << std::endl;
    //                         sum += (1 << pos);
    //                     }
    //                     pos++;
    //                 }
    //             }
    //
    //             // If we have samples, then we consider this a valid
    //             // patch and store the index corresponding to a unique
    //             // sampling pattern.
    //             if (sum > 0) {
    //                 patterns.insert(std::pair <unsigned long long int, unsigned int> (sum, idx));
    //             }
    //         }
    //     }
    // }
    //
    // // Get all unique keys corresponding to all unique sampling
    // // patterns
    // std::vector<unsigned long long int> keys;
    // for (std::multimap<unsigned long long int, unsigned int>::const_iterator iter = patterns.begin(); iter != patterns.end(); iter = patterns.upper_bound(iter->first)) {
    //     keys.push_back(iter->first);
    //
    //     // std::cout << iter->first << std::endl;
    //     // std::cout << std::bitset<25>(iter->first) << std::endl;
    //
    //     if (std::bitset<25>(iter->first) == std::bitset<25>("0101010101010101010101010")) {
    //         std::cout << "WE WIN!" << std::endl;
    //     }
    //
    //
    //
    //     // unsigned int pos = 0;
    //     // for (ii = 0; ii < 5; ii++) {
    //     //     for (jj = 0; jj < 5; jj++) {
    //     //         std::cout << std::bitset<25>(iter->first)[pos];
    //     //         pos++;
    //     //     }
    //     //     std::cout << std::endl;
    //     // }
    //
    // }
    //
    // // Now for each sampling pattern (key) we need to train a kernel!
}

/*

00000
00000
00000
00000
00000

0 1 0 1 0
1 0 1 0 1
0 1 0 1 0
1 0 1 0 1
0 1 0 1 0

*/
