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

    // Initializations
    std::multimap<unsigned long long int, unsigned int> patterns;
    unsigned int ii, jj, idx;
    int ksx2, ksy2;
    ksx2 = ksx/2;
    ksy2 = ksy/2;

    // Iterate through all possible overlapping patches of the mask
    // to find unqiue sampling patterns.  Assume zero-padded edges.
    for (ii = 0; ii < kx; ii++) {
        for (jj = 0; jj < ky; jj++) {

            // Find index of center patch
            idx = ii*kx + jj;

            // If the center is a hole, it might be valid patch!
            if (mask[idx] == 0) {
                // Look at the entire patch: if it's all zeros, we
                // don't care about it.  Treat sum as a binary number
                // and flip the bits corresponding to filled locations
                // within the patch.  Then sum also tells us what the
                // mask is, since the binary representation maps
                // directly to sampled and unsampled pixels.
                // Because sum is an unsigned long long, this will
                // break for inordinately large kernel sizes!
                unsigned long long int sum;
                unsigned int pos;
                int xx, yy;
                sum = 0;
                pos = 0;
                for (xx = -ksx2; xx <= (int)ksx2; xx++) {
                    for (yy = -ksy2; yy <= (int)ksy2; yy++) {
                        int wx, wy;
                        wx = (int)ii + xx;
                        wy = (int)jj + yy;
                        // Make sure the index is within bounds:
                        if ((wx >= 0) && (wy >= 0) && (wx < (int)kx) && (wy < (int)ky)) {
                            if (mask[wx*kx + wy]) {
                                sum += (1 << pos);
                            }
                        }
                        pos++;
                    }
                }

                // If we have samples, then we consider this a valid
                // patch and store the index corresponding to a unique
                // sampling pattern.
                if (sum > 0) {
                    patterns.insert(std::pair <unsigned long long int, unsigned int> (sum, idx));
                }
            }
        }
    }

    // For each unique sampling pattern we need to train a kernel!
    std::multimap<unsigned long long int, unsigned int>::const_iterator iter;
    for (iter = patterns.begin(); iter != patterns.end(); iter = patterns.upper_bound(iter->first)) {
        // std::cout << std::bitset<25>(iter->first) << std::endl;

        std::bitset<25> d(iter->first);

        for (ii = 0; ii < ksx; ii++) {
            for (jj = 0; jj < ksy; jj++) {
                std::cout << d[ii*ksx + jj];
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

}
