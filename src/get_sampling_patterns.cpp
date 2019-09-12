#include "get_sampling_patterns.h"
#include <vector>
#include <map>
#include <climits>
#include <stdio.h>

/* get_sampling_patterns:
    Given binary mask, find unique kernel-sized sampling patterns.

    From the sampling mask, map each unsampled k-space location to a
    unique kernel-sized sampling pattern.

    Parameters
    ----------
    mask[kx, ky] : int
        Sampling mask.
    kx, ky : unsigned int
        Size of 2D k-space coil data array (and mask).
    ksx, ksy : unsigned int
        Size of kernel: (ksx, ksy).

    Returns
    -------
    res : map<unsigned long long int, vecctor<unsigned int> >
        Maps vectors of indices to sampling patterns.

    Notes
    -----
    Each sampling pattern is a ksx by ksy patch.  We use a binary
    number to encode each pixel this patch.  This number is an
    unsigned long long int, so if the patch size ksx*ksy > ULLONG_MAX,
    then we run into issues.  Although unlikely, we check for this
    right at the start, and if it is an issue we use a default kernel
    size of (5, 5).

*/
std::map<unsigned long long int, std::vector<unsigned int> > get_sampling_patterns(
    int mask[],
    unsigned int kx, unsigned int ky,
    unsigned int ksx, unsigned int ksy)
{

    // Check to make sure we're fine (we should be unless the user
    // tries something stupid):
    if ((unsigned long long int)(ksx)*(unsigned long long)(ksy) > ULLONG_MAX) {
        fprintf(stderr, "Something wild is happening with kernel size, choosing (ksx, ksy) = (5, 5).\n");
        ksx = ksy = 5;
    }

    // Initializations
    std::multimap<unsigned long long int, unsigned int> patterns;
    unsigned int ii, jj, idx;
    int ksx2, ksy2, adjx, adjy;
    ksx2 = ksx/2;
    ksy2 = ksy/2;
    adjx = ksx % 2; // same adjustment issue for even/odd kernel sizes
    adjy = ksy % 2; // see grappa.py source for complete discussion.

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
                for (xx = -ksx2; xx < (int)(ksx2 + adjx); xx++) {
                    for (yy = -ksy2; yy < (int)(ksy2 + adjy); yy++) {
                        int wx, wy;
                        wx = (int)ii + xx;
                        wy = (int)jj + yy;
                        // Make sure the index is within bounds:
                        if ((wx >= 0) && (wy >= 0) && ((unsigned int)wx < kx) && ((unsigned int)wy < ky)) {
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
                if (sum) {
                    patterns.insert(std::pair <unsigned long long int, unsigned int> (sum, idx));
                }
            }
        }
    }

    // For each unique sampling pattern we need to train a kernel!
    // Iterate through each unique key (sampling pattern) and store
    // the vector of indices that use that pattern as a map entry.
    std::map<unsigned long long int, std::vector<unsigned int> > res;
    typedef std::multimap<unsigned long long int, unsigned int>::const_iterator iter_t;
    for (iter_t iter = patterns.begin(); iter != patterns.end(); iter = patterns.upper_bound(iter->first)) {
        std::vector<unsigned int> idxs0;
        std::pair<iter_t, iter_t> idxs = patterns.equal_range(iter->first);
        for (iter_t it = idxs.first; it != idxs.second; it++) {
            idxs0.push_back(it->second);
        }
        // res.emplace(iter->first, idxs0); // C++11 only
        res.insert(make_pair(iter->first, idxs0));
    }

    return res;
}
