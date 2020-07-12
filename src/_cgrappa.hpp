#include <complex>

typedef std::complex<float> complex64;
typedef std::complex<double> complex128;

enum GRAPPA_STATUS { SUCCESS = 0, FAIL };

/// Useful constants
static const float fone = 1;
static const float fzero = 0;
static const double done = 1;
static const double dzero = 0;

extern "C" {
// LAPACK: solves overdetermined or underdetermined systems AX=B (for
// std::complex<float>)
void cgels_(
    char *TRANS, // 'N' (no transpose) or 'C' (hermitian transpose)
    int *M,      // num rows of A
    int *N,      // num cols of A
    int *NRHS,   // num cols of B and X
    float *A,    // M x N matrix; overwritten by QR or LQ factorization
    int *LDA,    // max(1, M)
    float *B, // LDB x NRHS matrix; first M (or N) rows overwritten by solution
    int *LDB, // max(1, M, N)
    float *WORK, // max(1, LWORK) std::complex<float> vector
    int *LWORK, // for optimal perf: max( 1, MN + max( MN, NRHS )*NB ), where MN
                // = min(M,N)
    int *INFO); // 0 on success
}

template <class T>
GRAPPA_STATUS _cgrappa(const std::size_t ndim, const std::size_t *kspace_dims,
                       const std::size_t *calib_dims, const T *kspace,
                       const T *calib, const std::size_t *kernel_size,
                       T *recon);

extern "C" GRAPPA_STATUS
_cgrappa_complex64(const std::size_t ndim, const std::size_t *kspace_dims,
                   const std::size_t *calib_dims, const complex64 *kspace,
                   const complex64 *calib, const std::size_t *kernel_size,
                   complex64 *recon);

extern "C" GRAPPA_STATUS
_cgrappa_complex128(const std::size_t ndim, const std::size_t *kspace_dims,
                    const std::size_t *calib_dims, const complex128 *kspace,
                    const complex128 *calib, const std::size_t *kernel_size,
                    complex128 *recon);
