#include <complex>

typedef std::complex<float> complex64;

enum GRAPPA_STATUS {
  SUCCESS = 0,
  FAIL
};

template<class T>
GRAPPA_STATUS _cgrappa(const std::size_t ndim,
		       const std::size_t* kspace_dims,
		       const std::size_t* calib_dims,
		       const T* kspace,
		       const T* calib,
		       const std::size_t* kernel_size,
		       T* recon);

extern "C"
GRAPPA_STATUS _cgrappa_complex64(const std::size_t ndim,
				const std::size_t* kspace_dims,
				const std::size_t* calib_dims,
				const complex64* kspace,
				const complex64* calib,
				const std::size_t* kernel_size,
				complex64* recon);
