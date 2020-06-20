
#define DEBUG

#include <algorithm> // for std::transform
#include <numeric> // for std::accumulate
#include <functional> // for std::multiplies
#include <iostream> // for std::cout
#include <memory> // for std::unique_ptr, std::make_unique
#include <unordered_map> // for std::unordered_map
#include <vector> // for std::vector

#include "_cgrappa.hpp"

#ifdef DEBUG

void _print_unordered_map(const std::unordered_map<std::vector<bool>, std::vector<std::size_t> >& m) {
  for (auto it = m.cbegin(); it != m.cend(); ++it) {
    for (std::size_t ii = 0; ii < it->first.size(); ++ii) {
      std::cout << it->first[ii];
    }
    std::cout << " : ";
    for (auto const & el : it->second) {
      std::cout << el << " ";
    }
    std::cout << std::endl;
  }
}

template<class T>
void _print_vector(const std::vector<T>& vec) {
  // need to do raw loop for std::vector<bool>
  for (std::size_t ii = 0; ii < vec.size(); ++ii) {
    std::cout << vec[ii] << " ";
  }
  std::cout << std::endl;
}

template<class T>
void _print_array1d(const std::size_t dim, const T* arr) {
  for (std::size_t ii = 0; ii < dim; ++ii) {
    std::cout << arr[ii] << " ";
  }
  std::cout << std::endl;
}

template<class T>
void _print_array2d(const std::size_t* dims, const T* arr) {
  for (std::size_t ii = 0; ii < dims[0]; ++ii) {
    for (std::size_t jj = 0; jj < dims[1]; ++jj) {
      std::cout << arr[ii + jj*dims[0]] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}
#else
#define _print_unordered_map(m)
#define _print_vector(vec)
#define _print_array1d(dim, arr)
#define _print_array2d(dims, arr)
#endif

template<class T>
void extract_patch_2d(const std::size_t idx, // center index of patch w.r.t. src
		      const std::size_t coil_idx, // desired coil
		      const std::size_t* dims, // dimensions of src (assumes 2+1d -- coil dim at end)
		      const std::size_t* patch_shape, // dimensions of patch
		      const std::size_t* patch_shape2, // patch_shape/2 -- integer division
		      const std::size_t patch_size, // number of samples in patch
		      const T* src,
		      T* out) {
  // We are extracting hypercubes centered at idx from src;
  // convert flat index to (x, y)
  std::size_t x = idx % dims[0];
  std::size_t y = idx/dims[0]; // integer division
  std::size_t coil_offset = dims[0]*dims[1]*coil_idx;

#ifdef DEBUG
  std::cout << "(x, y): (" << x << ", " << y << ")" << std::endl;
#endif

  std::size_t x_start = 0;
  std::size_t x_end = patch_shape[0];
  std::size_t y_start = 0;
  std::size_t y_end = patch_shape[1];

  // Any idx that needs left padding will satisfy:
  //     x - patch_shape2[0] < 0
  int margin = x - patch_shape2[0];
  if (margin < 0) {
    x_start = -1*margin;
    for (std::size_t yy = 0; yy < patch_shape[1]; ++yy) {
      for (std::size_t xx = 0; xx < x_start; ++xx) {
	out[xx + yy*patch_shape[0]] = 0;
      }
    }
  }

  // Any idx that needs right padding will satisfy:
  //     x + patch_shape2[0] >= dims[0]
  margin = x + patch_shape2[0];
  if (margin >= (int)dims[0]) {
    x_end -= margin - dims[0];
    for (std::size_t yy = 0; yy < patch_shape[1]; ++yy) {
      for (std::size_t xx = x_end; xx < patch_shape[0]; ++xx) {
	out[xx + yy*patch_shape[0]] = 0;
      }
    }
  }

  // Any idx that needs top padding will satisfy:
  //     y - patch_shape2[1] < 0
  margin = y - patch_shape2[1];
  if (margin < 0) {
    y_start = -1*margin;
    for (std::size_t yy = 0; yy < y_start; ++yy) {
      for (std::size_t xx = x_start; xx < x_end; ++xx) {
	out[xx + yy*patch_shape[0]] = 0;
      }
    }
  }

  // Any idx that needs bottom padding will satisfy:
  //     y + patch_shape2[1] >= dims[1]
  margin = y + patch_shape2[1];
  if (margin >= (int)dims[1]) {
    y_end -= margin - dims[1];
    for (std::size_t yy = y_end; yy < patch_shape[1]; ++yy) {
      for (std::size_t xx = x_start; xx < x_start; ++xx) {
	out[xx + yy*patch_shape[0]] = 0;
      }
    }
  }

  // Get contents of patch
  std::size_t access;
  for (std::size_t yy = y_start; yy < y_end; ++yy) {
    for (std::size_t xx = x_start; xx < x_end; ++xx) {
      access = x - patch_shape2[0] + xx + (y - patch_shape2[1] + yy)*dims[0] + coil_offset;
      // std::cout << "access: " << access << std::endl;
      out[xx + yy*patch_shape[0]] = src[access];
    }
  }

  return;
}

template<class T>
GRAPPA_STATUS _cgrappa(const std::size_t ndim,
		       const std::size_t* kspace_dims, // coil axis is last
		       const std::size_t* calib_dims, // coil axis is last
		       const T* kspace, // row-major
		       const T* calib, // row-major
		       const std::size_t* kernel_shape,
		       T* recon) {

  // Get some useful calculations out of the way
  std::size_t ncoils = kspace_dims[ndim-1];
  std::size_t kspace_size = std::accumulate(
    kspace_dims,
    kspace_dims + ndim,
    1, std::multiplies<std::size_t>());
  std::size_t kernel_size = std::accumulate(
    kernel_shape,
    kernel_shape + ndim - 1,
    1, std::multiplies<std::size_t>());
  auto kernel_shape2 = std::make_unique<std::size_t[]>(ndim-1);
  std::transform(
    kernel_shape,
    kernel_shape + ndim - 1,
    kernel_shape2.get(),
    [](std::size_t x0) { return x0/2; });
  auto adjs = std::make_unique<std::size_t[]>(ndim-1);
  std::transform(
    kernel_shape,
    kernel_shape + ndim - 1,
    adjs.get(),
    [](std::size_t x0) { return x0 % 2; });

  void (*extract_patch)(const std::size_t,
			const std::size_t,
			const std::size_t*,
			const std::size_t*,
			const std::size_t*,
			const std::size_t,
			const T*, T*);
  if (ndim-1 == 2) {
    extract_patch = &extract_patch_2d;
  }
  else {
    std::cerr << "NotImplementedError" << std::endl;
    return GRAPPA_STATUS::FAIL;
  }

  // Find the unique sampling patterns and save where they are
  // (Note: using std::vector<bool> as a dynamic bitset)
  auto P = std::unordered_map<std::vector<bool>, std::vector<std::size_t> >();
  auto pattern = std::vector<bool>(kernel_size);
  auto patch = std::make_unique<T[]>(kernel_size);
  std::size_t coil_idx = 0; // any coil will do for determining sampling patterns
  std::size_t ctr = kernel_shape2[0] + kernel_shape2[1]*kernel_shape[0];
  for (std::size_t idx = 0; idx < kspace_size/ncoils; ++idx) {
    // note: row-major ordering
    extract_patch(
      idx,
      coil_idx,
      kspace_dims,
      kernel_shape,
      kernel_shape2.get(),
      kernel_size,
      kspace,
      patch.get());
    _print_array2d(kernel_shape, patch.get()); // DEBUG macro

    // We only care if this patch has a hole in the center
    if (std::abs(patch[ctr]) > 0) {
      continue;
    }

    // Convert patch to mask
    // (Note: can't use std::transform because of std::vector<bool>)
    for (std::size_t ii = 0; ii < kernel_size; ++ii) {
      pattern[ii] = std::abs(patch[ii]) > 0;
    }
    // _print_vector(pattern); // DEBUG macro

    // store the sampling pattern
    P[pattern].emplace_back(idx);
  }
  // Throw away any all zero sampling patterns
  auto zeros = std::vector<bool>(pattern.size(), 0);
  P.erase(zeros);
  _print_unordered_map(P); // DEBUG macro

  // TODO(mckib2): use same nnz criteria for P as in mdgrappa

  // TODO(mckib2): Get all overlapping patches from calibration data
  auto A = std::make_unique<T[]>(kernel_size*ncoils); // could be quite large

  // TODO(mckib2): Train and apply weights for each pattern;
  //     it->first : sampling pattern
  //     it->second : indices of holes in kspace
  for (auto it = P.cbegin(); it != P.cend(); ++it) {

    // TODO(mckib2): use LAPACK functions from numpy (hopefully)

  }

  return GRAPPA_STATUS::SUCCESS;
}

// Resolve the templates to make available for Cython:
extern "C" {
  GRAPPA_STATUS _cgrappa_complex64(const std::size_t ndim,
				   const std::size_t* kspace_dims,
				   const std::size_t* calib_dims,
				   const complex64* kspace,
				   const complex64* calib,
				   const std::size_t* kernel_size,
				   complex64* recon) {
    return _cgrappa(ndim, kspace_dims, calib_dims, kspace, calib, kernel_size, recon);
  }
}
