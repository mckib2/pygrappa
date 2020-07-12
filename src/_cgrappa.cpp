
#define DEBUG

#include <algorithm> // for std::transform
#include <cblas.h>
#include <functional> // for std::multiplies
#include <iostream>   // for std::cout
#include <memory> // for std::unique_ptr -- don't use make_unique as it value initializes!
#include <numeric>       // for std::accumulate
#include <type_traits>   // for std::is_same
#include <unordered_map> // for std::unordered_map
#include <vector>        // for std::vector

#include "_cgrappa.hpp"

/// Convert 2+1d index to flattened index
const std::size_t ind2to1(const std::size_t &x, const std::size_t &y,
                          const std::size_t &dim1, const std::size_t &dim2) {
  return dim2 * (y + x * dim1);
}

/// Convert 1d index to flattened index
const std::size_t ind2to1(const std::size_t &x, const std::size_t &y,
                          const std::size_t &dim1) {
  return y + x * dim1;
}

/// Convert 3d index to flattened index
const std::size_t ind3to1(const std::size_t &x, const std::size_t &y,
                          const std::size_t &z, const std::size_t &dim1,
                          const std::size_t &dim2) {
  return z + dim2 * (y + x * dim1);
}

#ifdef DEBUG

void _print_unordered_map(
    const std::unordered_map<std::vector<bool>, std::vector<std::size_t>> &m) {
  for (auto it = m.cbegin(); it != m.cend(); ++it) {
    for (std::size_t ii = 0; ii < it->first.size(); ++ii) {
      std::cout << it->first[ii];
    }
    std::cout << " : ";
    for (auto const &el : it->second) {
      std::cout << el << " ";
    }
    std::cout << std::endl;
  }
}

template <class T> void _print_vector(const std::vector<T> &vec) {
  // need to do raw loop for std::vector<bool>
  for (std::size_t ii = 0; ii < vec.size(); ++ii) {
    std::cout << vec[ii] << " ";
  }
  std::cout << std::endl;
}

template <class T> void _print_array1d(const std::size_t dim, const T *arr) {
  for (std::size_t ii = 0; ii < dim; ++ii) {
    std::cout << arr[ii] << " ";
  }
  std::cout << std::endl;
}

template <class T> void _print_array2d(const std::size_t *dims, const T *arr) {
  for (std::size_t ii = 0; ii < dims[0]; ++ii) {
    for (std::size_t jj = 0; jj < dims[1]; ++jj) {
      std::cout << arr[ind2to1(ii, jj, dims[1])] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

template <class T> void _print_coil2d(const std::size_t *dims, const T *arr) {
  for (std::size_t ii = 0; ii < dims[0]; ++ii) {
    for (std::size_t jj = 0; jj < dims[1]; ++jj) {
      std::cout << arr[ind2to1(ii, jj, dims[1], dims[2])] << " ";
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

template <class T>
void extract_patch_2d(
    const std::size_t idx,      // center index of patch w.r.t. src
    const std::size_t coil_idx, // desired coil
    const std::size_t
        *dims, // dimensions of src (assumes 2+1d -- coil dim at end)
    const std::size_t *patch_shape,  // dimensions of patch
    const std::size_t *patch_shape2, // patch_shape/2 -- integer division
    const std::size_t patch_size,    // number of samples in patch
    const std::size_t *adjs,         // kernel_shape % 2
    const T *src, T *out) {
  // We are extracting hypercubes centered at idx from src;
  // convert flat index to (x, y)
  std::size_t x = idx % dims[0];
  std::size_t y = idx / dims[0]; // integer division

#ifdef DEBUG
  std::cout << "(x, y, coil): (" << x << ", " << y << ", " << coil_idx << ")"
            << std::endl;
#endif

  std::size_t x_start = 0;
  std::size_t x_end = patch_shape[0];
  std::size_t y_start = 0;
  std::size_t y_end = patch_shape[1];

  // Any idx that needs top padding will satisfy:
  //     x - patch_shape2[0] < 0
  int margin = x - patch_shape2[0];
  if (margin < 0) {
    x_start = -1 * margin;
    for (std::size_t xx = 0; xx < x_start; ++xx) {
      for (std::size_t yy = 0; yy < patch_shape[1]; ++yy) {
        out[ind2to1(xx, yy, patch_shape[1])] = 0;
      }
    }
  }

  // Any idx that needs bottom padding will satisfy:
  //     x + patch_shape2[0] >= dims[0]
  margin = x + patch_shape2[0] + adjs[0];
  if (margin >= (int)dims[0]) {
    x_end -= margin - dims[0];
    for (std::size_t xx = x_end; xx < patch_shape[0]; ++xx) {
      for (std::size_t yy = 0; yy < patch_shape[1]; ++yy) {
        out[ind2to1(xx, yy, patch_shape[1])] = 0;
      }
    }
  }

  // Any idx that needs left padding will satisfy:
  //     y - patch_shape2[1] < 0
  margin = y - patch_shape2[1];
  if (margin < 0) {
    y_start = -1 * margin;
    for (std::size_t xx = x_start; xx < x_end; ++xx) {
      for (std::size_t yy = 0; yy < y_start; ++yy) {
        out[ind2to1(xx, yy, patch_shape[1])] = 0;
      }
    }
  }

  // Any idx that needs right padding will satisfy:
  //     y + patch_shape2[1] >= dims[1]
  margin = y + patch_shape2[1] + adjs[1];
  if (margin >= (int)dims[1]) {
    y_end -= margin - dims[1];
    for (std::size_t xx = x_start; xx < x_end; ++xx) {
      for (std::size_t yy = y_end; yy < patch_shape[1]; ++yy) {
        out[ind2to1(xx, yy, patch_shape[1])] = 0;
      }
    }
  }

  // Get contents of patch
  std::size_t access;
  for (std::size_t xx = x_start; xx < x_end; ++xx) {
    for (std::size_t yy = y_start; yy < y_end; ++yy) {
      access = ind2to1(x - patch_shape2[0] + xx, y - patch_shape2[1] + yy,
                       dims[1], dims[2]);
      // std::cout << "access: " << access << std::endl;
      out[ind2to1(xx, yy, patch_shape[1])] = src[access];
    }
  }

  return;
}

/// Define templated function
template <typename T>
void gemm(const int M, const int N, const int K, T *A, T *B, T *C) {}

/// Multiply two matrices with type std::complex<float>
template <>
void gemm(const int M, const int N, const int K, std::complex<float> *A,
          std::complex<float> *B, std::complex<float> *C) {

  static const float one = 1;
  static const float zero = 0;

  // reinterpret_cast: std::complex uses correct layout to make this work
  cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, &one,
              reinterpret_cast<float *>(A), 1, reinterpret_cast<float *>(B), 1,
              &zero, reinterpret_cast<float *>(C), M);
}

/// Multiply two matrices with type std::complex<double>
template <>
void gemm(const int M, const int N, const int K, std::complex<double> *A,
          std::complex<double> *B, std::complex<double> *C) {

  static const double one = 1;
  static const double zero = 0;

  // reinterpret_cast: std::complex uses correct layout to make this work
  cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, &one,
              reinterpret_cast<double *>(A), 1, reinterpret_cast<double *>(B),
              1, &zero, reinterpret_cast<double *>(C), M);
}

template <class T>
GRAPPA_STATUS _cgrappa(const std::size_t ndim,
                       const std::size_t *kspace_dims, // coil axis is last
                       const std::size_t *calib_dims,  // coil axis is last
                       const T *kspace,                // row-major
                       const T *calib,                 // row-major
                       const std::size_t *kernel_shape,
                       T *recon) { // row-major

  // Make sure we only take complex types
  static_assert(std::is_same<T, complex64>::value ||
                    std::is_same<T, complex128>::value,
                "T must be std::complex<[float|double]>");

  // Get some useful calculations out of the way
  std::size_t ncoils = kspace_dims[ndim - 1];
  std::size_t kspace_size = std::accumulate(kspace_dims, kspace_dims + ndim, 1,
                                            std::multiplies<std::size_t>());
  std::size_t kspace_sizeN_1 = std::accumulate(
      kspace_dims, kspace_dims + ndim - 1, 1, std::multiplies<std::size_t>());
  std::size_t calib_size = std::accumulate(calib_dims, calib_dims + ndim, 1,
                                           std::multiplies<std::size_t>());
  std::size_t calib_sizeN_1 = std::accumulate(
      calib_dims, calib_dims + ndim - 1, 1, std::multiplies<std::size_t>());
  std::size_t kernel_size = std::accumulate(
      kernel_shape, kernel_shape + ndim - 1, 1, std::multiplies<std::size_t>());
  std::unique_ptr<std::size_t[]> kernel_shape2(new std::size_t[ndim - 1]);
  std::transform(kernel_shape, kernel_shape + ndim - 1, kernel_shape2.get(),
                 [](std::size_t x0) { return x0 / 2; });
  std::unique_ptr<std::size_t[]> adjs(new std::size_t[ndim - 1]);
  std::transform(kernel_shape, kernel_shape + ndim - 1, adjs.get(),
                 [](std::size_t x0) { return x0 % 2; });

  // Choose the patch extraction/apply function: 2D or 3D
  void (*extract_patch)(const std::size_t, const std::size_t,
                        const std::size_t *, const std::size_t *,
                        const std::size_t *, const std::size_t,
                        const std::size_t *, const T *, T *);
  if (ndim - 1 == 2) {
    extract_patch = &extract_patch_2d;
  } else {
    std::cerr << "NotImplementedError" << std::endl;
    return GRAPPA_STATUS::FAIL;
  }

  // Find the unique sampling patterns and save where they are
  // (Note: using std::vector<bool> as a dynamic bitset)
  auto P = std::unordered_map<std::vector<bool>, std::vector<std::size_t>>();
  auto pattern = std::vector<bool>(kernel_size);
  std::unique_ptr<T[]> patch(new T[kernel_size]);
  std::size_t coil_idx =
      0; // any coil will do for determining sampling patterns
  std::size_t ctr = kernel_shape2[0] + kernel_shape2[1] * kernel_shape[0];
  for (std::size_t idx = 0; idx < kspace_size / ncoils; ++idx) {
    // note: row-major ordering
    extract_patch(idx, coil_idx, kspace_dims, kernel_shape, kernel_shape2.get(),
                  kernel_size, adjs.get(), kspace, patch.get());
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
  zeros.clear();           // done with zeros

  // TODO(mckib2): use same nnz criteria for P as in mdgrappa

  // Get all overlapping patches from calibration data; could be quite large;
  // TODO(mckib2): block-wise regression for large datasets
  // shape: (num_patches_per_coil, kernel_size, ncoils)
  std::unique_ptr<T[]> A(new T[calib_size * kernel_size]);
  std::size_t num_patches_per_coil = calib_size / ncoils;
  for (std::size_t idx = 0; idx < num_patches_per_coil; ++idx) {
    for (std::size_t coil_idx = 0; coil_idx < ncoils; ++coil_idx) {
      extract_patch(idx, coil_idx, calib_dims, kernel_shape,
                    kernel_shape2.get(), kernel_size, adjs.get(), calib,
                    patch.get());
      for (std::size_t patch_idx = 0; patch_idx < kernel_size; ++patch_idx) {
        A[ind3to1(idx, patch_idx, coil_idx, kernel_size, ncoils)] =
            patch[patch_idx];
      }
    }
  }

  // TODO(mckib2): Train and apply weights for each pattern;
  //     it->first : sampling pattern
  //     it->second : indices of holes in kspace
  // Initialize sources, targets, and weights;
  //     potentially more space than we need for sources,
  //     but beats dynamically allocating each iteration
  std::unique_ptr<T[]> S(new T[calib_size * kernel_size]); // sources
  std::unique_ptr<T[]> Tgt(new T[calib_size]);             // targets
  std::unique_ptr<T[]> W(new T[ncoils * ncoils]);          // weights
  // std::unique_ptr<T[]> WORK(new T[]);
  for (auto it = P.cbegin(); it != P.cend(); ++it) {

    // Populate masked sources and targets
    //     Split into inner and outer loop to populate Tgt alongside S
    // TODO(mckib2): might be a more efficient way to update by comparing
    // which patch elements changed iteration to iteration and only
    // updating the ones that change
    std::size_t local_idx;
    for (std::size_t ii = 0; ii < calib_size; ++ii) {
      for (std::size_t jj = 0; jj < kernel_size; ++jj) {
        local_idx = ii + jj * calib_size;
        S[local_idx] = A[local_idx] *
                       (T)(std::abs(it->first[local_idx % kernel_size]) > 0);
      }
      Tgt[ii + ctr * calib_size] =
          A[ii + ctr * calib_size]; // TODO(mckib2): take ctr*calib_size out of
                                    // the loops?
    }

    // TODO(mckib2): solve for weights using LAPACK
    // S @ W = Tgt, solve for W
    // cgels_(
    //  'N',
    //  &M,
    //  &N,
    //  &NRHS,
    //  reinterpret_cast<float*>(S.get()),
    //  &LDA,
    //  reinterpret_cast<float*>(Tgt.get()),
    //  &LDB,
    //  reinterpret_cast<float*>(WORK.get()), int *LWORK, int *INFO)

    // Apply kernel to fill each hole
    for (const auto &idx : it->second) {
      // Fill up source
      for (std::size_t coil_idx = 0; coil_idx < ncoils; ++coil_idx) {
        extract_patch(idx, coil_idx, kspace_dims, kernel_shape,
                      kernel_shape2.get(), kernel_size, adjs.get(), kspace,
                      patch.get());
        for (std::size_t patch_idx = 0; patch_idx < kernel_size; ++patch_idx) {
          S[patch_idx + coil_idx * kernel_size] =
              patch[patch_idx]; // TODO(mckib2): move coil_idx*kernel_size
                                // outside of loops; could avoid multiply by
                                // adding kernel_size each iteration
        }
      }

      // Tgt := S @ W
      gemm(kernel_size, ncoils, ncoils, S.get(), W.get(), Tgt.get());

      // Fill the result
      for (std::size_t coil_idx = 0; coil_idx < ncoils; ++coil_idx) {
        recon[idx + kspace_sizeN_1 * coil_idx] = Tgt[coil_idx];
      }
    }
  }

  _print_coil2d(kspace_dims, recon); // take a gander at what happened
  return GRAPPA_STATUS::SUCCESS;
}

// Resolve the templates to make available for Cython:
extern "C" {
GRAPPA_STATUS
_cgrappa_complex64(const std::size_t ndim, const std::size_t *kspace_dims,
                   const std::size_t *calib_dims, const complex64 *kspace,
                   const complex64 *calib, const std::size_t *kernel_size,
                   complex64 *recon) {
  return _cgrappa(ndim, kspace_dims, calib_dims, kspace, calib, kernel_size,
                  recon);
}
}
