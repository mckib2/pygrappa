pglib = py3.extension_module('train_kernels',
  cython_gen_cpp.process('train_kernels.pyx'),
  cpp_args: cython_cpp_args,
  include_directories: [inc_np],
  install: true,
  subdir: 'pygrappa'
)

pglib = py3.extension_module('cgrappa',
  [
    cython_gen_cpp.process('cgrappa.pyx'),
    'get_sampling_patterns.cpp'
  ],
  cpp_args: cython_cpp_args,
  include_directories: [inc_np],
  install: true,
  subdir: 'pygrappa'
)

pglib = py3.extension_module('grog_powers',
  [
    cython_gen_cpp.process('grog_powers.pyx'),
    '_grog_powers_template.cpp'
  ],
  cpp_args: cython_cpp_args,
  include_directories: [inc_np],
  install: true,
  subdir: 'pygrappa'
)

pglib = py3.extension_module('grog_gridding',
  cython_gen_cpp.process('grog_gridding.pyx'),
  cpp_args: cython_cpp_args,
  include_directories: [inc_np],
  install: true,
  subdir: 'pygrappa'
)
