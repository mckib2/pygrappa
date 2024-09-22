project(
  'pygrappa',
  'c', 'cpp', 'cython',
  version: '0.26.3',
  license: 'MIT',
  meson_version: '>= 1.5.0',
  default_options: [
    'buildtype=release',
    'b_ndebug=if-release',
    'c_std=c99',
    'cpp_std=c++14',
  ],
)

cc = meson.get_compiler('c')
cpp = meson.get_compiler('cpp')
cy = meson.get_compiler('cython')
cython = find_program(cy.cmd_array()[0])
if not cy.version().version_compare('>=3.0.8')
  error('SciPy requires Cython >= 3.0.8')
endif

is_windows = host_machine.system() == 'windows'

# https://mesonbuild.com/Python-module.html
py3 = import('python').find_installation(pure: false)
py3_dep = py3.dependency()

# NumPy include directory - needed in all submodules
incdir_numpy = run_command(py3,
    [
      '-c',
      '''import os
#os.chdir(os.path.join("..", "tools"))
import numpy as np
try:
    incdir = os.path.relpath(np.get_include())
except Exception:
    incdir = np.get_include()
print(incdir)
    '''
    ],
    check: true
).stdout().strip()
message(incdir_numpy)

inc_np = include_directories(incdir_numpy)
numpy_nodepr_api = '-DNPY_NO_DEPRECATED_API=NPY_1_9_API_VERSION'

cython_args = ['-3', '--fast-fail', '--output-file', '@OUTPUT@', '--include-dir', '@BUILD_ROOT@', '@INPUT@']
if cy.version().version_compare('>=3.1.0')
  cython_args += ['-Xfreethreading_compatible=True']
endif
cython_cplus_args = ['--cplus'] + cython_args

cython_gen_cpp = generator(cython,
  arguments : cython_cplus_args,
  output : '@BASENAME@.cpp',
  depends : [])

# C warning flags
Wno_maybe_uninitialized = cc.get_supported_arguments('-Wno-maybe-uninitialized')
Wno_discarded_qualifiers = cc.get_supported_arguments('-Wno-discarded-qualifiers')
Wno_empty_body = cc.get_supported_arguments('-Wno-empty-body')
Wno_implicit_function_declaration = cc.get_supported_arguments('-Wno-implicit-function-declaration')
Wno_parentheses = cc.get_supported_arguments('-Wno-parentheses')
Wno_switch = cc.get_supported_arguments('-Wno-switch')
Wno_unused_label = cc.get_supported_arguments('-Wno-unused-label')
Wno_unused_variable = cc.get_supported_arguments('-Wno-unused-variable')

# C++ warning flags
_cpp_Wno_cpp = cpp.get_supported_arguments('-Wno-cpp')
_cpp_Wno_deprecated_declarations = cpp.get_supported_arguments('-Wno-deprecated-declarations')
_cpp_Wno_class_memaccess = cpp.get_supported_arguments('-Wno-class-memaccess')
_cpp_Wno_format_truncation = cpp.get_supported_arguments('-Wno-format-truncation')
_cpp_Wno_non_virtual_dtor = cpp.get_supported_arguments('-Wno-non-virtual-dtor')
_cpp_Wno_sign_compare = cpp.get_supported_arguments('-Wno-sign-compare')
_cpp_Wno_switch = cpp.get_supported_arguments('-Wno-switch')
_cpp_Wno_terminate = cpp.get_supported_arguments('-Wno-terminate')
_cpp_Wno_unused_but_set_variable = cpp.get_supported_arguments('-Wno-unused-but-set-variable')
_cpp_Wno_unused_function = cpp.get_supported_arguments('-Wno-unused-function')
_cpp_Wno_unused_local_typedefs = cpp.get_supported_arguments('-Wno-unused-local-typedefs')
_cpp_Wno_unused_variable = cpp.get_supported_arguments('-Wno-unused-variable')
_cpp_Wno_int_in_bool_context = cpp.get_supported_arguments('-Wno-int-in-bool-context')

# Deal with M_PI & friends; add `use_math_defines` to c_args or cpp_args
# Cython doesn't always get this right itself (see, e.g., gh-16800), so
# explicitly add the define as a compiler flag for Cython-generated code.
if is_windows
  use_math_defines = ['-D_USE_MATH_DEFINES']
else
  use_math_defines = []
endif

# Suppress warning for deprecated Numpy API.
# (Suppress warning messages emitted by #warning directives).
# Replace with numpy_nodepr_api after Cython 3.0 is out
cython_c_args = [_cpp_Wno_cpp, use_math_defines]
cython_cpp_args = cython_c_args

subdir('pygrappa')
