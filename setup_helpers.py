'''Shamelessly borrow code from scipy's setup.py.

Use to run pre-build hooks to inject compiler directives
for c++11/14.

See: https://github.com/scipy/scipy/blob/master/setup.py
'''

import sys
import os
import sysconfig


def get_build_ext_override():
    """
    Custom build_ext command to tweak extension building.
    """
    from numpy.distutils.command.build_ext import build_ext as old_build_ext

    class build_ext(old_build_ext):
        def finalize_options(self):
            super().finalize_options()

            # Disable distutils parallel build, due to race conditions
            # in numpy.distutils (Numpy issue gh-15957)
            if self.parallel:
                print("NOTE: -j build option not supportd. Set NPY_NUM_BUILD_JOBS=4 "
                      "for parallel build.")
            self.parallel = None

        def build_extension(self, ext):
            # When compiling with GNU compilers, use a version script to
            # hide symbols during linking.
            if self.__is_using_gnu_linker(ext):
                export_symbols = self.get_export_symbols(ext)
                text = '{global: %s; local: *; };' % (';'.join(export_symbols),)

                script_fn = os.path.join(self.build_temp, 'link-version-{}.map'.format(ext.name))
                with open(script_fn, 'w') as f:
                    f.write(text)
                    # line below fixes gh-8680
                    ext.extra_link_args = [arg for arg in ext.extra_link_args if "version-script" not in arg]
                    ext.extra_link_args.append('-Wl,--version-script=' + script_fn)

            # Allow late configuration
            hooks = getattr(ext, '_pre_build_hook', ())
            _run_pre_build_hooks(hooks, (self, ext))

            old_build_ext.build_extension(self, ext)

        def __is_using_gnu_linker(self, ext):
            if not sys.platform.startswith('linux'):
                return False

            # Fortran compilation with gfortran uses it also for
            # linking. For the C compiler, we detect gcc in a similar
            # way as distutils does it in
            # UnixCCompiler.runtime_library_dir_option
            if ext.language == 'f90':
                is_gcc = (self._f90_compiler.compiler_type in ('gnu', 'gnu95'))
            elif ext.language == 'f77':
                is_gcc = (self._f77_compiler.compiler_type in ('gnu', 'gnu95'))
            else:
                is_gcc = False
                if self.compiler.compiler_type == 'unix':
                    cc = sysconfig.get_config_var("CC")
                    if not cc:
                        cc = ""
                    compiler_name = os.path.basename(cc.split(" ")[0])
                    is_gcc = "gcc" in compiler_name or "g++" in compiler_name
            return is_gcc and sysconfig.get_config_var('GNULD') == 'yes'

    return build_ext


def _run_pre_build_hooks(hooks, args):
    """Call a sequence of pre-build hooks, if any"""
    if hooks is None:
        hooks = ()
    elif not hasattr(hooks, '__iter__'):
        hooks = (hooks,)
    for hook in hooks:
        hook(*args)
