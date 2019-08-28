Windows 10 Installation
=======================

If you are using Windows, then, first of all: sorry.  This is not
ideal, but I understand that it might not be your fault.  I will
assume you are trying to get pygrappa installed on Windows 10. I will
further assume that you are using Python 3.7 64-bit build.  We will
need a C++ compiler to install pygrappa, so officially you should
have "Microsoft Visual C++ Build Tools" installed. I haven't tried
this, but it should work with VS build tools installed.

However, if you are not able to install the build tools, we can do it
using the MinGW compiler instead.  It'll be a little more involved
than a simple `pip install`, but that's what you get for choosing
Windows.

Steps:

- | Download 64-bit fork of MinGW from
  | https://sourceforge.net/projects/mingw-w64/
- | Follow this guide:
  | https://github.com/orlp/dev-on-windows/wiki/Installing-GCC--&-MSYS2
- Now you should be able to use gcc/g++/etc. from CMD-line
- | Modify cygwinccompiler.py similar to
  | https://github.com/tgalal/yowsup/issues/2494#issuecomment-388439162
  | but using the version number `1916`:

.. code-block:: python

    def get_msvcr():
        """Include the appropriate MSVC runtime library if Python
        was built with MSVC 7.0 or later.
        """
        msc_pos = sys.version.find('MSC v.')
        if msc_pos != -1:
            msc_ver = sys.version[msc_pos+6:msc_pos+10]
            if msc_ver == '1300':
                # MSVC 7.0
                return ['msvcr70']
            elif msc_ver == '1310':
                # MSVC 7.1
                return ['msvcr71']
            elif msc_ver == '1400':
                # VS2005 / MSVC 8.0
                return ['msvcr80']
            elif msc_ver == '1500':
                # VS2008 / MSVC 9.0
                return ['msvcr90']
            elif msc_ver == '1600':
                # VS2010 / MSVC 10.0
                return ['msvcr100']
            elif msc_ver == '1916': # <- ADD THIS CONDITION
                # Visual Studio 2015 / Visual C++ 14.0
                return ['vcruntime140']
            else:
                raise ValueError(
                    "Unknown MS Compiler version %s " % msc_ver)

- now run the command:

.. code-block:: bash

    pip install --global-option build_ext --global-option \
        --compiler=mingw32 --global-option -DMS_WIN64 pygrappa

Hopefully this works for you.  Refer to
https://github.com/mckib2/pygrappa/issues/17 for a more detailed
discussion.
