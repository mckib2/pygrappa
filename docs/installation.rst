.. installation:

Installation
============

.. toctree::
   :hidden:
   :maxdepth: 1

   windows_installation

This package is developed in Ubuntu 18.04 using Python 3.6.8.  That's
not to say it won't work on other things.  You should submit an issue
when it doesn't work like it says it should.  The whole idea was to
have an easy to use, pip-install-able GRAPPA module, so let's try to
do that.

In general, it's a good idea to work inside virtual environments.  I
create and activate mine like this:

.. code-block:: bash

    python3 -m venv /venvs/pygrappa
    source /venvs/pygrappa/bin/activate

More information can be found in the `venv documentation <https://docs.python.org/3/library/venv.html>`_.

Installation under a Unix-based platform should then be as easy as:

.. code-block:: bash

    pip install pygrappa

You will need a C/C++ compiler that supports the C++14 standard.
See :doc:`windows_installation` for more info on installing under Windows.
