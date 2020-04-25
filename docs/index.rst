.. index:

========
pygrappa
========

About
=====

GRAPPA is a popular parallel imaging reconstruction algorithm.
Unfortunately there aren't a lot of easy to use Python
implementations of it or its many variants available, so I decided to
release this simple package.

There are also a couple reference SENSE-like implementations that
have made their way into the package.  This is to be expected -- a
lot of later parallel imaging algorithms have hints of both GRAPPA-
and SENSE-like inspirations.

Installation
------------

.. toctree::
   :hidden:
   :maxdepth: 1

   installation

.. code-block:: python

   pip install pygrappa

There are C/C++ extensions to be compiled, so you will need a compiler
that supports either the C++11 or C++14 standard.
See :doc:`installation` for more instructions.

API Reference
-------------

.. toctree::
   :hidden:
   :maxdepth: 1

   pygrappa

The exact API of all functions and classes, as given by the docstrings. The API
documents expected types and allowed features for all functions, and all
parameters available for the algorithms.

A full catalog can be found in the :doc:`pygrappa` page.

Usage
=====

.. toctree::
   :hidden:

   usage

See the :doc:`usage` page.  Also see the `examples` module.
It has several scripts showing basic usage.  Docstrings are also a
great resource -- check them out for all possible arguments and
usage info.

You can run examples from the command line by calling them like this:

.. code-block:: bash

    python -m pygrappa.examples.[example-name]

    # For example, if I wanted to try out TGRAPPA:
    python -m pygrappa.examples.basic_tgrappa
