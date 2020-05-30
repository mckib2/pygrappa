.. pygrappa:

.. note::

   The upcoming `1.0.0` release will make changes to the API and simplify
   the interface considerably.  The plans are to collect all GRAPPA-like
   methods and SENSE-like methods in their own interfaces:

   .. code-block:: python

      pygrappa.grappa(
          kspace, calib=None, kernel_size=None,
          method='grappa', coil_axis=-1, options=None)
	  pygrappa.sense(kspace, sens, coil_axis=-1, options)

   The `method` parameter will allow the `grappa` interface to call the
   existing methods such as `tgrappa`, `mdgrappa`, etc. under the hood.
   The dictionary `options` can be used to pass in method-specific
   parameters. The SENSE interface will behave similarly.

   The gridding interface is still an open question.

   Progress on the `1.0.0` release can be found
   `here <https://github.com/mckib2/pygrappa/milestone/1>`_


API Reference
=============

.. toctree::
   :maxdepth: 1

   grappa
   cgrappa
   mdgrappa
   igrappa
   hpgrappa
   seggrappa
   tgrappa
   slicegrappa
   splitslicegrappa
   grappaop
   radialgrappaop
   ttgrappa
   pars
   grog
   nlgrappa_matlab
   gfactor
   sense1d
   cgsense
