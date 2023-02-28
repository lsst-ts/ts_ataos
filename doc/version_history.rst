.. py:currentmodule:: lsst.ts.ataos

.. _lsst.ts.ataos.version_history:

###############
Version History
###############

v1.10.5
-------

Changes:

  * Fix bug in ``disableCorrections``.

v1.10.4
-------

Changes:

  * Fix bug where if statement in ATSpectrograph correction loop could always be true if self.atspectrograph_corrections_required was less than zero. 


v1.10.3
-------

Changes:

  * Update conda recipe to properly handle builds for different versions of Python and to use ``pytest`` instead of ``py.test``.
  * Fix bug in ATAOS referencing ``self.self`` and update subsequent warning message to log the appropriate information.
  * Update pyproject.toml to include isort configuration.
  * Update pre-commit hooks configuration.
  * Run isort in the entire package to organize import statements.

v1.10.2
-------

Changes:

  * Increased timeout between opening valves and commanding a new air pressure to prevent a current race condition in the ATPneumatics.

v1.10.1
-------

Changes:

  * Rename ataos_csc script to run_ataos_csc script.
    Note that there was a typo in the method name, which gets superseeded by the rename.
  * Add gitigore file.

v1.10.0
------

Changes:

  * Corrected issue of detailedState handling when correction values are below the correction tolerance.
  * Switch to pyproject.toml.

v1.9.0
------

Changes:

  * Update CSC to salobj 7.
  * Add type annotations and enable mypy checking.
  * Modernize CSC unit tests.

v1.8.1
------

Changes:

  * Fix issue with sending resetOffset that would always reset the focus offset.

v1.8.0
------

Changes:

  * Update configuration schema to remove schema file and use module instead.
  * Add hexapod sensitivity matrix to the model.
    This allows users to specify linear cross-terms between different hexapod axis.
    For instance, if one wants to apply a tip/tilt correction when doing x/y translation, it is possible to specify the cross term in the sensitivity matrix.
  * Add feature that allow users to specify the valid range for LUTs.
    Any data beyond the limits return the value of the LUT in the limit.
    For instance, if minimum valid elevation for hexapod LUT is 30 and a correction is requested for elevation 20, it returns the value in elevation 30.
  * In `Model`, implement new feature in dealing with m1 correction when it is below the lower limit. Instead of returning a fixed correction, it will return a value that linearly approaches zero as elevation goes to zero.
  * Fix `Model.get_correction_m1` docstring, which said elevation parameter was ignored, where it should be azimuth.
  * In `ATAOS.do_resetOffset`, fix issue where the command would be rejected when trying to reset a correction that is not hexapod and hexapod correction is not enabled.
  * Remove leading white space in `ATAOS.check_atspectrograph` docstring.
  * Refactor model class tests. Adds tests for sensitivity matrix and for get_lut_elevation.
  * Add property and setter for hexapod sensitivity matrix in `Model` class. This was missing from the original implementation which would cause the CSC to be unable to set the v
alue.
  * Fix docstrings of `get_lut_elevation` method in the `Model` class.
  * Fix typo in description field of configuration schema.
  * Add a new feature to limit the minimum pressure on m1.
    If the computed value is below the limit, the assigned value will be the one specified by the limit.
  * Fix issue in `end_disable`, where it would check if it needed to lower the mirrors after disabling all corrections, so it was never lowering the mirror.
    It now stores the values before disabling the corrections and use these to determine if it needs to lower m1 and m2.
  * In `begin_start`, check that the user provided a non-empty `settingsToApply` and raise an exception (this rejecting the command) if so.
    This is preferable to having a "non-default" configuration as the user would be presented with a cryptic "schema validation" error message.
    The error provides sufficient information for the user to understand what went wrong and how to correct it
  * Send ack in progress when executing start command.

v1.7.4
------

Changes:

  * Fix bug in focus offset accounting

v1.7.3
------

Changes:

  * Reformat code using black 20.
  * Enable pytest-black.
  * Pin version of ts-conda-build to 0.3 in conda recipe.
  * Cleanup documentation.
