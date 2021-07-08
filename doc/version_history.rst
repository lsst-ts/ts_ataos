.. py:currentmodule:: lsst.ts.ataos

.. _lsst.ts.ataos.version_history:

###############
Version History
###############

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
