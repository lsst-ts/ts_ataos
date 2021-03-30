.. py:currentmodule:: lsst.ts.ataos

.. _lsst.ts.ataos.developer_guide:

###############
Developer Guide
###############

The ATAOS CSC is implemented using `ts_salobj <https://github.com/lsst-ts/ts_salobj>`_.

.. _lsst.ts.ataos-api:

API
===

The primary class is:

* `ATAOS`: the CSC.

.. automodapi:: lsst.ts.ataos
   :no-main-docstr:
   :no-inheritance-diagram:

Build and Test
==============

This is a pure python package. There is nothing to build except the documentation.

.. code-block:: bash

    make_idl_files.py ATAOS ATCamera ATHexapod ATMCS ATPneumatics
    setup -r .
    pytest -v  # to run tests
    package-docs clean; package-docs build  # to build the documentation

Contributing
============

``ts_ataos`` is developed at https://github.com/lsst-ts/ts_ataos.
You can find Jira issues for this package using `labels=ts_ataos <https://jira.lsstcorp.org/issues/?jql=project%20%3D%20DM%20AND%20labels%20%20%3D%20ts_mtrotator>`_..
